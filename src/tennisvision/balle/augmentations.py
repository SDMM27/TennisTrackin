"""Augmentations d'entraînement du détecteur de balle, appliquées à un triplet d'images.

Tout se passe dans l'espace du modèle (images RVB uint8 de 288x512, positions en pixels de cet
espace). La même transformation est appliquée aux 3 images d'un triplet ; les augmentations
géométriques mettent à jour les positions de la balle (et la visibilité si elle sort du cadre), les
autres les laissent inchangées. Le recadrage reprend celui de WASB-SBDT
(`RandomCropping` de `src/dataloaders/seq_transforms.py`, MIT, Copyright (c) 2023 NTT
Communications Corporation) : agrandissement de `1 + max_rescale` puis recadrage aléatoire.

Chaque augmentation est une fonction `(echantillon, rng, params) -> Echantillon` qui s'applique
toujours ; `Augmenteur` tire la probabilité de chacune. Avec un `numpy.random.Generator` initialisé
par une graine, le résultat est déterministe.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ParametresAugmentation:
    """Probabilité (`p_*`) et force de chaque augmentation ; `0.0` désactive l'augmentation."""

    p_retournement: float = 0.5
    p_recadrage: float = 0.5
    recadrage_max: float = 0.125  # agrandissement maximal (WASB : 1/8)
    p_distracteurs: float = 0.3
    distracteurs_max: int = 3
    distracteurs_rayon: tuple[float, float] = (2.0, 5.0)  # px, espace modèle
    distracteurs_distance_min: float = 30.0  # px minimum entre un leurre et la vraie balle
    p_grille: float = 0.15
    grille_pas: tuple[float, float] = (6.0, 30.0)  # px
    grille_opacite: tuple[float, float] = (0.25, 0.7)
    p_flou: float = 0.3
    flou_longueur_max: int = 15  # px
    p_photometrie: float = 0.5
    photometrie_force: float = 0.2  # luminosité (±) et contraste (1 ± force)
    gamma_force: float = 0.3  # gamma = exp(±force)
    p_jpeg: float = 0.3
    jpeg_qualite: tuple[int, int] = (30, 95)

    def desactivees(self) -> "ParametresAugmentation":
        """Mêmes réglages de force, toutes les probabilités à zéro (validation)."""
        return ParametresAugmentation(
            **{
                nom: (0.0 if nom.startswith("p_") else valeur)
                for nom, valeur in self.__dict__.items()
            }
        )


@dataclass(frozen=True)
class Echantillon:
    """Un triplet : 3 images RVB uint8 (H, W, 3), positions (3, 2) en (x, y), visibilité (3,)."""

    frames: tuple[np.ndarray, ...]
    points: np.ndarray
    visibles: np.ndarray

    def avec(self, frames=None, points=None, visibles=None) -> "Echantillon":
        return Echantillon(
            tuple(self.frames if frames is None else frames),
            self.points if points is None else points,
            self.visibles if visibles is None else visibles,
        )


def _dans_le_cadre(points: np.ndarray, hauteur: int, largeur: int) -> np.ndarray:
    return (
        (points[:, 0] >= 0)
        & (points[:, 0] < largeur)
        & (points[:, 1] >= 0)
        & (points[:, 1] < hauteur)
    )


def retourner_horizontalement(ech: Echantillon, rng, params) -> Echantillon:
    """Miroir gauche-droite : x -> (largeur - 1) - x (coordonnées en indices de pixels)."""
    largeur = ech.frames[0].shape[1]
    points = ech.points.copy()
    points[:, 0] = (largeur - 1) - points[:, 0]
    return ech.avec([f[:, ::-1].copy() for f in ech.frames], points)


def recadrer(ech: Echantillon, rng, params) -> Echantillon:
    """Agrandit de `1 + recadrage_max` puis recadre à la taille d'origine (décalage aléatoire)."""
    hauteur, largeur = ech.frames[0].shape[:2]
    echelle = 1.0 + params.recadrage_max
    nouvelle_l, nouvelle_h = int(largeur * echelle), int(hauteur * echelle)
    x1 = round(rng.uniform(0, nouvelle_l - largeur))
    y1 = round(rng.uniform(0, nouvelle_h - hauteur))
    frames = []
    for f in ech.frames:
        grande = cv2.resize(f, (nouvelle_l, nouvelle_h), interpolation=cv2.INTER_LINEAR)
        frames.append(np.ascontiguousarray(grande[y1 : y1 + hauteur, x1 : x1 + largeur]))
    fx, fy = nouvelle_l / largeur, nouvelle_h / hauteur
    points = ech.points.copy()
    points[:, 0] = (points[:, 0] + 0.5) * fx - 0.5 - x1  # convention des centres de pixels (cv2)
    points[:, 1] = (points[:, 1] + 0.5) * fy - 0.5 - y1
    visibles = ech.visibles & _dans_le_cadre(points, hauteur, largeur)
    return ech.avec(frames, points, visibles)


def _noyau_de_bouge(longueur: int, angle_deg: float) -> np.ndarray:
    longueur |= 1  # impair, pour un noyau centré
    noyau = np.zeros((longueur, longueur), np.float32)
    c = (longueur - 1) / 2.0
    dx, dy = np.cos(np.radians(angle_deg)) * c, np.sin(np.radians(angle_deg)) * c
    cv2.line(
        noyau,
        (round(c - dx), round(c - dy)),
        (round(c + dx), round(c + dy)),
        1.0,
        1,
    )
    return noyau / noyau.sum()


def flou_de_bouge(ech: Echantillon, rng, params) -> Echantillon:
    """Flou de mouvement le long d'une direction aléatoire (même noyau sur les 3 images)."""
    longueur = int(rng.integers(3, max(4, params.flou_longueur_max + 1)))
    noyau = _noyau_de_bouge(longueur, float(rng.uniform(0.0, 180.0)))
    frames = [cv2.filter2D(f, -1, noyau, borderType=cv2.BORDER_REFLECT) for f in ech.frames]
    return ech.avec(frames)


def recompression_jpeg(ech: Echantillon, rng, params) -> Echantillon:
    """Recompression JPEG à une qualité aléatoire (même qualité sur les 3 images)."""
    bas, haut = params.jpeg_qualite
    qualite = int(rng.integers(bas, haut + 1))
    frames = []
    for f in ech.frames:
        ok, tampon = cv2.imencode(
            ".jpg", np.ascontiguousarray(f[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, qualite]
        )
        if not ok:
            raise RuntimeError("encodage JPEG impossible")
        frames.append(np.ascontiguousarray(cv2.imdecode(tampon, cv2.IMREAD_COLOR)[:, :, ::-1]))
    return ech.avec(frames)


def photometrie(ech: Echantillon, rng, params) -> Echantillon:
    """Gamma, contraste et luminosité aléatoires (même courbe sur les 3 images)."""
    force = params.photometrie_force
    gamma = float(np.exp(rng.uniform(-params.gamma_force, params.gamma_force)))
    contraste = 1.0 + float(rng.uniform(-force, force))
    luminosite = float(rng.uniform(-force, force))
    v = np.arange(256, dtype=np.float32) / 255.0
    v = (v**gamma - 0.5) * contraste + 0.5 + luminosite
    table = np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return ech.avec([cv2.LUT(f, table) for f in ech.frames])


def grille(ech: Echantillon, rng, params) -> Echantillon:
    """Grillage ou filet : fines lignes sombres de deux directions, pas et opacité aléatoires.

    La grille est fixe d'une image à l'autre (le grillage ne bouge pas).
    """
    hauteur, largeur = ech.frames[0].shape[:2]
    diagonale = float(np.hypot(hauteur, largeur))
    centre = np.array([largeur / 2.0, hauteur / 2.0])
    masque = np.zeros((hauteur, largeur), np.uint8)
    angle = float(rng.uniform(0.0, 180.0))
    for decalage_angle in (0.0, float(rng.uniform(60.0, 120.0))):
        a = np.radians(angle + decalage_angle)
        direction = np.array([np.cos(a), np.sin(a)])
        normale = np.array([-direction[1], direction[0]])
        pas = float(rng.uniform(*params.grille_pas))
        t = -diagonale + float(rng.uniform(0.0, pas))
        while t < diagonale:
            p1 = centre + normale * t - direction * diagonale
            p2 = centre + normale * t + direction * diagonale
            cv2.line(
                masque,
                (round(p1[0]), round(p1[1])),
                (round(p2[0]), round(p2[1])),
                255,
                1,
                cv2.LINE_AA,
            )
            t += pas
    alpha = (masque.astype(np.float32) / 255.0 * float(rng.uniform(*params.grille_opacite)))[
        :, :, None
    ]
    sombre = float(rng.uniform(0.0, 40.0))
    frames = [
        np.clip(f * (1.0 - alpha) + sombre * alpha + 0.5, 0, 255).astype(np.uint8)
        for f in ech.frames
    ]
    return ech.avec(frames)


def distracteurs(ech: Echantillon, rng, params) -> Echantillon:
    """Colle des leurres (petits disques jaune-vert flous) loin de la vraie balle.

    Fixes ou en mouvement rectiligne d'une image à l'autre ; ils n'entrent jamais dans la cible
    (positions et visibilité inchangées) : le réseau apprend à ignorer les balles des courts
    voisins.
    """
    hauteur, largeur = ech.frames[0].shape[:2]
    vraies = ech.points[ech.visibles]
    frames = [f.astype(np.float32) for f in ech.frames]
    for _ in range(int(rng.integers(1, max(2, params.distracteurs_max + 1)))):
        rayon = float(rng.uniform(*params.distracteurs_rayon))
        flou = float(rng.uniform(0.6, 1.3))
        couleur = np.clip(np.array([205.0, 235.0, 50.0]) + rng.uniform(-20, 20, 3), 0, 255)
        vitesse = rng.uniform(-12, 12, 2) if rng.random() < 0.5 else np.zeros(2)
        depart = None
        for _essai in range(20):
            candidat = rng.uniform([8, 8], [largeur - 8, hauteur - 8])
            trajet = [candidat + vitesse * i for i in range(len(frames))]
            if all(
                0 <= p[0] < largeur
                and 0 <= p[1] < hauteur
                and (
                    len(vraies) == 0
                    or np.min(np.hypot(*(vraies - p).T)) >= params.distracteurs_distance_min
                )
                for p in trajet
            ):
                depart = trajet
                break
        if depart is None:
            continue
        for i, p in enumerate(depart):
            alpha = np.zeros((hauteur, largeur), np.float32)
            cv2.circle(alpha, (round(p[0]), round(p[1])), max(1, round(rayon)), 1.0, -1)
            alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), flou) * 1.4, 0, 1)[:, :, None]
            frames[i] = frames[i] * (1.0 - alpha) + couleur * alpha
    return ech.avec([np.clip(f + 0.5, 0, 255).astype(np.uint8) for f in frames])


# Ordre d'application : géométrie, éléments de scène, dégradations de la prise de vue, JPEG en dernier.
ETAPES = (
    ("p_retournement", retourner_horizontalement),
    ("p_recadrage", recadrer),
    ("p_distracteurs", distracteurs),
    ("p_grille", grille),
    ("p_flou", flou_de_bouge),
    ("p_photometrie", photometrie),
    ("p_jpeg", recompression_jpeg),
)


class Augmenteur:
    """Applique chaque augmentation avec sa probabilité ; déterministe pour un `rng` donné."""

    def __init__(self, params: ParametresAugmentation):
        self.params = params

    def __call__(self, ech: Echantillon, rng: np.random.Generator) -> Echantillon:
        for nom_probabilite, fonction in ETAPES:
            if rng.random() < getattr(self.params, nom_probabilite):
                ech = fonction(ech, rng, self.params)
        return ech
