"""Détection de la balle par cartes de chaleur sur 3 images (WASB / TrackNetV2), en flux.

Prétraitement et post-traitement identiques à ceux du dépôt WASB-SBDT à l'inférence
(https://github.com/nttcom/WASB-SBDT, licence MIT, Copyright (c) 2023 NTT Communications
Corporation) : `get_affine_transform` / `get_transform` (centre = milieu de l'image, échelle =
max(h, w)), `cv2.INTER_LINEAR`, normalisation ImageNet, puis `TracknetV2Postprocessor` avec
`blob_det_method: concomp` et `use_hm_weight: True`.

Les images sont poussées dans l'ordre ; chaque image est prédite exactement une fois, par triplets
disjoints (WASB `step: 3`) : la carte i d'un triplet est celle de l'image i. À 60 fps, les triplets
sont formés d'images espacées de `pas_temporel` pour retrouver la cadence d'entraînement (~30 fps).
"""

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from tennisvision.balle.reseaux import HAUTEUR_ENTREE, LARGEUR_ENTREE, NB_IMAGES

MOYENNE_IMAGENET = (0.485, 0.456, 0.406)
ECART_TYPE_IMAGENET = (0.229, 0.224, 0.225)
FPS_ENTRAINEMENT = 30.0


@dataclass(frozen=True)
class Candidat:
    """Un candidat balle : centre en pixels de l'image d'origine, score (pic de la carte) dans [0, 1]."""

    frame_idx: int
    x: float
    y: float
    score: float


def _direction(point: list[float], angle: float) -> list[float]:
    sn, cs = np.sin(angle), np.cos(angle)
    return [point[0] * cs - point[1] * sn, point[0] * sn + point[1] * cs]


def _troisieme_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return b + np.array([-d[1], d[0]], dtype=np.float32)


def matrice_affine(
    centre: np.ndarray,
    echelle: float,
    taille_sortie: tuple[int, int],
    inverse: bool = False,
) -> np.ndarray:
    """Transformation affine 2x3 de WASB (`get_affine_transform`, sans rotation ni décalage).

    `taille_sortie` est (largeur, hauteur). `inverse=True` donne le passage espace modèle -> image.
    """
    dst_w, dst_h = taille_sortie
    src_dir = _direction([0, echelle * -0.5], 0.0)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = centre
    src[1, :] = centre + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    src[2:, :] = _troisieme_point(src[0, :], src[1, :])
    dst[2:, :] = _troisieme_point(dst[0, :], dst[1, :])
    if inverse:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def transformations(hauteur: int, largeur: int) -> tuple[np.ndarray, np.ndarray]:
    """(image -> modèle, modèle -> image) pour une image `hauteur` x `largeur` (WASB `get_transform`)."""
    centre = np.array([largeur / 2.0, hauteur / 2.0], dtype=np.float32)
    echelle = float(max(hauteur, largeur))
    taille = (LARGEUR_ENTREE, HAUTEUR_ENTREE)
    return (
        matrice_affine(centre, echelle, taille),
        matrice_affine(centre, echelle, taille, inverse=True),
    )


def appliquer_affine(point: tuple[float, float], matrice: np.ndarray) -> tuple[float, float]:
    """Applique une matrice affine 2x3 à un point (x, y)."""
    x, y = matrice @ np.array([point[0], point[1], 1.0])
    return float(x), float(y)


def extraire_candidats(
    carte: np.ndarray, inverse: np.ndarray, seuil: float, frame_idx: int
) -> list[Candidat]:
    """Blobs d'une carte de chaleur (probabilités) : composantes connexes au-dessus de `seuil`.

    Centre = barycentre pondéré par la carte (`use_hm_weight`), ramené aux pixels de l'image
    d'origine par `inverse` ; score = valeur maximale de la carte dans le blob.
    """
    if not float(carte.max()) > seuil:
        return []
    binaire = (carte > seuil).astype(np.uint8)
    n_etiquettes, etiquettes = cv2.connectedComponents(binaire)
    if n_etiquettes <= 1:
        return []
    ys, xs = np.nonzero(binaire)
    poids = carte[ys, xs].astype(np.float64)
    ids = etiquettes[ys, xs]
    somme = np.bincount(ids, weights=poids, minlength=n_etiquettes)
    somme_x = np.bincount(ids, weights=poids * xs, minlength=n_etiquettes)
    somme_y = np.bincount(ids, weights=poids * ys, minlength=n_etiquettes)
    pics = np.zeros(n_etiquettes)
    np.maximum.at(pics, ids, poids)
    candidats = []
    for m in range(1, n_etiquettes):
        x, y = appliquer_affine((somme_x[m] / somme[m], somme_y[m] / somme[m]), inverse)
        candidats.append(Candidat(frame_idx, x, y, float(pics[m])))
    return candidats


def resoudre_device(device: str | None) -> str:
    """`None` -> cuda si disponible, sinon cpu ; « 0 » -> `cuda:0` (comme `--device` du CLI)."""
    import torch

    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return f"cuda:{device}" if device.isdigit() else device


@dataclass
class _Image:
    """Image réduite à la taille du modèle (RVB, uint8) et son passage inverse vers l'image."""

    frame_idx: int
    pixels: np.ndarray
    inverse: np.ndarray


class DetecteurBalle:
    """Détecteur de balle en flux : `pousser` chaque image dans l'ordre, puis `terminer`.

    `reseau` permet d'injecter un module déjà construit (tests) ; sinon `modele` est chargé depuis
    le registre (ou un chemin de fichier, avec `architecture`).
    """

    def __init__(
        self,
        modele: str = "wasb_tennis",
        device: str | None = None,
        seuil_score: float = 0.5,
        taille_lot: int = 8,
        fps: float | None = None,
        architecture: str | None = None,
        reseau=None,
    ):
        import torch

        from tennisvision.balle.reseaux import charger_reseau

        self.device = resoudre_device(device)
        self.seuil_score = seuil_score
        self.taille_lot = max(1, taille_lot)
        self.pas_temporel = max(1, round(fps / FPS_ENTRAINEMENT)) if fps else 1
        if reseau is None:
            self._reseau, self.modele_decrit = charger_reseau(modele, self.device, architecture)
        else:
            self._reseau = reseau.to(self.device).eval()
            self.modele_decrit = {"nom": modele, "version": None, "sha256": None}
        self._moyenne = torch.tensor(MOYENNE_IMAGENET, device=self.device).view(1, 1, 3, 1, 1)
        self._ecart = torch.tensor(ECART_TYPE_IMAGENET, device=self.device).view(1, 1, 3, 1, 1)
        self._transformations: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._compteur = 0
        # Une file d'images en attente par sous-séquence (images t, t+k, t+2k, ...).
        self._partielles: list[list[_Image]] = [[] for _ in range(self.pas_temporel)]
        self._triplets: list[list[_Image]] = []  # triplets complets, en attente de lot
        self._ordre: deque[int] = deque()  # images poussées pas encore restituées
        self._resultats: dict[int, list[Candidat]] = {}

    def _preparer(self, frame_idx: int, image: np.ndarray) -> _Image:
        forme = image.shape[:2]
        if forme not in self._transformations:
            self._transformations[forme] = transformations(*forme)
        aller, retour = self._transformations[forme]
        reduite = cv2.warpAffine(
            image, aller, (LARGEUR_ENTREE, HAUTEUR_ENTREE), flags=cv2.INTER_LINEAR
        )
        return _Image(frame_idx, cv2.cvtColor(reduite, cv2.COLOR_BGR2RGB), retour)

    def _inferer(self, lot: list[list[_Image]]) -> np.ndarray:
        """Cartes de probabilités (B, 3, H, W) pour un lot de triplets."""
        import torch

        brut = np.stack([[img.pixels for img in triplet] for triplet in lot])  # (B,3,H,W,3)
        x = torch.from_numpy(brut).to(self.device).permute(0, 1, 4, 2, 3).float() / 255.0
        x = ((x - self._moyenne) / self._ecart).flatten(1, 2)  # (B, 9, H, W)
        with torch.inference_mode():
            return torch.sigmoid(self._reseau(x)).cpu().numpy()

    def _traiter(self, lot: list[list[_Image]], reels: list[int]) -> None:
        """Infère un lot ; `reels[i]` = nombre d'images non dupliquées du triplet i."""
        cartes = self._inferer(lot)
        for triplet, n_reels, cartes_triplet in zip(lot, reels, cartes, strict=True):
            for image, carte in zip(triplet[:n_reels], cartes_triplet, strict=False):
                self._resultats[image.frame_idx] = extraire_candidats(
                    carte, image.inverse, self.seuil_score, image.frame_idx
                )

    def _vider_lots(self, force: bool) -> None:
        while len(self._triplets) >= self.taille_lot or (force and self._triplets):
            lot, self._triplets = (
                self._triplets[: self.taille_lot],
                self._triplets[self.taille_lot :],
            )
            self._traiter(lot, [NB_IMAGES] * len(lot))

    def _restituer(self) -> list[Candidat]:
        """Candidats des images prédites, dans l'ordre de poussée, tant qu'il n'y a pas de trou."""
        sortie: list[Candidat] = []
        while self._ordre and self._ordre[0] in self._resultats:
            sortie.extend(self._resultats.pop(self._ordre.popleft()))
        return sortie

    def pousser(self, frame_idx: int, image: np.ndarray) -> list[Candidat]:
        """Ajoute une image BGR ; renvoie les candidats des images devenues prédites."""
        sous_sequence = self._partielles[self._compteur % self.pas_temporel]
        self._compteur += 1
        self._ordre.append(frame_idx)
        sous_sequence.append(self._preparer(frame_idx, image))
        if len(sous_sequence) == NB_IMAGES:
            self._triplets.append(sous_sequence.copy())
            sous_sequence.clear()
            self._vider_lots(force=False)
        return self._restituer()

    def terminer(self) -> list[Candidat]:
        """Prédit les images restantes (triplet final complété en répétant la dernière image)."""
        self._vider_lots(force=True)
        for sous_sequence in self._partielles:
            if sous_sequence:
                n_reels = len(sous_sequence)
                triplet = sous_sequence + [sous_sequence[-1]] * (NB_IMAGES - n_reels)
                self._traiter([triplet], [n_reels])
                sous_sequence.clear()
        return self._restituer()
