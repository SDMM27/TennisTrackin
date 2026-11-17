"""Tests du détecteur de balle : affine, post-traitement, flux (réseau factice) et poids réels."""

import cv2
import numpy as np
import pytest
import torch

from tennisvision.balle.detecteur import (
    Candidat,
    DetecteurBalle,
    appliquer_affine,
    extraire_candidats,
    transformations,
)
from tennisvision.balle.reseaux import HAUTEUR_ENTREE, LARGEUR_ENTREE
from tennisvision.models import ModeleAbsent, resoudre_modele


def gaussienne(x, y, sigma=3.0, pic=0.9, carte=None):
    carte = np.zeros((HAUTEUR_ENTREE, LARGEUR_ENTREE), np.float32) if carte is None else carte
    yy, xx = np.mgrid[0:HAUTEUR_ENTREE, 0:LARGEUR_ENTREE]
    return np.maximum(carte, pic * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2)))


@pytest.mark.parametrize("forme", [(1080, 1920), (720, 1280), (360, 640), (1920, 1080)])
def test_aller_retour_affine(forme):
    aller, retour = transformations(*forme)
    h, w = forme
    for point in [(w / 2, h / 2), (100.0, 50.0), (w - 10.0, h - 10.0)]:
        retrouve = appliquer_affine(appliquer_affine(point, aller), retour)
        assert retrouve == pytest.approx(point, abs=1.0)


def test_affine_16_9_couvre_le_modele():
    aller, _ = transformations(1080, 1920)
    assert appliquer_affine((0, 0), aller) == pytest.approx((0, 0), abs=1e-3)
    assert appliquer_affine((1920, 1080), aller) == pytest.approx((512, 288), abs=1e-3)


def test_un_blob_un_candidat_aux_coordonnees_d_origine():
    _, retour = transformations(1080, 1920)
    candidats = extraire_candidats(gaussienne(300, 150), retour, 0.5, 7)
    assert len(candidats) == 1
    c = candidats[0]
    assert c.frame_idx == 7
    assert (c.x, c.y) == pytest.approx((960 + 44 * 3.75, 540 + 6 * 3.75), abs=1.0)
    assert c.score == pytest.approx(0.9, abs=1e-3)


def test_deux_blobs_deux_candidats():
    _, retour = transformations(1080, 1920)
    carte = gaussienne(100, 60, pic=0.95)
    carte = gaussienne(400, 220, pic=0.7, carte=carte)
    candidats = extraire_candidats(carte, retour, 0.5, 0)
    assert len(candidats) == 2
    assert sorted(round(c.score, 2) for c in candidats) == [0.7, 0.95]


def test_sous_le_seuil_aucun_candidat():
    _, retour = transformations(1080, 1920)
    assert extraire_candidats(gaussienne(300, 150, pic=0.4), retour, 0.5, 0) == []
    # Le seuil est configurable.
    assert len(extraire_candidats(gaussienne(300, 150, pic=0.4), retour, 0.3, 0)) == 1


class ReseauFactice(torch.nn.Module):
    """Enregistre les triplets reçus (valeur de gris = indice d'image) ; un blob par carte."""

    def __init__(self):
        super().__init__()
        self.triplets = []
        self.tailles_lot = []

    def forward(self, x):
        rouge = x[:, 0::3] * 0.229 + 0.485  # (B, 3, H, W) : canal rouge de chaque image
        valeurs = (rouge.mean(dim=(2, 3)) * 255).round().long()
        self.triplets += [tuple(t) for t in valeurs.tolist()]
        self.tailles_lot.append(len(x))
        logits = torch.full((len(x), 3, HAUTEUR_ENTREE, LARGEUR_ENTREE), -10.0)
        logits[:, :, 100, 200] = 10.0
        return logits


def image_grise(valeur, hauteur=72, largeur=128):
    return np.full((hauteur, largeur, 3), valeur, np.uint8)


def pousser_tout(detecteur, n, decalage=0):
    """Pousse `n` images (valeur de gris = rang) ; renvoie tous les candidats, dans l'ordre reçu."""
    candidats = []
    for i in range(n):
        candidats += detecteur.pousser(decalage + i, image_grise(i))
    return candidats + detecteur.terminer()


@pytest.mark.parametrize("n", [1, 2, 3, 7, 10, 24])
def test_une_prediction_par_image_ordre_conserve(n):
    reseau = ReseauFactice()
    detecteur = DetecteurBalle(device="cpu", taille_lot=2, reseau=reseau)
    candidats = pousser_tout(detecteur, n, decalage=100)
    assert [c.frame_idx for c in candidats] == list(range(100, 100 + n))
    assert all(isinstance(c, Candidat) and 0 <= c.score <= 1 for c in candidats)
    # Triplets disjoints, dans l'ordre temporel ; le dernier est complété par répétition.
    attendus = [tuple(range(i, i + 3)) for i in range(0, n - n % 3, 3)]
    if n % 3:
        reste = list(range(n - n % 3, n))
        attendus.append(tuple(reste + [reste[-1]] * (3 - len(reste))))
    assert reseau.triplets == attendus


def test_memoire_bornee_et_lots():
    reseau = ReseauFactice()
    detecteur = DetecteurBalle(device="cpu", taille_lot=4, reseau=reseau)
    for i in range(300):
        detecteur.pousser(i, image_grise(i % 250))
        en_attente = len(detecteur._triplets) * 3 + sum(len(p) for p in detecteur._partielles)
        assert en_attente <= 4 * 3 + 2
        assert len(detecteur._ordre) <= en_attente
        assert len(detecteur._resultats) <= 4 * 3
    detecteur.terminer()
    assert max(reseau.tailles_lot) == 4


def test_pas_temporel_selon_fps():
    kwargs = {"device": "cpu", "reseau": ReseauFactice()}
    assert DetecteurBalle(**kwargs).pas_temporel == 1
    assert DetecteurBalle(fps=25, **kwargs).pas_temporel == 1
    assert DetecteurBalle(fps=30, **kwargs).pas_temporel == 1
    assert DetecteurBalle(fps=50, **kwargs).pas_temporel == 2
    assert DetecteurBalle(fps=60, **kwargs).pas_temporel == 2
    assert DetecteurBalle(fps=120, **kwargs).pas_temporel == 4


@pytest.mark.parametrize("n", [6, 7, 11, 13])
def test_flux_pas_temporel_2(n):
    reseau = ReseauFactice()
    detecteur = DetecteurBalle(device="cpu", taille_lot=1, fps=60, reseau=reseau)
    assert detecteur.pas_temporel == 2
    candidats = pousser_tout(detecteur, n)
    assert [c.frame_idx for c in candidats] == list(range(n))
    # Chaque triplet couvre t, t+2, t+4 d'une même sous-séquence (pas de mélange pair / impair).
    for triplet in reseau.triplets:
        distinctes = list(dict.fromkeys(triplet))
        assert distinctes == list(range(distinctes[0], distinctes[0] + 2 * len(distinctes), 2))
    reels = [v for triplet in reseau.triplets for v in dict.fromkeys(triplet)]
    assert sorted(reels) == list(range(n))


def test_blob_factice_ramene_aux_pixels_d_origine():
    candidats = pousser_tout(DetecteurBalle(device="cpu", reseau=ReseauFactice()), 3)
    attendu = appliquer_affine((200, 100), transformations(72, 128)[1])
    assert (candidats[0].x, candidats[0].y) == pytest.approx(attendu, abs=1e-3)


def poids_ou_skip(nom):
    try:
        return resoudre_modele(nom)
    except ModeleAbsent:
        pytest.skip(f"poids {nom} absents du cache")


def sequence_synthetique(n=12, largeur=640, hauteur=360):
    """Disque jaune-vert sur fond vert foncé, en mouvement ; renvoie (images BGR, positions)."""
    images, positions = [], []
    for i in range(n):
        image = np.full((hauteur, largeur, 3), (40, 90, 40), np.uint8)
        x, y = 120 + 30 * i, 220 - 8 * i
        cv2.circle(image, (x, y), 5, (50, 230, 200), -1)
        images.append(image)
        positions.append((x, y))
    return images, positions


@pytest.mark.parametrize("nom", ["wasb_tennis", "tracknetv2_tennis"])
def test_poids_reels_sequence_synthetique(nom):
    poids_ou_skip(nom)
    images, positions = sequence_synthetique()
    detecteur = DetecteurBalle(nom, device="cpu", taille_lot=2)
    assert detecteur.modele_decrit["nom"] == nom
    candidats = []
    for i, image in enumerate(images):
        candidats += detecteur.pousser(i, image)
    candidats += detecteur.terminer()
    indices = [c.frame_idx for c in candidats]
    assert indices == sorted(indices)
    assert set(indices) <= set(range(len(images)))
    for c in candidats:
        assert 0.5 < c.score <= 1.0
        assert 0 <= c.x < 640 and 0 <= c.y < 360
    proches = sum(
        any(np.hypot(c.x - px, c.y - py) < 15 for c in candidats if c.frame_idx == i)
        for i, (px, py) in enumerate(positions)
    )
    print(f"{nom} : {proches}/{len(images)} images avec la balle synthétique trouvée")
