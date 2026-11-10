"""Tests de l'annotation des frames : traînée de la balle et bandeau."""

import numpy as np

from tennisvision.tracking.detector import Detection
from tennisvision.tracking.draw import BallTrail, _texte_bandeau, draw_frame

LARGEUR, HAUTEUR = 200, 100


def _est_jaune(image: np.ndarray) -> np.ndarray:
    # Tolérance : les traits sont anticrénelés (LINE_AA)
    return (image[..., 0] < 60) & (image[..., 1] > 100) & (image[..., 2] > 100)


def _balle(frame_idx: int, x: float, y: float) -> Detection:
    return Detection(frame_idx, None, 32, "ball", 0.9, x - 2, y - 2, x + 2, y + 2)


def _image() -> np.ndarray:
    return np.full((HAUTEUR, LARGEUR, 3), 40, dtype=np.uint8)


def test_trainee_s_efface_apres_max_age_sans_balle():
    trail = BallTrail(max_age=3, max_gap=5)
    trail.update(0, (10, 10))
    trail.update(1, None)
    trail.update(3, None)
    assert trail.segments() == [[(10, 10)]]
    trail.update(4, None)
    assert trail.segments() == []


def test_detections_separees_de_plus_de_max_gap_non_reliees():
    trail = BallTrail(max_age=50, max_gap=5)
    trail.update(0, (10, 10))
    trail.update(6, (150, 80))
    assert trail.segments() == [[(10, 10)], [(150, 80)]]


def test_trou_inferieur_ou_egal_a_max_gap_relie():
    trail = BallTrail(max_age=50, max_gap=5)
    trail.update(0, (10, 10))
    trail.update(5, (50, 40))
    assert trail.segments() == [[(10, 10), (50, 40)]]


def test_current_none_quand_la_balle_manque():
    trail = BallTrail()
    trail.update(0, (10, 20))
    assert trail.current() == (10, 20)
    trail.update(1, None)
    assert trail.current() is None


def test_draw_frame_copie_et_pixels_jaunes_seulement_sur_la_trainee():
    trail = BallTrail(max_age=50, max_gap=2)
    for idx, pos in [(0, (20, 60)), (1, (40, 60)), (10, (160, 80)), (11, (180, 80))]:
        trail.update(idx, pos)
    entree = _image()
    copie_entree = entree.copy()
    sortie = draw_frame(entree, [_balle(11, 180, 80)], trail, 11)

    assert sortie is not entree
    assert sortie.shape == entree.shape
    assert np.array_equal(entree, copie_entree)
    jaune = _est_jaune(sortie)
    assert jaune[59:62, 30].any()  # segment 1
    assert jaune[79:82, 170].any()  # segment 2
    assert not jaune[:, 60:150].any()  # rien entre les deux segments


def test_texte_bandeau_avec_temps_reel():
    assert _texte_bandeau(10, 61234.0) == "Frame 10  01:01.2"


def test_texte_bandeau_sans_temps():
    assert _texte_bandeau(10) == "Frame 10"
