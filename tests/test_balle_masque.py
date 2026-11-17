"""Tests du masque de court (court.json et polygone manuel)."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tennisvision.balle.masque import MasqueCourt, polygone_court_etendu

COURT_JSON = Path(__file__).parent / "fixtures" / "contrats" / "valide" / "court.json"


@pytest.fixture
def court() -> dict:
    return json.loads(COURT_JSON.read_text(encoding="utf-8"))


def test_court_json_centre_dedans_cote_dehors():
    masque = MasqueCourt.depuis_court_json(COURT_JSON)
    assert masque.contient(0, 960, 600)  # centre du court
    assert masque.contient(0, 960, 1000)  # près de la ligne de fond proche
    assert not masque.contient(0, 20, 600)  # loin sur le côté (court voisin)
    assert not masque.contient(0, 1900, 300)
    assert not masque.contient(0, 100, 100)


def test_court_json_extension_verticale_cote_loin():
    masque = MasqueCourt.depuis_court_json(COURT_JSON)
    # Fond loin à y = 300 ; une balle haute apparaît bien au-dessus, jusqu'en haut de l'image
    assert masque.contient(0, 960, 100)
    assert masque.contient(0, 960, 5)
    # Mais pas au-dessus d'un point situé hors des x du court étendu
    assert not masque.contient(0, 300, 100)


def test_marges_configurables(court):
    etroit = MasqueCourt.depuis_court_json(court, marge_laterale_m=0.0, marge_fond_m=0.0)
    large = MasqueCourt.depuis_court_json(court)
    assert not etroit.contient(0, 330, 600)
    assert large.contient(0, 330, 600)


def test_polygone_clippe_a_l_image(court):
    polygone = polygone_court_etendu(court["segments"][0]["homography"], (1920, 1080))
    assert polygone[:, 0].min() >= 0 and polygone[:, 0].max() <= 1920
    assert polygone[:, 1].min() >= 0 and polygone[:, 1].max() <= 1080


def test_segments_par_frame(court):
    premier = court["segments"][0]
    premier["valid_to_frame"] = 99
    second = copy.deepcopy(premier)
    second["valid_from_frame"] = 100
    second["valid_to_frame"] = None
    # Deuxième segment : caméra décalée, le court apparaît 200 px plus à droite
    translation = np.array([[1, 0, -200], [0, 1, 0], [0, 0, 1]], dtype=float)
    second["homography"] = (np.array(premier["homography"]) @ translation).tolist()
    court["segments"] = [premier, second]
    masque = MasqueCourt.depuis_court_json(court)
    # Le bord gauche du masque (x = 260 à y = 600) a bougé de 200 px au segment 2
    assert masque.contient(50, 330, 600)
    assert not masque.contient(150, 330, 600)
    assert masque.contient(150, 600, 600)
    # Frame hors de tous les segments : segment le plus proche
    assert masque.contient(5000, 600, 600)
    assert not masque.contient(5000, 330, 600)
    resultat = masque.contient_vec(
        np.array([50, 150, 150]), np.array([330.0, 330.0, 600.0]), np.array([600.0] * 3)
    )
    assert resultat.tolist() == [True, False, True]


def test_polygone_manuel_fichier_et_vectorise(tmp_path):
    fichier = tmp_path / "masque.json"
    fichier.write_text(json.dumps({"polygone": [[100, 100], [500, 100], [500, 400], [100, 400]]}))
    masque = MasqueCourt.depuis_polygone(fichier)
    assert masque.contient(0, 300, 200)
    assert masque.contient(9999, 300, 200)  # valable pour toutes les frames
    assert not masque.contient(0, 600, 200)
    dedans = masque.contient_vec(
        np.array([0, 1, 2]), np.array([300.0, 600.0, 120.0]), np.array([200.0, 200.0, 390.0])
    )
    assert dedans.tolist() == [True, False, True]
    assert masque.contient_vec([], [], []).shape == (0,)


def test_entrees_invalides(court):
    with pytest.raises(ValueError):
        MasqueCourt.depuis_polygone({"polygone": [[0, 0], [1, 1]]})
    with pytest.raises(ValueError):
        MasqueCourt.depuis_polygone({"autre": 1})
    with pytest.raises(ValueError):
        MasqueCourt.depuis_court_json({"segments": []})
    with pytest.raises(ValueError):
        MasqueCourt([])
