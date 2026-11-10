"""Tests de l'appariement hongrois et des scores de détection."""

import pytest

from tennisvision.eval.appariement import apparier_instants, apparier_intervalles, scores_detection


def test_hongrois_evite_le_piege_du_glouton():
    # Le glouton prendrait 12 -> 11 (écart 1) et laisserait 10 sans partenaire dans la tolérance
    assert apparier_instants([10, 12], [11, 13], tolerance=1) == [(0, 0), (1, 1)]


def test_hongrois_maximise_d_abord_le_nombre_de_paires():
    # 12 est plus proche de 11, mais l'optimum global apparie 10-11 et 12-13
    assert sorted(apparier_instants([12, 10], [11, 13], tolerance=1)) == [(0, 1), (1, 0)]


def test_tolerance_incluse_puis_exclue():
    assert apparier_instants([10], [12], tolerance=2) == [(0, 0)]
    assert apparier_instants([10], [13], tolerance=2) == []


def test_un_pour_un():
    assert apparier_instants([10, 10], [10], tolerance=3) == [(0, 0)]


def test_instants_vides():
    assert apparier_instants([], [1, 2], tolerance=5) == []
    assert apparier_instants([1, 2], [], tolerance=5) == []


def test_intervalles_iou_seuil_inclus():
    # [0, 9] et [0, 4] : intersection 5, union 10 -> IoU 0,5 (seuil inclus)
    assert apparier_intervalles([(0, 9)], [(0, 4)], iou_min=0.5) == [(0, 0, 0.5)]


def test_intervalles_sous_le_seuil():
    # [0, 9] et [5, 14] : intersection 5, union 15 -> IoU 1/3
    assert apparier_intervalles([(0, 9)], [(5, 14)], iou_min=0.5) == []
    ((i, j, iou),) = apparier_intervalles([(0, 9)], [(5, 14)], iou_min=0.3)
    assert (i, j) == (0, 0)
    assert iou == pytest.approx(1 / 3)


def test_intervalles_disjoints_et_vides():
    assert apparier_intervalles([(0, 9)], [(10, 19)], iou_min=0.0) == []
    assert apparier_intervalles([], [(0, 1)]) == []


def test_intervalles_appariement_optimal():
    # Prédictions [0, 9] et [8, 17] ; gt [0, 9] : seul le premier est apparié (IoU 1)
    assert apparier_intervalles([(8, 17), (0, 9)], [(0, 9)]) == [(1, 0, 1.0)]


def test_scores_detection():
    s = scores_detection(2, 4, 3)
    assert s["tp"] == 2
    assert s["precision"] == 0.5
    assert s["rappel"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(4 / 7)


def test_scores_vides():
    assert scores_detection(0, 0, 0) == {
        "tp": 0,
        "n_pred": 0,
        "n_gt": 0,
        "precision": None,
        "rappel": None,
        "f1": None,
    }


def test_prediction_vide_face_a_une_verite_non_vide():
    s = scores_detection(0, 0, 3)
    assert s["precision"] is None
    assert s["rappel"] == 0
    assert s["f1"] == 0
