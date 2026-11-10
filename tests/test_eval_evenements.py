"""Tests des métriques d'événements (cas construits à la main)."""

import math

import pytest

from tennisvision.eval.evenements import (
    distance_ligne_m,
    evaluer_evenements,
    evaluer_frappes,
    evaluer_points,
    evaluer_rebonds,
    evaluer_services,
    f1_par_classe,
)


def test_frappes(evenement, doc_events):
    gt = doc_events(
        evenement("hit", 10, player_id="P1", stroke="forehand"),
        evenement("hit", 30, player_id="P2", stroke="backhand"),
        evenement("hit", 50, player_id="P1", stroke="forehand"),
        evenement("hit", 70, player_id="P2", stroke="smash"),
    )
    pred = doc_events(
        evenement("hit", 200, player_id="P1", stroke="serve"),  # parasite
        evenement("hit", 52, player_id="P2", stroke="forehand"),  # mauvais joueur
        evenement("hit", 11, player_id="P1", stroke="forehand"),
        evenement("hit", 33, player_id="P2", stroke="forehand"),  # mauvais coup
    )
    r = evaluer_frappes(pred, gt)
    # ±2 : 10-11 (écart 1) et 50-52 (écart 2) ; 30-33 (écart 3) exclu
    t2 = r["tolerances"]["2"]
    assert (t2["tp"], t2["n_pred"], t2["n_gt"]) == (2, 4, 4)
    assert t2["f1"] == 0.5
    assert t2["ecart_moyen_frames"] == 1.5
    # ±5 : 30-33 s'ajoute
    t5 = r["tolerances"]["5"]
    assert t5["tp"] == 3
    assert t5["precision"] == 0.75
    assert t5["rappel"] == 0.75
    assert t5["f1"] == 0.75
    assert t5["ecart_moyen_frames"] == 2.0
    assert r["exactitude_joueur"] == {"n_ok": 2, "n": 3, "valeur": pytest.approx(2 / 3)}
    assert r["n_paires_coup"] == 3
    assert r["matrice_confusion"] == {"backhand": {"forehand": 1}, "forehand": {"forehand": 2}}
    # forehand : tp 2, fp 1, fn 0 -> 4/5 ; backhand : tp 0, fn 1 -> 0
    assert r["f1_par_classe"] == {"backhand": 0.0, "forehand": pytest.approx(0.8)}


def test_frappes_coup_non_classe_exclu_de_la_matrice(evenement, doc_events):
    gt = doc_events(evenement("hit", 10, player_id="P1", stroke=None))
    pred = doc_events(evenement("hit", 10, player_id="P1", stroke="forehand"))
    r = evaluer_frappes(pred, gt)
    assert r["n_paires_coup"] == 0
    assert r["matrice_confusion"] == {}


def test_f1_par_classe_direct():
    matrice = {"a": {"a": 3, "b": 1}, "b": {"b": 2}}
    # a : tp 3, fn 1, fp 0 -> 6/7 ; b : tp 2, fp 1, fn 0 -> 4/5
    assert f1_par_classe(matrice) == {"a": pytest.approx(6 / 7), "b": pytest.approx(4 / 5)}


def test_services(evenement, doc_events):
    gt = doc_events(
        evenement("serve", 100, player_id="P1", serve_number=1, result="fault"),
        evenement("serve", 140, player_id="P1", serve_number=2, result="in"),
        evenement("serve", 300, player_id="P2", serve_number=1, result="in"),
    )
    pred = doc_events(
        evenement("serve", 103, player_id="P1", serve_number=1, result="fault"),
        evenement("serve", 138, player_id="P1", serve_number=1, result="in"),
        evenement("serve", 320, player_id="P2", serve_number=1, result="in"),  # hors tolérance
    )
    r = evaluer_services(pred, gt)
    assert (r["tp"], r["n_pred"], r["n_gt"]) == (2, 3, 3)
    assert r["f1"] == pytest.approx(2 / 3)
    assert r["ecart_moyen_frames"] == 2.5
    assert r["exactitude_result"]["n_ok"] == 2
    assert r["exactitude_serve_number"] == {"n_ok": 1, "n": 2, "valeur": 0.5}
    assert r["exactitude_player_id"]["valeur"] == 1.0


@pytest.mark.parametrize(
    ("x", "y", "attendu"),
    [
        (0.0, 0.0, 0.0),  # sur la ligne médiane
        (2.0, 0.0, 2.0),  # médiane à 2 m, côté à 2,115 m
        (4.115, 0.0, 0.0),  # sur la ligne de côté
        (5.0, 0.0, 0.885),  # dans le couloir : ligne de côté
        (0.0, 12.0, 0.115),  # derrière le fond : ligne de fond
        (3.0, 7.0, 0.6),  # au-delà de la ligne de service
        (5.0, 12.0, math.hypot(0.885, 0.115)),  # hors du coin : distance au coin
        (0.0, -6.4, 0.0),  # sur la ligne de service loin
    ],
)
def test_distance_ligne(x, y, attendu):
    assert distance_ligne_m(x, y) == pytest.approx(attendu, abs=1e-9)


def test_rebonds(evenement, doc_events):
    gt = doc_events(
        evenement("bounce", 10, court_xy=[1.0, -6.3], in_out="in"),  # à 10 cm de la ligne
        evenement("bounce", 50, court_xy=[2.0, 4.0], in_out="in"),  # loin des lignes
        evenement("bounce", 90, court_xy=[3.0, 9.0], in_out="in"),  # manqué
    )
    pred = doc_events(
        evenement("bounce", 11, court_xy=[1.0, -6.0], in_out="out"),  # 30 cm d'erreur
        evenement("bounce", 52, court_xy=[2.4, 4.3], in_out="in"),  # 50 cm d'erreur
        evenement("bounce", 300, court_xy=[0.0, 1.0], in_out="in"),  # parasite
    )
    r = evaluer_rebonds(pred, gt)
    for tol in ("2", "5"):
        assert (r["tolerances"][tol]["tp"], r["tolerances"][tol]["n_pred"]) == (2, 3)
    assert r["erreur_position_cm"]["moyenne"] == pytest.approx(40.0)
    assert r["erreur_position_cm"]["max"] == pytest.approx(50.0)
    assert r["erreur_position_cm"]["n"] == 2
    assert r["exactitude_in_out"] == {"n_ok": 1, "n": 2, "valeur": 0.5}
    # seul le premier rebond gt est à moins de 30 cm d'une ligne, et son in/out est faux
    assert r["exactitude_in_out_pres_ligne"] == {"n_ok": 0, "n": 1, "valeur": 0.0}


def test_points_et_jeux(evenement, doc_events):
    score_a = {"sets": [], "games": [1, 0], "points": ["0", "0"]}
    score_b = {"sets": [], "games": [1, 1], "points": ["0", "0"]}

    def duree(type_, debut, fin, **champs):
        return evenement(type_, debut, end_t_ms=fin * 40.0, end_frame_idx=fin, **champs)

    gt = doc_events(
        duree("point", 0, 100, winner_id="P1"),
        duree("point", 200, 300, winner_id=None),
        duree("point", 400, 500, winner_id="P2"),  # manqué
        duree("game", 0, 150, winner_id="P1", score_after=score_a),
        duree("game", 200, 350, winner_id="P2", score_after=score_b),
        duree("game", 400, 550, winner_id="P1"),  # sans score
    )
    pred = doc_events(
        duree("point", 5, 105, winner_id="P1"),
        duree("point", 210, 290, winner_id="P2"),
        duree("point", 600, 700, winner_id="P1"),  # parasite
        duree("game", 0, 150, winner_id="P1", score_after=score_a),
        duree("game", 200, 350, winner_id="P2", score_after=score_a),  # score faux
    )
    r = evaluer_points(pred, gt)
    assert (r["tp"], r["n_pred"], r["n_gt"]) == (2, 3, 3)
    assert r["f1"] == pytest.approx(2 / 3)
    # IoU : [5,105]/[0,100] -> 96/106 ; [210,290]/[200,300] -> 81/101
    assert r["iou_moyen"]["moyenne"] == pytest.approx((96 / 106 + 81 / 101) / 2)
    assert r["iou_moyen"]["n"] == 2
    # le gagnant n'est évalué que si le gt le connaît (premier point seulement)
    assert r["exactitude_gagnant"] == {"n_ok": 1, "n": 1, "valeur": 1.0}
    assert (r["jeux"]["tp"], r["jeux"]["n_pred"], r["jeux"]["n_gt"]) == (2, 2, 3)
    assert r["jeux"]["pct_jeux_score_juste"] == {"n_ok": 1, "n": 2, "valeur": 0.5}


def test_points_sans_jeu_gt(evenement, doc_events):
    gt = doc_events(evenement("point", 0, end_t_ms=4.0, end_frame_idx=100, winner_id="P1"))
    assert evaluer_points(gt, gt)["jeux"] is None


def test_evaluer_evenements_blocs_absents(evenement, doc_events):
    gt = doc_events(evenement("hit", 10, player_id="P1", stroke="forehand"))
    r = evaluer_evenements(gt, gt)
    assert r["frappes"]["tolerances"]["5"]["f1"] == 1.0
    assert r["services"] is None
    assert r["rebonds"] is None
    assert r["points"] is None


def test_evenements_non_tries_et_source_quelconque(evenement, doc_events):
    gt = doc_events(
        evenement("hit", 30, player_id="P2", stroke="backhand", source="annotation"),
        evenement("hit", 10, player_id="P1", stroke="forehand", source="annotation"),
    )
    pred = doc_events(
        evenement("hit", 10, player_id="P1", stroke="forehand", source="auto"),
        evenement("hit", 30, player_id="P2", stroke="backhand", source="user"),
    )
    assert evaluer_frappes(pred, gt)["tolerances"]["2"]["f1"] == 1.0


def test_prediction_vide(evenement, doc_events):
    gt = doc_events(evenement("hit", 10, player_id="P1", stroke="forehand"))
    r = evaluer_evenements(doc_events(), gt)
    assert r["frappes"]["tolerances"]["5"]["rappel"] == 0
    assert r["frappes"]["tolerances"]["5"]["f1"] == 0
