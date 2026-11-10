"""Tests des indicateurs vus par l'utilisateur."""

import pytest

from tennisvision.eval.indicateurs import (
    agreger_indicateurs,
    calculer_indicateurs,
    comparer_indicateurs,
)


@pytest.fixture
def doc(evenement, doc_events):
    def fin(frame):
        return {"end_t_ms": frame * 40.0, "end_frame_idx": frame}

    return doc_events(
        # P1 : 1er service in, 1er fault, let ignoré, 2e in, 2e fault ; P2 : 1er in
        evenement("serve", 1, player_id="P1", serve_number=1, result="in"),
        evenement("serve", 2, player_id="P1", serve_number=1, result="fault"),
        evenement("serve", 3, player_id="P1", serve_number=1, result="let"),
        evenement("serve", 4, player_id="P1", serve_number=2, result="in"),
        evenement("serve", 5, player_id="P1", serve_number=2, result="fault"),
        evenement("serve", 6, player_id="P2", serve_number=1, result="in"),
        evenement("point", 0, **fin(100), winner_id="P1", server_id="P1", end_reason="ace"),
        evenement("point", 200, **fin(300), winner_id="P2", server_id="P1", end_reason="winner"),
        evenement("point", 400, **fin(500), winner_id=None),
        evenement("hit", 10, player_id="P1", stroke="forehand", outcome="winner"),
        evenement("hit", 20, player_id="P2", stroke="backhand", outcome="unforced_error"),
        evenement("hit", 30, player_id="P1", stroke="forehand_volley", outcome="in_play"),
        evenement("hit", 40, player_id="P1", stroke="backhand_volley", outcome="in_play"),
        evenement("hit", 50, player_id="P2", stroke="smash", outcome="in_play"),
        evenement("hit", 250, player_id="P1", stroke="backhand", outcome="unforced_error"),
        evenement("bounce", 11, court_xy=[0, 1], in_out="in"),
        evenement("bounce", 21, court_xy=[0, 1], in_out="in"),
        evenement("bounce", 31, court_xy=[0, 1], in_out="out"),
        evenement("bounce", 41, court_xy=[0, 1], in_out="too_close"),
    )


def val(stats, stat, joueur):
    return stats[(stat, joueur)]["value"], stats[(stat, joueur)]["n"]


def test_services(doc):
    s = calculer_indicateurs(doc)
    assert val(s, "first_serve_pct", "P1") == (50.0, 2)  # in, fault ; let exclu
    assert val(s, "second_serve_pct", "P1") == (50.0, 2)
    assert val(s, "first_serve_pct", "P2") == (100.0, 1)
    assert val(s, "second_serve_pct", "P2") == (None, 0)
    assert val(s, "double_faults", "P1")[0] == 1
    assert val(s, "double_faults", "P2")[0] == 0


def test_points(doc):
    s = calculer_indicateurs(doc)
    assert val(s, "aces", "P1")[0] == 1
    assert val(s, "aces", "P2")[0] == 0
    # n = points dont le vainqueur est connu
    assert val(s, "points_won", "P1") == (1, 2)
    assert val(s, "points_won", "P2") == (1, 2)


def test_coups_et_issues(doc):
    s = calculer_indicateurs(doc)
    assert val(s, "forehands", "P1")[0] == 1
    assert val(s, "forehands", "P2")[0] == 0
    assert val(s, "backhands", "P1")[0] == 1
    assert val(s, "backhands", "P2")[0] == 1
    assert val(s, "volleys", "P1")[0] == 2
    assert val(s, "smashes", "P2")[0] == 1
    assert val(s, "smashes", "P1")[0] == 0
    assert val(s, "winners", "P1")[0] == 1
    assert val(s, "unforced_errors", "P1")[0] == 1
    assert val(s, "unforced_errors", "P2")[0] == 1
    assert s[("forehands", "P1")]["unit"] == "count"


def test_stats_de_match(doc):
    s = calculer_indicateurs(doc)
    # hits par point : 5 (frames 10 à 50), 1 (250), 0 -> moyenne 2
    assert val(s, "rally_length_mean", None) == (2.0, 3)
    assert s[("rally_length_mean", None)]["unit"] == "ratio"
    valeur, n = val(s, "bounce_in_pct", None)
    assert valeur == pytest.approx(200 / 3)  # 2 in sur 3 (too_close exclu)
    assert n == 3


def test_toutes_les_cles_toujours_presentes(doc_events):
    s = calculer_indicateurs(doc_events())
    assert len(s) == 11 * 2 + 2
    assert s[("first_serve_pct", "P1")] == {"value": None, "n": 0, "unit": "pct"}
    assert s[("rally_length_mean", None)]["value"] is None
    assert set(s) == set(calculer_indicateurs(doc_events()))


def test_comparaison(evenement, doc_events):
    gt = doc_events(*[evenement("hit", f, player_id="P1", stroke="forehand") for f in range(4)])
    pred = doc_events(
        *[evenement("hit", f, player_id="P1", stroke="forehand") for f in range(3)],
        *[evenement("hit", f, player_id="P2", stroke="forehand") for f in range(10, 12)],
    )
    lignes = {(c["id"], c["player_id"]): c for c in comparer_indicateurs(pred, gt)}
    f1 = lignes[("forehands", "P1")]
    assert (f1["pred"], f1["gt"], f1["n_gt"]) == (3, 4, 4)
    assert f1["erreur_abs"] == 1
    assert f1["erreur_rel"] == 0.25
    assert f1["unit"] == "count"
    f2 = lignes[("forehands", "P2")]  # gt = 0 : erreur relative indéfinie
    assert f2["erreur_abs"] == 2
    assert f2["erreur_rel"] is None
    indefini = lignes[("first_serve_pct", "P1")]  # indéfini des deux côtés
    assert indefini["erreur_abs"] is None
    assert indefini["erreur_rel"] is None


def test_comparaison_parfaite(doc):
    assert all(c["erreur_abs"] in (0, None) for c in comparer_indicateurs(doc, doc))


def ligne(abs_, rel):
    return {
        "id": "forehands",
        "player_id": "P1",
        "unit": "count",
        "erreur_abs": abs_,
        "erreur_rel": rel,
    }


def test_agregation():
    par_match = {
        "m1": [ligne(2, 0.2)],
        "m2": [ligne(4, 0.1)],
        "m3": [ligne(None, None)],
    }
    (agr,) = agreger_indicateurs(par_match)
    assert agr["id"] == "forehands"
    assert agr["erreur_abs_moyenne"] == 3
    assert agr["erreur_abs_max"] == 4
    assert agr["erreur_rel_moyenne"] == pytest.approx(0.15)
    assert agr["erreur_rel_max"] == 0.2
    assert agr["n_matchs"] == 2
    assert agr["pire_match"] == "m2"


def test_agregation_sans_valeur():
    (agr,) = agreger_indicateurs({"m1": [ligne(None, None)]})
    assert agr["erreur_abs_moyenne"] is None
    assert agr["n_matchs"] == 0
    assert agr["pire_match"] is None
