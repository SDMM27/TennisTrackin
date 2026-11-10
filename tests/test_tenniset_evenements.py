"""Tests de la conversion des annotations TenniSet complètes en events.json."""

import json
from pathlib import Path

import pytest

from tennisvision.datasets import (
    charger_evenements_tenniset,
    charger_verites_tenniset,
    filtrer_split,
    intervalles_split_tenniset,
)
from tennisvision.validate import valider_fichier

DATA = Path(__file__).parent / "fixtures" / "eval" / "tenniset"
V901 = DATA / "annotations" / "V901.json"
REEL = Path(__file__).parent.parent / "data" / "annotations" / "V006.json"


@pytest.fixture(scope="module")
def conversion():
    return charger_evenements_tenniset(V901, fps=25.0)


def par_type(doc, type_):
    return [e for e in doc["events"] if e["type"] == type_]


def test_joueurs_et_enveloppe(conversion):
    doc, joueurs = conversion
    assert joueurs == {"Alpha": "P1", "Beta": "P2"}  # P1 = premier serveur
    assert doc["schema_version"] == "1.0"
    assert doc["video"] == "V901.mp4"  # sans chemin
    assert len(doc["events"]) == 16
    frames = [e["frame_idx"] for e in doc["events"]]
    assert frames == sorted(frames)


def test_document_valide_au_schema(conversion, tmp_path):
    chemin = tmp_path / "events.json"
    chemin.write_text(json.dumps(conversion[0]), encoding="utf-8")
    assert valider_fichier(chemin) == []


def test_champs_communs(conversion):
    doc, _ = conversion
    hit = par_type(doc, "hit")[0]
    assert hit["id"] == "hit_0001"
    assert hit["frame_idx"] == 160  # 160.0 arrondi
    assert isinstance(hit["frame_idx"], int)
    assert hit["end_frame_idx"] == 170
    assert hit["t_ms"] == 160 * 1000 / 25
    assert hit["end_t_ms"] == 170 * 40
    assert hit["source"] == "annotation"
    assert hit["confidence"] is None
    assert "desc" not in hit
    assert len({e["id"] for e in doc["events"]}) == len(doc["events"])


def test_points(conversion):
    p1, p2 = par_type(conversion[0], "point")
    assert (p1["frame_idx"], p1["end_frame_idx"]) == (100, 300)
    assert p1["winner_id"] == "P1"
    assert p1["server_id"] == "P1"
    assert p1["desc"] == "Premier point."
    assert p2["winner_id"] is None  # Winner vide
    assert p2["server_id"] == "P2"  # premier service dans l'intervalle du point


def test_jeux_et_sets(conversion):
    doc, _ = conversion
    g1, g2 = par_type(doc, "game")
    assert (g1["winner_id"], g1["server_id"]) == ("P1", "P1")
    assert (g2["winner_id"], g2["server_id"]) == ("P2", "P2")
    assert "score_after" not in g1
    (manche,) = par_type(doc, "set")
    assert manche["winner_id"] == "P1"


def test_services_numeros_et_resultats(conversion):
    services = par_type(conversion[0], "serve")
    assert [(s["frame_idx"], s["player_id"], s["serve_number"], s["result"]) for s in services] == [
        (105, "P1", 1, "fault"),
        (150, "P1", 2, "in"),  # après une faute
        (405, "P2", 1, "let"),
        (440, "P2", 1, "in"),  # le let garde le numéro courant
        (700, "P1", 1, "in"),  # hors de tout point
    ]


def test_aucun_hit_invente_pour_les_services(conversion):
    doc, _ = conversion
    assert len(par_type(doc, "hit")) == 6
    assert all("hit_id" not in s for s in par_type(doc, "serve"))


def test_frappes_stroke_et_spin(conversion):
    hits = par_type(conversion[0], "hit")
    assert [(h["player_id"], h["stroke"], h["spin"]) for h in hits] == [
        ("P2", "forehand", "topspin"),
        ("P1", "backhand", "slice"),
        ("P2", "backhand_volley", None),
        ("P1", "smash", None),
        ("P1", "forehand", "topspin"),
        ("P2", "backhand", "unknown"),
    ]
    assert all("outcome" not in h for h in hits)


def test_ids_dedoublonnes(tmp_path):
    brut = json.loads(V901.read_text(encoding="utf-8"))
    brut["classes"]["Hit"].append(dict(brut["classes"]["Hit"][0], start=900, end=905))
    chemin = tmp_path / "V902.json"
    chemin.write_text(json.dumps(brut), encoding="utf-8")
    doc, _ = charger_evenements_tenniset(chemin)
    ids = [e["id"] for e in doc["events"]]
    assert len(set(ids)) == len(ids)
    assert "hit_0001_2" in ids


def test_noms_inconnus_donnent_null(tmp_path):
    brut = json.loads(V901.read_text(encoding="utf-8"))
    brut["classes"]["Point"][0]["custom"]["Winner"] = "40-30"
    brut["classes"]["Game"][0]["custom"]["Winner"] = "Tiebreak"
    chemin = tmp_path / "V902.json"
    chemin.write_text(json.dumps(brut), encoding="utf-8")
    doc, joueurs = charger_evenements_tenniset(chemin)
    assert len(joueurs) == 2
    assert par_type(doc, "point")[0]["winner_id"] is None
    assert par_type(doc, "game")[0]["winner_id"] is None


def test_deux_joueurs_exactement(tmp_path):
    brut = json.loads(V901.read_text(encoding="utf-8"))
    brut["classes"]["Hit"][0]["custom"]["Player"] = "Gamma"
    chemin = tmp_path / "V902.json"
    chemin.write_text(json.dumps(brut), encoding="utf-8")
    with pytest.raises(ValueError, match="Deux joueurs"):
        charger_evenements_tenniset(chemin)


def test_intervalles_de_split():
    assert intervalles_split_tenniset(V901) == {
        "train": [(80, 350)],
        "val": [(351, 379)],
        "test": [(380, 720)],
    }


def test_filtrer_split(conversion):
    doc, _ = conversion
    train = filtrer_split(doc, [(80, 350)])
    assert {e["id"] for e in train["events"]} == {
        "point_0001",
        "game_0001",
        "set_0001",
        "serve_0001",
        "serve_0002",
        "hit_0001",
        "hit_0002",
        "hit_0003",
        "hit_0004",
    }
    assert filtrer_split(doc, [])["events"] == []
    assert train["video"] == doc["video"]


def test_charger_verites_par_split():
    test = charger_verites_tenniset(DATA, "test")
    assert list(test) == ["V901"]
    v = test["V901"]
    assert v["hauteur_image"] == 720  # pas de vidéo : valeur par défaut
    assert (v["ball"], v["court"], v["players"]) == (None, None, None)
    assert v["intervalles"] == [(380, 720)]
    ids = {e["id"] for e in v["events"]["events"]}
    assert ids == {
        "point_0002",
        "game_0002",
        "serve_0003",
        "serve_0004",
        "serve_0005",
        "hit_0005",
        "hit_0006",
    }
    tout = charger_verites_tenniset(DATA, "all")["V901"]
    assert len(tout["events"]["events"]) == 16
    assert tout["intervalles"] is None


def test_charger_verites_sans_annotation(tmp_path):
    with pytest.raises(FileNotFoundError):
        charger_verites_tenniset(tmp_path, "all")


def test_charger_verites_ignore_generalised(tmp_path):
    annotations = tmp_path / "annotations"
    (annotations / "generalised").mkdir(parents=True)
    (annotations / "generalised" / "V001.json").write_text("{}", encoding="utf-8")
    (annotations / "V901.json").write_text(V901.read_text(encoding="utf-8"), encoding="utf-8")
    assert list(charger_verites_tenniset(tmp_path, "all")) == ["V901"]


@pytest.mark.skipif(not REEL.is_file(), reason="données TenniSet absentes")
def test_v006_reel(tmp_path):
    doc, joueurs = charger_evenements_tenniset(REEL)
    assert len(par_type(doc, "hit")) == 220
    assert len(par_type(doc, "serve")) == 112
    assert len(par_type(doc, "point")) == 81
    assert sorted(joueurs.values()) == ["P1", "P2"]
    chemin = tmp_path / "events.json"
    chemin.write_text(json.dumps(doc), encoding="utf-8")
    assert valider_fichier(chemin) == []
