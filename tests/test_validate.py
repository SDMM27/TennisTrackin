"""Tests de la validation des contrats de données."""

import json
import math
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator

from tennisvision import schemas, validate
from tennisvision.schemas import CONTRATS, COURT_KEYPOINTS_M, charger_schema
from tennisvision.validate import Probleme, valider_dossier, valider_fichier

VALIDE = Path(__file__).parent / "fixtures" / "contrats" / "valide"


@pytest.fixture
def dossier(tmp_path) -> Path:
    """Copie modifiable de la fixture valide."""
    cible = tmp_path / "sortie"
    shutil.copytree(VALIDE, cible)
    return cible


def _modifier_json(chemin: Path, fonction) -> None:
    doc = json.loads(chemin.read_text(encoding="utf-8"))
    fonction(doc)
    chemin.write_text(json.dumps(doc), encoding="utf-8")


def _remplacer(chemin: Path, ancien: str, nouveau: str) -> None:
    texte = chemin.read_text(encoding="utf-8")
    assert ancien in texte
    chemin.write_text(texte.replace(ancien, nouveau, 1), encoding="utf-8")


def _modifier_cellule(chemin: Path, ligne: int, colonne: str, valeur: str) -> None:
    """Remplace une cellule d'un CSV (ligne = numéro de ligne du fichier, en-tête = 1)."""
    lignes = chemin.read_text(encoding="utf-8").splitlines()
    cellules = lignes[ligne - 1].split(",")
    cellules[lignes[0].split(",").index(colonne)] = valeur
    lignes[ligne - 1] = ",".join(cellules)
    chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")


def _lieux(problemes: list[Probleme]) -> list[str]:
    return [p.emplacement for p in problemes]


# --- Schémas et fixture valide ---------------------------------------------------------------


@pytest.mark.parametrize("nom", sorted(CONTRATS))
def test_schema_valide_et_chargeable(nom):
    schema = charger_schema(CONTRATS[nom])
    Draft202012Validator.check_schema(schema)
    assert schema["$schema"].endswith("2020-12/schema")


def test_schema_inconnu():
    with pytest.raises(ValueError, match="inconnu"):
        charger_schema("nexiste.schema.json")


def test_dossier_valide():
    resultat = valider_dossier(VALIDE)
    assert set(resultat) == set(CONTRATS)
    assert all(problemes == [] for problemes in resultat.values()), resultat


@pytest.mark.parametrize("nom", sorted(CONTRATS))
def test_fichier_valide(nom):
    assert valider_fichier(VALIDE / nom) == []


def test_nom_de_fichier_inconnu(tmp_path):
    chemin = tmp_path / "autre.json"
    chemin.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="autre.json"):
        valider_fichier(chemin)


def test_dossier_avec_fichier_absent(dossier):
    (dossier / "ball.csv").unlink()
    assert valider_dossier(dossier)["ball.csv"] is None


# --- Cas invalides : match.json --------------------------------------------------------------


def test_match_handedness_invalide(dossier):
    _modifier_json(
        dossier / "match.json", lambda d: d["players"][1].__setitem__("handedness", "ambidextre")
    )
    problemes = valider_fichier(dossier / "match.json")
    assert _lieux(problemes) == ["players[1].handedness"]
    assert "ambidextre" in problemes[0].message


def test_match_joueurs_dupliques_et_serveur_inconnu(dossier):
    def corrompre(doc):
        doc["players"][1]["id"] = "P1"
        doc["first_server"] = "P2"

    _modifier_json(dossier / "match.json", corrompre)
    lieux = _lieux(valider_fichier(dossier / "match.json"))
    assert "players[1].id" in lieux and "first_server" in lieux


def test_toutes_les_erreurs_sont_rapportees(dossier):
    def corrompre(doc):
        doc["players"][0]["handedness"] = "x"
        doc["players"][1]["handedness"] = "y"
        doc["surface"] = "lune"

    _modifier_json(dossier / "match.json", corrompre)
    assert len(valider_fichier(dossier / "match.json")) == 3


def test_json_illisible(dossier):
    (dossier / "match.json").write_text("{pas du json", encoding="utf-8")
    problemes = valider_fichier(dossier / "match.json")
    assert len(problemes) == 1 and "illisible" in problemes[0].message
    assert valider_dossier(dossier)["match.json"] == problemes


# --- court.json ------------------------------------------------------------------------------


def test_court_segments_chevauchants(dossier):
    def corrompre(doc):
        premier = doc["segments"][0]
        premier["valid_to_frame"] = 50
        second = json.loads(json.dumps(premier))
        second["valid_from_frame"], second["valid_to_frame"] = 40, None
        doc["segments"].append(second)

    _modifier_json(dossier / "court.json", corrompre)
    problemes = valider_fichier(dossier / "court.json")
    assert _lieux(problemes) == ["segments[1].valid_from_frame"]
    assert "chevauche" in problemes[0].message


def test_court_segment_ouvert_suivi_dun_autre(dossier):
    def corrompre(doc):
        second = json.loads(json.dumps(doc["segments"][0]))
        second["valid_from_frame"] = 100
        doc["segments"].append(second)

    # le premier segment va jusqu'à la fin de la vidéo (valid_to_frame null)
    _modifier_json(dossier / "court.json", corrompre)
    assert _lieux(valider_fichier(dossier / "court.json")) == ["segments[1].valid_from_frame"]


def test_court_fin_avant_debut_et_keypoints_incomplets(dossier):
    def corrompre(doc):
        doc["segments"][0]["valid_from_frame"] = 10
        doc["segments"][0]["valid_to_frame"] = 5
        doc["segments"][0]["keypoints_px"].pop()

    _modifier_json(dossier / "court.json", corrompre)
    lieux = _lieux(valider_fichier(dossier / "court.json"))
    assert "segments[0].valid_to_frame" in lieux
    assert "segments[0].keypoints_px" in lieux


# --- events.json -----------------------------------------------------------------------------


def test_events_serve_number_invalide(dossier):
    _modifier_json(dossier / "events.json", lambda d: d["events"][2].__setitem__("serve_number", 3))
    assert _lieux(valider_fichier(dossier / "events.json")) == ["events[2].serve_number"]


def test_events_id_duplique(dossier):
    _modifier_json(dossier / "events.json", lambda d: d["events"][1].__setitem__("id", "e1"))
    problemes = valider_fichier(dossier / "events.json")
    assert _lieux(problemes) == ["events[1].id"]
    assert "e1" in problemes[0].message


def test_events_fin_avant_debut(dossier):
    def corrompre(doc):
        doc["events"][0]["frame_idx"] = 5
        doc["events"][0]["end_frame_idx"] = 3
        doc["events"][0]["t_ms"] = 200
        doc["events"][0]["end_t_ms"] = 100

    _modifier_json(dossier / "events.json", corrompre)
    lieux = _lieux(valider_fichier(dossier / "events.json"))
    assert lieux == ["events[0].end_frame_idx", "events[0].end_t_ms"]


def test_events_hit_id_invalide(dossier):
    def corrompre(doc):
        doc["events"][4]["hit_id"] = "e99"  # serve : id inexistant
        doc["events"][6]["hit_id"] = "e3"  # bounce : e3 existe mais c'est un serve
        doc["events"][8]["hit_id"] = "e99"  # net

    _modifier_json(dossier / "events.json", corrompre)
    problemes = valider_fichier(dossier / "events.json")
    assert _lieux(problemes) == ["events[4].hit_id", "events[6].hit_id", "events[8].hit_id"]
    assert "hit" in problemes[0].message


# --- stats.json ------------------------------------------------------------------------------


def test_stats_n_negatif(dossier):
    _modifier_json(dossier / "stats.json", lambda d: d["stats"][0].__setitem__("n", -1))
    assert _lieux(valider_fichier(dossier / "stats.json")) == ["stats[0].n"]


# --- ball.csv --------------------------------------------------------------------------------


def test_ball_colonne_manquante(dossier):
    _remplacer(dossier / "ball.csv", ",score", "")
    problemes = valider_fichier(dossier / "ball.csv")
    assert len(problemes) == 1
    assert problemes[0].emplacement == "ligne 1"
    assert "manquantes : score" in problemes[0].message


def test_ball_visible_sans_coordonnee(dossier):
    _modifier_cellule(dossier / "ball.csv", 2, "x", "")
    assert _lieux(valider_fichier(dossier / "ball.csv")) == ["ligne 2, colonne x"]


def test_ball_entier_decimal_refuse_et_nombre_illisible(dossier):
    _modifier_cellule(dossier / "ball.csv", 3, "frame_idx", "1.5")
    _modifier_cellule(dossier / "ball.csv", 3, "x", "abc")
    lieux = _lieux(valider_fichier(dossier / "ball.csv"))
    assert "ligne 3, colonne frame_idx" in lieux and "ligne 3, colonne x" in lieux


def test_ball_frame_idx_non_croissant(dossier):
    _modifier_cellule(dossier / "ball.csv", 4, "frame_idx", "1")
    assert _lieux(valider_fichier(dossier / "ball.csv")) == ["ligne 4, colonne frame_idx"]


def test_ball_score_hors_bornes(dossier):
    _modifier_cellule(dossier / "ball.csv", 2, "score", "1.5")
    assert _lieux(valider_fichier(dossier / "ball.csv")) == ["ligne 2, colonne score"]


def test_limite_de_problemes(dossier):
    lignes = ["frame_idx,t_ms,x,y,visible,score,source"]
    lignes += [f"{i},{i},1.0,1.0,1,0.5,lune" for i in range(80)]
    (dossier / "ball.csv").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    problemes = valider_fichier(dossier / "ball.csv")
    assert len(problemes) == validate.MAX_PROBLEMES + 1
    # 2 erreurs par ligne (enum de base + branche « visible ») : 160 - 50 = 110
    assert problemes[-1].message == "… 110 autres problèmes"


# --- players.csv -----------------------------------------------------------------------------


def test_players_doublon_et_bbox_inversee(dossier):
    _modifier_cellule(dossier / "players.csv", 3, "player_id", "P1")
    _modifier_cellule(dossier / "players.csv", 3, "bbox_x2", "800.0")
    lieux = _lieux(valider_fichier(dossier / "players.csv"))
    assert "ligne 3, colonne player_id" in lieux
    assert "ligne 3, colonne bbox_x2" in lieux


def test_players_confiance_keypoint_hors_bornes(dossier):
    _modifier_cellule(dossier / "players.csv", 2, "kp3_c", "1.7")
    assert _lieux(valider_fichier(dossier / "players.csv")) == ["ligne 2, colonne kp3_c"]


# --- frames.parquet --------------------------------------------------------------------------


def _ecrire_frames(chemin: Path, frame_idx: list[int], t_ms: list[float]) -> None:
    pq.write_table(
        pa.table({"frame_idx": pa.array(frame_idx, pa.int64()), "t_ms": pa.array(t_ms)}), chemin
    )


def test_frames_t_ms_non_croissant(dossier):
    _ecrire_frames(dossier / "frames.parquet", [0, 1, 2, 3], [0.0, 33.0, 33.0, 100.0])
    assert _lieux(valider_fichier(dossier / "frames.parquet")) == ["index 2, colonne t_ms"]


def test_frames_non_contigu(dossier):
    _ecrire_frames(dossier / "frames.parquet", [0, 1, 3], [0.0, 33.0, 67.0])
    assert _lieux(valider_fichier(dossier / "frames.parquet")) == ["index 2, colonne frame_idx"]


def test_frames_colonne_en_trop(dossier):
    pq.write_table(
        pa.table({"frame_idx": [0], "t_ms": [0.0], "pts": [1]}), dossier / "frames.parquet"
    )
    problemes = valider_fichier(dossier / "frames.parquet")
    assert len(problemes) == 1 and "en trop : pts" in problemes[0].message


def test_parquet_illisible(dossier):
    (dossier / "frames.parquet").write_bytes(b"pas du parquet")
    problemes = valider_fichier(dossier / "frames.parquet")
    assert len(problemes) == 1 and "illisible" in problemes[0].message


# --- Contrôles croisés -----------------------------------------------------------------------


def test_joueur_inconnu_de_match(dossier):
    # match.json ne déclare plus que P1 (P2 renommé P1) : P2 devient inconnu partout
    def corrompre(doc):
        doc["players"][1]["id"] = "P1"
        doc["first_server"] = "P1"

    _modifier_json(dossier / "match.json", corrompre)
    resultat = valider_dossier(dossier)
    assert "players[1].id" in _lieux(resultat["match.json"])
    # P2 apparaît comme player_id (hit e6), winner_id (point e1, game e2)
    lieux_events = _lieux(resultat["events.json"])
    assert "events[5].player_id" in lieux_events
    assert "events[0].winner_id" in lieux_events
    assert any("colonne player_id" in lieu for lieu in _lieux(resultat["players.csv"]))
    assert all(
        "inconnu de match.json" in p.message
        for p in resultat["events.json"] + resultat["players.csv"]
    )


def test_joueur_p3_dans_events_et_stats(dossier):
    _modifier_json(dossier / "events.json", lambda d: d["events"][3].__setitem__("player_id", "P3"))
    _modifier_json(dossier / "stats.json", lambda d: d["stats"][0].__setitem__("player_id", "P3"))
    resultat = valider_dossier(dossier)
    assert "events[3].player_id" in _lieux(resultat["events.json"])
    assert _lieux(resultat["stats.json"]).count("stats[0].player_id") == 2  # motif + croisé


# --- Géométrie du court ----------------------------------------------------------------------


def test_points_cles_du_court():
    pts = COURT_KEYPOINTS_M
    assert len(pts) == 14
    assert (schemas.L, schemas.Wd, schemas.Ws, schemas.S) == (11.885, 5.485, 4.115, 6.40)
    # longueur 23,77 m, largeur double 10,97 m, largeur simple 8,23 m (ITF)
    assert math.isclose(pts[3][1] - pts[1][1], 23.77)
    assert math.isclose(pts[1][0] - pts[0][0], 10.97)
    assert math.isclose(pts[6][0] - pts[4][0], 8.23)
    # symétries gauche/droite et loin/proche
    ensemble = {(round(x, 3), round(y, 3)) for x, y in pts}
    assert len(ensemble) == 14
    for x, y in pts:
        assert (round(-x, 3), round(y, 3)) in ensemble
        assert (round(x, 3), round(-y, 3)) in ensemble
