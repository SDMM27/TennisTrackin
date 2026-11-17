"""Tests de l'import CVAT -> contrats et de la configuration des étiquettes."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tennisvision import cli
from tennisvision.annotation import labels
from tennisvision.annotation.cvat import lire_export
from tennisvision.annotation.importeur import importer_cvat
from tennisvision.schemas import COURT_KEYPOINTS_M
from tennisvision.validate import valider_dossier

EXPORT = Path(__file__).parent / "fixtures" / "annotation" / "export_cvat.xml"
N_FRAMES = 40


def t_ms(frame: int) -> float:
    """Temps irrégulier (pas de frame_idx / fps) : détecte un calcul à cadence fixe."""
    return frame * 40.0 + (frame % 3)


def ecrire_frames(chemin: Path, n: int = N_FRAMES) -> Path:
    table = pa.table(
        {
            "frame_idx": pa.array(range(n), type=pa.int64()),
            "t_ms": pa.array([t_ms(i) for i in range(n)], type=pa.float64()),
        }
    )
    pq.write_table(table, chemin)
    return chemin


@pytest.fixture
def match(tmp_path) -> Path:
    dossier = tmp_path / "M1"
    dossier.mkdir()
    ecrire_frames(dossier / "frames.parquet")
    return dossier


def lire_ball(dossier: Path) -> list[dict]:
    with (dossier / "ball.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def evenements(dossier: Path, type_: str) -> list[dict]:
    doc = json.loads((dossier / "events.json").read_text(encoding="utf-8"))
    return [e for e in doc["events"] if e["type"] == type_]


# --- Configuration des étiquettes ----------------------------------------------------------


def test_configuration_livree_identique_aux_constantes():
    assert labels.charger_configuration() == labels.configuration_cvat()


def test_configuration_etiquettes_et_valeurs():
    config = {e["name"]: e for e in labels.charger_configuration()}
    assert set(config) == {
        labels.BALLE,
        labels.COURT,
        labels.FRAPPE,
        labels.REBOND,
        labels.FILET,
        labels.SERVICE,
        labels.DEBUT_POINT,
        labels.FIN_POINT,
        labels.SCORE,
    }
    assert {n: e["type"] for n, e in config.items()} == {
        "balle": "points",
        "court": "points",
        "frappe": "tag",
        "rebond": "points",
        "filet": "tag",
        "service": "tag",
        "debut_point": "tag",
        "fin_point": "tag",
        "score": "tag",
    }
    attributs = {e["name"]: {a["name"]: a for a in e["attributes"]} for e in config.values()}
    assert attributs["balle"]["visible"]["mutable"] is True
    assert attributs["court"]["kp"]["values"] == [str(i) for i in range(14)]
    assert attributs["frappe"]["coup"]["values"] == list(labels.COUPS)
    assert attributs["rebond"]["in_out"]["values"] == ["in", "out", "too_close"]
    assert set(attributs["fin_point"]) == {"gagnant", "serveur", "raison"}
    assert set(attributs["score"]) == {"jeux", "sets"}


# --- Lecture du XML ------------------------------------------------------------------------


def test_lire_export():
    export = lire_export(EXPORT)
    assert export.taille_image == (1280, 720)
    assert sorted(p.label for p in export.pistes).count("court") == 4
    balle = next(p for p in export.pistes if p.label == "balle")
    assert [p.frame for p in balle.points] == [2, 3, 4, 5, 8, 10, 12]
    assert not balle.points[1].keyframe and balle.points[3].outside
    assert {t.label for t in export.tags} >= {"frappe", "service", "fin_point", "score"}


def test_export_au_format_images_refuse(tmp_path):
    xml = tmp_path / "images.xml"
    xml.write_text('<annotations><image id="0" name="a.jpg"/></annotations>', encoding="utf-8")
    with pytest.raises(ValueError, match="CVAT for video 1.1"):
        lire_export(xml)


def test_xml_illisible_nomme_le_fichier(tmp_path):
    xml = tmp_path / "casse.xml"
    xml.write_text("<annotations><track", encoding="utf-8")
    with pytest.raises(ValueError, match="casse.xml : XML illisible"):
        lire_export(xml)


# --- Import complet ------------------------------------------------------------------------


def test_import_niveau_3(match):
    resultat = importer_cvat(EXPORT, match, annotateur="A1")
    assert resultat.niveau == 3
    assert resultat.fichiers == ["ball.csv", "court.json", "events.json"]
    # Tous les fichiers produits respectent les contrats
    problemes = valider_dossier(match)
    assert all(not p for p in problemes.values() if p is not None), problemes
    assert problemes["events.json"] == [] and problemes["ball.csv"] == []


def test_import_balle(match):
    importer_cvat(EXPORT, match)
    lignes = lire_ball(match)
    assert [int(r["frame_idx"]) for r in lignes] == list(range(2, 13))
    par_frame = {int(r["frame_idx"]): r for r in lignes}
    # t_ms vient de frames.parquet
    assert all(float(r["t_ms"]) == t_ms(int(r["frame_idx"])) for r in lignes)
    assert (par_frame[2]["visible"], par_frame[2]["source"]) == ("1", "annotation")
    assert (par_frame[3]["x"], par_frame[3]["y"], par_frame[3]["source"]) == (
        "110.0",
        "210.0",
        "interpolated",
    )
    assert par_frame[4]["source"] == "annotation"
    for f in (5, 6, 7, 9, 10, 11, 12):  # hors image, trou ou visible=false
        assert (par_frame[f]["visible"], par_frame[f]["x"], par_frame[f]["y"]) == ("0", "", "")
    assert par_frame[5]["source"] == "annotation" and par_frame[6]["source"] == "interpolated"
    assert par_frame[10]["source"] == "annotation"
    assert par_frame[8]["visible"] == "1"
    assert all(r["score"] == "" for r in lignes)
    assert list(lignes[0]) == ["frame_idx", "t_ms", "x", "y", "visible", "score", "source"]


def test_import_court(match):
    importer_cvat(EXPORT, match)
    court = json.loads((match / "court.json").read_text(encoding="utf-8"))
    assert court["image_size"] == [1280, 720]
    premier, second = court["segments"]
    assert (premier["valid_from_frame"], premier["valid_to_frame"]) == (0, 19)
    assert (second["valid_from_frame"], second["valid_to_frame"]) == (20, None)
    assert premier["keypoints_px"][:4] == [[400, 200], [880, 200], [150, 650], [1130, 650]]
    assert premier["keypoints_px"][4:] == [None] * 10
    assert premier["source"] == "annotation" and premier["confidence"] is None
    assert premier["reprojection_error_px"] < 1e-3  # 4 points : homographie exacte
    h = np.array(premier["homography"])
    for kp, (x, y) in enumerate(COURT_KEYPOINTS_M[:4]):
        px, py, w = h @ np.array([*premier["keypoints_px"][kp], 1.0])
        assert (px / w, py / w) == pytest.approx((x, y), abs=1e-3)


def test_import_evenements(match):
    importer_cvat(EXPORT, match)
    hits = evenements(match, "hit")
    assert [(h["id"], h["frame_idx"], h["player_id"], h["stroke"]) for h in hits] == [
        ("hit_0001", 6, "P1", "serve"),
        ("hit_0002", 10, "P2", "forehand"),
        ("hit_0003", 14, "P1", "backhand"),
    ]
    assert [h["outcome"] for h in hits] == ["in_play", "winner", "unforced_error"]
    assert all(e["t_ms"] == t_ms(e["frame_idx"]) for e in hits)
    assert all(h["source"] == "annotation" and h["confidence"] is None for h in hits)

    services = evenements(match, "serve")
    assert [(s["frame_idx"], s.get("hit_id")) for s in services] == [
        (6, "hit_0001"),
        (30, None),
    ]
    assert services[1]["fault_type"] == "net" and services[1]["serve_number"] == 2

    (net,) = evenements(match, "net")
    assert net["hit_id"] == "hit_0002"

    avant, rebond = evenements(match, "bounce")  # le premier jeu de court vaut dès la frame 0
    assert (avant["frame_idx"], avant["in_out"], "hit_id" in avant) == (2, "in", False)
    assert (rebond["frame_idx"], rebond["in_out"], rebond["hit_id"]) == (15, "out", "hit_0003")
    assert rebond["image_xy"] == [640, 400]
    court = json.loads((match / "court.json").read_text(encoding="utf-8"))
    h = np.array(court["segments"][0]["homography"])
    px, py, w = h @ np.array([640.0, 400.0, 1.0])
    assert rebond["court_xy"] == pytest.approx([px / w, py / w])


def test_import_points_et_jeux(match):
    importer_cvat(EXPORT, match)
    p1, p2 = evenements(match, "point")
    assert (p1["frame_idx"], p1["end_frame_idx"], p1["winner_id"], p1["server_id"]) == (
        5,
        16,
        "P2",
        "P1",
    )
    assert p1["rally_length"] == 3 and p1["desc"] == "double rebond"
    assert p1["end_t_ms"] == t_ms(16)
    assert (p2["frame_idx"], p2["end_frame_idx"], p2["winner_id"]) == (25, 38, "P1")
    assert p2["score_after"] == {"sets": [], "games": [1, 0], "points": ["0", "0"]}
    (jeu,) = evenements(match, "game")
    assert (jeu["frame_idx"], jeu["end_frame_idx"], jeu["winner_id"], jeu["server_id"]) == (
        5,
        38,
        "P1",
        "P1",
    )


def test_import_avertissements_et_rapport(match, capsys):
    resultat = importer_cvat(EXPORT, match, annotateur="A1")
    texte = "\n".join(resultat.avertissements)
    assert "fin_point (frame 1) sans debut_point" in texte
    assert "deux debut_point de suite (frames 20 et 25)" in texte
    assert "rebonds perdus" not in texte
    assert "service (frame 30) : aucune frappe de P2" in texte
    rapport = json.loads((match / "import_report.json").read_text(encoding="utf-8"))
    assert rapport["avertissements"] == resultat.avertissements
    assert rapport["annotateur"] == "A1" and rapport["niveau_annotation"] == 3
    assert rapport["export"] == "export_cvat.xml" and "\\" not in (
        match / "import_report.json"
    ).read_text(encoding="utf-8")
    assert rapport["comptes"]["hit"] == 3 and rapport["comptes"]["balle_frames"] == 11


def test_import_respecte_match_json(match):
    shutil.copy(
        Path(__file__).parent / "fixtures" / "contrats" / "valide" / "match.json",
        match / "match.json",
    )
    importer_cvat(EXPORT, match)
    assert all(not p for p in valider_dossier(match).values() if p is not None)


# --- Niveaux 1 et 2, erreurs ---------------------------------------------------------------


def xml(corps: str) -> str:
    return (
        '<?xml version="1.0"?><annotations><version>1.1</version><meta><task><original_size>'
        f"<width>640</width><height>360</height></original_size></task></meta>{corps}</annotations>"
    )


def importer_xml(tmp_path: Path, match: Path, corps: str):
    export = tmp_path / "e.xml"
    export.write_text(xml(corps), encoding="utf-8")
    return importer_cvat(export, match)


def test_niveau_1_points_seulement(tmp_path, match):
    corps = (
        '<tag frame="3" label="debut_point"/>'
        '<tag frame="9" label="fin_point"><attribute name="gagnant">P1</attribute>'
        '<attribute name="serveur">P1</attribute></tag>'
    )
    resultat = importer_xml(tmp_path, match, corps)
    assert resultat.niveau == 1
    assert resultat.fichiers == ["events.json"] and not (match / "ball.csv").exists()
    assert resultat.avertissements == []


def test_niveau_2_frappes(tmp_path, match):
    corps = (
        '<tag frame="4" label="frappe"><attribute name="joueur">P2</attribute>'
        '<attribute name="coup">smash</attribute><attribute name="issue">winner</attribute></tag>'
    )
    assert importer_xml(tmp_path, match, corps).niveau == 2


def test_attributs_invalides_signales_sans_planter(tmp_path, match):
    corps = (
        '<tag frame="4" label="frappe"><attribute name="joueur">P9</attribute></tag>'
        '<tag frame="5" label="frappe"><attribute name="joueur">P1</attribute>'
        '<attribute name="coup">lob</attribute></tag>'
        '<tag frame="99" label="frappe"><attribute name="joueur">P1</attribute></tag>'
    )
    resultat = importer_xml(tmp_path, match, corps)
    texte = "\n".join(resultat.avertissements)
    assert (
        "joueur invalide" in texte and "coup inconnu" in texte and "hors de frames.parquet" in texte
    )
    (hit,) = evenements(match, "hit")
    assert hit["stroke"] is None


def test_aucune_annotation_exploitable(tmp_path, match):
    with pytest.raises(ValueError, match="aucune annotation exploitable"):
        importer_xml(tmp_path, match, '<tag frame="1" label="autre"/>')


def test_frames_parquet_absent(tmp_path):
    match = tmp_path / "M2"
    match.mkdir()
    with pytest.raises(ValueError, match="frames.parquet"):
        importer_cvat(EXPORT, match)


def test_frames_parquet_explicite(tmp_path):
    match = tmp_path / "M3"
    match.mkdir()
    frames = ecrire_frames(tmp_path / "ailleurs.parquet")
    assert importer_cvat(EXPORT, match, frames=frames).niveau == 3


def test_sortie_invalide_rien_n_est_ecrit(match, monkeypatch):
    from tennisvision.annotation import importeur
    from tennisvision.validate import Probleme

    monkeypatch.setattr(
        importeur,
        "valider_dossier",
        lambda dossier: {"events.json": [Probleme("events.json", "events[0]", "cassé")]},
    )
    with pytest.raises(ValueError, match="rien n'est écrit"):
        importer_cvat(EXPORT, match)
    assert sorted(p.name for p in match.iterdir()) == ["frames.parquet"]


# --- CLI -----------------------------------------------------------------------------------


def test_cli_import_cvat(match, capsys):
    code = cli.main(["annotation", "import-cvat", str(EXPORT), "--match", str(match)])
    sortie = capsys.readouterr().out
    assert code == 0
    assert "Niveau d'annotation détecté : 3" in sortie
    assert "Avertissement : " in sortie
    assert (match / "import_report.json").is_file()


def test_cli_import_cvat_erreurs(tmp_path, match, capsys):
    sans_frames = tmp_path / "M9"
    sans_frames.mkdir()
    erreurs = [
        ["annotation", "import-cvat", str(tmp_path / "absent.xml"), "--match", str(match)],
        ["annotation", "import-cvat", str(EXPORT), "--match", str(tmp_path / "absent")],
        ["annotation", "import-cvat", str(EXPORT), "--match", str(sans_frames)],
    ]
    for argv in erreurs:
        assert cli.main(argv) == 2, argv
    err = capsys.readouterr().err
    assert err.count("Erreur :") == 3 and "frames.parquet" in err


# --- Revue : rebonds sans court, export partiel, occlusion, ré-import --------------------------

CORPS_COURT = "".join(
    f'<track id="c{kp}" label="court"><points frame="10" keyframe="1" outside="0" occluded="0" '
    f'points="{x},{y}"><attribute name="kp">{kp}</attribute></points></track>'
    for kp, (x, y) in enumerate([(100, 50), (500, 50), (50, 300), (550, 300)])
)
REBOND_2 = (
    '<track id="r" label="rebond"><points frame="2" keyframe="1" outside="0" occluded="0" '
    'points="200,200"><attribute name="in_out">in</attribute></points></track>'
)


def test_rebond_avant_le_premier_jeu_de_court_garde(tmp_path, match):
    importer_xml(tmp_path, match, CORPS_COURT + REBOND_2)
    court = json.loads((match / "court.json").read_text(encoding="utf-8"))
    assert court["segments"][0]["valid_from_frame"] == 0  # annoté à la frame 10
    (rebond,) = evenements(match, "bounce")
    assert rebond["frame_idx"] == 2


def test_rebonds_sans_aucun_court_erreur(tmp_path, match):
    with pytest.raises(ValueError, match="14 repères"):
        importer_xml(tmp_path, match, REBOND_2)
    assert sorted(p.name for p in match.iterdir()) == ["frames.parquet"]


def test_rebond_perdu_resume_et_rapport(tmp_path, match, monkeypatch):
    from tennisvision.annotation import importeur

    monkeypatch.setattr(importeur, "_projeter", lambda *a: None)
    frappe = '<tag frame="1" label="frappe"><attribute name="joueur">P1</attribute></tag>'
    resultat = importer_xml(tmp_path, match, CORPS_COURT + REBOND_2 + frappe)
    assert "1 rebonds perdus (pas de calibration du court) : frames 2" in resultat.avertissements
    rapport = json.loads((match / "import_report.json").read_text(encoding="utf-8"))
    assert rapport["rebonds_perdus"] == [2]


@pytest.mark.parametrize(
    "meta", ["<start_frame>5</start_frame>", "<frame_filter>step=2</frame_filter>"]
)
def test_export_partiel_refuse(tmp_path, match, meta):
    export = tmp_path / "p.xml"
    export.write_text(
        xml('<tag frame="3" label="debut_point"/>').replace("<task>", f"<task>{meta}"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="toute la vidéo"):
        importer_cvat(export, match)


@pytest.mark.parametrize(
    "meta", ["<start_frame>0</start_frame>", "<frame_filter>step=1</frame_filter>"]
)
def test_export_complet_accepte(tmp_path, match, meta):
    export = tmp_path / "c.xml"
    export.write_text(
        xml('<tag frame="3" label="debut_point"/>').replace("<task>", f"<task>{meta}"),
        encoding="utf-8",
    )
    importer_cvat(export, match)


def test_balle_occluded_invisible(tmp_path, match):
    corps = (
        '<track id="0" label="balle">'
        '<points frame="1" keyframe="1" outside="0" occluded="0" points="10,10"/>'
        '<points frame="2" keyframe="1" outside="0" occluded="1" points="12,12"/>'
        '<points frame="3" keyframe="1" outside="0" occluded="0" points="14,14"/></track>'
    )
    importer_xml(tmp_path, match, corps)
    assert [(r["frame_idx"], r["visible"], r["x"]) for r in lire_ball(match)] == [
        ("1", "1", "10.0"),
        ("2", "0", ""),
        ("3", "1", "14.0"),
    ]


def test_reimport_supprime_les_fichiers_perimes(tmp_path, match):
    importer_cvat(EXPORT, match)
    assert (match / "ball.csv").is_file() and (match / "court.json").is_file()
    corps = (
        '<tag frame="3" label="debut_point"/>'
        '<tag frame="9" label="fin_point"><attribute name="gagnant">P1</attribute></tag>'
    )
    resultat = importer_xml(tmp_path, match, corps)
    assert set(resultat.supprimes) == {"ball.csv", "court.json"}
    assert not (match / "ball.csv").exists() and not (match / "court.json").exists()
    rapport = json.loads((match / "import_report.json").read_text(encoding="utf-8"))
    assert sorted(rapport["fichiers_supprimes"]) == ["ball.csv", "court.json"]
    assert any("supprimé" in a for a in resultat.avertissements)
