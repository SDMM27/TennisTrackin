"""Tests des étiquettes faibles TenniSet et de l'échantillon de `Side`."""

import csv
import json
from pathlib import Path

import pytest

from tennisvision import cli
from tennisvision.datasets.etiquettes_faibles import (
    COLONNES_ECHANTILLON,
    COLONNES_ETIQUETTES,
    analyser_desc,
    couverture,
    echantillon_cotes,
)

PHRASES = [
    # (phrase, fin, dernier_coup, retour, direction)
    ("Williams a good serve, Sharapova is unable to return it.", "ace", None, None, None),
    ("Quick serve out wide is an ace.", "ace", None, None, None),
    ("Good serve aimed at T, it's an ace.", "ace", None, None, None),
    ("Double Fault.", "double_faute", None, None, None),
    ("Second serve doesn't make it over the net, double fault.", "double_faute", None, None, None),
    (
        (
            "Sharapova serves a good one, Williams puts back a forehand return, good rally, "
            "Serena hits a forehand winner down the line."
        ),
        "gagnant",
        "forehand",
        "forehand",
        "down-the-line",
    ),
    (
        (
            "Williams hits a good serve, Sharapova struggles with it, before hitting a backhand "
            "into the net."
        ),
        "faute_filet",
        "backhand",
        None,
        None,
    ),
    (
        (
            "Good serve aimed at T, Williams only reaches to it, short rally, Serena cross-court "
            "fails to clear the net in the middle."
        ),
        "faute_filet",
        None,
        None,
        "cross-court",
    ),
    (
        (
            "High kick serve, Sharapova returns a forehand return, short rally, Sharapova "
            "cross-court backhand lands out-side the court."
        ),
        "faute_out",
        "backhand",
        "forehand",
        "cross-court",
    ),
    (
        (
            "Good serve wide, Sharapova returns a quick forehand return, short rally, Sharapova "
            "hits a backhand long and out of court."
        ),
        "faute_out",
        "backhand",
        "forehand",
        None,
    ),
    (  # faute de frappe observée dans les données : « forehandhand »
        (
            "Fine serve, Williams puts back a forehandhand return, brief rally, Serena sends a "
            "forehand wide of the court."
        ),
        "faute_out",
        "forehand",
        "forehand",
        None,
    ),
    (
        (
            "Good serve in the middle, Williams smashes a backhand return which Sharapova is "
            "unable to return."
        ),
        "gagnant",
        "backhand",
        "backhand",
        None,
    ),
    ("", "inconnu", None, None, None),
    ("Sharapova serves, Serena replies with a puzzling shot.", "inconnu", None, None, None),
]


@pytest.mark.parametrize(("phrase", "fin", "coup", "retour", "direction"), PHRASES)
def test_analyser_desc(phrase, fin, coup, retour, direction):
    etiquettes = analyser_desc(phrase)
    assert etiquettes["fin"] == fin
    assert etiquettes["dernier_coup"] == coup
    assert etiquettes["retour"] == retour
    assert etiquettes["direction"] == direction


def test_analyser_desc_types_de_coups():
    assert analyser_desc("Serve, Serena lobs a forehand lob over the net into the court.")["lob"]
    assert analyser_desc("Serve, Williams returns, Serena puts a backhand volley into the net.")[
        "volley"
    ]
    assert analyser_desc("Serve, Williams returns, Serena smashes a forehand winner.")["smash"]
    assert analyser_desc("Serve, Williams returns, Serena slices a backhand long.")["slice"]
    neutre = analyser_desc("Serve, Williams returns, Serena hits a forehand winner.")
    assert not any(neutre[k] for k in ("lob", "volley", "smash", "slice"))


def test_couverture():
    lignes = [{"fin": "ace"}, {"fin": "inconnu"}, {"fin": "gagnant"}, {"fin": "faute_out"}]
    assert couverture(lignes) == 0.75
    assert couverture([]) is None


def annotation(points: list[tuple[str, str, int, int]], hits: list[dict] | None = None) -> dict:
    return {
        "video": "data/videos/V900.mp4",
        "classes": {
            "Point": [
                {"name": n, "start": s, "end": e, "desc": d, "custom": {}} for n, d, s, e in points
            ],
            "Hit": hits or [],
        },
    }


@pytest.fixture
def data_dir(tmp_path) -> Path:
    dossier = tmp_path / "data"
    (dossier / "annotations").mkdir(parents=True)
    doc = annotation(
        [
            ("0001", "Quick serve is an ace.", 10, 40),
            ("0002", "Double Fault.", 50, 80),
            ("0003", "Serve, Williams returns a backhand return, Serena nets a forehand.", 90, 150),
            ("0004", "", 160, 200),
        ]
    )
    (dossier / "annotations" / "V900.json").write_text(json.dumps(doc), encoding="utf-8")
    return dossier


def test_cli_weak_labels(data_dir, tmp_path, capsys):
    sortie = tmp_path / "weak.csv"
    code = cli.main(
        ["tenniset", "weak-labels", "--data-dir", str(data_dir), "--sortie", str(sortie)]
    )
    assert code == 0
    assert "Couverture : 75.0 %" in capsys.readouterr().out
    with sortie.open(encoding="utf-8", newline="") as f:
        lignes = list(csv.DictReader(f))
    assert list(lignes[0]) == COLONNES_ETIQUETTES
    assert [r["fin"] for r in lignes] == ["ace", "double_faute", "faute_filet", "inconnu"]
    assert lignes[0]["video"] == "V900" and lignes[0]["frame_debut"] == "10"
    assert lignes[2]["retour"] == "backhand" and lignes[2]["dernier_coup"] == "forehand"
    assert lignes[2]["lob"] == "0"


def test_cli_weak_labels_sans_annotation(tmp_path, capsys):
    assert cli.main(["tenniset", "weak-labels", "--data-dir", str(tmp_path)]) == 2
    assert "Aucune annotation" in capsys.readouterr().err


def test_cli_weak_labels_annotation_invalide(tmp_path, capsys):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "V900.json").write_text('{"video": "x"}', encoding="utf-8")
    assert cli.main(["tenniset", "weak-labels", "--data-dir", str(tmp_path)]) == 2
    assert "V900.json" in capsys.readouterr().err


# --- Échantillon de Side -------------------------------------------------------------------


def hit(nom: str, debut: int, fin: int, joueur: str, side: str, type_: str) -> dict:
    custom = {"Player": joueur, "Side": side, "Type": type_}
    return {"name": nom, "start": debut, "end": fin, "desc": "", "custom": custom}


@pytest.fixture
def data_side(tmp_path, video_synthetique) -> Path:
    dossier = tmp_path / "data"
    (dossier / "annotations").mkdir(parents=True)
    (dossier / "videos").mkdir()
    (dossier / "videos" / "V900.mp4").write_bytes(video_synthetique.read_bytes())
    hits = [hit(f"{i:04d}", 2 * i, 2 * i + 4, "Alpha", "Forehand", "Topspin") for i in range(10)]
    for nom, h in (("V900", hits), ("V901", hits[:3])):  # V901 : pas de vidéo
        doc = annotation([], h)
        (dossier / "annotations" / f"{nom}.json").write_text(json.dumps(doc), encoding="utf-8")
    return dossier


def test_echantillon_reproductible(data_side):
    a = echantillon_cotes(data_side, n=5, graine=1)
    b = echantillon_cotes(data_side, n=5, graine=1)
    c = echantillon_cotes(data_side, n=5, graine=2)
    assert a == b and a != c
    assert len(a) == 5 and len({(h["video"], h["frame_debut"]) for h in a}) == 5
    assert len(echantillon_cotes(data_side, n=500)) == 13  # plus de frappes que demandé
    assert list(a[0]) == COLONNES_ECHANTILLON
    assert a[0]["verif_coup"] == "" and a[0]["verif_cote_image"] == ""


def test_cli_side_sample_avec_images(data_side, tmp_path, capsys):
    sortie, images = tmp_path / "side.csv", tmp_path / "images"
    code = cli.main(
        ["tenniset", "side-sample", "--data-dir", str(data_side), "--n", "13",
         "--sortie", str(sortie), "--images", str(images)]
    )  # fmt: skip
    assert code == 0
    sortie_texte = capsys.readouterr().out
    assert "13 frappes" in sortie_texte
    assert "vidéo V901 absente" in sortie_texte  # avertissement, pas une erreur
    with sortie.open(encoding="utf-8", newline="") as f:
        lignes = list(csv.DictReader(f))
    assert lignes[0]["side"] == "Forehand" and lignes[0]["type"] == "Topspin"
    jpg = sorted(p.name for p in images.glob("*.jpg"))
    assert len(jpg) == 10  # les 10 frappes de V900, dont la vidéo existe
    assert "V900_0_4.jpg" in jpg


def test_cli_side_sample_sans_images(data_side, tmp_path):
    sortie = tmp_path / "side.csv"
    code = cli.main(
        [
            "tenniset",
            "side-sample",
            "--data-dir",
            str(data_side),
            "--n",
            "4",
            "--sortie",
            str(sortie),
        ]
    )
    assert code == 0 and sortie.is_file()


@pytest.mark.parametrize(
    ("phrase", "fin"),
    [
        (
            "Good serve, Williams returns, Serena comes to the net and hits a forehand wide.",
            "faute_out",
        ),
        (
            "Fine serve, Sharapova returns, Serena runs to the net, hits a backhand long.",
            "faute_out",
        ),
        (
            "Good serve, Sharapova returns, Serena rushes to net but fails to keep a forehand in play.",
            "faute_out",
        ),
        (
            "Fine serve, Williams returns, Serena approaches the net, a forehand goes out.",
            "faute_out",
        ),
        (
            "Fast serve, Williams returns, Serena moves closer to net and nets a backhand.",
            "faute_filet",
        ),
        (
            "Fine serve, Williams returns, Serena comes to the net, smashes a forehand winner.",
            "gagnant",
        ),
    ],
)
def test_approche_du_filet_n_est_pas_une_faute_de_filet(phrase, fin):
    assert analyser_desc(phrase)["fin"] == fin
