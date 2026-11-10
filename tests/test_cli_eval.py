"""Tests de bout en bout des commandes `eval` et `tenniset export-events`."""

import json
from pathlib import Path

import pytest

from tennisvision import cli

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
TENNISET = FIXTURES / "tenniset"
AMATEUR = FIXTURES / "amateur"


def rapport_json(sortie: Path) -> dict:
    (chemin,) = sortie.glob("*/rapport.json")
    return json.loads(chemin.read_text(encoding="utf-8"))


def test_export_puis_eval_tenniset(tmp_path, capsys):
    export = tmp_path / "export"
    assert (
        cli.main(["tenniset", "export-events", "--data-dir", str(TENNISET), "-o", str(export)]) == 0
    )
    doc = json.loads((export / "V901" / "events.json").read_text(encoding="utf-8"))
    assert len(doc["events"]) == 16  # split « all » par défaut

    sortie = tmp_path / "rapports"
    args = ["eval", "--dataset", "tenniset", "--data-dir", str(TENNISET)]
    code = cli.main([*args, "--predictions", str(export), "--output-dir", str(sortie)])
    assert code == 0
    assert "Rapport :" in capsys.readouterr().out

    (dossier,) = [d for d in sortie.iterdir() if d.is_dir()]
    assert (dossier / "rapport.json").is_file()
    assert (dossier / "rapport.html").is_file()
    assert (sortie / "historique.jsonl").is_file()

    rapport = rapport_json(sortie)
    assert rapport["meta"]["split"] == "test"
    ev = rapport["agrege"]["evenements"]
    assert ev["frappes"]["tolerances"]["2"]["f1"] == 1.0  # oracle : les prédictions hors split
    assert ev["frappes"]["tolerances"]["2"]["n_gt"] == 2  # sont écartées, pas comptées en FP
    assert ev["services"]["f1"] == 1.0
    assert ev["points"]["f1"] == 1.0
    assert ev["points"]["jeux"]["f1"] == 1.0
    assert rapport["agrege"]["evenements"]["rebonds"] is None


def test_export_events_selon_le_split(tmp_path):
    cli.main(
        ["tenniset", "export-events", "--data-dir", str(TENNISET), "--split", "train",
         "-o", str(tmp_path)]
    )  # fmt: skip
    doc = json.loads((tmp_path / "V901" / "events.json").read_text(encoding="utf-8"))
    assert {e["frame_idx"] for e in doc["events"]} <= set(range(80, 351))
    assert len(doc["events"]) == 9


def test_eval_tenniset_prediction_vide(tmp_path):
    (tmp_path / "vide").mkdir()
    sortie = tmp_path / "rapports"
    code = cli.main(
        ["eval", "--dataset", "tenniset", "--data-dir", str(TENNISET),
         "--predictions", str(tmp_path / "vide"), "--output-dir", str(sortie)]
    )  # fmt: skip
    assert code == 0
    rapport = rapport_json(sortie)
    assert rapport["agrege"]["evenements"]["frappes"]["tolerances"]["5"]["rappel"] == 0
    assert rapport["avertissements"] == ["V901 : events.json absent, prédiction vide"]


def test_eval_amateur(tmp_path, capsys):
    sortie = tmp_path / "rapports"
    code = cli.main(
        ["eval", "--dataset", "amateur", "--gt-dir", str(AMATEUR),
         "--predictions", str(AMATEUR), "--output-dir", str(sortie)]
    )  # fmt: skip
    assert code == 0
    assert "F1 balle_10px : 1.000" in capsys.readouterr().out
    rapport = rapport_json(sortie)
    assert rapport["meta"]["dataset"] == "amateur"
    assert rapport["agrege"]["joueurs"]["idf1"] == 1.0
    assert rapport["agrege"]["evenements"]["rebonds"]["tolerances"]["2"]["f1"] == 1.0


def test_erreurs_d_usage(tmp_path, capsys):
    sortie = ["--output-dir", str(tmp_path / "rapports")]
    erreurs = [
        # dossier de prédictions introuvable
        ["eval", "--dataset", "tenniset", "--data-dir", str(TENNISET),
         "--predictions", str(tmp_path / "absent"), *sortie],
        # --gt-dir manquant pour amateur
        ["eval", "--dataset", "amateur", "--predictions", str(AMATEUR), *sortie],
        # dossier de vérité terrain introuvable
        ["eval", "--dataset", "amateur", "--gt-dir", str(tmp_path / "absent"),
         "--predictions", str(AMATEUR), *sortie],
        # split sans match (« val » est vide dans splits.json)
        ["eval", "--dataset", "amateur", "--gt-dir", str(AMATEUR), "--split", "val",
         "--predictions", str(AMATEUR), *sortie],
        # aucune annotation TenniSet
        ["eval", "--dataset", "tenniset", "--data-dir", str(tmp_path),
         "--predictions", str(AMATEUR), *sortie],
        ["tenniset", "export-events", "--data-dir", str(tmp_path), "-o", str(tmp_path / "x")],
    ]  # fmt: skip
    for argv in erreurs:
        assert cli.main(argv) == 2, argv
    assert not (tmp_path / "rapports").exists()
    assert "Erreur" in capsys.readouterr().err


def test_split_absent_de_splits_json(tmp_path):
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / "splits.json").write_text('{"test": ["M1"]}', encoding="utf-8")
    code = cli.main(
        ["eval", "--dataset", "amateur", "--gt-dir", str(gt), "--split", "train",
         "--predictions", str(tmp_path), "--output-dir", str(tmp_path / "r")]
    )  # fmt: skip
    assert code == 2


def test_aide_eval(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["eval", "--help"])
    assert "--predictions" in capsys.readouterr().out
