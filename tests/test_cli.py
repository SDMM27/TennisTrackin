"""Tests de la CLI (sans charger YOLO)."""

import hashlib
import shutil
from pathlib import Path

import pytest

from tennisvision import cli, config, models
from tennisvision.models import ModeleRef

RESUME = {
    "video": "v.mp4",
    "fps_nominal": 25.0,
    "resolution": [64, 48],
    "rotation": 0,
    "intervalle": [0, 9],
    "intervalle_ms": [0.0, 360.0],
    "duree_ms": 360.0,
    "n_frames_traitees": 10,
    "n_identites": 2,
    "top_identites": [{"track_id": 1, "n_frames": 10}],
    "n_detections_balle": 3,
    "taux_detection_balle": 0.3,
    "modele": {"nom": "yolo11n", "version": "11", "sha256": "a" * 64},
    "fps_traitement": 12.0,
    "sorties": {
        "annotated": None,
        "tracks": "tracks.csv",
        "frames": "frames.parquet",
        "summary": "summary.json",
        "match": None,
    },
    "dossier": "out",
}


@pytest.fixture
def faux_analyze(monkeypatch):
    """Remplace `analyze` : enregistre les appels, retourne un résumé fixe."""
    appels = {}

    def remplacant(video, **kwargs):
        appels.update(kwargs, video=video)
        return RESUME

    monkeypatch.setattr(cli, "analyze", remplacant)
    return appels


@pytest.fixture
def dossier_tenniset(tmp_path):
    racine = tmp_path / "tenniset"
    annotations = config.points_file(racine)
    annotations.parent.mkdir(parents=True)
    annotations.write_text("P00000001\tV006\t100\t200\t0001\n", encoding="utf-8")
    video = config.tenniset_video(racine, "V006")
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    return racine


def test_parsing_analyze():
    args = cli.build_parser().parse_args(
        [
            "analyze",
            "v.mp4",
            "--start",
            "3",
            "--end",
            "9",
            "--match",
            "match.json",
            "--model",
            "yolo11n",
            "--conf",
            "0.4",
            "--device",
            "cpu",
            "--no-video",
            "-v",
        ]
    )
    assert (args.start, args.end, args.conf, args.device) == (3, 9, 0.4, "cpu")
    assert args.match == Path("match.json") and args.model == "yolo11n"
    assert args.no_video and args.verbose


def test_analyze_appelle_pipeline(faux_analyze, capsys):
    code = cli.main(["analyze", "v.mp4", "--start", "3", "--end", "9", "--no-video"])
    assert code == 0
    assert faux_analyze["start"] == 3 and faux_analyze["end"] == 9
    assert faux_analyze["write_video"] is False and faux_analyze["match"] is None
    sortie = capsys.readouterr().out
    assert "30.0 %" in sortie and "yolo11n" in sortie and "out" in sortie


def test_analyze_transmet_match(faux_analyze):
    assert cli.main(["analyze", "v.mp4", "--match", "m.json"]) == 0
    assert faux_analyze["match"] == Path("m.json")


def test_analyze_n_a_plus_d_option_point(faux_analyze):
    assert cli.main(["analyze", "v.mp4", "--point", "P00000001"]) == 2
    assert "video" not in faux_analyze


def test_tenniset_analyze_resout_video_et_bornes(dossier_tenniset, faux_analyze):
    code = cli.main(["tenniset", "analyze", "P00000001", "--data-dir", str(dossier_tenniset)])
    assert code == 0
    assert faux_analyze["video"] == dossier_tenniset / "videos" / "V006.mp4"
    assert (faux_analyze["start"], faux_analyze["end"]) == (100, 200)


def test_tenniset_analyze_accepte_les_options_communes(dossier_tenniset, faux_analyze):
    code = cli.main(
        ["tenniset", "analyze", "P00000001", "--data-dir", str(dossier_tenniset)]
        + ["--no-video", "--conf", "0.5", "--match", "m.json"]
    )
    assert code == 0
    assert faux_analyze["write_video"] is False and faux_analyze["conf"] == 0.5
    assert faux_analyze["match"] == Path("m.json")


def test_tenniset_data_dir_depuis_l_environnement(dossier_tenniset, faux_analyze, monkeypatch):
    monkeypatch.setenv(config.ENV_TENNISET_DIR, str(dossier_tenniset))
    assert cli.main(["tenniset", "analyze", "P00000001"]) == 0
    assert faux_analyze["video"] == dossier_tenniset / "videos" / "V006.mp4"


def test_tenniset_point_inconnu(dossier_tenniset, faux_analyze, capsys):
    code = cli.main(["tenniset", "analyze", "P99999999", "--data-dir", str(dossier_tenniset)])
    assert code == 2
    assert "P99999999" in capsys.readouterr().err
    assert "video" not in faux_analyze


def test_tenniset_sans_annotations(tmp_path, capsys):
    assert cli.main(["tenniset", "analyze", "P00000001", "--data-dir", str(tmp_path)]) == 2
    assert "points.txt" in capsys.readouterr().err


def test_tenniset_clip_exige_la_sortie(dossier_tenniset):
    assert cli.main(["tenniset", "clip", "P00000001", "--data-dir", str(dossier_tenniset)]) == 2


def test_tenniset_clip_appelle_cut_clip(dossier_tenniset, tmp_path, monkeypatch):
    appels = []

    def faux_cut_clip(video, dst, start, end):
        appels.append((video, dst, start, end))
        return dst

    monkeypatch.setattr(cli, "cut_clip", faux_cut_clip)
    sortie = tmp_path / "c.mp4"
    code = cli.main(
        ["tenniset", "clip", "P00000001", "--data-dir", str(dossier_tenniset), "-o", str(sortie)]
    )
    assert code == 0
    assert appels == [(dossier_tenniset / "videos" / "V006.mp4", sortie, 100, 200)]


def test_video_inexistante(tmp_path, capsys):
    code = cli.main(["analyze", str(tmp_path / "absente.mp4"), "--no-video"])
    assert code == 2
    assert "introuvable" in capsys.readouterr().err


def test_bornes_invalides_code_2_sans_dossier(video_synthetique, tmp_path, capsys):
    sortie = tmp_path / "sortie"
    code = cli.main(
        ["analyze", str(video_synthetique), "--start", "9", "--end", "3"]
        + ["--output-dir", str(sortie)]
    )
    assert code == 2 and not sortie.exists()
    assert "start" in capsys.readouterr().err


def test_modele_absent_indique_models_pull(video_synthetique, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(config.ENV_MODELS_DIR, str(tmp_path / "cache_vide"))
    code = cli.main(["analyze", str(video_synthetique), "--output-dir", str(tmp_path / "s")])
    assert code == 2
    assert "models pull" in capsys.readouterr().err


def test_argument_invalide():
    assert cli.main(["analyze"]) == 2


def test_clip(video_synthetique, tmp_path, capsys):
    sortie = tmp_path / "clip.mp4"
    code = cli.main(
        ["clip", str(video_synthetique), "--start", "2", "--end", "6", "-o", str(sortie)]
    )
    assert code == 0 and sortie.exists()


def test_clip_sans_fin(video_synthetique, tmp_path, capsys):
    code = cli.main(["clip", str(video_synthetique), "-o", str(tmp_path / "c.mp4")])
    assert code == 2


def test_clip_n_a_plus_d_option_point(video_synthetique, tmp_path):
    code = cli.main(
        ["clip", str(video_synthetique), "--point", "P00000001", "-o", str(tmp_path / "c.mp4")]
    )
    assert code == 2


# --- models ---------------------------------------------------------------------------------

CONTENU = b"poids factices" * 100


@pytest.fixture
def registre_test(monkeypatch, tmp_path):
    source = tmp_path / "source" / "yolo11n.pt"
    source.parent.mkdir()
    source.write_bytes(CONTENU)
    ref = ModeleRef(
        nom="yolo11n",
        version="11",
        fichier="yolo11n.pt",
        url=source.as_uri(),
        sha256=hashlib.sha256(CONTENU).hexdigest(),
        licence="MIT",
        description="modèle de test",
    )
    monkeypatch.setattr(models, "registre", lambda: {"yolo11n": ref})
    monkeypatch.setattr(cli, "registre", lambda: {"yolo11n": ref})
    return source


def test_models_list_absent_puis_present(registre_test, tmp_path, capsys):
    cache = tmp_path / "cache"
    assert cli.main(["models", "list", "--models-dir", str(cache)]) == 0
    sortie = capsys.readouterr().out
    assert str(cache) in sortie and "yolo11n" in sortie and "MIT" in sortie
    assert "absent" in sortie

    assert cli.main(["models", "pull", "--models-dir", str(cache)]) == 0
    assert (cache / "yolo11n.pt").read_bytes() == CONTENU
    capsys.readouterr()
    assert cli.main(["models", "list", "--models-dir", str(cache)]) == 0
    assert "présent" in capsys.readouterr().out


def test_models_pull_nom_explicite(registre_test, tmp_path):
    cache = tmp_path / "cache"
    assert cli.main(["models", "pull", "yolo11n", "--models-dir", str(cache)]) == 0
    assert (cache / "yolo11n.pt").is_file()


def test_models_pull_utilise_la_variable_d_environnement(registre_test, tmp_path, monkeypatch):
    monkeypatch.setenv(config.ENV_MODELS_DIR, str(tmp_path / "env"))
    assert cli.main(["models", "pull"]) == 0
    assert (tmp_path / "env" / "yolo11n.pt").is_file()


def test_models_pull_modele_inconnu(registre_test, tmp_path, capsys):
    assert cli.main(["models", "pull", "inconnu", "--models-dir", str(tmp_path / "c")]) == 2
    assert "inconnu" in capsys.readouterr().err


def test_models_import(registre_test, tmp_path):
    cache = tmp_path / "cache"
    args = ["models", "import", "yolo11n", str(registre_test), "--models-dir", str(cache)]
    assert cli.main(args) == 0
    assert (cache / "yolo11n.pt").read_bytes() == CONTENU


def test_models_import_somme_incorrecte(registre_test, tmp_path, capsys):
    faux = tmp_path / "faux.pt"
    faux.write_bytes(b"autre contenu")
    args = ["models", "import", "yolo11n", str(faux), "--models-dir", str(tmp_path / "c")]
    assert cli.main(args) == 2
    assert "SHA-256" in capsys.readouterr().err


def test_modele_resolu_apres_import(registre_test, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv(config.ENV_MODELS_DIR, str(cache))
    assert cli.main(["models", "import", "yolo11n", str(registre_test)]) == 0
    assert models.resoudre_modele(None) == cache / "yolo11n.pt"


# --- validate -------------------------------------------------------------------------------

CONTRATS_VALIDE = Path(__file__).parent / "fixtures" / "contrats" / "valide"


def test_validate_dossier_valide(capsys):
    assert cli.main(["validate", str(CONTRATS_VALIDE)]) == 0
    sortie = capsys.readouterr().out
    assert "OK  events.json" in sortie and "INVALIDE" not in sortie


def test_validate_dossier_corrompu(tmp_path, capsys):
    cible = tmp_path / "sortie"
    shutil.copytree(CONTRATS_VALIDE, cible)
    (cible / "ball.csv").unlink()
    match = (CONTRATS_VALIDE / "match.json").read_text(encoding="utf-8")
    (cible / "match.json").write_text(match.replace("left", "ambidextre"), encoding="utf-8")
    assert cli.main(["validate", str(cible)]) == 1
    sortie = capsys.readouterr().out
    assert "ABSENT  ball.csv" in sortie
    assert "INVALIDE  match.json (1 problèmes)" in sortie
    assert "  - players[1].handedness : " in sortie


def test_validate_fichier_unique(capsys):
    assert cli.main(["validate", str(CONTRATS_VALIDE / "court.json")]) == 0
    assert capsys.readouterr().out.strip() == "OK  court.json"


def test_validate_chemin_inexistant(tmp_path, capsys):
    assert cli.main(["validate", str(tmp_path / "absent")]) == 2
    assert "introuvable" in capsys.readouterr().err


def test_validate_dossier_vide(tmp_path, capsys):
    assert cli.main(["validate", str(tmp_path)]) == 2
    assert "Aucun fichier de contrat" in capsys.readouterr().err


def test_validate_nom_de_fichier_inconnu(tmp_path, capsys):
    (tmp_path / "autre.json").write_text("{}", encoding="utf-8")
    assert cli.main(["validate", str(tmp_path / "autre.json")]) == 2
    assert "autre.json" in capsys.readouterr().err
