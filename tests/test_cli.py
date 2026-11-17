"""Tests de la CLI (sans charger YOLO)."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tennisvision import cli, config, models
from tennisvision.models import ModeleRef
from tennisvision.validate import valider_fichier

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
    "modele": {"nom": "yolo11n", "version": "11", "sha256": "a" * 64},
    "balle": {
        "modele": {"nom": "wasb_tennis", "version": "bmvc2023", "sha256": "b" * 64},
        "pas_temporel": 1,
        "seuil_score": 0.5,
        "taux_visible": 0.3,
        "n_detectes": 2,
        "n_interpoles": 1,
        "diagnostic": {"n_candidats": 8},
        "masque": None,
    },
    "fps_traitement": 12.0,
    "sorties": {
        "annotated": None,
        "tracks": "tracks.csv",
        "ball": "ball.csv",
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
    assert "wasb_tennis" in sortie and "2 détectées, 1 interpolées" in sortie


def test_parsing_options_balle():
    args = cli.build_parser().parse_args(
        ["analyze", "v.mp4", "--ball-model", "aucun", "--ball-threshold", "0.3"]
        + ["--court", "court.json"]
    )
    assert (args.ball_model, args.seuil_balle, args.court, args.mask) == (
        "aucun",
        0.3,
        Path("court.json"),
        None,
    )
    defaut = cli.build_parser().parse_args(["analyze", "v.mp4", "--mask", "m.json"])
    assert (defaut.ball_model, defaut.seuil_balle, defaut.mask) == (
        "wasb_tennis",
        0.5,
        Path("m.json"),
    )


def test_analyze_transmet_les_options_balle(faux_analyze):
    code = cli.main(["analyze", "v.mp4", "--ball-model", "aucun", "--mask", "m.json"])
    assert code == 0
    assert faux_analyze["modele_balle"] == "aucun" and faux_analyze["masque"] == Path("m.json")
    assert faux_analyze["seuil_balle"] == 0.5 and faux_analyze["court"] is None


def test_tenniset_analyze_transmet_les_options_balle(dossier_tenniset, faux_analyze):
    code = cli.main(
        ["tenniset", "analyze", "P00000001", "--data-dir", str(dossier_tenniset)]
        + ["--ball-threshold", "0.7", "--court", "c.json"]
    )
    assert code == 0
    assert faux_analyze["seuil_balle"] == 0.7 and faux_analyze["court"] == Path("c.json")


def test_analyze_balle_desactivee_affichage(monkeypatch, capsys):
    monkeypatch.setattr(cli, "analyze", lambda video, **kw: {**RESUME, "balle": None})
    assert cli.main(["analyze", "v.mp4", "--ball-model", "aucun"]) == 0
    assert "Balle : désactivée" in capsys.readouterr().out


def test_analyze_court_et_mask_refuses(video_vfr, tmp_path, capsys):
    court = Path(__file__).parent / "fixtures" / "contrats" / "valide" / "court.json"
    masque = tmp_path / "m.json"
    masque.write_text('{"polygone": [[0, 0], [10, 0], [10, 10]]}', encoding="utf-8")
    code = cli.main(
        ["analyze", str(video_vfr), "--output-dir", str(tmp_path / "s"), "--ball-model", "aucun"]
        + ["--court", str(court), "--mask", str(masque)]
    )
    assert code == 2
    assert "s'excluent" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()


def test_analyze_poids_balle_absents(video_vfr, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(config.ENV_MODELS_DIR, str(tmp_path / "cache_vide"))
    code = cli.main(["analyze", str(video_vfr), "--output-dir", str(tmp_path / "s")])
    assert code == 2
    assert "models pull wasb_tennis" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()


def test_ball_track_appelle_le_pipeline_sans_personnes(monkeypatch, capsys):
    appels = {}

    def remplacant(video, **kwargs):
        appels.update(kwargs, video=video)
        return {**RESUME, "modele": None, "sorties": {"ball": "ball.csv"}}

    monkeypatch.setattr(cli, "analyze", remplacant)
    code = cli.main(
        ["ball", "track", "v.mp4", "--start", "5", "--end", "9", "--model", "tracknetv2_tennis"]
        + ["--device", "cpu", "--threshold", "0.4", "--mask", "m.json", "--output-dir", "o"]
    )
    assert code == 0
    assert appels["personnes"] is False and appels["write_video"] is False
    assert appels["modele_balle"] == "tracknetv2_tennis"
    assert (appels["start"], appels["end"], appels["seuil_balle"]) == (5, 9, 0.4)
    assert appels["masque"] == Path("m.json") and appels["output_dir"] == Path("o")
    sortie = capsys.readouterr().out
    assert "Balle (" in sortie and "Identités" not in sortie


def test_ball_bench_affiche_du_json(monkeypatch, capsys):
    appels = {}

    def faux_debit(modele, device=None, n_triplets=8, taille_lot=4, architecture=None):
        appels.update(modele=modele, device=device, n_triplets=n_triplets, taille_lot=taille_lot)
        return {"modele": modele, "fps_reseau": 12.5}

    monkeypatch.setattr("tennisvision.balle.export.mesurer_debit", faux_debit)
    args = ["ball", "bench", "--model", "tracknetv2_tennis", "--batch", "2", "--triplets", "6"]
    assert cli.main(args) == 0
    assert appels == {
        "modele": "tracknetv2_tennis",
        "device": None,
        "n_triplets": 6,
        "taille_lot": 2,
    }
    assert json.loads(capsys.readouterr().out) == {
        "modele": "tracknetv2_tennis",
        "fps_reseau": 12.5,
    }


def test_ball_export_onnx_transmet_les_options(monkeypatch, tmp_path, capsys):
    appels = {}

    def faux_export(modele, chemin, architecture=None, opset=17):
        appels.update(modele=modele, chemin=chemin, opset=opset)
        return chemin

    monkeypatch.setattr("tennisvision.balle.export.exporter_onnx", faux_export)
    sortie = tmp_path / "m.onnx"
    assert cli.main(["ball", "export-onnx", str(sortie), "--opset", "16"]) == 0
    assert appels == {"modele": "wasb_tennis", "chemin": sortie, "opset": 16}
    assert str(sortie) in capsys.readouterr().out


def test_ball_export_onnx_dependances_absentes(monkeypatch, tmp_path, capsys):
    def sans_onnx(*args, **kwargs):
        raise ImportError('pip install "tennisvision[onnx]"')

    monkeypatch.setattr("tennisvision.balle.export.exporter_onnx", sans_onnx)
    assert cli.main(["ball", "export-onnx", str(tmp_path / "m.onnx")]) == 2
    assert "tennisvision[onnx]" in capsys.readouterr().err


def test_ball_track_reel(video_vfr, tmp_path, capsys):
    """Essai de bout en bout avec les vrais poids (ignoré s'ils ne sont pas dans le cache)."""
    if not models.chemin_modele("wasb_tennis").is_file():
        pytest.skip("poids wasb_tennis absents du cache")
    sortie = tmp_path / "s"
    args = ["ball", "track", str(video_vfr), "--output-dir", str(sortie), "--device", "cpu"]
    assert cli.main(args) == 0
    assert {p.name for p in sortie.iterdir()} == {"ball.csv", "frames.parquet", "summary.json"}
    assert len(pd.read_csv(sortie / "ball.csv")) == 25
    assert valider_fichier(sortie / "ball.csv") == []
    resume = json.loads((sortie / "summary.json").read_text("utf-8"))
    assert resume["balle"]["modele"]["nom"] == "wasb_tennis"
    assert "Balle (wasb_tennis" in capsys.readouterr().out


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

    assert cli.main(["models", "pull", "yolo11n", "--models-dir", str(cache)]) == 0
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
    assert cli.main(["models", "pull", "yolo11n"]) == 0
    assert (tmp_path / "env" / "yolo11n.pt").is_file()


def test_models_pull_sans_argument_recupere_les_modeles_d_analyze(monkeypatch, tmp_path):
    demandes = []
    monkeypatch.setattr(cli, "telecharger", lambda nom, dossier: demandes.append(nom) or tmp_path)
    assert cli.main(["models", "pull"]) == 0
    assert demandes == ["yolo11n", "wasb_tennis"]


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


def test_ball_train_construit_la_config_et_appelle_entrainer(monkeypatch, tmp_path, capsys):
    appels = {}

    def faux_entrainer(config, reseau=None):
        appels["config"] = config
        return config.dossier_sortie / "best.pth"

    monkeypatch.setattr("tennisvision.balle.entrainement.entrainer", faux_entrainer)
    args = ["ball", "train", "--source", "tracknet:/d/tennis", "--source", "amateur:/d/gt"]
    args += ["--output-dir", str(tmp_path), "--architecture", "unet", "--init", "aucun"]
    args += ["--epochs", "3", "--batch", "2", "--lr", "0.01", "--device", "cpu", "--seed", "7"]
    args += ["--val-fraction", "0.2", "--temporal-step", "2", "--workers", "1"]
    args += ["--cache-dir", "c", "--videos-dir", "v", "--no-interpolated", "--max-batches", "4"]
    args += ["--no-augmentation"]
    assert cli.main(args) == 0
    c = appels["config"]
    assert c.sources == ("tracknet:/d/tennis", "amateur:/d/gt")
    assert c.dossier_sortie == tmp_path and c.architecture == "unet"
    assert c.poids_initiaux is None and (c.epochs, c.taille_lot, c.lr) == (3, 2, 0.01)
    assert (c.device, c.graine, c.fraction_validation) == ("cpu", 7, 0.2)
    assert (c.pas_temporel, c.nb_workers, c.max_lots) == (2, 1, 4)
    assert c.dossier_cache == Path("c") and c.videos_dir == Path("v") and not c.interpolees
    assert c.augmentation.p_retournement == 0.0 and c.augmentation.p_jpeg == 0.0
    assert str(tmp_path / "best.pth") in capsys.readouterr().out


def test_ball_train_valeurs_par_defaut(monkeypatch, tmp_path):
    appels = {}
    monkeypatch.setattr(
        "tennisvision.balle.entrainement.entrainer",
        lambda config, reseau=None: appels.update(config=config) or tmp_path / "best.pth",
    )
    assert cli.main(["ball", "train", "--source", "tracknet:/t", "--output-dir", "o"]) == 0
    c = appels["config"]
    assert c.poids_initiaux == "wasb_tennis" and c.architecture == "hrnet" and c.epochs == 10
    assert c.augmentation.p_retournement == 0.5 and c.interpolees


def test_ball_train_erreur_de_configuration(capsys):
    assert cli.main(["ball", "train", "--source", "n_importe_quoi", "--output-dir", "o"]) == 2
    assert "tracknet" in capsys.readouterr().err
    assert cli.main(["ball", "train", "--output-dir", "o"]) == 2  # --source obligatoire


def test_ball_architecture_transmise_au_pipeline(monkeypatch):
    appels = []
    monkeypatch.setattr(
        cli, "analyze", lambda video, **kw: appels.append(kw) or {**RESUME, "sorties": {}}
    )
    assert (
        cli.main(["ball", "track", "v.mp4", "--model", "m.pth", "--ball-architecture", "unet"]) == 0
    )
    assert (
        cli.main(["analyze", "v.mp4", "--ball-model", "m.pth", "--ball-architecture", "hrnet"]) == 0
    )
    assert [a["architecture_balle"] for a in appels] == ["unet", "hrnet"]
