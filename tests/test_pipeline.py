"""Tests du pipeline avec un faux tracker (aucun modèle chargé)."""

import json
import shutil
from pathlib import Path

import av
import pandas as pd
import pytest

from tennisvision import cli, pipeline
from tennisvision.io.video import frame_index
from tennisvision.tracking.detector import Detection
from tennisvision.validate import valider_dossier

MATCH = Path(__file__).parent / "fixtures" / "contrats" / "valide" / "match.json"


class FauxTracker:
    """Deux personnes à chaque frame, une balle une frame sur deux (frames paires)."""

    def update(self, frame_idx, image):
        dets = [
            Detection(frame_idx, 1, 0, "person", 0.9, 5.0, 5.0, 15.0, 40.0),
            Detection(frame_idx, 2, 0, "person", 0.8, 30.0, 5.0, 40.0, 40.0),
        ]
        if frame_idx % 2 == 0:
            dets.append(Detection(frame_idx, None, 32, "ball", 0.7, 20.0, 20.0, 24.0, 24.0))
        return dets


def _chaines(valeur):
    """Toutes les chaînes d'une structure JSON (clés et valeurs, récursivement)."""
    if isinstance(valeur, str):
        yield valeur
    elif isinstance(valeur, dict):
        for cle, sous in valeur.items():
            yield cle
            yield from _chaines(sous)
    elif isinstance(valeur, list):
        for sous in valeur:
            yield from _chaines(sous)


def _summary(dossier: Path) -> dict:
    return json.loads((dossier / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture
def sortie(video_vfr, tmp_path):
    dossier = tmp_path / "sortie"
    resume = pipeline.analyze(video_vfr, dossier, match=MATCH, tracker=FauxTracker())
    return dossier, resume


def test_fichiers_produits(sortie):
    dossier, _ = sortie
    attendus = {"annotated.mp4", "tracks.csv", "frames.parquet", "summary.json", "match.json"}
    assert {p.name for p in dossier.iterdir()} == attendus


def test_dossier_valide_selon_les_contrats(sortie):
    dossier, _ = sortie
    resultats = valider_dossier(dossier)
    assert resultats["frames.parquet"] == []
    assert resultats["match.json"] == []


def test_frames_parquet_est_l_index_complet(sortie, video_vfr):
    dossier, _ = sortie
    df = pd.read_parquet(dossier / "frames.parquet")
    assert df["t_ms"].tolist() == pytest.approx(frame_index(video_vfr))


def test_tracks_csv_porte_t_ms(sortie, video_vfr):
    dossier, _ = sortie
    df = pd.read_csv(dossier / "tracks.csv")
    assert list(df.columns)[:2] == ["frame_idx", "t_ms"]
    t_ms = frame_index(video_vfr)
    assert df["t_ms"].tolist() == pytest.approx([t_ms[i] for i in df["frame_idx"]])


def test_summary_sans_chemin_absolu(sortie, tmp_path):
    dossier, _ = sortie
    contenu = _summary(dossier)
    chaines = list(_chaines(contenu))
    assert chaines
    for chaine in chaines:
        assert not Path(chaine).is_absolute(), chaine
        assert str(tmp_path) not in chaine
        assert "\\" not in chaine
    assert contenu["video"] == "vfr.mp4"
    assert contenu["sorties"] == {
        "annotated": "annotated.mp4",
        "tracks": "tracks.csv",
        "frames": "frames.parquet",
        "summary": "summary.json",
        "match": "match.json",
    }


def test_duree_depuis_les_t_ms_reels(sortie, video_vfr):
    dossier, _ = sortie
    contenu = _summary(dossier)
    t_ms = frame_index(video_vfr)
    assert contenu["intervalle"] == [0, len(t_ms) - 1]
    assert contenu["intervalle_ms"] == pytest.approx([t_ms[0], t_ms[-1]])
    assert abs(contenu["duree_ms"] - (t_ms[-1] - t_ms[0])) < 1


def test_indicateurs(sortie):
    dossier, retour = sortie
    contenu = _summary(dossier)
    assert contenu["n_frames_traitees"] == 25
    assert contenu["taux_detection_balle"] == pytest.approx(13 / 25)  # frames paires de 0 à 24
    assert contenu["n_identites"] == 2
    assert contenu["modele"]["nom"] == "yolo11n" and len(contenu["modele"]["sha256"]) == 64
    assert contenu["version_tennisvision"]
    # Le retour ajoute le dossier absolu pour l'affichage, sans qu'il soit écrit
    assert retour["dossier"] == str(dossier.resolve()) and "dossier" not in contenu


def test_taux_balle_une_frame_sur_deux(video_vfr, tmp_path):
    resume = pipeline.analyze(
        video_vfr, tmp_path / "s", start=0, end=9, write_video=False, tracker=FauxTracker()
    )
    assert resume["taux_detection_balle"] == pytest.approx(0.5)
    assert resume["sorties"]["annotated"] is None and resume["sorties"]["match"] is None


def test_video_annotee_h264(sortie, video_vfr):
    dossier, _ = sortie
    with av.open(str(dossier / "annotated.mp4")) as conteneur:
        flux = conteneur.streams.video[0]
        assert flux.codec_context.name == "h264"
        assert flux.frames == len(frame_index(video_vfr))


@pytest.mark.parametrize(("start", "end"), [(5, 2), (-1, 3), (999, None)])
def test_bornes_invalides_ne_creent_aucun_dossier(video_vfr, tmp_path, start, end):
    dossier = tmp_path / "ne_doit_pas_exister"
    with pytest.raises(ValueError):
        pipeline.analyze(video_vfr, dossier, start=start, end=end, tracker=FauxTracker())
    assert not dossier.exists()


def test_bornes_invalides_dossier_par_defaut(video_vfr, tmp_path, monkeypatch):
    racine = tmp_path / "racine"
    monkeypatch.setenv("TENNISVISION_OUTPUT_DIR", str(racine))
    with pytest.raises(ValueError):
        pipeline.analyze(video_vfr, start=5, end=2, tracker=FauxTracker())
    assert not racine.exists()


def test_match_invalide_leve_avant_traitement(video_vfr, tmp_path):
    match = tmp_path / "mon_match.json"
    match.write_text(MATCH.read_text(encoding="utf-8").replace("left", "ambidextre"), "utf-8")
    dossier = tmp_path / "s"
    with pytest.raises(ValueError, match=r"players\[1\]\.handedness"):
        pipeline.analyze(video_vfr, dossier, match=match, tracker=FauxTracker())
    assert not dossier.exists()


def test_match_introuvable(video_vfr, tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.analyze(video_vfr, tmp_path / "s", match=tmp_path / "absent.json")


def test_dossier_par_defaut_hors_du_depot(video_vfr, tmp_path, monkeypatch, capsys):
    """Lancé depuis un autre dossier, `analyze` écrit sous ./tennisvision-sorties."""
    depot = Path(cli.__file__).resolve().parents[2]
    avant = (depot / "tennisvision-sorties").exists()
    ailleurs = tmp_path / "ailleurs"
    ailleurs.mkdir()
    shutil.copyfile(video_vfr, ailleurs / "match.mp4")
    monkeypatch.chdir(ailleurs)
    monkeypatch.delenv("TENNISVISION_OUTPUT_DIR", raising=False)
    monkeypatch.setattr(pipeline, "Tracker", lambda **kwargs: FauxTracker())

    assert cli.main(["analyze", "match.mp4", "--no-video"]) == 0

    produit = ailleurs / "tennisvision-sorties" / "match_0-24"
    assert (produit / "summary.json").is_file()
    assert (depot / "tennisvision-sorties").exists() == avant
    assert str(produit) in capsys.readouterr().out
