"""Tests du pipeline avec un faux tracker et un faux détecteur de balle (aucun modèle chargé)."""

import json
import shutil
from pathlib import Path

import av
import pandas as pd
import pytest

from tennisvision import cli, pipeline
from tennisvision.balle.detecteur import Candidat
from tennisvision.balle.trajectoire import ParamsTrajectoire
from tennisvision.io.video import frame_index
from tennisvision.models import ModeleAbsent
from tennisvision.tracking.detector import Detection
from tennisvision.validate import valider_dossier, valider_fichier

FIXTURES = Path(__file__).parent / "fixtures" / "contrats" / "valide"
MATCH = FIXTURES / "match.json"
COURT = FIXTURES / "court.json"

STATIQUE = (10.0, 8.0)  # distracteur immobile, présent à chaque frame
# Durée statique réduite : la vidéo de test dure moins d'une seconde
PARAMS = ParamsTrajectoire(duree_statique_ms=300.0)


class FauxTracker:
    """Deux personnes à chaque frame."""

    def update(self, frame_idx, image):
        return [
            Detection(frame_idx, 1, 0, "person", 0.9, 5.0, 5.0, 15.0, 40.0),
            Detection(frame_idx, 2, 0, "person", 0.8, 30.0, 5.0, 40.0, 40.0),
        ]


class FauxDetecteurBalle:
    """Parabole en fonction du temps réel + un distracteur immobile, à chaque frame.

    Les candidats d'une frame sont rendus à la poussée suivante (les derniers par `terminer`),
    comme un vrai détecteur à triplets.
    """

    modele_decrit = {"nom": "faux", "version": None, "sha256": None}  # noqa: RUF012
    pas_temporel = 1

    def __init__(self, t_ms):
        self.t_ms = t_ms
        self.formes = []
        self._en_attente = []

    def _candidats(self, frame_idx):
        t = self.t_ms[frame_idx]
        return [
            Candidat(frame_idx, 4.0 + 0.05 * t, 30.0 + 5e-5 * (t - 450.0) ** 2, 0.9),
            Candidat(frame_idx, *STATIQUE, 0.8),
        ]

    def pousser(self, frame_idx, image):
        self.formes.append(image.shape)
        sortie, self._en_attente = self._en_attente, self._candidats(frame_idx)
        return sortie

    def terminer(self):
        sortie, self._en_attente = self._en_attente, []
        return sortie


@pytest.fixture
def faux_balle(temps_vfr_ms):
    return FauxDetecteurBalle(temps_vfr_ms)


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
def sortie(video_vfr, tmp_path, faux_balle):
    dossier = tmp_path / "sortie"
    resume = pipeline.analyze(
        video_vfr,
        dossier,
        match=MATCH,
        tracker=FauxTracker(),
        detecteur_balle=faux_balle,
        params_trajectoire=PARAMS,
    )
    return dossier, resume


def test_fichiers_produits(sortie):
    dossier, _ = sortie
    attendus = {
        "annotated.mp4",
        "tracks.csv",
        "ball.csv",
        "frames.parquet",
        "summary.json",
        "match.json",
    }
    assert {p.name for p in dossier.iterdir()} == attendus


def test_dossier_valide_selon_les_contrats(sortie):
    dossier, _ = sortie
    resultats = valider_dossier(dossier)
    assert resultats["frames.parquet"] == []
    assert resultats["match.json"] == []
    assert resultats["ball.csv"] == []


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


def test_tracks_csv_sans_balle(sortie):
    dossier, _ = sortie
    df = pd.read_csv(dossier / "tracks.csv")
    assert set(df["label"]) == {"person"}


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
        "ball": "ball.csv",
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
    assert "taux_detection_balle" not in contenu and "n_detections_balle" not in contenu
    assert contenu["n_identites"] == 2
    assert contenu["modele"]["nom"] == "yolo11n" and len(contenu["modele"]["sha256"]) == 64
    assert contenu["version_tennisvision"]
    assert contenu["fps_traitement"] > 0
    # Le retour ajoute le dossier absolu pour l'affichage, sans qu'il soit écrit
    assert retour["dossier"] == str(dossier.resolve()) and "dossier" not in contenu


def test_ball_csv_une_ligne_par_frame_sans_distracteur(sortie, video_vfr):
    dossier, _ = sortie
    assert valider_fichier(dossier / "ball.csv") == []
    ball = pd.read_csv(dossier / "ball.csv")
    assert ball["frame_idx"].tolist() == list(range(25))
    assert ball["t_ms"].tolist() == pytest.approx(frame_index(video_vfr))
    visibles = ball[ball["visible"] == 1]
    assert len(visibles) >= 20
    # Le distracteur immobile ne passe jamais ; la trajectoire suit la parabole
    ecart = (visibles["x"] - STATIQUE[0]).abs() + (visibles["y"] - STATIQUE[1]).abs()
    assert (ecart > 5).all()
    assert (visibles["y"] > 29).all()
    assert visibles["x"].is_monotonic_increasing


def test_summary_bloc_balle(sortie, faux_balle):
    dossier, retour = sortie
    balle = _summary(dossier)["balle"]
    ball = pd.read_csv(dossier / "ball.csv")
    assert balle["modele"] == FauxDetecteurBalle.modele_decrit
    assert balle["pas_temporel"] == 1 and balle["seuil_score"] == 0.5
    assert balle["masque"] is None
    assert balle["taux_visible"] == pytest.approx(ball["visible"].sum() / 25)
    assert balle["n_detectes"] == (ball["source"] == "detected").sum()
    assert balle["n_interpoles"] == (ball["source"] == "interpolated").sum()
    assert balle["diagnostic"]["n_candidats"] == 50
    assert balle["diagnostic"]["n_statiques"] == 25
    assert retour["balle"] == balle
    # Une seule passe de décodage : chaque frame est poussée une fois, en taille d'origine
    assert faux_balle.formes == [(48, 64, 3)] * 25


def test_video_annotee_dessine_la_balle(sortie):
    dossier, _ = sortie
    ball = pd.read_csv(dossier / "ball.csv")
    detectes = ball[ball["source"] == "detected"]
    ligne = detectes.iloc[len(detectes) // 2]
    with av.open(str(dossier / "annotated.mp4")) as conteneur:
        images = [f.to_ndarray(format="bgr24") for f in conteneur.decode(video=0)]
    pixel = images[int(ligne["frame_idx"])][int(ligne["y"]), int(ligne["x"])].astype(int)
    assert pixel[2] > 150 and pixel[1] > 150 and pixel[0] < 120  # jaune (BGR)


def test_sans_ecriture_video_pas_de_seconde_passe(video_vfr, tmp_path, faux_balle):
    resume = pipeline.analyze(
        video_vfr,
        tmp_path / "s",
        start=0,
        end=9,
        write_video=False,
        tracker=FauxTracker(),
        detecteur_balle=faux_balle,
        params_trajectoire=PARAMS,
    )
    assert resume["sorties"]["annotated"] is None and resume["sorties"]["match"] is None
    assert not (tmp_path / "s" / "annotated.mp4").exists()
    assert resume["n_frames_traitees"] == 10
    assert len(pd.read_csv(tmp_path / "s" / "ball.csv")) == 10


@pytest.mark.parametrize("desactive", [None, "aucun", "AUCUN"])
def test_balle_desactivee(video_vfr, tmp_path, desactive):
    dossier = tmp_path / "s"
    resume = pipeline.analyze(
        video_vfr, dossier, write_video=False, tracker=FauxTracker(), modele_balle=desactive
    )
    assert resume["balle"] is None and resume["sorties"]["ball"] is None
    assert not (dossier / "ball.csv").exists()
    assert _summary(dossier)["balle"] is None


def test_court_et_masque_s_excluent(video_vfr, tmp_path, faux_balle):
    masque = tmp_path / "masque.json"
    masque.write_text('{"polygone": [[0, 0], [60, 0], [60, 40]]}', encoding="utf-8")
    dossier = tmp_path / "s"
    with pytest.raises(ValueError, match="s'excluent"):
        pipeline.analyze(
            video_vfr,
            dossier,
            tracker=FauxTracker(),
            detecteur_balle=faux_balle,
            court=COURT,
            masque=masque,
        )
    assert not dossier.exists()


def test_masque_polygone_ecarte_les_candidats(video_vfr, tmp_path, faux_balle):
    masque = tmp_path / "masque.json"
    masque.write_text('{"polygone": [[0, 20], [64, 20], [64, 48], [0, 48]]}', encoding="utf-8")
    resume = pipeline.analyze(
        video_vfr,
        tmp_path / "s",
        write_video=False,
        tracker=FauxTracker(),
        detecteur_balle=faux_balle,
        masque=masque,
    )
    balle = resume["balle"]
    assert balle["masque"] == "polygone"
    assert balle["diagnostic"]["n_hors_masque"] == 25  # le distracteur (y = 8) est hors masque


def test_masque_court_json(video_vfr, tmp_path, faux_balle):
    resume = pipeline.analyze(
        video_vfr,
        tmp_path / "s",
        write_video=False,
        tracker=FauxTracker(),
        detecteur_balle=faux_balle,
        court=COURT,
    )
    assert resume["balle"]["masque"] == "court.json"


def test_masque_introuvable(video_vfr, tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.analyze(video_vfr, tmp_path / "s", masque=tmp_path / "absent.json")
    assert not (tmp_path / "s").exists()


def test_poids_balle_absents_levent_avant_le_dossier(video_vfr, tmp_path, monkeypatch):
    monkeypatch.setenv("TENNISVISION_MODELS_DIR", str(tmp_path / "cache_vide"))
    dossier = tmp_path / "ne_doit_pas_exister"
    with pytest.raises(ModeleAbsent, match="models pull wasb_tennis"):
        pipeline.analyze(video_vfr, dossier, tracker=FauxTracker())
    assert not dossier.exists()


def test_sans_personnes_ni_balle(video_vfr, tmp_path):
    with pytest.raises(ValueError, match="rien à analyser"):
        pipeline.analyze(video_vfr, tmp_path / "s", personnes=False, modele_balle=None)


def test_balle_seule_sans_personnes(video_vfr, tmp_path, faux_balle):
    dossier = tmp_path / "s"
    resume = pipeline.analyze(
        video_vfr,
        dossier,
        write_video=False,
        personnes=False,
        detecteur_balle=faux_balle,
        params_trajectoire=PARAMS,
    )
    assert {p.name for p in dossier.iterdir()} == {"ball.csv", "frames.parquet", "summary.json"}
    assert resume["modele"] is None and "n_identites" not in resume
    assert resume["sorties"]["tracks"] is None and resume["sorties"]["ball"] == "ball.csv"


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

    assert cli.main(["analyze", "match.mp4", "--no-video", "--ball-model", "aucun"]) == 0

    produit = ailleurs / "tennisvision-sorties" / "match_0-24"
    assert (produit / "summary.json").is_file()
    assert (depot / "tennisvision-sorties").exists() == avant
    assert str(produit) in capsys.readouterr().out
