"""Tests de l'ingestion vidéo."""

from pathlib import Path

import pytest

from conftest import FPS, HAUTEUR, LARGEUR, N_FRAMES
from tennisvision.io.video import cut_clip, iter_frames, probe

TOLERANCE = 12  # la compression mp4v altère les valeurs de pixels


def _valeur(frame) -> float:
    return float(frame.mean())


def test_probe(video_synthetique):
    info = probe(video_synthetique)
    assert (info.width, info.height) == (LARGEUR, HAUTEUR)
    assert info.n_frames == N_FRAMES
    assert info.fps == pytest.approx(FPS)
    assert info.duration_s == pytest.approx(N_FRAMES / FPS)


def test_probe_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "absent.mp4")


def test_probe_fichier_illisible(tmp_path):
    faux = tmp_path / "faux.mp4"
    faux.write_bytes(b"pas une video")
    with pytest.raises(ValueError):
        probe(faux)


def test_iter_frames_tout(video_synthetique):
    frames = list(iter_frames(video_synthetique))
    assert [i for i, _ in frames] == list(range(N_FRAMES))


def test_iter_frames_bornes_inclusives_et_ordre(video_synthetique):
    frames = list(iter_frames(video_synthetique, start=10, end=14))
    assert [i for i, _ in frames] == [10, 11, 12, 13, 14]
    for i, frame in frames:
        assert _valeur(frame) == pytest.approx(5 * i, abs=TOLERANCE)


def test_iter_frames_step(video_synthetique):
    indices = [i for i, _ in iter_frames(video_synthetique, start=2, end=12, step=5)]
    assert indices == [2, 7, 12]


def test_iter_frames_end_au_dela_de_la_fin(video_synthetique):
    indices = [i for i, _ in iter_frames(video_synthetique, start=25, end=1000)]
    assert indices == list(range(25, N_FRAMES))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": 10, "end": 5},
        {"start": N_FRAMES},
        {"start": -1},
        {"step": 0},
    ],
)
def test_iter_frames_bornes_invalides(video_synthetique, kwargs):
    with pytest.raises(ValueError):
        list(iter_frames(video_synthetique, **kwargs))


def test_cut_clip(video_synthetique, tmp_path):
    dst = tmp_path / "sous_dossier" / "clip.mp4"
    resultat = cut_clip(video_synthetique, dst, 5, 14)
    assert isinstance(resultat, Path) and resultat.exists()
    assert probe(dst).n_frames == 10
    premiere = next(iter_frames(dst))[1]
    assert _valeur(premiere) == pytest.approx(25, abs=TOLERANCE)
