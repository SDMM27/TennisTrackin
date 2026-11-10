"""Tests de l'ingestion vidéo."""

import subprocess
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from conftest import FPS, HAUTEUR, LARGEUR, N_FRAMES, ecrire_video
from tennisvision.io import video as mod_video
from tennisvision.io.video import (
    Frame,
    VideoWriter,
    cut_clip,
    frame_index,
    iter_frames,
    probe,
    resolve_interval,
    write_frames_parquet,
)
from tennisvision.validate import valider_fichier

TOLERANCE = 12  # la compression altère les valeurs de pixels
V006 = Path(__file__).resolve().parent.parent / "data" / "videos" / "V006.mp4"


def _valeur(frame) -> float:
    return float(frame.mean())


def test_probe(video_synthetique):
    info = probe(video_synthetique)
    assert (info.width, info.height) == (LARGEUR, HAUTEUR)
    assert info.rotation == 0
    assert info.n_frames == N_FRAMES
    assert info.fps == info.fps_nominal == pytest.approx(FPS)
    # durée = dernière t_ms - première : la durée d'une frame n'est pas ajoutée
    assert info.duration_ms == pytest.approx((N_FRAMES - 1) / FPS * 1000)
    assert info.codec


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
    assert all(isinstance(f, Frame) for f in frames)
    assert [f.idx for f in frames] == list(range(N_FRAMES))
    assert frames[0].image.shape == (HAUTEUR, LARGEUR, 3)
    assert frames[0].image.dtype == np.uint8


def test_iter_frames_bornes_inclusives_et_ordre(video_synthetique):
    frames = list(iter_frames(video_synthetique, start=10, end=14))
    assert [f.idx for f in frames] == [10, 11, 12, 13, 14]
    for f in frames:
        assert _valeur(f.image) == pytest.approx(5 * f.idx, abs=TOLERANCE)


def test_iter_frames_step(video_synthetique):
    indices = [f.idx for f in iter_frames(video_synthetique, start=2, end=12, step=5)]
    assert indices == [2, 7, 12]


def test_iter_frames_end_au_dela_de_la_fin(video_synthetique):
    indices = [f.idx for f in iter_frames(video_synthetique, start=25, end=1000)]
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


def test_resolve_interval(video_synthetique):
    info = probe(video_synthetique)
    assert resolve_interval(info) == (0, N_FRAMES - 1)
    assert resolve_interval(info, 3, 1000) == (3, N_FRAMES - 1)
    assert resolve_interval(info, 3, 7) == (3, 7)
    for start, end in [(-1, None), (N_FRAMES, None), (10, 5)]:
        with pytest.raises(ValueError):
            resolve_interval(info, start, end)


def test_cut_clip(video_synthetique, tmp_path):
    dst = tmp_path / "sous_dossier" / "clip.mp4"
    resultat = cut_clip(video_synthetique, dst, 5, 14)
    assert isinstance(resultat, Path) and resultat.exists()
    assert probe(dst).n_frames == 10
    premiere = next(iter_frames(dst)).image
    assert _valeur(premiere) == pytest.approx(25, abs=TOLERANCE)


def test_cut_clip_conserve_les_intervalles(video_vfr, tmp_path):
    t_src = frame_index(video_vfr)
    dst = cut_clip(video_vfr, tmp_path / "clip.mp4", 4, 12)
    t_clip = frame_index(dst)
    attendu = np.diff(t_src[4:13])
    assert np.diff(t_clip) == pytest.approx(attendu, abs=1.0)


# --- cadence variable -------------------------------------------------------------------------


def test_frame_index_vfr_intervalles_exacts(video_vfr, temps_vfr_ms):
    t_ms = frame_index(video_vfr)
    assert len(t_ms) == len(temps_vfr_ms)
    assert t_ms[0] == 0
    erreurs = np.abs(np.diff(t_ms) - np.diff(temps_vfr_ms))
    assert erreurs.max() < 1.0
    assert probe(video_vfr).duration_ms == pytest.approx(temps_vfr_ms[-1], abs=1.0)


def test_frame_index_cache(video_vfr):
    mod_video._index_cache.cache_clear()
    frame_index(video_vfr)
    frame_index(video_vfr)
    infos = mod_video._index_cache.cache_info()
    assert (infos.misses, infos.hits) == (1, 1)


def test_frames_parquet_conforme_au_contrat(video_vfr, temps_vfr_ms, tmp_path):
    dst = write_frames_parquet(video_vfr, tmp_path / "sortie" / "frames.parquet")
    assert valider_fichier(dst) == []
    import pyarrow.parquet as pq

    table = pq.read_table(dst)
    assert table.column_names == ["frame_idx", "t_ms"]
    assert str(table.schema.field("frame_idx").type) == "int64"
    assert str(table.schema.field("t_ms").type) == "double"
    assert table.num_rows == len(temps_vfr_ms)


# --- accès direct ------------------------------------------------------------------------------


def test_saut_identique_a_la_lecture_sequentielle(video_vfr):
    sequentielle = list(iter_frames(video_vfr))
    for k in (0, 1, 4, 6, 7, 13, len(sequentielle) - 1):  # dont des frames hors keyframe
        premiere = next(iter_frames(video_vfr, start=k))
        assert premiere.idx == k
        assert premiere.t_ms == pytest.approx(sequentielle[k].t_ms)
        assert np.array_equal(premiere.image, sequentielle[k].image)
        assert _valeur(premiere.image) == pytest.approx(8 * k + 10, abs=TOLERANCE)


def test_saut_avec_end_et_step(video_vfr):
    sequentielle = list(iter_frames(video_vfr))
    frames = list(iter_frames(video_vfr, start=7, end=17, step=5))
    assert [f.idx for f in frames] == [7, 12, 17]
    for f in frames:
        assert np.array_equal(f.image, sequentielle[f.idx].image)


# --- rotation ----------------------------------------------------------------------------------


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotation_conforme_a_ffmpeg(video_tournee, ffmpeg, tmp_path, rotation):
    # Taille stockée choisie pour que l'image affichée soit en paysage
    stockee = (64, 48) if rotation == 180 else (48, 64)
    chemin = video_tournee(rotation, stockee)
    info = probe(chemin)
    assert info.rotation == rotation
    assert (info.width, info.height) == (64, 48)

    ref = tmp_path / "ref.png"
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(chemin), "-frames:v", "1", str(ref)],
        check=True,
    )
    attendu = cv2.imread(str(ref))
    image = next(iter_frames(chemin)).image
    assert image.shape == attendu.shape == (48, 64, 3)
    assert np.abs(image.astype(int) - attendu.astype(int)).mean() < 3


def test_portrait_refuse(video_tournee):
    chemin = video_tournee(90, (64, 48))  # stockée 64x48, affichée 48x64
    with pytest.raises(ValueError, match="portrait"):
        probe(chemin)


def test_portrait_natif_refuse(tmp_path):
    chemin = ecrire_video(tmp_path / "portrait.mp4", [0, 40], [10, 20], size=(48, 64))
    with pytest.raises(ValueError, match=r"portrait \(48x64"):
        probe(chemin)


# --- écriture ----------------------------------------------------------------------------------


def test_ecriture_h264_et_t_ms_relus(tmp_path):
    t_ecrits = [1000.0, 1033.0, 1067.0, 1117.0, 1134.0, 1167.0]
    dst = tmp_path / "ecrit.mp4"
    with VideoWriter(dst, (LARGEUR, HAUTEUR)) as writer:
        for i, t in enumerate(t_ecrits):
            writer.write(np.full((HAUTEUR, LARGEUR, 3), 20 * i, dtype=np.uint8), t)
        assert writer.codec == "libx264"
    with av.open(str(dst)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        assert container.streams.video[0].codec_context.pix_fmt == "yuv420p"
    attendu = [t - t_ecrits[0] for t in t_ecrits]
    assert frame_index(dst) == pytest.approx(attendu, abs=1e-6)


def test_ecriture_dimensions_impaires(tmp_path):
    dst = tmp_path / "impair.mp4"
    with VideoWriter(dst, (65, 49)) as writer:
        for i in range(3):
            writer.write(np.full((49, 65, 3), 200, dtype=np.uint8), 40.0 * i)
    info = probe(dst)
    assert (info.width, info.height) == (66, 50)
    image = next(iter_frames(dst)).image
    assert image[:49, :65].mean() > 150
    assert image[49:, :].mean() < 30  # bande de bord noire (tolérance de compression)


def test_ecriture_t_ms_non_croissant(tmp_path):
    with VideoWriter(tmp_path / "x.mp4", (LARGEUR, HAUTEUR)) as writer:
        image = np.zeros((HAUTEUR, LARGEUR, 3), dtype=np.uint8)
        writer.write(image, 0.0)
        writer.write(image, 40.0)
        with pytest.raises(ValueError, match="croissant"):
            writer.write(image, 40.0)
        with pytest.raises(ValueError, match="croissant"):
            writer.write(image, 10.0)


# --- vidéo réelle ------------------------------------------------------------------------------


@pytest.mark.skipif(not V006.is_file(), reason="data/videos/V006.mp4 absent")
def test_v006_saut_identique_a_opencv():
    debut = time.perf_counter()
    info = probe(V006)
    duree_index = time.perf_counter() - debut
    print(f"\nindexation V006 : {duree_index:.1f} s pour {info.n_frames} frames")
    assert (info.width, info.height) == (1280, 720)

    cap = cv2.VideoCapture(str(V006))
    try:
        for k in (997, 3001):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, attendue = cap.read()
            assert ok
            frame = next(iter_frames(V006, start=k, end=k))
            assert frame.idx == k
            assert np.array_equal(frame.image, attendue)
    finally:
        cap.release()
