"""Fixtures partagées : petites vidéos synthétiques."""

import shutil
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

N_FRAMES = 30
LARGEUR, HAUTEUR = 64, 48
FPS = 10.0

# Intervalles irréguliers connus (ms) de la vidéo à cadence variable : 24 intervalles, 25 frames
INTERVALLES_VFR_MS = [33, 34, 33, 50, 17, 33, 100, 33, 33, 34, 66, 16, 33, 33, 34, 40, 27, 33, 33,
                      34, 80, 20, 33, 33]  # fmt: skip


def ecrire_video(
    chemin: Path,
    t_ms: Sequence[int],
    valeurs: Sequence[int],
    size: tuple[int, int] = (LARGEUR, HAUTEUR),
    gop: int | None = None,
    b_frames: int | None = None,
) -> Path:
    """Écrit un mp4 H.264 de pts exacts `t_ms` ; la frame i est uniformément remplie de valeurs[i]."""
    container = av.open(str(chemin), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height = size
    stream.pix_fmt = "yuv420p"
    stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
    options = {"crf": "10", "preset": "veryfast"}
    if gop is not None:
        options["g"] = str(gop)
    if b_frames is not None:
        options["bf"] = str(b_frames)
    stream.options = options
    for t, valeur in zip(t_ms, valeurs, strict=True):
        image = np.full((size[1], size[0], 3), valeur, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24").reformat(format="yuv420p")
        frame.pts = int(t)
        frame.time_base = Fraction(1, 1000)
        for paquet in stream.encode(frame):
            container.mux(paquet)
    for paquet in stream.encode(None):
        container.mux(paquet)
    container.close()
    return chemin


def image_asymetrique(size: tuple[int, int]) -> np.ndarray:
    """Image BGR (largeur, hauteur) à quadrants de couleurs plates distinctes."""
    largeur, hauteur = size
    image = np.zeros((hauteur, largeur, 3), dtype=np.uint8)
    h, w = hauteur // 2, largeur // 2
    image[:h, :w] = (0, 0, 255)  # haut-gauche : rouge
    image[:h, w:] = (0, 255, 0)  # haut-droite : vert
    image[h:, :w] = (255, 0, 0)  # bas-gauche : bleu
    image[h:, w:] = (255, 255, 255)  # bas-droite : blanc
    image[: h // 2, : w // 2] = (0, 255, 255)  # coin haut-gauche : jaune
    return image


@pytest.fixture
def video_synthetique(tmp_path: Path) -> Path:
    """Vidéo de 30 frames 64x48 ; la frame i est uniformément remplie de la valeur 5*i."""
    chemin = tmp_path / "synth.mp4"
    writer = cv2.VideoWriter(str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (LARGEUR, HAUTEUR))
    for i in range(N_FRAMES):
        writer.write(np.full((HAUTEUR, LARGEUR, 3), 5 * i, dtype=np.uint8))
    writer.release()
    return chemin


@pytest.fixture
def temps_vfr_ms() -> list[int]:
    """t_ms des frames de `video_vfr` (cumul des intervalles connus, à partir de 0)."""
    return [0, *np.cumsum(INTERVALLES_VFR_MS).tolist()]


@pytest.fixture
def video_vfr(tmp_path: Path, temps_vfr_ms: list[int]) -> Path:
    """Vidéo H.264 à cadence variable (GOP de 5, B-frames) ; la frame i vaut 8*i + 10."""
    valeurs = [8 * i + 10 for i in range(len(temps_vfr_ms))]
    return ecrire_video(tmp_path / "vfr.mp4", temps_vfr_ms, valeurs, gop=5, b_frames=2)


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    """Chemin du CLI ffmpeg (test ignoré s'il est absent)."""
    chemin = shutil.which("ffmpeg")
    if chemin is None:
        pytest.skip("ffmpeg absent du PATH")
    return chemin


@pytest.fixture
def video_tournee(tmp_path: Path, ffmpeg: str):
    """Fabrique de vidéos avec métadonnée de rotation : `fabrique(rotation, size_stockee)`."""

    def fabrique(rotation: int, size: tuple[int, int]) -> Path:
        # Image asymétrique encodée par ffmpeg (png -> mp4), puis rotation par métadonnée
        png = tmp_path / f"img_{rotation}.png"
        cv2.imwrite(str(png), image_asymetrique(size))
        source = tmp_path / f"src_{rotation}.mp4"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-frames:v", "3",
             "-r", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
            check=True,
        )  # fmt: skip
        sortie = tmp_path / f"rot_{rotation}.mp4"
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-display_rotation", str(rotation),
             "-i", str(source), "-c", "copy", str(sortie)],
            check=True,
        )  # fmt: skip
        return sortie

    return fabrique
