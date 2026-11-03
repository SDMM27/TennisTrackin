"""Fixtures partagées : petite vidéo synthétique."""

from pathlib import Path

import cv2
import numpy as np
import pytest

N_FRAMES = 30
LARGEUR, HAUTEUR = 64, 48
FPS = 10.0


@pytest.fixture
def video_synthetique(tmp_path: Path) -> Path:
    """Vidéo de 30 frames 64x48 ; la frame i est uniformément remplie de la valeur 5*i."""
    chemin = tmp_path / "synth.mp4"
    writer = cv2.VideoWriter(str(chemin), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (LARGEUR, HAUTEUR))
    for i in range(N_FRAMES):
        writer.write(np.full((HAUTEUR, LARGEUR, 3), 5 * i, dtype=np.uint8))
    writer.release()
    return chemin
