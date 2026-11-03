"""Ingestion vidéo : métadonnées, lecture de frames, écriture et extraction de clips."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FPS_PAR_DEFAUT = 25.0


@dataclass(frozen=True)
class VideoInfo:
    """Métadonnées d'une vidéo."""

    path: Path
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def duration_s(self) -> float:
        """Durée en secondes."""
        return self.n_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    """Lit les métadonnées d'une vidéo.

    Lève FileNotFoundError si le fichier n'existe pas et ValueError s'il est illisible.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo : {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if width <= 0 or height <= 0 or n_frames <= 0:
        raise ValueError(f"Vidéo illisible ou vide : {path}")
    if fps <= 0:
        logger.warning("FPS invalide pour %s, repli sur %s", path, FPS_PAR_DEFAUT)
        fps = FPS_PAR_DEFAUT
    return VideoInfo(path=path, fps=fps, width=width, height=height, n_frames=n_frames)


def iter_frames(
    path: str | Path, start: int = 0, end: int | None = None, step: int = 1
) -> Iterator[tuple[int, np.ndarray]]:
    """Itère sur les frames d'une vidéo : (index absolu, image BGR).

    `end` est inclusif ; None signifie jusqu'à la fin de la vidéo.
    """
    if step < 1:
        raise ValueError(f"step doit être >= 1 (reçu {step})")
    info = probe(path)
    if start < 0:
        raise ValueError(f"start doit être >= 0 (reçu {start})")
    if start >= info.n_frames:
        raise ValueError(f"start ({start}) hors vidéo ({info.n_frames} frames)")
    if end is None or end >= info.n_frames:
        end = info.n_frames - 1
    if start > end:
        raise ValueError(f"start ({start}) > end ({end})")

    cap = cv2.VideoCapture(str(info.path))
    try:
        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while idx <= end:
            ok, frame = cap.read()
            if not ok:
                logger.warning("Lecture interrompue à la frame %d", idx)
                return
            if (idx - start) % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


class VideoWriter:
    """Écrivain vidéo mp4v, utilisable comme gestionnaire de contexte."""

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        """`size` est (largeur, hauteur)."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, size)
        if not self._writer.isOpened():
            raise ValueError(f"Impossible de créer la vidéo : {self.path}")

    def write(self, frame: np.ndarray) -> None:
        """Ajoute une frame BGR."""
        self._writer.write(frame)

    def release(self) -> None:
        """Ferme le fichier."""
        self._writer.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def cut_clip(src: str | Path, dst: str | Path, start: int, end: int) -> Path:
    """Extrait les frames [start, end] (inclus) de `src` vers `dst`."""
    info = probe(src)
    dst = Path(dst)
    with VideoWriter(dst, info.fps, (info.width, info.height)) as writer:
        for _, frame in iter_frames(src, start, end):
            writer.write(frame)
    return dst
