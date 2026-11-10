"""Ingestion vidéo (PyAV) : métadonnées, index des frames, lecture, écriture et clips.

Le temps de référence est l'horodatage de présentation (pts) de chaque frame, en millisecondes
depuis la première frame (`frame_index`) : jamais `frame_idx / fps`, car les vidéos de téléphone
sont à cadence variable. `frame_idx` suit l'ordre de présentation des paquets du conteneur.
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple, Self

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

FPS_PAR_DEFAUT = 25.0
TIME_BASE_ECRITURE = Fraction(1, 1000)  # pts écrits en millisecondes
CRF = "23"
PRESET = "veryfast"


@dataclass(frozen=True)
class VideoInfo:
    """Métadonnées d'une vidéo.

    `width` et `height` sont les dimensions APRÈS application de la rotation. `duration_ms` est
    t_ms(dernière frame) − t_ms(première frame) : la durée moyenne d'une frame n'y est PAS ajoutée.
    """

    path: Path
    width: int
    height: int
    rotation: int
    fps_nominal: float
    n_frames: int
    duration_ms: float
    codec: str

    @property
    def fps(self) -> float:
        """Cadence nominale (compatibilité) : ne pas utiliser pour calculer des durées."""
        return self.fps_nominal

    @property
    def duration_s(self) -> float:
        """Durée en secondes (voir `duration_ms`)."""
        return self.duration_ms / 1000.0


class Frame(NamedTuple):
    """Une frame décodée : index, horodatage (ms) et image BGR uint8 (rotation appliquée)."""

    idx: int
    t_ms: float
    image: np.ndarray


def _verifier_fichier(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Vidéo introuvable : {path}")


def _ouvrir(path: Path) -> av.container.InputContainer:
    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError) as e:
        raise ValueError(f"Impossible d'ouvrir la vidéo : {path} ({e})") from e
    if not container.streams.video:
        container.close()
        raise ValueError(f"Aucun flux vidéo dans : {path}")
    return container


@lru_cache(maxsize=8)
def _index_cache(chemin: str, mtime_ns: int, taille: int) -> tuple[tuple[int, ...], Fraction]:
    """Pts triés (ordre de présentation) et time_base du flux vidéo ; clé = chemin+mtime+taille."""
    container = _ouvrir(Path(chemin))
    try:
        stream = container.streams.video[0]
        pts: list[int] = []
        for paquet in container.demux(stream):
            if paquet.size == 0:
                continue  # paquet de vidage de fin de flux
            ts = paquet.pts if paquet.pts is not None else paquet.dts
            if ts is None:
                continue
            pts.append(int(ts))
        time_base = Fraction(stream.time_base)
    finally:
        container.close()
    pts.sort()
    return tuple(pts), time_base


def _pts_index(path: Path) -> tuple[tuple[int, ...], Fraction]:
    _verifier_fichier(path)
    st = path.stat()
    return _index_cache(str(path.resolve()), st.st_mtime_ns, st.st_size)


def frame_index(path: str | Path) -> list[float]:
    """Horodatages t_ms de chaque frame (index de la liste = `frame_idx`).

    Démultiplexe les paquets sans décoder : pts (ou dts si absent), trié en ordre de
    présentation, paquets vides ou sans timestamp ignorés. t_ms = (pts − pts_min) × time_base.
    Résultat mis en cache par chemin+mtime+taille.
    """
    pts, time_base = _pts_index(Path(path))
    if not pts:
        return []
    facteur = float(time_base) * 1000.0
    return [(p - pts[0]) * facteur for p in pts]


def _rotation(frame: av.VideoFrame) -> int:
    return round(frame.rotation or 0) % 360 // 90 * 90


def _appliquer_rotation(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, rotation // 90))


def probe(path: str | Path) -> VideoInfo:
    """Lit les métadonnées d'une vidéo.

    Lève FileNotFoundError si le fichier n'existe pas et ValueError s'il est illisible, vide ou
    en portrait (après rotation).
    """
    path = Path(path)
    _verifier_fichier(path)
    container = _ouvrir(path)
    try:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        cadence = stream.average_rate
        premiere = None
        try:
            for premiere in container.decode(stream):
                break
        except av.error.FFmpegError as e:
            raise ValueError(f"Vidéo illisible : {path} ({e})") from e
        if premiere is None:
            raise ValueError(f"Vidéo illisible ou vide : {path}")
        rotation = _rotation(premiere)
        width, height = premiere.width, premiere.height
    finally:
        container.close()
    if rotation in (90, 270):
        width, height = height, width

    t_ms = frame_index(path)
    if not t_ms or width <= 0 or height <= 0:
        raise ValueError(f"Vidéo illisible ou vide : {path}")
    if height > width:
        raise ValueError(f"Vidéo en portrait ({width}x{height} après rotation) : filmez en paysage")
    if not cadence or float(cadence) <= 0:
        logger.warning("Cadence nominale absente pour %s, repli sur %s", path, FPS_PAR_DEFAUT)
        fps_nominal = FPS_PAR_DEFAUT
    else:
        fps_nominal = float(cadence)
    return VideoInfo(
        path=path,
        width=width,
        height=height,
        rotation=rotation,
        fps_nominal=fps_nominal,
        n_frames=len(t_ms),
        duration_ms=t_ms[-1] - t_ms[0],
        codec=codec,
    )


def write_frames_parquet(path_video: str | Path, dst: str | Path) -> Path:
    """Écrit `frames.parquet` (`frame_idx` int64, `t_ms` float64) pour une vidéo."""
    t_ms = frame_index(path_video)
    table = pa.table(
        {
            "frame_idx": pa.array(range(len(t_ms)), type=pa.int64()),
            "t_ms": pa.array(t_ms, type=pa.float64()),
        }
    )
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)
    return dst


def resolve_interval(info: VideoInfo, start: int = 0, end: int | None = None) -> tuple[int, int]:
    """Valide et normalise l'intervalle [start, end] (inclus).

    `end` None ou ≥ n_frames désigne la dernière frame. Lève ValueError si l'intervalle est
    incohérent.
    """
    if start < 0:
        raise ValueError(f"start doit être >= 0 (reçu {start})")
    if start >= info.n_frames:
        raise ValueError(f"start ({start}) hors vidéo ({info.n_frames} frames)")
    if end is None or end >= info.n_frames:
        end = info.n_frames - 1
    if start > end:
        raise ValueError(f"start ({start}) > end ({end})")
    return start, end


def iter_frames(
    path: str | Path, start: int = 0, end: int | None = None, step: int = 1
) -> Iterator[Frame]:
    """Itère sur les frames d'une vidéo : `Frame(idx, t_ms, image BGR)`, rotation appliquée.

    `end` est inclus ; None signifie jusqu'à la fin. L'accès à `start` se fait par seek sur la
    keyframe précédente puis décodage jusqu'au pts exact ; l'index d'une frame est déterminé par
    son pts dans l'index des frames, donc un saut donne la même image qu'une lecture séquentielle.
    """
    if step < 1:
        raise ValueError(f"step doit être >= 1 (reçu {step})")
    info = probe(path)
    start, end = resolve_interval(info, start, end)
    pts, time_base = _pts_index(info.path)
    position = {p: i for i, p in enumerate(pts)}
    facteur = float(time_base) * 1000.0

    container = _ouvrir(info.path)
    try:
        stream = container.streams.video[0]
        if start > 0:
            container.seek(pts[start], stream=stream, backward=True, any_frame=False)
        for decodee in container.decode(stream):
            if decodee.pts is None:
                continue
            idx = position.get(int(decodee.pts))
            if idx is None or idx < start:
                continue
            if idx > end:
                break
            if (idx - start) % step:
                continue
            image = _appliquer_rotation(decodee.to_ndarray(format="bgr24"), info.rotation)
            yield Frame(idx, (pts[idx] - pts[0]) * facteur, image)
    finally:
        container.close()


class VideoWriter:
    """Écrivain H.264 (repli mpeg4), utilisable comme gestionnaire de contexte.

    Les pts sont les t_ms (time_base 1/1000) pour conserver la cadence réelle ; le premier t_ms
    écrit est ramené à 0. Les dimensions impaires sont complétées d'1 pixel noir (bord droit
    et/ou bas), H.264 yuv420p exigeant des dimensions paires.
    """

    def __init__(self, path: str | Path, size: tuple[int, int], fps_nominal: float = 30.0) -> None:
        """`size` est (largeur, hauteur)."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._size = size
        self._width = size[0] + size[0] % 2
        self._height = size[1] + size[1] % 2
        self._container = av.open(str(self.path), "w", options={"movflags": "+faststart"})
        cadence = Fraction(fps_nominal).limit_denominator(1001)
        try:
            self._stream = self._container.add_stream("libx264", rate=cadence)
            self._stream.options = {"crf": CRF, "preset": PRESET}
        except (av.error.FFmpegError, ValueError):
            logger.warning("libx264 indisponible, repli sur mpeg4 (non lisible partout)")
            self._stream = self._container.add_stream("mpeg4", rate=cadence)
        self._stream.width = self._width
        self._stream.height = self._height
        self._stream.pix_fmt = "yuv420p"
        self._stream.time_base = TIME_BASE_ECRITURE
        self._stream.codec_context.time_base = TIME_BASE_ECRITURE
        self._origine: float | None = None
        self._dernier_pts: int | None = None
        self._ferme = False

    @property
    def codec(self) -> str:
        """Nom du codec réellement utilisé."""
        return self._stream.codec_context.name

    def write(self, image_bgr: np.ndarray, t_ms: float) -> None:
        """Ajoute une frame BGR à l'instant `t_ms` (strictement croissant)."""
        if self._origine is None:
            self._origine = t_ms
        pts = round(t_ms - self._origine)
        if self._dernier_pts is not None and pts <= self._dernier_pts:
            raise ValueError(
                f"t_ms doit être strictement croissant (reçu {t_ms}, précédent écrit "
                f"{self._dernier_pts + self._origine})"
            )
        if image_bgr.shape[1] != self._size[0] or image_bgr.shape[0] != self._size[1]:
            raise ValueError(
                f"Taille d'image {image_bgr.shape[1]}x{image_bgr.shape[0]} "
                f"!= {self._size[0]}x{self._size[1]}"
            )
        if (self._width, self._height) != self._size:
            image_bgr = np.pad(
                image_bgr,
                ((0, self._height - self._size[1]), (0, self._width - self._size[0]), (0, 0)),
            )
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image_bgr), format="bgr24")
        frame = frame.reformat(format="yuv420p")
        frame.pts = pts
        frame.time_base = TIME_BASE_ECRITURE
        self._dernier_pts = pts
        for paquet in self._stream.encode(frame):
            self._container.mux(paquet)

    def release(self) -> None:
        """Vide l'encodeur et ferme le fichier."""
        if self._ferme:
            return
        self._ferme = True
        try:
            for paquet in self._stream.encode(None):
                self._container.mux(paquet)
        finally:
            self._container.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def cut_clip(src: str | Path, dst: str | Path, start: int, end: int) -> Path:
    """Extrait les frames [start, end] (inclus) de `src` vers `dst`.

    Les intervalles de temps d'origine sont conservés.
    """
    info = probe(src)
    dst = Path(dst)
    with VideoWriter(dst, (info.width, info.height), info.fps_nominal) as writer:
        for frame in iter_frames(src, start, end):
            writer.write(frame.image, frame.t_ms)
    return dst
