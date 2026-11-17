"""Annotation des frames : joueurs, traînée de la balle et bandeau d'information."""

from collections.abc import Iterable

import cv2
import numpy as np

from tennisvision.tracking.detector import Detection

VERT = (0, 255, 0)
JAUNE = (0, 255, 255)
NOIR = (0, 0, 0)
BLANC = (255, 255, 255)


class BallTrail:
    """Traînée de la balle : points récents, regroupés en segments sans trou trop long.

    Un point est oublié quand son âge (frame courante - frame du point) dépasse `max_age` ;
    deux points consécutifs ne sont reliés que si leurs frames sont séparées d'au plus `max_gap`.

    Un point peut être marqué interpolé (position ajustée, pas détectée par le réseau) : la
    position courante est alors dessinée en cercle creux.
    """

    def __init__(self, max_age: int = 15, max_gap: int = 5) -> None:
        self.max_age = max_age
        self.max_gap = max_gap
        self._points: list[tuple[int, tuple[int, int], bool]] = []
        self._frame_idx: int | None = None

    def update(
        self, frame_idx: int, position: tuple[float, float] | None, interpole: bool = False
    ) -> None:
        """Enregistre la position de la balle à cette frame (None si absente) et purge."""
        self._frame_idx = frame_idx
        if position is not None:
            point = (int(position[0]), int(position[1]))
            self._points.append((frame_idx, point, interpole))
        self._points = [pt for pt in self._points if frame_idx - pt[0] <= self.max_age]

    def segments(self) -> list[list[tuple[int, int]]]:
        """Points regroupés en segments (un nouveau segment démarre après un trou > `max_gap`)."""
        segments: list[list[tuple[int, int]]] = []
        precedente: int | None = None
        for frame, point, _ in self._points:
            if precedente is None or frame - precedente > self.max_gap:
                segments.append([])
            segments[-1].append(point)
            precedente = frame
        return segments

    def current(self) -> tuple[int, int] | None:
        """Position à la frame courante, None si la balle n'y est pas."""
        if self._points and self._points[-1][0] == self._frame_idx:
            return self._points[-1][1]
        return None

    def courant_interpole(self) -> bool:
        """Vrai si la position à la frame courante est interpolée."""
        return bool(self._points and self._points[-1][0] == self._frame_idx and self._points[-1][2])


def _texte_bandeau(frame_idx: int, t_ms: float | None = None) -> str:
    """Texte du bandeau : `Frame {idx}  mm:ss.d` si le temps est connu, sinon `Frame {idx}`."""
    if t_ms is None:
        return f"Frame {frame_idx}"
    dixiemes = int(max(t_ms, 0.0) // 100)
    minutes, reste = divmod(dixiemes, 600)
    return f"Frame {frame_idx}  {minutes:02d}:{reste // 10:02d}.{reste % 10}"


def _dessiner_trainee(
    out: np.ndarray,
    segments: list[list[tuple[int, int]]],
    courant: tuple[int, int] | None,
    interpole: bool = False,
) -> None:
    for segment in segments:
        if len(segment) > 1:
            cv2.polylines(out, [np.array(segment, dtype=np.int32)], False, JAUNE, 2, cv2.LINE_AA)
    if courant is not None:
        if interpole:  # position ajustée et non détectée : cercle creux
            cv2.circle(out, courant, 6, JAUNE, 2, cv2.LINE_AA)
        else:
            cv2.circle(out, courant, 6, JAUNE, -1, cv2.LINE_AA)
        cv2.circle(out, courant, 8, NOIR, 1, cv2.LINE_AA)


def draw_frame(
    frame: np.ndarray,
    detections: Iterable[Detection],
    trail: BallTrail,
    frame_idx: int,
    t_ms: float | None = None,
) -> np.ndarray:
    """Retourne une copie annotée de `frame`.

    Seules les détections « person » sont dessinées ; la balle vient de `trail`, une `BallTrail`
    tenue par l'appelant et déjà mise à jour pour cette frame.
    Le bandeau affiche l'indice de frame et, si `t_ms` est donné, le temps réel.
    """
    out = frame.copy()
    detections = list(detections)

    for det in detections:
        if det.label != "person":
            continue
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(out, p1, p2, VERT, 2)
        ident = det.track_id if det.track_id is not None else "-"
        cv2.putText(
            out,
            f"ID {ident} {det.conf:.2f}",
            (p1[0], max(p1[1] - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            VERT,
            1,
            cv2.LINE_AA,
        )

    _dessiner_trainee(out, trail.segments(), trail.current(), trail.courant_interpole())

    texte = _texte_bandeau(frame_idx, t_ms)
    (w, h), base = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(out, (0, 0), (w + 16, h + base + 12), NOIR, -1)
    cv2.putText(out, texte, (8, h + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLANC, 1, cv2.LINE_AA)
    return out
