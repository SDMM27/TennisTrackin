"""Annotation des frames : joueurs, balle, traînée et bandeau d'information."""

import itertools
from collections import deque
from collections.abc import Iterable

import cv2
import numpy as np

from tennisvision.tracking.detector import Detection

VERT = (0, 255, 0)
JAUNE = (0, 255, 255)
NOIR = (0, 0, 0)
BLANC = (255, 255, 255)


def draw_frame(
    frame: np.ndarray,
    detections: Iterable[Detection],
    ball_trail: deque[tuple[int, int]],
    frame_idx: int | None = None,
    fps: float = 25.0,
) -> np.ndarray:
    """Retourne une copie annotée de `frame`.

    `ball_trail` est une deque (maxlen = longueur de traînée) tenue par l'appelant :
    la position de la balle de cette frame y est ajoutée.
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

    balles = [d for d in detections if d.label == "ball"]
    if balles:
        meilleure = max(balles, key=lambda d: d.conf)
        ball_trail.append((int(meilleure.cx), int(meilleure.cy)))

    points = list(ball_trail)
    for a, b in itertools.pairwise(points):
        cv2.line(out, a, b, JAUNE, 2, cv2.LINE_AA)
    if balles:
        cv2.circle(out, points[-1], 6, JAUNE, -1, cv2.LINE_AA)
        cv2.circle(out, points[-1], 8, NOIR, 1, cv2.LINE_AA)

    if frame_idx is not None:
        secondes = int(frame_idx / fps) if fps else 0
        texte = f"Frame {frame_idx}  {secondes // 60:02d}:{secondes % 60:02d}"
        (w, h), base = cv2.getTextSize(texte, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(out, (0, 0), (w + 16, h + base + 12), NOIR, -1)
        cv2.putText(out, texte, (8, h + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLANC, 1, cv2.LINE_AA)
    return out
