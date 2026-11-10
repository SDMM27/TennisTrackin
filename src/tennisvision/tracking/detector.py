"""Détection et suivi (YOLO + ByteTrack via ultralytics), frame par frame."""

from dataclasses import dataclass

import numpy as np

from tennisvision.config import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_TRACKER,
    PERSON,
    SPORTS_BALL,
)
from tennisvision.models import resoudre_modele

LABELS = {PERSON: "person", SPORTS_BALL: "ball"}


@dataclass(frozen=True)
class Detection:
    """Une détection (boîte en pixels) associée à une frame."""

    frame_idx: int
    track_id: int | None
    cls: int
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        """Abscisse du centre de la boîte."""
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        """Ordonnée du centre de la boîte."""
        return (self.y1 + self.y2) / 2


class Tracker:
    """Enveloppe d'un modèle YOLO avec suivi ByteTrack persistant entre les frames."""

    def __init__(
        self,
        model_path: str | None = None,
        conf: float = DEFAULT_CONF,
        tracker: str = DEFAULT_TRACKER,
        device: str | None = None,
        imgsz: int = DEFAULT_IMGSZ,
    ) -> None:
        from ultralytics import YOLO  # import tardif : lourd

        self.model = YOLO(str(resoudre_modele(model_path)))
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker
        self.device = device

    def update(self, frame_idx: int, frame: np.ndarray) -> list[Detection]:
        """Détecte et suit les objets d'une frame (à appeler dans l'ordre des frames)."""
        result = self.model.track(
            frame,
            persist=True,
            classes=[PERSON, SPORTS_BALL],
            conf=self.conf,
            imgsz=self.imgsz,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        ids = None if boxes.id is None else boxes.id.cpu().numpy().astype(int)

        detections = []
        for i in range(len(xyxy)):
            cls = int(classes[i])
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            detections.append(
                Detection(
                    frame_idx=frame_idx,
                    track_id=None if ids is None else int(ids[i]),
                    cls=cls,
                    label=LABELS.get(cls, str(cls)),
                    conf=float(confs[i]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections
