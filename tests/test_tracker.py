"""Tests du `Tracker` avec un faux YOLO (aucun poids chargé)."""

import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from tennisvision.config import PERSON
from tennisvision.tracking import detector
from tennisvision.tracking.detector import Tracker


class FauxYolo:
    appels: ClassVar[list[dict]] = []

    def __init__(self, chemin):
        self.chemin = chemin

    def track(self, frame, **kwargs):
        FauxYolo.appels.append(kwargs)
        return [SimpleNamespace(boxes=None)]


def test_le_tracker_ne_suit_que_les_personnes(monkeypatch):
    FauxYolo.appels = []
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FauxYolo))
    monkeypatch.setattr(detector, "resoudre_modele", lambda nom: "poids.pt")
    tracker = Tracker()
    assert tracker.update(0, np.zeros((48, 64, 3), dtype=np.uint8)) == []
    assert FauxYolo.appels[0]["classes"] == [PERSON]
