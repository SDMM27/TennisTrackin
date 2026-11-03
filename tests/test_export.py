"""Tests de l'export et du résumé des détections."""

from pathlib import Path

import pytest

from tennisvision.io.video import VideoInfo
from tennisvision.tracking.detector import Detection
from tennisvision.tracking.export import detections_to_dataframe, summarize


def _det(frame, track_id, label, x=0.0):
    cls = 0 if label == "person" else 32
    return Detection(frame, track_id, cls, label, 0.9, x, 0.0, x + 10, 20.0)


@pytest.fixture
def info():
    return VideoInfo(Path("v.mp4"), 25.0, 1280, 720, 1000)


def test_detection_centre():
    d = _det(0, 1, "person", x=10.0)
    assert (d.cx, d.cy) == (15.0, 10.0)


def test_dataframe_vide():
    df = detections_to_dataframe([])
    assert df.empty and "track_id" in df.columns


def test_dataframe_track_id_nul():
    df = detections_to_dataframe([_det(0, None, "person"), _det(0, 3, "person")])
    assert df["track_id"].isna().sum() == 1
    assert df["track_id"].dropna().iloc[0] == 3


def test_summarize(info):
    dets = [
        # identité 1 : 4 frames, identité 2 : 2 frames, une personne sans id
        *[_det(f, 1, "person") for f in range(4)],
        *[_det(f, 2, "person") for f in range(2)],
        _det(0, None, "person"),
        # balle sur 2 frames, 3 détections
        _det(0, None, "ball"),
        _det(0, None, "ball"),
        _det(3, 7, "ball"),
    ]
    resume = summarize(detections_to_dataframe(dets), info, (0, 9))
    assert resume["n_frames_traitees"] == 10
    assert resume["n_identites"] == 2
    assert resume["top_identites"] == [
        {"track_id": 1, "n_frames": 4},
        {"track_id": 2, "n_frames": 2},
    ]
    assert resume["n_detections_balle"] == 3
    assert resume["taux_detection_balle"] == pytest.approx(0.2)
    assert resume["fps"] == 25.0
    assert resume["resolution"] == [1280, 720]
    assert resume["intervalle"] == [0, 9]


def test_summarize_top_limite_a_cinq(info):
    dets = [_det(0, i, "person") for i in range(1, 9)]
    resume = summarize(detections_to_dataframe(dets), info, (0, 0))
    assert len(resume["top_identites"]) == 5
    assert resume["n_identites"] == 8


def test_summarize_sans_detection(info):
    resume = summarize(detections_to_dataframe([]), info, (5, 14))
    assert resume["n_identites"] == 0
    assert resume["taux_detection_balle"] == 0.0
