"""Tests de l'export et du résumé des détections."""

from pathlib import Path

import pytest

from tennisvision.io.video import VideoInfo
from tennisvision.tracking.detector import Detection
from tennisvision.tracking.export import detections_to_dataframe, summarize


def _t_ms(dets):
    """Horodatage fictif : 40 ms par frame."""
    return {d.frame_idx: 40.0 * d.frame_idx for d in dets}


def _det(frame, track_id, label, x=0.0):
    cls = 0 if label == "person" else 32
    return Detection(frame, track_id, cls, label, 0.9, x, 0.0, x + 10, 20.0)


@pytest.fixture
def info():
    return VideoInfo(Path("dossier/v.mp4"), 1280, 720, 0, 25.0, 1000, 39960.0, "h264")


def test_detection_centre():
    d = _det(0, 1, "person", x=10.0)
    assert (d.cx, d.cy) == (15.0, 10.0)


def test_dataframe_vide():
    df = detections_to_dataframe([], {})
    assert df.empty and "track_id" in df.columns


def test_dataframe_colonne_t_ms_apres_frame_idx():
    df = detections_to_dataframe([_det(3, 1, "person")], {3: 123.5})
    assert list(df.columns)[:2] == ["frame_idx", "t_ms"]
    assert df.loc[0, "t_ms"] == 123.5


def test_dataframe_track_id_nul():
    dets = [_det(0, None, "person"), _det(0, 3, "person")]
    df = detections_to_dataframe(dets, _t_ms(dets))
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
    df = detections_to_dataframe(dets, _t_ms(dets))
    resume = summarize(df, info, (0, 9), 10, (0.0, 360.0))
    assert resume["n_frames_traitees"] == 10
    assert resume["n_identites"] == 2
    assert resume["top_identites"] == [
        {"track_id": 1, "n_frames": 4},
        {"track_id": 2, "n_frames": 2},
    ]
    assert resume["n_detections_balle"] == 3
    assert resume["taux_detection_balle"] == pytest.approx(0.2)
    assert resume["video"] == "v.mp4"  # nom seul, jamais le chemin
    assert resume["fps_nominal"] == 25.0
    assert resume["rotation"] == 0
    assert resume["resolution"] == [1280, 720]
    assert resume["intervalle"] == [0, 9]
    assert resume["intervalle_ms"] == [0.0, 360.0]
    assert resume["duree_ms"] == 360.0


def test_summarize_taux_balle_sur_frames_traitees(info):
    # 2 frames avec balle sur 4 traitées
    dets = [_det(0, None, "ball"), _det(0, None, "ball"), _det(2, None, "ball")]
    df = detections_to_dataframe(dets, _t_ms(dets))
    resume = summarize(df, info, (0, 3), 4, (0.0, 120.0))
    assert resume["n_frames_traitees"] == 4
    assert resume["taux_detection_balle"] == pytest.approx(0.5)


def test_summarize_top_limite_a_cinq(info):
    dets = [_det(0, i, "person") for i in range(1, 9)]
    resume = summarize(detections_to_dataframe(dets, _t_ms(dets)), info, (0, 0), 1, (0.0, 0.0))
    assert len(resume["top_identites"]) == 5
    assert resume["n_identites"] == 8


def test_summarize_sans_detection(info):
    resume = summarize(detections_to_dataframe([], {}), info, (5, 14), 10, (200.0, 560.0))
    assert resume["n_identites"] == 0
    assert resume["taux_detection_balle"] == 0.0
    assert resume["duree_ms"] == 360.0
