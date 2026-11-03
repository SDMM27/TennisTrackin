"""Tests de la CLI (sans charger YOLO)."""

import pytest

from tennisvision import cli
from tennisvision.datasets.tenniset import get_point

RESUME = {
    "video": "v.mp4",
    "fps": 25.0,
    "resolution": [64, 48],
    "intervalle": [0, 9],
    "n_frames_traitees": 10,
    "n_identites": 2,
    "top_identites": [{"track_id": 1, "n_frames": 10}],
    "n_detections_balle": 3,
    "taux_detection_balle": 0.3,
    "fps_traitement": 12.0,
    "sorties": {"dossier": "out", "video_annotee": None, "tracks": "t.csv", "resume": "s.json"},
}


@pytest.fixture
def points(tmp_path, monkeypatch):
    chemin = tmp_path / "points.txt"
    chemin.write_text("P00000001\tV006\t100\t200\t0001\n", encoding="utf-8")
    monkeypatch.setattr(cli, "get_point", lambda pid: get_point(pid, chemin))
    return chemin


def test_parsing_analyze():
    args = cli.build_parser().parse_args(
        [
            "analyze",
            "v.mp4",
            "--start",
            "3",
            "--end",
            "9",
            "--conf",
            "0.4",
            "--device",
            "cpu",
            "--no-video",
            "-v",
        ]
    )
    assert (args.start, args.end, args.conf, args.device) == (3, 9, 0.4, "cpu")
    assert args.no_video and args.verbose


def test_analyze_appelle_pipeline(tmp_path, monkeypatch, capsys):
    appels = {}

    def faux_analyze(video, **kwargs):
        appels.update(kwargs, video=video)
        return RESUME

    monkeypatch.setattr(cli, "analyze", faux_analyze)
    code = cli.main(["analyze", "v.mp4", "--start", "3", "--end", "9", "--no-video"])
    assert code == 0
    assert appels["start"] == 3 and appels["end"] == 9 and appels["write_video"] is False
    sortie = capsys.readouterr().out
    assert "30.0 %" in sortie


def test_point_incoherent(points, monkeypatch, capsys):
    monkeypatch.setattr(cli, "analyze", lambda *a, **k: pytest.fail("ne doit pas être appelé"))
    code = cli.main(["analyze", "data/V007.mp4", "--point", "P00000001"])
    assert code == 2
    assert "V006" in capsys.readouterr().err


def test_point_resolu(points, monkeypatch):
    appels = {}
    monkeypatch.setattr(cli, "analyze", lambda video, **kw: appels.update(kw) or RESUME)
    assert cli.main(["analyze", "data/V006.mp4", "--point", "P00000001"]) == 0
    assert (appels["start"], appels["end"]) == (100, 200)


def test_point_inconnu(points, capsys):
    assert cli.main(["analyze", "V006.mp4", "--point", "P99999999"]) == 2
    assert "P99999999" in capsys.readouterr().err


def test_point_et_start_incompatibles(points, capsys):
    assert cli.main(["analyze", "V006.mp4", "--point", "P00000001", "--start", "1"]) == 2


def test_video_inexistante(tmp_path, capsys):
    code = cli.main(["analyze", str(tmp_path / "absente.mp4"), "--no-video"])
    assert code == 2
    assert "introuvable" in capsys.readouterr().err


def test_argument_invalide():
    assert cli.main(["analyze"]) == 2


def test_clip(video_synthetique, tmp_path, capsys):
    sortie = tmp_path / "clip.mp4"
    code = cli.main(
        ["clip", str(video_synthetique), "--start", "2", "--end", "6", "-o", str(sortie)]
    )
    assert code == 0 and sortie.exists()


def test_clip_sans_fin(video_synthetique, tmp_path, capsys):
    code = cli.main(["clip", str(video_synthetique), "-o", str(tmp_path / "c.mp4")])
    assert code == 2
