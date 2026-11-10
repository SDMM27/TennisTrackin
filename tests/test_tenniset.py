"""Tests du chargement des annotations de points."""

import pytest

from tennisvision import config
from tennisvision.datasets.tenniset import get_point, load_points, resoudre_point


@pytest.fixture
def points_file(tmp_path):
    chemin = tmp_path / "points.txt"
    chemin.write_text(
        "P00000001\tV006\t25740\t25984\t0001\nP00000002\tV007\t100\t200\t0002\n",
        encoding="utf-8",
    )
    return chemin


def test_load_points(points_file):
    df = load_points(points_file)
    assert list(df.columns) == ["point_id", "video", "start", "end", "number"]
    assert len(df) == 2
    assert df.loc[0, "number"] == "0001"  # zéros initiaux conservés
    assert df["start"].dtype.kind == "i"


def test_get_point(points_file):
    assert get_point("P00000002", points_file) == ("V007", 100, 200)


def test_get_point_inconnu(points_file):
    with pytest.raises(KeyError, match="P99999999"):
        get_point("P99999999", points_file)


@pytest.fixture
def dossier_tenniset(tmp_path):
    racine = tmp_path / "tenniset"
    annotations = config.points_file(racine)
    annotations.parent.mkdir(parents=True)
    annotations.write_text("P00000001\tV006\t100\t200\t0001\n", encoding="utf-8")
    video = config.tenniset_video(racine, "V006")
    video.parent.mkdir(parents=True)
    video.write_bytes(b"")
    return racine


def test_load_points_exige_le_chemin():
    with pytest.raises(TypeError):
        load_points()


def test_resoudre_point(dossier_tenniset):
    video, debut, fin = resoudre_point("P00000001", dossier_tenniset)
    assert video == dossier_tenniset / "videos" / "V006.mp4"
    assert (debut, fin) == (100, 200)


def test_resoudre_point_inconnu(dossier_tenniset):
    with pytest.raises(KeyError, match="P99999999"):
        resoudre_point("P99999999", dossier_tenniset)


def test_resoudre_point_sans_annotations(tmp_path):
    with pytest.raises(FileNotFoundError, match="points.txt"):
        resoudre_point("P00000001", tmp_path)


def test_resoudre_point_sans_video(dossier_tenniset):
    config.tenniset_video(dossier_tenniset, "V006").unlink()
    with pytest.raises(FileNotFoundError, match="V006.mp4"):
        resoudre_point("P00000001", dossier_tenniset)
