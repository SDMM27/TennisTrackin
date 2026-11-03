"""Tests du chargement des annotations de points."""

import pytest

from tennisvision.datasets.tenniset import get_point, load_points


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
