"""Annotations des points (fichier TSV sans en-tête)."""

from pathlib import Path

import pandas as pd

from tennisvision.config import POINTS_FILE

COLONNES = ["point_id", "video", "start", "end", "number"]


def load_points(path: str | Path = POINTS_FILE) -> pd.DataFrame:
    """Charge les points annotés : point_id, video, start, end, number."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=COLONNES,
        dtype={"point_id": str, "video": str, "number": str},
    )


def get_point(point_id: str, path: str | Path = POINTS_FILE) -> tuple[str, int, int]:
    """Retourne (video, frame_debut, frame_fin) d'un point. KeyError si inconnu."""
    points = load_points(path)
    ligne = points[points["point_id"] == point_id]
    if ligne.empty:
        raise KeyError(f"Point inconnu : {point_id}")
    row = ligne.iloc[0]
    return str(row["video"]), int(row["start"]), int(row["end"])
