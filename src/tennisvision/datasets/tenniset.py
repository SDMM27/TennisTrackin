"""Annotations des points (fichier TSV sans en-tête) et résolution vers les vidéos."""

from pathlib import Path

import pandas as pd

from tennisvision.config import points_file, tenniset_video

COLONNES = ["point_id", "video", "start", "end", "number"]


def load_points(path: str | Path) -> pd.DataFrame:
    """Charge les points annotés : point_id, video, start, end, number."""
    return pd.read_csv(
        path,
        sep="	",
        header=None,
        names=COLONNES,
        dtype={"point_id": str, "video": str, "number": str},
    )


def get_point(point_id: str, path: str | Path) -> tuple[str, int, int]:
    """Retourne (video, frame_debut, frame_fin) d'un point. KeyError si inconnu."""
    points = load_points(path)
    ligne = points[points["point_id"] == point_id]
    if ligne.empty:
        raise KeyError(f"Point inconnu : {point_id}")
    row = ligne.iloc[0]
    return str(row["video"]), int(row["start"]), int(row["end"])


def resoudre_point(point_id: str, dossier_tenniset: str | Path) -> tuple[Path, int, int]:
    """Retourne (chemin_video, frame_debut, frame_fin) d'un point d'un dossier TenniSet.

    KeyError si le point est inconnu, FileNotFoundError si le fichier d'annotations ou la
    vidéo est absent.
    """
    dossier = Path(dossier_tenniset)
    fichier = points_file(dossier)
    if not fichier.is_file():
        raise FileNotFoundError(
            f"Fichier d'annotations introuvable : {fichier} (dossier TenniSet : {dossier})"
        )
    nom, debut, fin = get_point(point_id, fichier)
    video = tenniset_video(dossier, nom)
    if not video.is_file():
        raise FileNotFoundError(f"Vidéo du point {point_id} introuvable : {video}")
    return video, debut, fin
