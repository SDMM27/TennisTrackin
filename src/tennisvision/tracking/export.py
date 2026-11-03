"""Export des détections (DataFrame) et résumé statistique."""

from collections.abc import Iterable

import pandas as pd

from tennisvision.io.video import VideoInfo
from tennisvision.tracking.detector import Detection

COLONNES = ["frame_idx", "track_id", "cls", "label", "conf", "x1", "y1", "x2", "y2", "cx", "cy"]


def detections_to_dataframe(dets: Iterable[Detection]) -> pd.DataFrame:
    """Convertit des détections en DataFrame (une ligne par détection)."""
    lignes = [
        {
            "frame_idx": d.frame_idx,
            "track_id": d.track_id,
            "cls": d.cls,
            "label": d.label,
            "conf": d.conf,
            "x1": d.x1,
            "y1": d.y1,
            "x2": d.x2,
            "y2": d.y2,
            "cx": d.cx,
            "cy": d.cy,
        }
        for d in dets
    ]
    df = pd.DataFrame(lignes, columns=COLONNES)
    # Int64 nullable : conserve les track_id entiers malgré les valeurs manquantes
    df["track_id"] = df["track_id"].astype("Int64")
    return df


def summarize(df: pd.DataFrame, video_info: VideoInfo, frame_range: tuple[int, int]) -> dict:
    """Résume une analyse sur l'intervalle de frames [début, fin] (inclus)."""
    debut, fin = frame_range
    n_frames = fin - debut + 1

    personnes = df[(df["label"] == "person") & df["track_id"].notna()]
    par_identite = personnes.groupby("track_id")["frame_idx"].nunique()
    top = par_identite.sort_values(ascending=False, kind="stable").head(5)

    balles = df[df["label"] == "ball"]
    frames_balle = balles["frame_idx"].nunique()

    return {
        "video": str(video_info.path),
        "fps": video_info.fps,
        "resolution": [video_info.width, video_info.height],
        "intervalle": [debut, fin],
        "n_frames_traitees": n_frames,
        "n_identites": int(par_identite.size),
        "top_identites": [{"track_id": int(tid), "n_frames": int(n)} for tid, n in top.items()],
        "n_detections_balle": len(balles),
        "taux_detection_balle": frames_balle / n_frames if n_frames > 0 else 0.0,
    }
