"""Export des détections (DataFrame) et résumé statistique."""

from collections.abc import Iterable, Mapping

import pandas as pd

from tennisvision.io.video import VideoInfo
from tennisvision.tracking.detector import Detection

COLONNES = [
    "frame_idx",
    "t_ms",
    "track_id",
    "cls",
    "label",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
]


def detections_to_dataframe(dets: Iterable[Detection], t_ms: Mapping[int, float]) -> pd.DataFrame:
    """Convertit des détections en DataFrame (une ligne par détection).

    `t_ms` donne l'horodatage réel de chaque frame (`frame_idx` -> ms).
    """
    lignes = [
        {
            "frame_idx": d.frame_idx,
            "t_ms": t_ms[d.frame_idx],
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


def summarize(
    df: pd.DataFrame,
    video_info: VideoInfo,
    frame_range: tuple[int, int],
    n_frames_traitees: int,
    t_range_ms: tuple[float, float],
) -> dict:
    """Résume une analyse sur l'intervalle de frames [début, fin] (inclus).

    `n_frames_traitees` est le nombre de frames réellement traitées ; `t_range_ms` les t_ms de
    la première et de la dernière frame traitées (la durée vient de leur différence).
    """
    debut, fin = frame_range
    t_debut, t_fin = t_range_ms

    personnes = df[(df["label"] == "person") & df["track_id"].notna()]
    par_identite = personnes.groupby("track_id")["frame_idx"].nunique()
    top = par_identite.sort_values(ascending=False, kind="stable").head(5)

    balles = df[df["label"] == "ball"]
    frames_balle = balles["frame_idx"].nunique()

    return {
        "video": video_info.path.name,
        "fps_nominal": video_info.fps_nominal,
        "resolution": [video_info.width, video_info.height],
        "rotation": video_info.rotation,
        "intervalle": [debut, fin],
        "intervalle_ms": [t_debut, t_fin],
        "duree_ms": t_fin - t_debut,
        "n_frames_traitees": n_frames_traitees,
        "n_identites": int(par_identite.size),
        "top_identites": [{"track_id": int(tid), "n_frames": int(n)} for tid, n in top.items()],
        "n_detections_balle": len(balles),
        "taux_detection_balle": frames_balle / n_frames_traitees if n_frames_traitees else 0.0,
    }
