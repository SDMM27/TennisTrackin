"""Métriques des joueurs : `players.csv` prédit contre vérité terrain."""

from pathlib import Path

import numpy as np
import pandas as pd

from tennisvision.eval.appariement import scores_detection

IOU_JOUEUR = 0.5
_BOITE = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


def lire_players_csv(chemin: str | Path) -> pd.DataFrame:
    """Lit un `players.csv` du contrat (cellule vide = NaN)."""
    return pd.read_csv(chemin, dtype={"player_id": str})


def _recouvrements(gt: pd.DataFrame, pred: pd.DataFrame, croise: bool) -> pd.DataFrame:
    """Lignes gt dont la boîte recouvre (IoU >= seuil) celle de pred, même frame et même joueur.

    `croise` : compare le gt d'un joueur à la prédiction de l'autre joueur.
    """
    g = gt[["frame_idx", "player_id", *_BOITE]]
    p = pred[["frame_idx", "player_id", *_BOITE]].copy()
    if croise:
        p["player_id"] = p["player_id"].map({"P1": "P2", "P2": "P1"})
    f = g.merge(p, on=["frame_idx", "player_id"], suffixes=("_g", "_p"))
    x1 = np.maximum(f["bbox_x1_g"], f["bbox_x1_p"])
    y1 = np.maximum(f["bbox_y1_g"], f["bbox_y1_p"])
    x2 = np.minimum(f["bbox_x2_g"], f["bbox_x2_p"])
    y2 = np.minimum(f["bbox_y2_g"], f["bbox_y2_p"])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aire_g = (f["bbox_x2_g"] - f["bbox_x1_g"]) * (f["bbox_y2_g"] - f["bbox_y1_g"])
    aire_p = (f["bbox_x2_p"] - f["bbox_x1_p"]) * (f["bbox_y2_p"] - f["bbox_y1_p"])
    union = aire_g + aire_p - inter
    iou = inter / union.where(union > 0)
    return f[iou >= IOU_JOUEUR]


def evaluer_joueurs(pred: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Présence des deux joueurs, IDF1 (identités fixes P1/P2) et échanges d'identité."""
    if len(pred) == 0:
        pred = pd.DataFrame(columns=["frame_idx", "player_id", *_BOITE])
    pred = pred.drop_duplicates(["frame_idx", "player_id"])
    gt = gt.drop_duplicates(["frame_idx", "player_id"])

    def deux_joueurs(table: pd.DataFrame) -> pd.Series:
        return table.groupby("frame_idx")["player_id"].agg(lambda s: {"P1", "P2"} <= set(s))

    present_gt = deux_joueurs(gt)
    frames_ref = present_gt[present_gt].index
    deux = int(deux_joueurs(pred).reindex(frames_ref, fill_value=False).sum())

    pred_sur_gt = pred[pred["frame_idx"].isin(gt["frame_idx"])]
    identite = scores_detection(
        len(_recouvrements(gt, pred_sur_gt, croise=False)), len(pred_sur_gt), len(gt)
    )
    # Échange : P1 prédit sur P2 gt ET P2 prédit sur P1 gt, à la même frame
    croises = _recouvrements(gt, pred_sur_gt, croise=True)
    par_frame = croises.groupby("frame_idx")["player_id"].nunique()
    return {
        "pct_frames_deux_joueurs": {
            "n_ok": deux,
            "n": len(frames_ref),
            "valeur": deux / len(frames_ref) if len(frames_ref) else None,
        },
        "identite": identite,
        "idf1": identite["f1"],
        "n_echanges_identite": int((par_frame == 2).sum()),
    }
