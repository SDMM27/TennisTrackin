"""Métriques de la balle : `ball.csv` prédit contre vérité terrain (protocole TrackNet)."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SEUILS_BALLE_PX = (4, 10)  # à 1080p
HAUTEUR_REFERENCE = 1080
TROU_LONG_FRAMES = 5


def lire_ball_csv(chemin: str | Path) -> pd.DataFrame:
    """Lit un `ball.csv` du contrat (cellule vide = NaN)."""
    return pd.read_csv(chemin)


def _visible(table: pd.DataFrame) -> np.ndarray:
    return ((table["visible"] == 1) & table["x"].notna() & table["y"].notna()).to_numpy()


def _trous(gt_visible: np.ndarray, manque: np.ndarray, frames: np.ndarray) -> list[int]:
    """Longueurs des suites maximales de frames contiguës où le gt est vu et la pred non."""
    longueurs: list[int] = []
    courant = 0
    precedente = None
    for en_trou, frame in zip(gt_visible & manque, frames, strict=True):
        if en_trou and courant and frame == precedente + 1:
            courant += 1
        elif en_trou:
            if courant:
                longueurs.append(courant)
            courant = 1
        elif courant:
            longueurs.append(courant)
            courant = 0
        precedente = frame
    if courant:
        longueurs.append(courant)
    return longueurs


def evaluer_balle(
    pred: pd.DataFrame,
    gt: pd.DataFrame,
    hauteur_image: int,
    seuils_px: Sequence[float] = SEUILS_BALLE_PX,
) -> dict:
    """Précision, rappel, F1 par seuil de distance, et taux de trous longs.

    Les seuils sont donnés à 1080p et mis à l'échelle par `hauteur_image / 1080`. Seules les
    frames du gt sont évaluées ; une frame absente de `pred` vaut « non visible ». Une
    détection visible trop éloignée compte à la fois comme FP et comme FN.
    """
    gt = gt.sort_values("frame_idx").reset_index(drop=True)
    pred_aligne = (
        pred.drop_duplicates("frame_idx").set_index("frame_idx").reindex(gt["frame_idx"])
        if len(pred)
        else pd.DataFrame({"x": np.nan, "y": np.nan, "visible": 0}, index=gt["frame_idx"])
    )
    gt_vu = _visible(gt)
    pred_vu = _visible(pred_aligne.reset_index())
    distance = np.hypot(
        gt["x"].to_numpy(dtype=float) - pred_aligne["x"].to_numpy(dtype=float),
        gt["y"].to_numpy(dtype=float) - pred_aligne["y"].to_numpy(dtype=float),
    )
    echelle = hauteur_image / HAUTEUR_REFERENCE
    par_seuil = {}
    for seuil in seuils_px:
        limite = seuil * echelle
        proche = gt_vu & pred_vu & (distance <= limite)
        n_tp = int(proche.sum())
        n_fp = int((pred_vu & ~proche).sum())
        n_fn = int((gt_vu & ~proche).sum())
        par_seuil[f"{seuil:g}"] = {
            "seuil_px_effectif": limite,
            "n_tp": n_tp,
            "n_fp": n_fp,
            "n_fn": n_fn,
            "precision": n_tp / (n_tp + n_fp) if n_tp + n_fp else None,
            "rappel": n_tp / (n_tp + n_fn) if n_tp + n_fn else None,
            "f1": 2 * n_tp / (2 * n_tp + n_fp + n_fn) if n_tp + n_fp + n_fn else None,
        }

    trous = _trous(gt_vu, ~pred_vu, gt["frame_idx"].to_numpy())
    longs = [t for t in trous if t > TROU_LONG_FRAMES]
    n_visibles = int(gt_vu.sum())
    return {
        "seuils": par_seuil,
        "n_frames_visibles": n_visibles,
        "n_frames_trous_longs": sum(longs),
        "n_trous_longs": len(longs),
        "taux_trous_longs": sum(longs) / n_visibles if n_visibles else None,
    }
