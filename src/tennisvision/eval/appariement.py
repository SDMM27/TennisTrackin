"""Appariement un-pour-un (algorithme hongrois) et scores de détection."""

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

# Coût d'une paire interdite : bien supérieur à toute somme de coûts autorisés (≤ n × 1),
# donc l'optimum maximise d'abord le nombre de paires valides
_PROHIBITIF = 1e9


def _resoudre(cout: np.ndarray, autorise: np.ndarray) -> list[tuple[int, int]]:
    """Affectation de coût minimal ; seules les paires autorisées sont conservées."""
    if cout.size == 0:
        return []
    lignes, colonnes = linear_sum_assignment(np.where(autorise, cout, _PROHIBITIF))
    return [(int(i), int(j)) for i, j in zip(lignes, colonnes, strict=True) if autorise[i, j]]


def apparier_instants(
    pred: Sequence[int | float], gt: Sequence[int | float], tolerance: float
) -> list[tuple[int, int]]:
    """Apparie des instants un-pour-un, au coût total |écart| minimal.

    Seules les paires dont l'écart est inférieur ou égal à `tolerance` sont gardées (l'égalité
    est incluse). Retourne des couples (indice_pred, indice_gt).
    """
    if len(pred) == 0 or len(gt) == 0:
        return []
    ecart = np.abs(np.asarray(pred, dtype=float)[:, None] - np.asarray(gt, dtype=float)[None, :])
    # Le coût d'une paire valide reste < 1 : la tolérance est ramenée à [0, 1) pour que
    # _PROHIBITIF domine toujours
    autorise = ecart <= tolerance
    cout = ecart / (tolerance + 1.0)
    return _resoudre(cout, autorise)


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    """IoU temporel de deux intervalles de frames, bornes incluses."""
    inter = min(a[1], b[1]) - max(a[0], b[0]) + 1
    if inter <= 0:
        return 0.0
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - inter
    return inter / union


def apparier_intervalles(
    pred: Sequence[tuple[int, int]], gt: Sequence[tuple[int, int]], iou_min: float = 0.5
) -> list[tuple[int, int, float]]:
    """Apparie des intervalles de frames un-pour-un, au coût total 1 − IoU minimal.

    Garde les paires d'IoU supérieure ou égale à `iou_min`. Retourne (i_pred, i_gt, iou).
    """
    if len(pred) == 0 or len(gt) == 0:
        return []
    iou = np.array([[_iou(tuple(p), tuple(g)) for g in gt] for p in pred])
    paires = _resoudre(1.0 - iou, (iou >= iou_min) & (iou > 0))
    return [(i, j, float(iou[i, j])) for i, j in paires]


def scores_detection(n_tp: int, n_pred: int, n_gt: int) -> dict:
    """Précision, rappel et F1 ; `None` quand le dénominateur est nul."""
    return {
        "tp": n_tp,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "precision": n_tp / n_pred if n_pred else None,
        "rappel": n_tp / n_gt if n_gt else None,
        "f1": 2 * n_tp / (n_pred + n_gt) if (n_pred + n_gt) else None,
    }
