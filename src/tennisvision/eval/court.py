"""Métriques du court : `court.json` prédit contre vérité terrain."""

import math

import numpy as np

from tennisvision.schemas import COURT_KEYPOINTS_M


def _moyenne_max(valeurs: list[float]) -> dict:
    n = len(valeurs)
    return {
        "moyenne": sum(valeurs) / n if n else None,
        "max": max(valeurs) if n else None,
        "n": n,
    }


def _segment_valide(segments: list[dict], frame: int) -> dict | None:
    """Segment dont l'intervalle contient `frame` (`valid_to_frame` null = jusqu'à la fin)."""
    for seg in segments:
        fin = seg.get("valid_to_frame")
        if seg["valid_from_frame"] <= frame and (fin is None or frame <= fin):
            return seg
    return None


def _projeter(homographie, point) -> tuple[float, float] | None:
    """Projette un point px en m par l'homographie 3x3 ; None si le point part à l'infini."""
    x, y, w = np.asarray(homographie, dtype=float) @ np.array([point[0], point[1], 1.0])
    if abs(w) < 1e-12:
        return None
    return float(x / w), float(y / w)


def evaluer_court(pred: dict, gt: dict) -> dict:
    """Erreur des points clés en px (pred contre gt) et en cm (gt projeté par l'homographie pred).

    Chaque segment gt est comparé au segment pred valide à son `valid_from_frame` ; sans segment
    pred, il est compté comme manqué.
    """
    erreurs_px: list[float] = []
    erreurs_cm: list[float] = []
    manques = 0
    segments_gt = gt.get("segments", [])
    for seg_gt in segments_gt:
        seg_pred = _segment_valide(pred.get("segments", []), seg_gt["valid_from_frame"])
        if seg_pred is None:
            manques += 1
            continue
        for k, pt_gt in enumerate(seg_gt["keypoints_px"]):
            if pt_gt is None:
                continue
            pt_pred = seg_pred["keypoints_px"][k]
            if pt_pred is not None:
                erreurs_px.append(math.dist(pt_pred, pt_gt))
            projete = _projeter(seg_pred["homography"], pt_gt)
            if projete is not None:
                erreurs_cm.append(math.dist(projete, COURT_KEYPOINTS_M[k]) * 100)
    return {
        "n_segments": len(segments_gt),
        "n_segments_manques": manques,
        "erreur_px": _moyenne_max(erreurs_px),
        "erreur_cm": _moyenne_max(erreurs_cm),
    }
