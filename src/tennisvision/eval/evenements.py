"""Métriques sur les événements : deux documents `events.json` d'un même match.

L'appariement est temporel (sur `frame_idx`) et un-pour-un. La vérité terrain et la prédiction
ont le même format ; la `source` des événements n'est pas regardée.
"""

import math

from tennisvision.eval.appariement import apparier_instants, apparier_intervalles, scores_detection
from tennisvision.schemas import L, S, Ws

TOLERANCES_FRAPPE = (2, 5)
TOLERANCE_SERVICE = 5
TOLERANCES_REBOND = (2, 5)
IOU_POINT = 0.5
MARGE_LIGNE_M = 0.30


def _evenements(doc: dict, type_: str) -> list[dict]:
    """Événements d'un type, triés par frame_idx (tri stable)."""
    return sorted(
        (e for e in doc.get("events", []) if e.get("type") == type_),
        key=lambda e: e["frame_idx"],
    )


def _exactitude(n_ok: int, n: int) -> dict:
    return {"n_ok": n_ok, "n": n, "valeur": n_ok / n if n else None}


def _moyenne_max(valeurs: list[float]) -> dict:
    n = len(valeurs)
    return {
        "moyenne": sum(valeurs) / n if n else None,
        "max": max(valeurs) if n else None,
        "n": n,
    }


def _scores_instants(pred: list[dict], gt: list[dict], tolerance: float) -> tuple[dict, list]:
    """Scores de détection à une tolérance, écart moyen en frames, et paires (pred, gt)."""
    indices = apparier_instants(
        [e["frame_idx"] for e in pred], [e["frame_idx"] for e in gt], tolerance
    )
    paires = [(pred[i], gt[j]) for i, j in indices]
    ecarts = [abs(p["frame_idx"] - g["frame_idx"]) for p, g in paires]
    scores = scores_detection(len(paires), len(pred), len(gt))
    scores["ecart_moyen_frames"] = sum(ecarts) / len(ecarts) if ecarts else None
    return scores, paires


def f1_par_classe(matrice: dict[str, dict[str, int]]) -> dict[str, float | None]:
    """F1 un-contre-tous de chaque classe d'une matrice {classe_gt: {classe_pred: n}}."""
    classes = set(matrice)
    for ligne in matrice.values():
        classes.update(ligne)
    resultat = {}
    for c in sorted(classes):
        tp = matrice.get(c, {}).get(c, 0)
        fn = sum(n for p, n in matrice.get(c, {}).items() if p != c)
        fp = sum(ligne.get(c, 0) for g, ligne in matrice.items() if g != c)
        resultat[c] = scores_detection(tp, tp + fp, tp + fn)["f1"]
    return resultat


def evaluer_frappes(pred: dict, gt: dict) -> dict:
    """Frappes (`hit`) : scores à chaque tolérance, joueur et type de coup sur la plus large."""
    p, g = _evenements(pred, "hit"), _evenements(gt, "hit")
    tolerances = {}
    paires: list = []
    for tol in TOLERANCES_FRAPPE:
        tolerances[str(tol)], paires = _scores_instants(p, g, tol)
    ok_joueur = sum(ep.get("player_id") == eg.get("player_id") for ep, eg in paires)
    brut: dict[str, dict[str, int]] = {}
    n_coup = 0
    for ep, eg in paires:
        sp, sg = ep.get("stroke"), eg.get("stroke")
        if sp and sg:
            n_coup += 1
            ligne = brut.setdefault(sg, {})
            ligne[sp] = ligne.get(sp, 0) + 1
    matrice = {c: dict(sorted(ligne.items())) for c, ligne in sorted(brut.items())}
    return {
        "tolerances": tolerances,
        "exactitude_joueur": _exactitude(ok_joueur, len(paires)),
        "n_paires_coup": n_coup,
        "matrice_confusion": matrice,
        "f1_par_classe": f1_par_classe(matrice),
    }


def evaluer_services(pred: dict, gt: dict) -> dict:
    """Services (`serve`) : scores, puis exactitude de `result`, `serve_number` et `player_id`."""
    p, g = _evenements(pred, "serve"), _evenements(gt, "serve")
    scores, paires = _scores_instants(p, g, TOLERANCE_SERVICE)
    for champ in ("result", "serve_number", "player_id"):
        ok = sum(ep.get(champ) == eg.get(champ) for ep, eg in paires)
        scores[f"exactitude_{champ}"] = _exactitude(ok, len(paires))
    return scores


def distance_ligne_m(x: float, y: float) -> float:
    """Distance (m) d'un point du sol à la ligne la plus proche du court de simple."""
    segments = [
        ((-Ws, -L), (-Ws, L)),  # lignes de côté
        ((Ws, -L), (Ws, L)),
        ((-Ws, -L), (Ws, -L)),  # lignes de fond
        ((-Ws, L), (Ws, L)),
        ((-Ws, -S), (Ws, -S)),  # lignes de service
        ((-Ws, S), (Ws, S)),
        ((0.0, -S), (0.0, S)),  # ligne médiane
    ]
    meilleur = math.inf
    for (x1, y1), (x2, y2) in segments:
        # Segments alignés sur les axes : le point le plus proche s'obtient en bornant x et y
        px = min(max(x, min(x1, x2)), max(x1, x2))
        py = min(max(y, min(y1, y2)), max(y1, y2))
        meilleur = min(meilleur, math.hypot(x - px, y - py))
    return meilleur


def evaluer_rebonds(pred: dict, gt: dict) -> dict:
    """Rebonds (`bounce`) : scores aux deux tolérances, position et in/out sur la plus large."""
    p, g = _evenements(pred, "bounce"), _evenements(gt, "bounce")
    tolerances = {}
    paires: list = []
    for tol in TOLERANCES_REBOND:
        tolerances[str(tol)], paires = _scores_instants(p, g, tol)
    erreurs = [
        math.dist(ep["court_xy"], eg["court_xy"]) * 100
        for ep, eg in paires
        if ep.get("court_xy") and eg.get("court_xy")
    ]
    ok = sum(ep.get("in_out") == eg.get("in_out") for ep, eg in paires)
    pres = [
        (ep, eg)
        for ep, eg in paires
        if eg.get("court_xy") and distance_ligne_m(*eg["court_xy"]) < MARGE_LIGNE_M
    ]
    ok_pres = sum(ep.get("in_out") == eg.get("in_out") for ep, eg in pres)
    return {
        "tolerances": tolerances,
        "erreur_position_cm": _moyenne_max(erreurs),
        "exactitude_in_out": _exactitude(ok, len(paires)),
        "exactitude_in_out_pres_ligne": _exactitude(ok_pres, len(pres)),
    }


def _intervalles(evenements: list[dict]) -> list[tuple[int, int]]:
    return [(e["frame_idx"], e.get("end_frame_idx", e["frame_idx"])) for e in evenements]


def evaluer_points(pred: dict, gt: dict) -> dict:
    """Points (`point`) puis jeux (`game`), appariés sur l'IoU temporel de leurs intervalles."""
    p, g = _evenements(pred, "point"), _evenements(gt, "point")
    paires = apparier_intervalles(_intervalles(p), _intervalles(g), IOU_POINT)
    bloc = scores_detection(len(paires), len(p), len(g))
    bloc["iou_moyen"] = _moyenne_max([iou for _, _, iou in paires])
    avec_gagnant = [(p[i], g[j]) for i, j, _ in paires if g[j].get("winner_id")]
    ok = sum(ep.get("winner_id") == eg["winner_id"] for ep, eg in avec_gagnant)
    bloc["exactitude_gagnant"] = _exactitude(ok, len(avec_gagnant))

    pj, gj = _evenements(pred, "game"), _evenements(gt, "game")
    if not gj:
        bloc["jeux"] = None
        return bloc
    appariement = {
        j: i for i, j, _ in apparier_intervalles(_intervalles(pj), _intervalles(gj), IOU_POINT)
    }
    avec_score = [(j, e["score_after"]) for j, e in enumerate(gj) if e.get("score_after")]
    juste = sum(
        j in appariement and pj[appariement[j]].get("score_after") == score
        for j, score in avec_score
    )
    jeux = scores_detection(len(appariement), len(pj), len(gj))
    jeux["pct_jeux_score_juste"] = _exactitude(juste, len(avec_score))
    bloc["jeux"] = jeux
    return bloc


def evaluer_evenements(pred: dict, gt: dict) -> dict:
    """Les quatre blocs d'un match ; un bloc vaut None si le gt n'a aucun événement du type."""

    def present(type_: str) -> bool:
        return any(e.get("type") == type_ for e in gt.get("events", []))

    return {
        "frappes": evaluer_frappes(pred, gt) if present("hit") else None,
        "services": evaluer_services(pred, gt) if present("serve") else None,
        "rebonds": evaluer_rebonds(pred, gt) if present("bounce") else None,
        "points": evaluer_points(pred, gt) if present("point") else None,
    }
