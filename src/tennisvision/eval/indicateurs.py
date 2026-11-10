"""Statistiques vues par l'utilisateur, calculées depuis un document `events.json`.

Implémentation de référence du banc : le Lot 10 (module de statistiques) la réutilisera ou la
remplacera. Le banc compare la valeur calculée sur les événements prédits à celle calculée sur la
vérité terrain, pour mesurer l'erreur de chaque statistique affichée.
"""

JOUEURS = ("P1", "P2")
STATS_JOUEUR = (
    "first_serve_pct",
    "second_serve_pct",
    "double_faults",
    "aces",
    "points_won",
    "forehands",
    "backhands",
    "volleys",
    "smashes",
    "winners",
    "unforced_errors",
)
STATS_MATCH = ("rally_length_mean", "bounce_in_pct")


def _stat(valeur: float | None, n: int, unite: str) -> dict:
    return {"value": valeur, "n": n, "unit": unite}


def _pct_services(evenements: list[dict], joueur: str, numero: int) -> dict:
    """% de services dans le jeu (`in`), lets exclus."""
    services = [
        e
        for e in evenements
        if e["type"] == "serve"
        and e.get("player_id") == joueur
        and e.get("serve_number") == numero
        and e.get("result") != "let"
    ]
    n = len(services)
    dans = sum(e.get("result") == "in" for e in services)
    return _stat(100 * dans / n if n else None, n, "pct")


def _compte(evenements: list[dict], predicat) -> dict:
    n = sum(1 for e in evenements if predicat(e))
    return _stat(float(n), n, "count")


def calculer_indicateurs(doc: dict) -> dict[tuple[str, str | None], dict]:
    """Calcule toutes les statistiques ; la clé est (id_stat, player_id), `player_id` valant
    None pour les statistiques de match.

    Chaque valeur est `{"value": float|None, "n": int, "unit": "pct"|"count"|"ratio"}`. Toutes les
    clés sont toujours présentes (`value` None quand la statistique est indéfinie), pour que
    prédiction et vérité terrain aient les mêmes. Les pourcentages sont sur 0-100.
    """
    ev = doc.get("events", [])
    hits = [e for e in ev if e["type"] == "hit"]
    points = [e for e in ev if e["type"] == "point"]
    resultat: dict[tuple[str, str | None], dict] = {}

    for j in JOUEURS:
        du_joueur = [e for e in hits if e.get("player_id") == j]
        resultat[("first_serve_pct", j)] = _pct_services(ev, j, 1)
        resultat[("second_serve_pct", j)] = _pct_services(ev, j, 2)
        resultat[("double_faults", j)] = _compte(
            ev,
            lambda e, j=j: (
                e["type"] == "serve"
                and e.get("player_id") == j
                and e.get("serve_number") == 2
                and e.get("result") == "fault"
            ),
        )
        resultat[("aces", j)] = _compte(
            points, lambda e, j=j: e.get("end_reason") == "ace" and e.get("server_id") == j
        )
        gagnes = _compte(points, lambda e, j=j: e.get("winner_id") == j)
        gagnes["n"] = sum(1 for e in points if e.get("winner_id"))
        resultat[("points_won", j)] = gagnes
        for nom, strokes in (
            ("forehands", {"forehand"}),
            ("backhands", {"backhand"}),
            ("volleys", {"forehand_volley", "backhand_volley"}),
            ("smashes", {"smash"}),
        ):
            resultat[(nom, j)] = _compte(du_joueur, lambda e, s=strokes: e.get("stroke") in s)
        resultat[("winners", j)] = _compte(du_joueur, lambda e: e.get("outcome") == "winner")
        resultat[("unforced_errors", j)] = _compte(
            du_joueur, lambda e: e.get("outcome") == "unforced_error"
        )

    longueurs = [
        sum(1 for h in hits if e["frame_idx"] <= h["frame_idx"] <= e["end_frame_idx"])
        for e in points
    ]
    resultat[("rally_length_mean", None)] = _stat(
        sum(longueurs) / len(longueurs) if longueurs else None, len(longueurs), "ratio"
    )
    bonds = [e["in_out"] for e in ev if e["type"] == "bounce" and e["in_out"] in ("in", "out")]
    resultat[("bounce_in_pct", None)] = _stat(
        100 * bonds.count("in") / len(bonds) if bonds else None, len(bonds), "pct"
    )
    return resultat


def comparer_indicateurs(pred_doc: dict, gt_doc: dict) -> list[dict]:
    """Compare les statistiques prédites et vraies d'un match, une ligne par statistique."""
    pred, gt = calculer_indicateurs(pred_doc), calculer_indicateurs(gt_doc)
    lignes = []
    for (stat, joueur), ref in gt.items():
        p, g = pred[(stat, joueur)]["value"], ref["value"]
        erreur = abs(p - g) if p is not None and g is not None else None
        lignes.append(
            {
                "id": stat,
                "player_id": joueur,
                "unit": ref["unit"],
                "pred": p,
                "gt": g,
                "n_gt": ref["n"],
                "erreur_abs": erreur,
                "erreur_rel": erreur / abs(g) if erreur is not None and g else None,
            }
        )
    return lignes


def agreger_indicateurs(par_match: dict[str, list[dict]]) -> list[dict]:
    """Erreurs par statistique sur plusieurs matchs : moyenne, pire cas et pire match.

    Les valeurs None (statistique indéfinie d'un côté) sont ignorées.
    """
    groupes: dict[tuple[str, str | None], list[tuple[str, dict]]] = {}
    for match in sorted(par_match):
        for ligne in par_match[match]:
            groupes.setdefault((ligne["id"], ligne["player_id"]), []).append((match, ligne))

    resultat = []
    for (stat, joueur), lignes in sorted(
        groupes.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
    ):
        absolues = [(m, l["erreur_abs"]) for m, l in lignes if l["erreur_abs"] is not None]
        relatives = [l["erreur_rel"] for _, l in lignes if l["erreur_rel"] is not None]
        pire = max(absolues, key=lambda x: x[1])[0] if absolues else None
        resultat.append(
            {
                "id": stat,
                "player_id": joueur,
                "unit": lignes[0][1]["unit"],
                "erreur_abs_moyenne": sum(e for _, e in absolues) / len(absolues)
                if absolues
                else None,
                "erreur_abs_max": max(e for _, e in absolues) if absolues else None,
                "erreur_rel_moyenne": sum(relatives) / len(relatives) if relatives else None,
                "erreur_rel_max": max(relatives) if relatives else None,
                "n_matchs": len(absolues),
                "pire_match": pire,
            }
        )
    return resultat
