"""Accord inter-annotateurs : deux annotations d'un même match (ou de mêmes matchs).

L'appariement réutilise celui du banc d'évaluation (`tennisvision.eval.appariement`) : on ne
refait pas l'algorithme hongrois. Les deux annotations sont symétriques (A et B sont deux
annotateurs, aucune n'est « la vérité »).
"""

import json
from pathlib import Path

from tennisvision.eval.appariement import apparier_instants, apparier_intervalles, scores_detection
from tennisvision.eval.evenements import IOU_POINT, TOLERANCE_SERVICE

TOLERANCE_FRAPPE = 2  # frames, comme le critère du Lot 2
TOLERANCE_REBOND = 2
OBJECTIFS = {"frappes_f1": 0.95, "kappa_coup": 0.9, "in_out": 0.95}


def _evenements(doc: dict, type_: str) -> list[dict]:
    return sorted(
        (e for e in doc.get("events", []) if e.get("type") == type_),
        key=lambda e: e["frame_idx"],
    )


def kappa_cohen(matrice: dict[str, dict[str, int]]) -> float | None:
    """κ de Cohen d'une matrice {étiquette A: {étiquette B: n}} ; None sans paire."""
    n = sum(sum(ligne.values()) for ligne in matrice.values())
    if n == 0:
        return None
    classes = set(matrice) | {b for ligne in matrice.values() for b in ligne}
    observe = sum(matrice.get(c, {}).get(c, 0) for c in classes) / n
    marge_a = {c: sum(matrice.get(c, {}).values()) / n for c in classes}
    marge_b = {c: sum(ligne.get(c, 0) for ligne in matrice.values()) / n for c in classes}
    attendu = sum(marge_a[c] * marge_b[c] for c in classes)
    if attendu == 1:
        return 1.0 if observe == 1 else None
    return (observe - attendu) / (1 - attendu)


def _exactitude(n_ok: int, n: int) -> dict:
    return {"n_ok": n_ok, "n": n, "valeur": n_ok / n if n else None}


def _paires(a: list[dict], b: list[dict], tolerance: float) -> list[tuple[dict, dict]]:
    indices = apparier_instants([e["frame_idx"] for e in a], [e["frame_idx"] for e in b], tolerance)
    return [(a[i], b[j]) for i, j in indices]


def accord_match(doc_a: dict, doc_b: dict) -> dict:
    """Comptes d'accord de deux documents `events.json` (sommables d'un match à l'autre)."""
    hits_a, hits_b = _evenements(doc_a, "hit"), _evenements(doc_b, "hit")
    paires = _paires(hits_a, hits_b, TOLERANCE_FRAPPE)
    matrice: dict[str, dict[str, int]] = {}
    for ea, eb in paires:
        sa, sb = ea.get("stroke"), eb.get("stroke")
        if sa and sb:
            ligne = matrice.setdefault(sa, {})
            ligne[sb] = ligne.get(sb, 0) + 1

    rebonds = _paires(_evenements(doc_a, "bounce"), _evenements(doc_b, "bounce"), TOLERANCE_REBOND)
    ok_in_out = sum(ea.get("in_out") == eb.get("in_out") for ea, eb in rebonds)

    pts_a, pts_b = _evenements(doc_a, "point"), _evenements(doc_b, "point")

    def intervalle(e: dict) -> tuple[int, int]:
        return e["frame_idx"], e.get("end_frame_idx", e["frame_idx"])

    appares = apparier_intervalles(
        [intervalle(e) for e in pts_a], [intervalle(e) for e in pts_b], IOU_POINT
    )
    avec_gagnant = [
        (pts_a[i], pts_b[j])
        for i, j, _ in appares
        if pts_a[i].get("winner_id") and pts_b[j].get("winner_id")
    ]
    ok_gagnant = sum(ea["winner_id"] == eb["winner_id"] for ea, eb in avec_gagnant)

    services = _paires(_evenements(doc_a, "serve"), _evenements(doc_b, "serve"), TOLERANCE_SERVICE)
    ok_service = sum(ea.get("result") == eb.get("result") for ea, eb in services)

    return {
        "frappes": {"n_a": len(hits_a), "n_b": len(hits_b), "tp": len(paires)},
        "matrice_coups": {a: dict(sorted(ligne.items())) for a, ligne in sorted(matrice.items())},
        "in_out": _exactitude(ok_in_out, len(rebonds)),
        "gagnant_point": _exactitude(ok_gagnant, len(avec_gagnant)),
        "resultat_service": _exactitude(ok_service, len(services)),
    }


def _resume(brut: dict) -> dict:
    """Pourcentages et verdicts à partir des comptes."""
    f = brut["frappes"]
    scores = scores_detection(f["tp"], f["n_a"], f["n_b"])
    return {
        **brut,
        "frappes": {
            **f,
            "pct_a_dans_b": f["tp"] / f["n_a"] if f["n_a"] else None,
            "pct_b_dans_a": f["tp"] / f["n_b"] if f["n_b"] else None,
            "f1": scores["f1"],
        },
        "kappa_coup": kappa_cohen(brut["matrice_coups"]),
    }


def _somme(blocs: list[dict]) -> dict:
    matrice: dict[str, dict[str, int]] = {}
    for bloc in blocs:
        for a, ligne in bloc["matrice_coups"].items():
            cible = matrice.setdefault(a, {})
            for b, n in ligne.items():
                cible[b] = cible.get(b, 0) + n
    somme = {
        "frappes": {c: sum(b["frappes"][c] for b in blocs) for c in ("n_a", "n_b", "tp")},
        "matrice_coups": {a: dict(sorted(ligne.items())) for a, ligne in sorted(matrice.items())},
    }
    for cle in ("in_out", "gagnant_point", "resultat_service"):
        somme[cle] = _exactitude(
            sum(b[cle]["n_ok"] for b in blocs), sum(b[cle]["n"] for b in blocs)
        )
    return somme


def _objectifs(agrege: dict) -> list[dict]:
    """Verdict de chaque objectif du Lot 2 (`ok` None quand il n'y a rien à mesurer)."""
    mesures = {
        "frappes_f1": ("instant de frappe ±2 frames (F1)", agrege["frappes"]["f1"]),
        "kappa_coup": ("type de coup (κ de Cohen)", agrege["kappa_coup"]),
        "in_out": ("rebonds in/out", agrege["in_out"]["valeur"]),
    }
    return [
        {
            "id": cle,
            "libelle": libelle,
            "cible": OBJECTIFS[cle],
            "valeur": valeur,
            "ok": None if valeur is None else valeur >= OBJECTIFS[cle],
        }
        for cle, (libelle, valeur) in mesures.items()
    ]


def _documents(dossier: Path) -> dict | None:
    chemin = dossier / "events.json"
    if not chemin.is_file():
        return None
    try:
        doc = json.loads(chemin.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{chemin} : JSON illisible : {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("events"), list):
        raise ValueError(f"{chemin} : clé « events » (liste) absente")  # noqa: TRY004
    return doc


def comparer(dir_a: str | Path, dir_b: str | Path) -> dict:
    """Accord de deux annotations : un match (deux dossiers portant `events.json`) ou des
    matchs de mêmes noms (sous-dossiers). ValueError si aucun match n'est comparable."""
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    paires: dict[str, tuple[dict, dict]] = {}
    doc_a, doc_b = _documents(dir_a), _documents(dir_b)
    if doc_a is not None and doc_b is not None:
        paires[dir_a.name] = (doc_a, doc_b)
    else:
        for sous in sorted(d for d in dir_a.iterdir() if d.is_dir()):
            if not (dir_b / sous.name).is_dir():
                continue
            a, b = _documents(sous), _documents(dir_b / sous.name)
            if a is not None and b is not None:
                paires[sous.name] = (a, b)
    if not paires:
        raise ValueError(
            f"Aucun events.json commun à {dir_a} et {dir_b} (deux dossiers de match, ou deux "
            "dossiers contenant des matchs de mêmes noms)"
        )
    par_match = {m: _resume(accord_match(a, b)) for m, (a, b) in sorted(paires.items())}
    agrege = _resume(_somme([accord_match(a, b) for a, b in paires.values()]))
    return {
        "n_matchs": len(par_match),
        "par_match": par_match,
        "agrege": agrege,
        "objectifs": _objectifs(agrege),
    }


def _pct(valeur: float | None) -> str:
    return "n/a" if valeur is None else f"{valeur * 100:.1f} %"


def formater_accord(rapport: dict) -> list[str]:
    """Lignes d'affichage du rapport de `dataset accord`."""

    def bloc(titre: str, r: dict) -> list[str]:
        f = r["frappes"]
        kappa = r["kappa_coup"]
        gagnant, service = r["gagnant_point"], r["resultat_service"]
        frappes = (
            f"  frappes ±{TOLERANCE_FRAPPE} frames : {f['tp']} appariées ({f['n_a']} en A, "
            f"{f['n_b']} en B) ; A dans B {_pct(f['pct_a_dans_b'])}, "
            f"B dans A {_pct(f['pct_b_dans_a'])}, F1 {_pct(f['f1'])}"
        )
        return [
            titre,
            frappes,
            f"  type de coup : κ {'n/a' if kappa is None else f'{kappa:.3f}'}",
            f"  rebonds in/out : {_pct(r['in_out']['valeur'])} ({r['in_out']['n']} appariés)",
            f"  gagnant du point : {_pct(gagnant['valeur'])} ({gagnant['n']} points)",
            f"  résultat du service : {_pct(service['valeur'])} ({service['n']} services)",
        ]

    sortie = []
    if rapport["n_matchs"] > 1:
        for nom, r in rapport["par_match"].items():
            sortie += bloc(f"Match {nom}", r)
    sortie += bloc("Agrégé" if rapport["n_matchs"] > 1 else "Accord", rapport["agrege"])
    sortie.append("Objectifs :")
    for o in rapport["objectifs"]:
        etat = "n/a" if o["ok"] is None else "OK" if o["ok"] else "ÉCHEC"
        valeur = "n/a" if o["valeur"] is None else f"{o['valeur']:.3f}"
        sortie.append(f"  [{etat}] {o['libelle']} : {valeur} (cible ≥ {o['cible']})")
    return sortie
