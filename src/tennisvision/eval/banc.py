"""Banc d'évaluation : charge vérités et prédictions, calcule les métriques, écrit le rapport.

**Dossier de vérité terrain** (jeu amateur, Lot 2) : `<gt_dir>/<match>/` contient tout ou partie
de `events.json`, `ball.csv`, `court.json`, `players.csv` (contrats de `docs/contracts.md`), plus
un `match.json` facultatif ; `<gt_dir>/splits.json` donne le découpage par match,
`{"train": [...], "val": [...], "test": [...]}`. La hauteur d'image (échelle des seuils de la
balle) vient de la clé `image_size` de `match.json` si elle existe, sinon de
`court.json.image_size`, sinon 1080.

**Dossier de prédictions** : `<pred_dir>/<match>/` avec les mêmes noms de fichiers (sorties du
pipeline). Un fichier absent alors que la vérité terrain l'a vaut prédiction vide, et le match est
listé dans `avertissements` du rapport.
"""

import html
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

import tennisvision
from tennisvision.datasets.tenniset import filtrer_split
from tennisvision.eval import balle, evenements, joueurs
from tennisvision.eval.appariement import scores_detection
from tennisvision.eval.balle import evaluer_balle, lire_ball_csv
from tennisvision.eval.court import evaluer_court
from tennisvision.eval.evenements import evaluer_evenements, f1_par_classe
from tennisvision.eval.indicateurs import agreger_indicateurs, comparer_indicateurs
from tennisvision.eval.joueurs import evaluer_joueurs, lire_players_csv

HAUTEUR_DEFAUT = 1080
NOTE_TENNISET = (
    "Les scores TenniSet sont optimistes : les découpages Train/Val/Test sont des intervalles "
    "d'une même vidéo (mêmes joueuses, même court, même caméra), cf. ADR-003. Seul le jeu "
    "amateur, découpé par match, fait foi."
)


def plis_leave_one_out(matchs: list[str]) -> list[tuple[list[str], str]]:
    """Un pli par match : (matchs d'entraînement, match de test), dans l'ordre trié."""
    tries = sorted(matchs)
    return [([m for m in tries if m != test], test) for test in tries]


# --- Vérités terrain amateur ---------------------------------------------------------------


def lire_splits(gt_dir: Path) -> dict[str, list[str]]:
    """Lit `splits.json` ; FileNotFoundError s'il manque."""
    chemin = gt_dir / "splits.json"
    if not chemin.is_file():
        raise FileNotFoundError(f"splits.json introuvable dans {gt_dir}")
    return json.loads(chemin.read_text(encoding="utf-8"))


def charger_verites_amateur(gt_dir: str | Path, split: str) -> dict[str, dict]:
    """Charge les vérités terrain des matchs d'un split (`all` = tous les matchs du fichier).

    ValueError si le split est absent de `splits.json` ou ne contient aucun match,
    FileNotFoundError si le dossier d'un match manque.
    """
    gt_dir = Path(gt_dir)
    splits = lire_splits(gt_dir)
    if split == "all":
        matchs = sorted({m for liste in splits.values() for m in liste})
    elif split in splits:
        matchs = sorted(splits[split])
    else:
        raise ValueError(f"Split « {split} » absent de {gt_dir / 'splits.json'}")
    if not matchs:
        raise ValueError(f"Aucun match dans le split « {split} » de {gt_dir / 'splits.json'}")

    verites = {}
    for match in matchs:
        dossier = gt_dir / match
        if not dossier.is_dir():
            raise FileNotFoundError(f"Dossier de vérité terrain introuvable : {dossier}")
        verites[match] = {
            "events": _lire_json(dossier / "events.json"),
            "ball": lire_ball_csv(dossier / "ball.csv")
            if (dossier / "ball.csv").is_file()
            else None,
            "court": _lire_json(dossier / "court.json"),
            "players": lire_players_csv(dossier / "players.csv")
            if (dossier / "players.csv").is_file()
            else None,
            "hauteur_image": _hauteur_image(dossier),
        }
    return verites


def _lire_json(chemin: Path) -> dict | None:
    if not chemin.is_file():
        return None
    return json.loads(chemin.read_text(encoding="utf-8"))


def _hauteur_image(dossier: Path) -> int:
    for nom in ("match.json", "court.json"):
        doc = _lire_json(dossier / nom)
        taille = doc.get("image_size") if doc else None
        if taille:
            return int(taille[1])
    return HAUTEUR_DEFAUT


# --- Prédictions ---------------------------------------------------------------------------


def _charger_predictions(dossier: Path, gt: dict, a_signaler: list[str]) -> dict:
    """Prédictions d'un match ; fichier absent => prédiction vide (nom signalé dans a_signaler)."""

    def charger(cle: str, fichier: str, lecteur, vide):
        chemin = dossier / fichier
        if chemin.is_file():
            return lecteur(chemin)
        if gt.get(cle) is not None:
            a_signaler.append(fichier)
        return vide

    return {
        "events": charger("events", "events.json", _lire_json, {"events": []}),
        "ball": charger("ball", "ball.csv", lire_ball_csv, pd.DataFrame()),
        "court": charger("court", "court.json", _lire_json, {"segments": []}),
        "players": charger("players", "players.csv", lire_players_csv, pd.DataFrame()),
    }


def _evaluer_match(gt: dict, pred: dict) -> dict:
    resultat = {}
    if gt.get("events") is not None:
        doc = pred["events"]
        if gt.get("intervalles") is not None:  # split interne à la vidéo (TenniSet)
            doc = filtrer_split(doc, gt["intervalles"])
        resultat["evenements"] = evaluer_evenements(doc, gt["events"])
        resultat["indicateurs"] = comparer_indicateurs(doc, gt["events"])
    if gt.get("ball") is not None:
        resultat["balle"] = evaluer_balle(pred["ball"], gt["ball"], gt["hauteur_image"])
    if gt.get("court") is not None:
        resultat["court"] = evaluer_court(pred["court"], gt["court"])
    if gt.get("players") is not None:
        resultat["joueurs"] = evaluer_joueurs(pred["players"], gt["players"])
    return resultat


# --- Agrégation ----------------------------------------------------------------------------

# Valeurs recalculées après la somme des effectifs
_DERIVEES = {
    "precision",
    "rappel",
    "f1",
    "valeur",
    "idf1",
    "taux_trous_longs",
    "f1_par_classe",
    "ecart_moyen_frames",
}


def _est_entier(valeur) -> bool:
    return isinstance(valeur, int) and not isinstance(valeur, bool)


def _fusion_moyenne(blocs: list[dict]) -> dict:
    """Fusionne des blocs `{moyenne, max, n}` : moyenne pondérée par n, maximum des maxima."""
    avec = [b for b in blocs if b["n"] and b["moyenne"] is not None]
    n = sum(b["n"] for b in avec)
    return {
        "moyenne": sum(b["moyenne"] * b["n"] for b in avec) / n if n else None,
        "max": max(b["max"] for b in avec) if avec else None,
        "n": n,
    }


def _somme_matrices(matrices: list[dict]) -> dict:
    somme: dict[str, dict[str, int]] = {}
    for matrice in matrices:
        for gt_classe, ligne in matrice.items():
            cible = somme.setdefault(gt_classe, {})
            for pred_classe, n in ligne.items():
                cible[pred_classe] = cible.get(pred_classe, 0) + n
    return {c: dict(sorted(ligne.items())) for c, ligne in sorted(somme.items())}


def _agreger(blocs: list):
    """Agrège des blocs de métriques de même forme (un par match), sans perdre les effectifs.

    Les effectifs entiers sont sommés puis les taux sont recalculés (micro-moyenne) ; les blocs
    `{moyenne, max, n}` sont fusionnés (moyenne pondérée). Les valeurs None sont ignorées.
    """
    blocs = [b for b in blocs if b is not None]
    if not blocs:
        return None
    if "moyenne" in blocs[0]:
        return _fusion_moyenne(blocs)
    sortie: dict = {}
    for cle in dict.fromkeys(k for b in blocs for k in b):
        valeurs = [b[cle] for b in blocs if cle in b]
        if cle in _DERIVEES:
            continue
        if cle == "matrice_confusion":
            sortie[cle] = _somme_matrices(valeurs)
        elif all(v is None or isinstance(v, dict) for v in valeurs):
            sortie[cle] = _agreger(valeurs)
        elif all(_est_entier(v) for v in valeurs):
            sortie[cle] = sum(valeurs)
        elif all(isinstance(v, float) for v in valeurs) and len(set(valeurs)) == 1:
            sortie[cle] = valeurs[0]
    if {"tp", "n_pred", "n_gt"} <= sortie.keys():
        sortie.update(scores_detection(sortie["tp"], sortie["n_pred"], sortie["n_gt"]))
        pondere = [
            (b["ecart_moyen_frames"], b["tp"])
            for b in blocs
            if b.get("ecart_moyen_frames") is not None
        ]
        total = sum(poids for _, poids in pondere)
        if "ecart_moyen_frames" in blocs[0]:
            sortie["ecart_moyen_frames"] = (
                sum(e * poids for e, poids in pondere) / total if total else None
            )
    if {"n_tp", "n_fp", "n_fn"} <= sortie.keys():
        tp, fp, fn = sortie["n_tp"], sortie["n_fp"], sortie["n_fn"]
        sortie["precision"] = tp / (tp + fp) if tp + fp else None
        sortie["rappel"] = tp / (tp + fn) if tp + fn else None
        sortie["f1"] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    if {"n_ok", "n"} <= sortie.keys():
        sortie["valeur"] = sortie["n_ok"] / sortie["n"] if sortie["n"] else None
    if sortie.get("identite"):
        sortie["idf1"] = sortie["identite"]["f1"]
    if "n_frames_trous_longs" in sortie:
        n = sortie["n_frames_visibles"]
        sortie["taux_trous_longs"] = sortie["n_frames_trous_longs"] / n if n else None
    if "matrice_confusion" in sortie:
        sortie["f1_par_classe"] = f1_par_classe(sortie["matrice_confusion"])
    return sortie


def _chemin(doc, *cles):
    """doc[c1][c2]... ou None si un maillon manque."""
    for cle in cles:
        if not isinstance(doc, dict) or cle not in doc:
            return None
        doc = doc[cle]
    return doc


def f1_principaux(agrege: dict) -> dict[str, float | None]:
    """F1 agrégés mis en avant (résumé console et historique)."""
    return {
        "frappes_pm5": _chemin(agrege, "evenements", "frappes", "tolerances", "5", "f1"),
        "services": _chemin(agrege, "evenements", "services", "f1"),
        "rebonds_pm5": _chemin(agrege, "evenements", "rebonds", "tolerances", "5", "f1"),
        "points": _chemin(agrege, "evenements", "points", "f1"),
        "balle_10px": _chemin(agrege, "balle", "seuils", "10", "f1"),
    }


# --- Évaluation ----------------------------------------------------------------------------


def _commit() -> str | None:
    """Hash court du commit courant (cwd), None si indisponible."""
    try:
        sortie = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return sortie.stdout.strip() or None


def _modeles(dossiers: list[Path]) -> dict[str, str | None]:
    """Modèles et version du paquet lus dans les `summary.json` / `stats.json` des prédictions."""
    versions: dict[str, set] = {}
    for dossier in dossiers:
        summary = _lire_json_tolerant(dossier / "summary.json")
        if isinstance(summary, dict):
            modele = summary.get("modele")
            if isinstance(modele, dict) and modele.get("nom"):
                versions.setdefault(modele["nom"], set()).add(modele.get("version"))
            if summary.get("version_tennisvision"):
                versions.setdefault("tennisvision", set()).add(summary["version_tennisvision"])
        stats = _lire_json_tolerant(dossier / "stats.json")
        generated = stats.get("generated_with") if isinstance(stats, dict) else None
        if isinstance(generated, dict):
            for nom, version in (generated.get("models") or {}).items():
                versions.setdefault(nom, set()).add(version)
    resultat = {}
    for nom in sorted(versions):
        connues = sorted(str(v) for v in versions[nom] if v is not None)
        resultat[nom] = ", ".join(connues) if connues else None
    return resultat


def _lire_json_tolerant(chemin: Path):
    try:
        return json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _tolerances() -> dict:
    return {
        "frappe_frames": list(evenements.TOLERANCES_FRAPPE),
        "service_frames": evenements.TOLERANCE_SERVICE,
        "rebond_frames": list(evenements.TOLERANCES_REBOND),
        "iou_point": evenements.IOU_POINT,
        "marge_ligne_m": evenements.MARGE_LIGNE_M,
        "seuils_balle_px_1080p": list(balle.SEUILS_BALLE_PX),
        "trou_long_frames": balle.TROU_LONG_FRAMES,
        "iou_joueur": joueurs.IOU_JOUEUR,
    }


def evaluer(
    verites: dict[str, dict],
    pred_dir: Path,
    dataset: str | None = None,
    split: str | None = None,
) -> dict:
    """Évalue les prédictions de `pred_dir` contre `verites` et retourne le rapport.

    `verites[match]` porte les objets déjà chargés : `events`, `ball`, `court`, `players`
    (None si la vérité terrain n'a pas ce fichier) et `hauteur_image` ; `intervalles`
    (facultatif) limite aussi les événements prédits à ces intervalles de frames.
    """
    pred_dir = Path(pred_dir)
    avertissements: list[str] = []
    par_match = {}
    for match in sorted(verites):
        manquants: list[str] = []
        pred = _charger_predictions(pred_dir / match, verites[match], manquants)
        avertissements += [f"{match} : {f} absent, prédiction vide" for f in manquants]
        par_match[match] = _evaluer_match(verites[match], pred)

    agrege: dict = {"n_matchs": len(par_match)}
    for brique in ("evenements", "balle", "court", "joueurs"):
        agrege[brique] = _agreger([r.get(brique) for r in par_match.values()])
    agrege["indicateurs"] = agreger_indicateurs(
        {m: r["indicateurs"] for m, r in par_match.items() if "indicateurs" in r}
    )
    return {
        "meta": {
            "tennisvision_version": tennisvision.__version__,
            "commit": _commit(),
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "dataset": dataset,
            "split": split,
            "tolerances": _tolerances(),
            "modeles": _modeles([pred_dir / m for m in sorted(verites)]),
        },
        "avertissements": avertissements,
        "par_match": par_match,
        "agrege": agrege,
    }


# --- Rapport -------------------------------------------------------------------------------


def _aplatir(valeur, prefixe: str = "") -> dict[str, object]:
    """Aplatit les dicts imbriqués en {chemin.pointé: valeur} (les listes sont ignorées)."""
    if not isinstance(valeur, dict):
        return {prefixe: valeur}
    plat = {}
    for cle, sous in valeur.items():
        plat.update(_aplatir(sous, f"{prefixe}.{cle}" if prefixe else str(cle)))
    return plat


def _texte(valeur) -> str:
    if valeur is None:
        return "—"
    if isinstance(valeur, float):
        return f"{valeur:.4f}"
    if isinstance(valeur, list):
        return ", ".join(_texte(v) for v in valeur)
    return str(valeur)


def _cellule(valeur, balise: str = "td") -> str:
    return f"<{balise}>{html.escape(_texte(valeur))}</{balise}>"


def _tableau(entetes: list[str], lignes: list[list]) -> str:
    tete = "".join(_cellule(e, "th") for e in entetes)
    corps = "".join("<tr>" + "".join(_cellule(c) for c in ligne) + "</tr>" for ligne in lignes)
    return f"<table><thead><tr>{tete}</tr></thead><tbody>{corps}</tbody></table>"


def _html(rapport: dict) -> str:
    meta, agrege = rapport["meta"], rapport["agrege"]
    sections = [
        "<h2>Métadonnées</h2>",
        _tableau(["clé", "valeur"], [[k, v] for k, v in _aplatir(meta).items()]),
    ]
    if meta.get("dataset") == "tenniset":
        sections.insert(0, f'<p class="note">{html.escape(NOTE_TENNISET)}</p>')
    if rapport["avertissements"]:
        items = "".join(f"<li>{html.escape(a)}</li>" for a in rapport["avertissements"])
        sections.append(f"<h2>Avertissements</h2><ul>{items}</ul>")

    sections.append("<h2>Résumé agrégé</h2>")
    for brique in ("evenements", "balle", "court", "joueurs"):
        if agrege.get(brique):
            lignes = [[k, v] for k, v in _aplatir(agrege[brique]).items()]
            sections += [f"<h3>{brique}</h3>", _tableau(["métrique", "valeur"], lignes)]

    if agrege.get("indicateurs"):
        cles = list(agrege["indicateurs"][0])
        sections += [
            "<h2>Indicateurs agrégés</h2>",
            _tableau(cles, [[ligne[c] for c in cles] for ligne in agrege["indicateurs"]]),
        ]

    sections.append("<h2>Détail par match</h2>")
    matchs = list(rapport["par_match"])
    for brique in ("evenements", "balle", "court", "joueurs"):
        plats = {m: _aplatir(r[brique]) for m, r in rapport["par_match"].items() if r.get(brique)}
        if not plats:
            continue
        chemins = list(dict.fromkeys(c for plat in plats.values() for c in plat))
        lignes = [[c, *[plats[m].get(c) if m in plats else None for m in matchs]] for c in chemins]
        sections += [f"<h3>{brique}</h3>", _tableau(["métrique", *matchs], lignes)]
    for match, resultat in rapport["par_match"].items():
        if resultat.get("indicateurs"):
            cles = list(resultat["indicateurs"][0])
            sections += [
                f"<h3>indicateurs — {html.escape(match)}</h3>",
                _tableau(cles, [[ligne[c] for c in cles] for ligne in resultat["indicateurs"]]),
            ]

    titre = html.escape(f"Rapport d'évaluation tennisvision — {meta.get('dataset')}")
    style = (
        "body{font-family:sans-serif;margin:2em;max-width:1200px}"
        "table{border-collapse:collapse;margin:.5em 0}"
        "th,td{border:1px solid #bbb;padding:2px 8px;text-align:left;font-size:.9em}"
        "th{background:#eee}.note{background:#fff4d6;padding:.6em;border:1px solid #e0c060}"
    )
    return (
        '<!DOCTYPE html>\n<html lang="fr"><head><meta charset="utf-8">'
        f"<title>{titre}</title><style>{style}</style></head><body>"
        f"<h1>{titre}</h1>{''.join(sections)}</body></html>\n"
    )


def ecrire_rapport(rapport: dict, dossier_sortie: str | Path) -> Path:
    """Écrit `rapport.json` et `rapport.html` dans `<sortie>/<date>_<commit>/` et ajoute une ligne
    à `<sortie>/historique.jsonl`. Retourne le chemin de `rapport.json`."""
    dossier_sortie = Path(dossier_sortie)
    meta = rapport["meta"]
    horodatage = datetime.fromisoformat(meta["date"]).strftime("%Y%m%d-%H%M%S")
    dossier = dossier_sortie / f"{horodatage}_{meta['commit'] or 'sans-commit'}"
    dossier.mkdir(parents=True, exist_ok=True)

    chemin = dossier / "rapport.json"
    chemin.write_text(
        json.dumps(rapport, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (dossier / "rapport.html").write_text(_html(rapport), encoding="utf-8", newline="\n")

    ligne = {
        "date": meta["date"],
        "commit": meta["commit"],
        "dataset": meta["dataset"],
        "split": meta["split"],
        "f1": f1_principaux(rapport["agrege"]),
    }
    with open(dossier_sortie / "historique.jsonl", "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(ligne, ensure_ascii=False) + "\n")
    return chemin
