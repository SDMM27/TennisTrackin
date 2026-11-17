"""Contrôle du dossier de vérité terrain amateur : registre, matchs, critères de réussite du Lot 2."""

import csv
import json
from collections import Counter
from pathlib import Path

from tennisvision.jeu_amateur.registre import NOM_REGISTRE, lire_registre, liste, problemes_ligne
from tennisvision.schemas import CONTRATS
from tennisvision.validate import valider_dossier

FICHIERS_MATCH = ("match.json", "events.json", "ball.csv", "court.json", "players.csv")
SURFACES_CIBLE = ("terre", "dur", "interieur")


def dossiers_matchs(gt_dir: Path) -> list[str]:
    """Sous-dossiers de `gt_dir` qui portent au moins un fichier de contrat."""
    return sorted(
        d.name
        for d in gt_dir.iterdir()
        if d.is_dir() and any((d / f).is_file() for f in CONTRATS if f != "frames.parquet")
    )


def _lire_json(chemin: Path) -> dict:
    try:
        doc = json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _frames_balle(chemin: Path) -> int:
    """Nombre de frames où la balle est visible dans un `ball.csv` (0 si illisible)."""
    try:
        with chemin.open(encoding="utf-8", newline="") as f:
            return sum(1 for ligne in csv.DictReader(f) if ligne.get("visible") == "1")
    except (OSError, csv.Error, UnicodeDecodeError):
        return 0


def stats_match(dossier: Path) -> dict:
    """Effectifs d'un dossier de match : événements par type, frames de balle, niveau constaté."""
    evenements = _lire_json(dossier / "events.json").get("events", [])
    if not isinstance(evenements, list):
        evenements = []
    types = Counter(e.get("type") for e in evenements if isinstance(e, dict))
    balle = _frames_balle(dossier / "ball.csv") if (dossier / "ball.csv").is_file() else 0
    if balle:
        niveau = 3
    elif types["hit"] or types["bounce"]:
        niveau = 2
    elif types["point"] or types["serve"] or types["game"]:
        niveau = 1
    else:
        niveau = 0
    return {
        "fichiers": [f for f in FICHIERS_MATCH if (dossier / f).is_file()],
        "hits": types["hit"],
        "rebonds": types["bounce"],
        "points": types["point"],
        "services": types["serve"],
        "jeux": types["game"],
        "balle_frames": balle,
        "niveau_constate": niveau,
    }


def _controler_match(gt_dir: Path, ligne: dict[str, str]) -> dict:
    nom = ligne["match"]
    dossier = gt_dir / nom
    resultat: dict = {"match": nom, "erreurs": [], "avertissements": []}
    erreurs, avert = resultat["erreurs"], resultat["avertissements"]
    erreurs += [f"registre : {p}" for p in problemes_ligne(ligne)]
    if not dossier.is_dir():
        erreurs.append(f"dossier {nom}/ introuvable")
        return resultat
    for fichier, problemes in valider_dossier(dossier).items():
        for pb in problemes or []:
            lieu = f"{pb.emplacement} : " if pb.emplacement else ""
            erreurs.append(f"{fichier} : {lieu}{pb.message}")
    stats = stats_match(dossier)
    resultat.update({k: v for k, v in stats.items() if k != "fichiers"})
    declare = int(ligne["niveau_annotation"]) if ligne["niveau_annotation"].isdigit() else 0
    if declare >= 1 and "events.json" not in stats["fichiers"]:
        erreurs.append("niveau déclaré ≥ 1 mais events.json absent")
    if declare >= 3:
        for requis in ("ball.csv", "court.json"):
            if requis not in stats["fichiers"]:
                erreurs.append(f"niveau 3 déclaré mais {requis} absent")
    if stats["niveau_constate"] < declare:
        erreurs.append(
            f"niveau déclaré {declare} mais les fichiers n'atteignent que le niveau "
            f"{stats['niveau_constate']}"
        )
    elif stats["niveau_constate"] > declare:
        avert.append(
            f"niveau constaté {stats['niveau_constate']} > niveau déclaré {declare} "
            "(registre à jour ?)"
        )
    return resultat


def _critere(ident: str, libelle: str, observe, ok: bool) -> dict:
    return {"id": ident, "libelle": libelle, "observe": observe, "ok": ok}


def evaluer_criteres(lignes: list[dict[str, str]], matchs: list[dict]) -> list[dict]:
    """Critères de réussite du Lot 2, d'après le registre et les effectifs constatés."""
    n = len(lignes)
    joueurs = {p for ligne in lignes for p in liste(ligne["joueurs"])}
    surfaces = {ligne["surface"] for ligne in lignes} & set(SURFACES_CIBLE)
    fps = set()
    for ligne in lignes:
        try:
            fps.add(round(float(ligne["fps_nominal"])))
        except ValueError:
            pass
    telephones = {ligne["telephone"] for ligne in lignes if ligne["telephone"]}
    niv2 = sum(ligne["niveau_annotation"] in ("2", "3") for ligne in lignes)
    niv3 = [m for m in matchs if m.get("niveau_constate") == 3]
    frames3 = sum(m.get("balle_frames", 0) for m in niv3)
    archives = sum(ligne["consentements_archives"] == "oui" for ligne in lignes)
    doubles = sum(ligne["double_annotation"] == "oui" for ligne in lignes)
    return [
        _critere("matchs", "≥ 10 matchs", n, n >= 10),
        _critere("joueurs", "≥ 6 joueurs distincts", len(joueurs), len(joueurs) >= 6),
        _critere(
            "surfaces",
            "≥ 3 surfaces parmi terre, dur, interieur",
            sorted(surfaces),
            len(surfaces) >= 3,
        ),
        _critere("fps", "30 et 60 fps présents", sorted(fps), {30, 60} <= fps),
        _critere("telephones", "≥ 2 modèles de téléphone", len(telephones), len(telephones) >= 2),
        _critere("niveau2", "niveau 2 sur ≥ 5 matchs", niv2, niv2 >= 5),
        _critere(
            "niveau3",
            "niveau 3 : ≥ 3000 frames de balle sur ≥ 5 matchs",
            {"matchs": len(niv3), "frames": frames3},
            len(niv3) >= 5 and frames3 >= 3000,
        ),
        _critere(
            "consentements",
            "tous les consentements archivés",
            f"{archives}/{n}",
            n > 0 and archives == n,
        ),
        _critere(
            "double_annotation",
            "~10 % des matchs doublement annotés",
            f"{doubles}/{n}",
            n > 0 and doubles / n >= 0.1,
        ),
    ]


def controler(gt_dir: str | Path) -> dict:
    """Rapport de contrôle d'un dossier de vérité terrain amateur.

    ValueError seulement si le registre est illisible (colonnes inattendues).
    """
    gt_dir = Path(gt_dir)
    lignes = lire_registre(gt_dir)
    avertissements: list[str] = []
    if lignes is None:
        avertissements.append(
            f"{NOM_REGISTRE} absent de {gt_dir.name}/ : modèle dans docs/dataset/"
        )
        lignes = []
    matchs = [_controler_match(gt_dir, ligne) for ligne in lignes]
    declares = {ligne["match"] for ligne in lignes}
    for nom in dossiers_matchs(gt_dir):
        if nom not in declares:
            avertissements.append(f"dossier {nom}/ absent du registre")
    doublons = [m for m, c in Counter(ligne["match"] for ligne in lignes).items() if c > 1]
    avertissements += [f"match {m} en double dans le registre" for m in doublons]
    return {
        "gt_dir": gt_dir.name,
        "n_matchs": len(lignes),
        "matchs": matchs,
        "criteres": evaluer_criteres(lignes, matchs),
        "avertissements": avertissements,
    }


def formater_controle(rapport: dict) -> list[str]:
    """Lignes d'affichage du rapport de `dataset check`."""
    sortie = [f"Matchs du registre : {rapport['n_matchs']}"]
    for m in rapport["matchs"]:
        etat = "ERREUR" if m["erreurs"] else "OK"
        detail = ""
        if "niveau_constate" in m:
            detail = (
                f" (niveau {m['niveau_constate']}, {m['hits']} frappes, {m['rebonds']} rebonds, "
                f"{m['points']} points, {m['balle_frames']} frames balle visibles (interpolées comprises))"
            )
        sortie.append(f"  {etat}  {m['match']}{detail}")
        sortie += [f"      - {e}" for e in m["erreurs"]]
        sortie += [f"      ~ {a}" for a in m["avertissements"]]
    sortie += [f"Avertissement : {a}" for a in rapport["avertissements"]]
    sortie.append("Critères de réussite du Lot 2 :")
    for c in rapport["criteres"]:
        sortie.append(f"  [{'OK' if c['ok'] else '--'}] {c['libelle']} : {c['observe']}")
    return sortie
