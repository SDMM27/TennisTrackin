"""Découpage train / val / test par match, avec jeu de test gelé (ADR-003)."""

import json
import random
from pathlib import Path

from tennisvision.jeu_amateur.controle import dossiers_matchs
from tennisvision.jeu_amateur.registre import lire_registre

NOM_SPLITS = "splits.json"


def _surfaces(gt_dir: Path) -> dict[str, str]:
    lignes = lire_registre(gt_dir) or []
    return {ligne["match"]: ligne["surface"] for ligne in lignes}


def _choisir_test(
    matchs: list[str], n_test: int, surfaces: dict[str, str], rng: random.Random
) -> list[str]:
    """Tirage glouton : un match par surface (surfaces les plus fréquentes d'abord), puis tour
    suivant jusqu'à n_test."""
    groupes: dict[str, list[str]] = {}
    for m in sorted(matchs):
        groupes.setdefault(surfaces.get(m, "inconnue"), []).append(m)
    for liste in groupes.values():
        rng.shuffle(liste)
    ordre = sorted(groupes, key=lambda s: (-len(groupes[s]), s))
    test: list[str] = []
    while len(test) < n_test:
        for surface in ordre:
            if groupes[surface] and len(test) < n_test:
                test.append(groupes[surface].pop())
    return sorted(test)


def _lire_splits(fichier: Path) -> dict:
    """`splits.json` existant ({} s'il n'existe pas) ; ValueError s'il est mal formé."""
    if not fichier.is_file():
        return {}
    try:
        doc = json.loads(fichier.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{fichier} : JSON illisible : {exc}") from exc
    if not isinstance(doc, dict) or not all(
        isinstance(v, list) and all(isinstance(m, str) for m in v) for v in doc.values()
    ):
        raise ValueError(
            f"{fichier} : objet {{train, val, test}} de listes de noms de matchs attendu"
        )
    return doc


def decouper(
    gt_dir: str | Path,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 0,
    refaire_test: bool = False,
    confirme: bool = False,
) -> dict[str, list[str]]:
    """Calcule et écrit `splits.json` ({"train", "val", "test"}, noms de matchs).

    Le test est gelé : une fois non vide, il ne change plus ; les nouveaux matchs vont en
    train ou val. ValueError si un match de test a disparu du disque, si les ratios sont
    invalides ou si `refaire_test` est demandé sans `confirme`.
    """
    gt_dir = Path(gt_dir)
    if not (0 <= test_ratio < 1 and 0 <= val_ratio < 1 and test_ratio + val_ratio < 1):
        raise ValueError("--test-ratio et --val-ratio doivent être dans [0, 1[ et de somme < 1")
    if refaire_test and not confirme:
        raise ValueError(
            "refaire le jeu de test brise le gel (ADR-003) : ajouter --je-sais-ce-que-je-fais"
        )
    matchs = dossiers_matchs(gt_dir)
    if not matchs:
        raise ValueError(f"Aucun dossier de match dans {gt_dir}")

    fichier = gt_dir / NOM_SPLITS
    existant = _lire_splits(fichier)
    test_gele = [] if refaire_test else list(existant.get("test", []))
    rng = random.Random(seed)
    n = len(matchs)

    if test_gele:
        absents = sorted(set(test_gele) - set(matchs))
        if absents:
            raise ValueError(
                f"match(s) de test absent(s) du disque : {', '.join(absents)} ; le jeu de test "
                "est gelé (refaire avec --refaire-test --je-sais-ce-que-je-fais)"
            )
        test = sorted(test_gele)
        connus = {m: s for s in ("train", "val") for m in existant.get(s, []) if m in matchs}
        n_val_cible = round(n * val_ratio)
        val = sorted(m for m, s in connus.items() if s == "val")
        train = sorted(m for m, s in connus.items() if s == "train")
        nouveaux = sorted(set(matchs) - set(test) - set(connus))
        rng.shuffle(nouveaux)
        for m in nouveaux:
            (val if len(val) < n_val_cible else train).append(m)
    else:
        n_test = max(1, round(n * test_ratio)) if n >= 2 and test_ratio > 0 else 0
        n_val = round(n * val_ratio) if n - n_test > 1 else 0
        test = _choisir_test(matchs, n_test, _surfaces(gt_dir), rng)
        reste = [m for m in matchs if m not in test]
        rng.shuffle(reste)
        val, train = reste[:n_val], reste[n_val:]

    resultat = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    fichier.write_text(
        json.dumps(resultat, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return resultat
