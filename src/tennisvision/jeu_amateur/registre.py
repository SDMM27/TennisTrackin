"""Registre du jeu amateur : `<gt_dir>/registre.csv` (métadonnées hors contrat, joueurs pseudonymisés)."""

import csv
import re
from pathlib import Path

NOM_REGISTRE = "registre.csv"
COLONNES = [
    "match",
    "date",
    "surface",
    "lieu_type",
    "telephone",
    "fps_nominal",
    "resolution",
    "joueurs",
    "niveau_jeu",
    "niveau_annotation",
    "consentements",
    "consentements_archives",
    "licence",
    "annotateurs",
    "double_annotation",
    "notes",
]
SURFACES = ("terre", "dur", "interieur", "gazon", "autre")
NIVEAUX_JEU = ("debutant", "intermediaire", "confirme", "classe")
NIVEAUX_ANNOTATION = ("1", "2", "3")
OUI_NON = ("oui", "non")


PSEUDONYME = re.compile(r"^J\d+$")
ID_ANNOTATEUR = re.compile(r"^A\d+$")


def lire_registre(gt_dir: str | Path) -> list[dict[str, str]] | None:
    """Lignes du registre (dictionnaires colonne -> texte), ou None si le fichier est absent.

    ValueError si les colonnes ne sont pas exactement celles du modèle, dans l'ordre.
    """
    chemin = Path(gt_dir) / NOM_REGISTRE
    if not chemin.is_file():
        return None
    with chemin.open(encoding="utf-8-sig", newline="") as f:
        lecteur = csv.reader(f)
        entete = next(lecteur, [])
        if entete != COLONNES:
            raise ValueError(
                f"{chemin} : colonnes inattendues. Attendu : {','.join(COLONNES)} ; "
                f"reçu : {','.join(entete)}"
            )
        lignes = []
        for numero, brut in enumerate(lecteur, start=2):
            if not any(c.strip() for c in brut):
                continue
            if len(brut) != len(COLONNES):
                raise ValueError(
                    f"{chemin} : ligne {numero} : {len(brut)} colonnes au lieu de {len(COLONNES)}"
                )
            lignes.append({c: v.strip() for c, v in zip(COLONNES, brut, strict=True)})
    return lignes


def liste(cellule: str) -> list[str]:
    """Éléments d'une cellule multi-valeurs séparés par « ; »."""
    return [e.strip() for e in cellule.split(";") if e.strip()]


def problemes_ligne(ligne: dict[str, str]) -> list[str]:
    """Valeurs hors vocabulaire d'une ligne du registre."""
    problemes = []
    for colonne, permis in (
        ("surface", SURFACES),
        ("niveau_jeu", NIVEAUX_JEU),
        ("niveau_annotation", NIVEAUX_ANNOTATION),
        ("consentements_archives", OUI_NON),
        ("double_annotation", OUI_NON),
    ):
        if ligne[colonne] not in permis:
            problemes.append(f"{colonne} : « {ligne[colonne]} » (attendu : {'|'.join(permis)})")
    try:
        if float(ligne["fps_nominal"]) <= 0:
            raise ValueError
    except ValueError:
        problemes.append(f"fps_nominal : « {ligne['fps_nominal']} » (nombre positif attendu)")
    if not liste(ligne["joueurs"]):
        problemes.append("joueurs : aucun pseudonyme")
    if not all(PSEUDONYME.match(j) for j in liste(ligne["joueurs"])):
        problemes.append("joueurs : pseudonymes J01, J02… attendus, jamais un nom réel")
    if not all(ID_ANNOTATEUR.match(a) for a in liste(ligne["annotateurs"])):
        problemes.append("annotateurs : identifiants A1, A2… attendus, jamais un nom réel")
    return problemes
