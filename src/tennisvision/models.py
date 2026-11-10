"""Registre de modèles versionnés : cache utilisateur, téléchargement explicite, somme SHA-256.

Aucun téléchargement implicite : un modèle absent du cache lève `ModeleAbsent`, ce qui garantit
un fonctionnement hors-ligne une fois les poids en place.
"""

import hashlib
import logging
import os
import shutil
import tomllib
import urllib.request
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

from tennisvision.config import DEFAULT_MODEL, models_dir

log = logging.getLogger(__name__)

_TAILLE_BLOC = 1024 * 1024


class ModeleAbsent(FileNotFoundError):
    """Le modèle du registre n'est pas présent dans le cache."""


@dataclass(frozen=True)
class ModeleRef:
    """Entrée du registre : un fichier de poids figé par sa version et sa somme de contrôle."""

    nom: str
    version: str
    fichier: str
    url: str
    sha256: str
    licence: str
    description: str


@cache
def registre() -> dict[str, ModeleRef]:
    """Charge `models.toml` (embarqué dans le paquet)."""
    texte = resources.files("tennisvision").joinpath("models.toml").read_text(encoding="utf-8")
    return {nom: ModeleRef(nom=nom, **champs) for nom, champs in tomllib.loads(texte).items()}


def _ref(nom: str) -> ModeleRef:
    modeles = registre()
    if nom not in modeles:
        raise ValueError(f"modèle inconnu : {nom} (disponibles : {', '.join(sorted(modeles))})")
    return modeles[nom]


def _sha256(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        while bloc := f.read(_TAILLE_BLOC):
            h.update(bloc)
    return h.hexdigest()


def chemin_modele(nom: str, dossier: Path | None = None) -> Path:
    """Emplacement attendu du modèle dans le cache (le fichier peut ne pas exister)."""
    return models_dir(dossier) / _ref(nom).fichier


def resoudre_modele(nom_ou_chemin: str | None, dossier: Path | None = None) -> Path:
    """Résout un modèle : chemin de fichier existant, ou nom du registre présent dans le cache."""
    if nom_ou_chemin is None:
        nom_ou_chemin = DEFAULT_MODEL
    chemin = Path(nom_ou_chemin)
    if chemin.is_file():
        return chemin
    if nom_ou_chemin in registre():
        cible = chemin_modele(nom_ou_chemin, dossier)
        if cible.is_file():
            return cible
        raise ModeleAbsent(
            f"Modèle {nom_ou_chemin} absent du cache ({cible}) : "
            f"lancez `tennisvision models pull {nom_ou_chemin}`"
        )
    raise ValueError(
        f"modèle inconnu : {nom_ou_chemin} (ni fichier existant, ni nom du registre : "
        f"{', '.join(sorted(registre()))})"
    )


def decrire_modele(nom_ou_chemin: str | None) -> dict:
    """Identité d'un modèle pour les sorties : nom, version et somme SHA-256.

    Un nom du registre donne la version et la somme du registre ; un fichier local donne son nom
    de fichier (sans dossier), une version `None` et la somme calculée sur ses octets.
    """
    if nom_ou_chemin is None:
        nom_ou_chemin = DEFAULT_MODEL
    chemin = Path(nom_ou_chemin)
    if chemin.is_file():
        return {"nom": chemin.name, "version": None, "sha256": _sha256(chemin)}
    ref = _ref(nom_ou_chemin)
    return {"nom": ref.nom, "version": ref.version, "sha256": ref.sha256}


def verifier(nom: str, dossier: Path | None = None) -> bool:
    """Vrai si le fichier en cache existe et que sa somme SHA-256 correspond au registre."""
    cible = chemin_modele(nom, dossier)
    return cible.is_file() and _sha256(cible) == _ref(nom).sha256


def telecharger(nom: str, dossier: Path | None = None, force: bool = False) -> Path:
    """Télécharge un modèle dans le cache après vérification de sa somme SHA-256."""
    ref = _ref(nom)
    cible = chemin_modele(nom, dossier)
    if not force and verifier(nom, dossier):
        log.info("Modèle %s déjà présent : %s", nom, cible)
        return cible

    cible.parent.mkdir(parents=True, exist_ok=True)
    partiel = cible.with_name(cible.name + ".part")
    h = hashlib.sha256()
    log.info("Téléchargement de %s depuis %s", nom, ref.url)
    try:
        with urllib.request.urlopen(ref.url) as reponse, open(partiel, "wb") as f:
            total = int(reponse.headers.get("Content-Length") or 0)
            lu = 0
            while bloc := reponse.read(_TAILLE_BLOC):
                f.write(bloc)
                h.update(bloc)
                lu += len(bloc)
                if total:
                    log.info("  %d %% (%d / %d octets)", 100 * lu // total, lu, total)
        if h.hexdigest() != ref.sha256:
            raise ValueError(
                f"somme SHA-256 incorrecte pour {nom} : attendue {ref.sha256}, "
                f"obtenue {h.hexdigest()}"
            )
        os.replace(partiel, cible)
    except BaseException:
        partiel.unlink(missing_ok=True)
        raise
    log.info("Modèle %s enregistré : %s", nom, cible)
    return cible


def importer(nom: str, source: Path, dossier: Path | None = None) -> Path:
    """Copie un fichier local dans le cache après vérification de sa somme SHA-256."""
    ref = _ref(nom)
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"fichier introuvable : {source}")
    obtenue = _sha256(source)
    if obtenue != ref.sha256:
        raise ValueError(
            f"somme SHA-256 incorrecte pour {source} : attendue {ref.sha256}, obtenue {obtenue}"
        )
    cible = chemin_modele(nom, dossier)
    cible.parent.mkdir(parents=True, exist_ok=True)
    partiel = cible.with_name(cible.name + ".part")
    try:
        shutil.copyfile(source, partiel)
        os.replace(partiel, cible)
    except BaseException:
        partiel.unlink(missing_ok=True)
        raise
    return cible
