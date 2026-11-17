"""Constantes et résolution des chemins (option explicite -> variable d'environnement -> défaut).

Aucun chemin ne dépend de l'emplacement du paquet : le code installé ne connaît pas le dépôt.
"""

import os
from pathlib import Path

from platformdirs import user_cache_dir

# Classe COCO suivie par YOLO (la balle a son propre détecteur : `tennisvision.balle`)
PERSON = 0

ENV_MODELS_DIR = "TENNISVISION_MODELS_DIR"
ENV_OUTPUT_DIR = "TENNISVISION_OUTPUT_DIR"
ENV_TENNISET_DIR = "TENNISVISION_TENNISET_DIR"

DEFAULT_MODEL = "yolo11n"  # nom dans le registre (models.toml)
DEFAULT_CONF = 0.25
DEFAULT_TRACKER = "bytetrack.yaml"
# Taille d'inférence YOLO pour les personnes (la balle est traitée par `tennisvision.balle`).
DEFAULT_IMGSZ = 640


def _resoudre(explicite: str | Path | None, variable: str) -> Path | None:
    """Retourne l'argument explicite, sinon la variable d'environnement, sinon None."""
    if explicite is not None:
        return Path(explicite)
    valeur = os.environ.get(variable)
    return Path(valeur) if valeur else None


def models_dir(explicite: str | Path | None = None) -> Path:
    """Dossier du cache des modèles (`TENNISVISION_MODELS_DIR`, sinon cache utilisateur)."""
    chemin = _resoudre(explicite, ENV_MODELS_DIR)
    if chemin is not None:
        return chemin
    return Path(user_cache_dir("tennisvision", appauthor=False)) / "models"


def output_root(explicite: str | Path | None = None) -> Path:
    """Dossier racine des sorties (`TENNISVISION_OUTPUT_DIR`, sinon `./tennisvision-sorties`)."""
    chemin = _resoudre(explicite, ENV_OUTPUT_DIR)
    if chemin is not None:
        return chemin
    return Path.cwd() / "tennisvision-sorties"


def tenniset_dir(explicite: str | Path | None = None) -> Path:
    """Dossier du jeu TenniSet (`TENNISVISION_TENNISET_DIR`, sinon `./data`)."""
    chemin = _resoudre(explicite, ENV_TENNISET_DIR)
    if chemin is not None:
        return chemin
    return Path.cwd() / "data"


def points_file(dossier_tenniset: Path) -> Path:
    """Fichier d'annotations des points d'un dossier TenniSet."""
    return dossier_tenniset / "annotations" / "points.txt"


def tenniset_video(dossier_tenniset: Path, nom: str) -> Path:
    """Vidéo `<nom>.mp4` d'un dossier TenniSet."""
    return dossier_tenniset / "videos" / f"{nom}.mp4"
