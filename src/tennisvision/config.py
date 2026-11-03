"""Constantes et chemins du projet."""

from pathlib import Path

# Classes COCO utilisées
PERSON = 0
SPORTS_BALL = 32

# Racine du dépôt : src/tennisvision/config.py -> remonter de trois niveaux
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
POINTS_FILE = DATA_DIR / "annotations" / "points.txt"

DEFAULT_MODEL = MODELS_DIR / "yolo11n.pt"
DEFAULT_CONF = 0.25
DEFAULT_TRACKER = "bytetrack.yaml"
# Taille d'inférence YOLO. Mesuré sur le point P00000001 de V006 : à 1280, le taux brut de
# « balle » monte à ~70 %, mais ce sont presque tous des faux positifs (roue de la caméra TV) ;
# la vraie balle reste détectée sur < 2 % des frames et le traitement est 2x plus lent.
# On garde donc 640 tant que la balle n'a pas son propre modèle (étape 1 de la ROADMAP).
DEFAULT_IMGSZ = 640


def default_model_path() -> str:
    """Retourne le modèle local s'il existe, sinon le nom pour téléchargement par ultralytics."""
    if DEFAULT_MODEL.exists():
        return str(DEFAULT_MODEL)
    return DEFAULT_MODEL.name
