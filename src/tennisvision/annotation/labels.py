"""Étiquettes CVAT du jeu amateur : noms, attributs et valeurs (constantes de référence).

`configuration_cvat()` construit la liste à coller dans l'éditeur « Raw » des étiquettes d'un
projet CVAT ; elle est identique au fichier `cvat_labels.json` livré avec le paquet (un test le
vérifie). L'importeur (`tennisvision.annotation.cvat`) s'appuie sur les mêmes constantes.
"""

import json
from importlib import resources

# Noms d'étiquettes
BALLE = "balle"
COURT = "court"
FRAPPE = "frappe"
REBOND = "rebond"
FILET = "filet"
SERVICE = "service"
DEBUT_POINT = "debut_point"
FIN_POINT = "fin_point"
SCORE = "score"

# Attributs
ATTR_VISIBLE = "visible"
ATTR_KP = "kp"
ATTR_JOUEUR = "joueur"
ATTR_COUP = "coup"
ATTR_ISSUE = "issue"
ATTR_IN_OUT = "in_out"
ATTR_NUMERO = "numero"
ATTR_RESULTAT = "resultat"
ATTR_FAUTE = "faute"
ATTR_SERVEUR = "serveur"
ATTR_GAGNANT = "gagnant"
ATTR_RAISON = "raison"
ATTR_JEUX = "jeux"
ATTR_SETS = "sets"

# Valeurs
JOUEURS = ("P1", "P2")
KEYPOINTS = tuple(str(i) for i in range(14))
COUPS = (
    "serve",
    "forehand",
    "backhand",
    "forehand_volley",
    "backhand_volley",
    "smash",
    "uncertain",
)
ISSUES = ("in_play", "winner", "unforced_error", "forced_error", "unknown")
IN_OUT = ("in", "out", "too_close")
NUMEROS_SERVICE = ("1", "2")
RESULTATS_SERVICE = ("in", "fault", "let")
FAUTES = ("net", "long", "wide", "foot", "unknown")

FICHIER_CONFIGURATION = "cvat_labels.json"


def _select(nom: str, valeurs: tuple[str, ...]) -> dict:
    return {
        "name": nom,
        "input_type": "select",
        "mutable": False,
        "values": list(valeurs),
        "default_value": valeurs[0],
    }


def _texte(nom: str) -> dict:
    return {
        "name": nom,
        "input_type": "text",
        "mutable": False,
        "values": [""],
        "default_value": "",
    }


def _etiquette(nom: str, type_: str, couleur: str, attributs: list[dict]) -> dict:
    return {"name": nom, "color": couleur, "type": type_, "attributes": attributs}


def configuration_cvat() -> list[dict]:
    """Liste des étiquettes du projet CVAT (format de l'éditeur « Raw »)."""
    visible = {
        "name": ATTR_VISIBLE,
        "input_type": "checkbox",
        "mutable": True,
        "values": ["true"],
        "default_value": "true",
    }
    return [
        _etiquette(BALLE, "points", "#ffd400", [visible]),
        _etiquette(COURT, "points", "#00a6ff", [_select(ATTR_KP, KEYPOINTS)]),
        _etiquette(
            FRAPPE,
            "tag",
            "#ff5a36",
            [
                _select(ATTR_JOUEUR, JOUEURS),
                _select(ATTR_COUP, COUPS),
                _select(ATTR_ISSUE, ISSUES),
            ],
        ),
        _etiquette(REBOND, "points", "#2ecc71", [_select(ATTR_IN_OUT, IN_OUT)]),
        _etiquette(FILET, "tag", "#9b59b6", []),
        _etiquette(
            SERVICE,
            "tag",
            "#e67e22",
            [
                _select(ATTR_NUMERO, NUMEROS_SERVICE),
                _select(ATTR_RESULTAT, RESULTATS_SERVICE),
                _select(ATTR_FAUTE, FAUTES),
                _select(ATTR_SERVEUR, JOUEURS),
            ],
        ),
        _etiquette(DEBUT_POINT, "tag", "#1abc9c", []),
        _etiquette(
            FIN_POINT,
            "tag",
            "#e74c3c",
            [_select(ATTR_GAGNANT, JOUEURS), _select(ATTR_SERVEUR, JOUEURS), _texte(ATTR_RAISON)],
        ),
        _etiquette(SCORE, "tag", "#34495e", [_texte(ATTR_JEUX), _texte(ATTR_SETS)]),
    ]


def charger_configuration() -> list[dict]:
    """Lit `cvat_labels.json` livré avec le paquet."""
    ressource = resources.files(__package__).joinpath(FICHIER_CONFIGURATION)
    return json.loads(ressource.read_text(encoding="utf-8"))
