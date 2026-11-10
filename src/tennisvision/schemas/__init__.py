"""Contrats de données : schémas JSON (draft 2020-12) et constantes de géométrie du court."""

import json
from functools import cache
from importlib import resources

# Nom du fichier de sortie -> nom du schéma qui le décrit
CONTRATS = {
    "match.json": "match.schema.json",
    "court.json": "court.schema.json",
    "events.json": "events.schema.json",
    "stats.json": "stats.schema.json",
    "frames.parquet": "frames.row.schema.json",
    "ball.csv": "ball.row.schema.json",
    "players.csv": "players.row.schema.json",
}

# Court de tennis (ITF), en mètres. Repère : origine au centre du filet, x vers la droite de
# l'image vue de la caméra, y positif du côté caméra (proche).
L = 11.885  # demi-longueur (23,77 m au total)
Wd = 5.485  # demi-largeur du double (10,97 m)
Ws = 4.115  # demi-largeur du simple (8,23 m)
S = 6.40  # distance filet - ligne de service

# 14 points clés du court : (x, y) en mètres, dans l'ordre du contrat
COURT_KEYPOINTS_M = (
    (-Wd, -L),
    (Wd, -L),
    (-Wd, L),
    (Wd, L),
    (-Ws, -L),
    (-Ws, L),
    (Ws, -L),
    (Ws, L),
    (-Ws, -S),
    (Ws, -S),
    (-Ws, S),
    (Ws, S),
    (0.0, -S),
    (0.0, S),
)


@cache
def charger_schema(nom: str) -> dict:
    """Charge un schéma du paquet par son nom (ex. `match.schema.json`). Résultat en cache :
    ne pas le modifier."""
    ressource = resources.files(__package__).joinpath(nom)
    if not ressource.is_file():
        raise ValueError(f"Schéma inconnu : {nom}")
    return json.loads(ressource.read_text(encoding="utf-8"))
