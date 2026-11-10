"""Validation des fichiers de sortie contre les contrats de données (schémas + cohérence)."""

import csv
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

from tennisvision.schemas import CONTRATS, charger_schema

MAX_PROBLEMES = 50


@dataclass(frozen=True)
class Probleme:
    """Un écart au contrat : fichier, emplacement lisible et message."""

    fichier: str
    emplacement: str
    message: str


# --- Utilitaires ---------------------------------------------------------------------------


def _chemin_json(chemin) -> str:
    """Chemin JSON lisible, ex. `events[3].serve_number`."""
    texte = ""
    for element in chemin:
        if isinstance(element, int):
            texte += f"[{element}]"
        else:
            texte += f".{element}" if texte else str(element)
    return texte or "(racine)"


def _limiter(fichier: str, problemes: list[Probleme]) -> list[Probleme]:
    """Tronque à MAX_PROBLEMES et ajoute une ligne finale « … N autres problèmes »."""
    if len(problemes) <= MAX_PROBLEMES:
        return problemes
    reste = len(problemes) - MAX_PROBLEMES
    return [*problemes[:MAX_PROBLEMES], Probleme(fichier, "", f"… {reste} autres problèmes")]


def _est_nombre(valeur) -> bool:
    return isinstance(valeur, int | float) and not isinstance(valeur, bool)


def _est_entier(valeur) -> bool:
    return isinstance(valeur, int) and not isinstance(valeur, bool)


def _erreurs_schema(schema: dict, instance) -> list[tuple[str, str]]:
    """Toutes les erreurs jsonschema (chemin lisible, message), triées par chemin."""
    erreurs = sorted(
        Draft202012Validator(schema).iter_errors(instance),
        key=lambda e: [f"{x:08d}" if isinstance(x, int) else str(x) for x in e.absolute_path],
    )
    return [(_chemin_json(e.absolute_path), e.message) for e in erreurs]


# --- Documents JSON ------------------------------------------------------------------------


def _verifier_match(nom: str, doc: dict) -> list[Probleme]:
    problemes = []
    joueurs = doc.get("players")
    if not isinstance(joueurs, list):
        return problemes
    ids = [j.get("id") for j in joueurs if isinstance(j, dict)]
    if len(ids) == 2 and ids[0] == ids[1]:
        problemes.append(Probleme(nom, "players[1].id", f"identifiant dupliqué : {ids[1]}"))
    premier = doc.get("first_server")
    if isinstance(premier, str) and ids and premier not in ids:
        problemes.append(
            Probleme(nom, "first_server", f"{premier} ne figure pas dans players ({ids})")
        )
    return problemes


def _verifier_court(nom: str, doc: dict) -> list[Probleme]:
    problemes = []
    segments = doc.get("segments")
    if not isinstance(segments, list):
        return problemes
    valides = []
    for i, seg in enumerate(segments):
        if not isinstance(seg, dict) or not _est_entier(seg.get("valid_from_frame")):
            continue
        debut, fin = seg["valid_from_frame"], seg.get("valid_to_frame")
        if _est_entier(fin) and fin < debut:
            problemes.append(
                Probleme(
                    nom,
                    f"segments[{i}].valid_to_frame",
                    f"{fin} < valid_from_frame ({debut})",
                )
            )
        valides.append((debut, i, fin))
    valides.sort(key=lambda s: (s[0], s[1]))
    for (_, i, fin), (debut_suivant, j, _) in itertools.pairwise(valides):
        if fin is None or (_est_entier(fin) and fin >= debut_suivant):
            fin_txt = "la fin de la vidéo" if fin is None else f"la frame {fin}"
            problemes.append(
                Probleme(
                    nom,
                    f"segments[{j}].valid_from_frame",
                    f"chevauche segments[{i}] (qui court jusqu'à {fin_txt})",
                )
            )
    return problemes


def _verifier_events(nom: str, doc: dict) -> list[Probleme]:
    problemes = []
    evenements = doc.get("events")
    if not isinstance(evenements, list):
        return problemes
    vus: dict[str, int] = {}
    ids_hits = {e.get("id") for e in evenements if isinstance(e, dict) and e.get("type") == "hit"}
    for i, ev in enumerate(evenements):
        if not isinstance(ev, dict):
            continue
        ident = ev.get("id")
        if isinstance(ident, str):
            if ident in vus:
                problemes.append(
                    Probleme(
                        nom,
                        f"events[{i}].id",
                        f"id « {ident} » déjà utilisé par events[{vus[ident]}]",
                    )
                )
            else:
                vus[ident] = i
        fin_frame, frame = ev.get("end_frame_idx"), ev.get("frame_idx")
        if _est_entier(fin_frame) and _est_entier(frame) and fin_frame < frame:
            problemes.append(
                Probleme(nom, f"events[{i}].end_frame_idx", f"{fin_frame} < frame_idx ({frame})")
            )
        fin_t, t = ev.get("end_t_ms"), ev.get("t_ms")
        if _est_nombre(fin_t) and _est_nombre(t) and fin_t < t:
            problemes.append(Probleme(nom, f"events[{i}].end_t_ms", f"{fin_t} < t_ms ({t})"))
        hit_id = ev.get("hit_id")
        if (
            isinstance(hit_id, str)
            and ev.get("type") in ("serve", "bounce", "net")
            and hit_id not in ids_hits
        ):
            problemes.append(
                Probleme(
                    nom,
                    f"events[{i}].hit_id",
                    f"« {hit_id} » ne référence aucun événement de type hit",
                )
            )
    return problemes


_VERIFICATEURS_JSON = {
    "match.json": _verifier_match,
    "court.json": _verifier_court,
    "events.json": _verifier_events,
}


def _valider_json(chemin: Path, schema: dict) -> tuple[list[Probleme], object]:
    nom = chemin.name
    try:
        doc = json.loads(chemin.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        return [Probleme(nom, "(fichier)", f"lecture impossible : {exc}")], None
    except json.JSONDecodeError as exc:
        return [Probleme(nom, f"ligne {exc.lineno}", f"JSON illisible : {exc.msg}")], None
    problemes = [Probleme(nom, lieu, msg) for lieu, msg in _erreurs_schema(schema, doc)]
    verificateur = _VERIFICATEURS_JSON.get(nom)
    if verificateur is not None and isinstance(doc, dict):
        problemes += verificateur(nom, doc)
    return problemes, doc


# --- Tables --------------------------------------------------------------------------------


def _types_colonne(schema: dict, colonne: str) -> tuple[set[str], list]:
    """Types déclarés et valeurs d'enum de la colonne (propriétés puis patternProperties)."""
    sous = schema.get("properties", {}).get(colonne)
    if sous is None:
        for motif, candidat in schema.get("patternProperties", {}).items():
            if re.search(motif, colonne):
                sous = candidat
                break
    sous = sous or {}
    types = sous.get("type", [])
    types = {types} if isinstance(types, str) else set(types)
    return types, sous.get("enum", [])


_ENTIER = re.compile(r"[+-]?\d+")


def _convertir(cellule: str, types: set[str], enum: list):
    """Convertit une cellule CSV selon le schéma ; lève ValueError si impossible."""
    if cellule == "":
        return None
    valeurs_enum = [v for v in enum if v is not None]
    if "integer" in types or (valeurs_enum and all(_est_entier(v) for v in valeurs_enum)):
        if not _ENTIER.fullmatch(cellule):
            raise ValueError(f"entier attendu, reçu « {cellule} »")
        return int(cellule)
    if "number" in types:
        try:
            return float(cellule)
        except ValueError:
            raise ValueError(f"nombre attendu, reçu « {cellule} »") from None
    return cellule


def _ecart_colonnes(reelles: list[str], attendues: list[str]) -> str | None:
    """Message décrivant l'écart aux colonnes attendues, ou None si identiques."""
    if list(reelles) == list(attendues):
        return None
    manquantes = [c for c in attendues if c not in reelles]
    en_trop = [c for c in reelles if c not in attendues]
    morceaux = []
    if manquantes:
        morceaux.append("manquantes : " + ", ".join(manquantes))
    if en_trop:
        morceaux.append("en trop : " + ", ".join(en_trop))
    if not morceaux:
        morceaux.append("mêmes colonnes mais ordre différent")
    return "colonnes non conformes (" + " ; ".join(morceaux) + ")"


def _lire_csv(chemin: Path, colonnes: list[str], schema: dict):
    """Lit un CSV : (problèmes de lecture, lignes [(« ligne N », dict)], lecture_ok)."""
    nom = chemin.name
    problemes: list[Probleme] = []
    lignes: list[tuple[str, dict]] = []
    try:
        with chemin.open(newline="", encoding="utf-8") as f:
            lecteur = csv.reader(f)
            entete = next(lecteur, None)
            if entete is None:
                return [Probleme(nom, "ligne 1", "fichier vide (en-tête manquant)")], [], False
            ecart = _ecart_colonnes(entete, colonnes)
            if ecart:
                return [Probleme(nom, "ligne 1", ecart)], [], False
            types = {c: _types_colonne(schema, c) for c in colonnes}
            for cellules in lecteur:
                if not cellules:  # ligne vide
                    continue
                emplacement = f"ligne {lecteur.line_num}"
                if len(cellules) != len(colonnes):
                    problemes.append(
                        Probleme(
                            nom, emplacement, f"{len(cellules)} cellules au lieu de {len(colonnes)}"
                        )
                    )
                    continue
                ligne = {}
                for colonne, cellule in zip(colonnes, cellules, strict=True):
                    try:
                        ligne[colonne] = _convertir(cellule, *types[colonne])
                    except ValueError as exc:
                        lieu = f"{emplacement}, colonne {colonne}"
                        problemes.append(Probleme(nom, lieu, str(exc)))
                        ligne[colonne] = None  # déjà signalé : pas de doublon côté schéma
                lignes.append((emplacement, ligne))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        return [Probleme(nom, "(fichier)", f"lecture impossible : {exc}")], [], False
    return problemes, lignes, True


def _lire_parquet(chemin: Path, colonnes: list[str]):
    """Lit un parquet : (problèmes de lecture, lignes [(« index N », dict)], lecture_ok)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    nom = chemin.name
    try:
        table = pq.read_table(chemin)
    except (pa.ArrowException, OSError) as exc:
        return [Probleme(nom, "(fichier)", f"parquet illisible : {exc}")], [], False
    ecart = _ecart_colonnes(table.column_names, colonnes)
    if ecart:
        return [Probleme(nom, "colonnes", ecart)], [], False
    lignes = [(f"index {i}", ligne) for i, ligne in enumerate(table.to_pylist())]
    return [], lignes, True


def _verifier_frames(nom: str, lignes: list[tuple[str, dict]]) -> list[Probleme]:
    problemes = []
    precedent = None
    for i, (emplacement, ligne) in enumerate(lignes):
        idx, t = ligne.get("frame_idx"), ligne.get("t_ms")
        if _est_entier(idx) and idx != i:
            problemes.append(
                Probleme(nom, f"{emplacement}, colonne frame_idx", f"attendu {i}, reçu {idx}")
            )
        if _est_nombre(t):
            if precedent is not None and t <= precedent:
                problemes.append(
                    Probleme(
                        nom,
                        f"{emplacement}, colonne t_ms",
                        f"t_ms non strictement croissant ({t} après {precedent})",
                    )
                )
            precedent = t
    return problemes


def _verifier_ball(nom: str, lignes: list[tuple[str, dict]]) -> list[Probleme]:
    problemes = []
    precedent = None
    for emplacement, ligne in lignes:
        idx = ligne.get("frame_idx")
        if not _est_entier(idx):
            continue
        if precedent is not None and idx <= precedent:
            problemes.append(
                Probleme(
                    nom,
                    f"{emplacement}, colonne frame_idx",
                    f"frame_idx non strictement croissant ({idx} après {precedent})",
                )
            )
        precedent = idx
    return problemes


def _verifier_players(nom: str, lignes: list[tuple[str, dict]]) -> list[Probleme]:
    problemes = []
    vus: dict[tuple, str] = {}
    for emplacement, ligne in lignes:
        cle = (ligne.get("frame_idx"), ligne.get("player_id"))
        if None not in cle:
            if cle in vus:
                problemes.append(
                    Probleme(
                        nom,
                        f"{emplacement}, colonne player_id",
                        f"doublon ({cle[1]}, frame {cle[0]}) déjà présent à {vus[cle]}",
                    )
                )
            else:
                vus[cle] = emplacement
        for axe in ("x", "y"):
            a, b = ligne.get(f"bbox_{axe}1"), ligne.get(f"bbox_{axe}2")
            if _est_nombre(a) and _est_nombre(b) and b <= a:
                problemes.append(
                    Probleme(
                        nom,
                        f"{emplacement}, colonne bbox_{axe}2",
                        f"bbox_{axe}2 ({b}) doit être > bbox_{axe}1 ({a})",
                    )
                )
    return problemes


_VERIFICATEURS_TABLES = {
    "frames.parquet": _verifier_frames,
    "ball.csv": _verifier_ball,
    "players.csv": _verifier_players,
}


def _valider_table(chemin: Path, schema: dict) -> tuple[list[Probleme], object]:
    nom = chemin.name
    colonnes = schema["x-columns"]
    if schema["x-format"] == "csv":
        problemes, lignes, ok = _lire_csv(chemin, colonnes, schema)
    else:
        problemes, lignes, ok = _lire_parquet(chemin, colonnes)
    if not ok:
        return problemes, None
    for emplacement, ligne in lignes:
        for lieu, message in _erreurs_schema(schema, ligne):
            if lieu == "(racine)":
                problemes.append(Probleme(nom, emplacement, message))
            else:
                problemes.append(Probleme(nom, f"{emplacement}, colonne {lieu}", message))
    problemes += _VERIFICATEURS_TABLES[nom](nom, lignes)
    return problemes, lignes


# --- API publique --------------------------------------------------------------------------


def _valider(chemin: Path) -> tuple[list[Probleme], object]:
    """Valide un fichier : (problèmes non tronqués, données lues ou None)."""
    nom = chemin.name
    if nom not in CONTRATS:
        connus = ", ".join(CONTRATS)
        raise ValueError(f"Fichier inconnu : {nom} (fichiers connus : {connus})")
    schema = charger_schema(CONTRATS[nom])
    if "x-format" in schema:
        return _valider_table(chemin, schema)
    return _valider_json(chemin, schema)


def valider_fichier(chemin: Path) -> list[Probleme]:
    """Valide un fichier d'après son nom ; liste vide si conforme.

    Lève ValueError si le nom ne correspond à aucun contrat.
    """
    chemin = Path(chemin)
    problemes, _ = _valider(chemin)
    return _limiter(chemin.name, problemes)


def _ids_joueurs(match) -> set[str] | None:
    """Identifiants de joueurs déclarés dans match.json (None si inexploitable)."""
    if not isinstance(match, dict) or not isinstance(match.get("players"), list):
        return None
    return {
        j["id"] for j in match["players"] if isinstance(j, dict) and isinstance(j.get("id"), str)
    }


def _croiser(nom: str, donnees, ids: set[str]) -> list[Probleme]:
    """Contrôles croisés : les identifiants de joueurs existent dans match.json."""
    problemes = []

    def verifier(emplacement: str, valeur) -> None:
        if isinstance(valeur, str) and valeur not in ids:
            message = f"joueur {valeur} inconnu de match.json ({sorted(ids)})"
            problemes.append(Probleme(nom, emplacement, message))

    if nom == "events.json" and isinstance(donnees, dict):
        evenements = donnees.get("events")
        for i, ev in enumerate(evenements if isinstance(evenements, list) else []):
            if isinstance(ev, dict):
                for champ in ("player_id", "winner_id", "server_id"):
                    verifier(f"events[{i}].{champ}", ev.get(champ))
    elif nom == "stats.json" and isinstance(donnees, dict):
        stats = donnees.get("stats")
        for i, st in enumerate(stats if isinstance(stats, list) else []):
            if isinstance(st, dict):
                verifier(f"stats[{i}].player_id", st.get("player_id"))
    elif nom == "players.csv" and isinstance(donnees, list):
        for emplacement, ligne in donnees:
            verifier(f"{emplacement}, colonne player_id", ligne.get("player_id"))
    return problemes


def valider_dossier(dossier: Path) -> dict[str, list[Probleme] | None]:
    """Valide chaque fichier de contrat d'un dossier ; None pour un fichier absent.

    Contrôles croisés (si match.json est présent) : les identifiants de joueurs de events.json,
    players.csv et stats.json doivent exister dans match.json.
    """
    dossier = Path(dossier)
    bruts = {}
    for nom in CONTRATS:
        chemin = dossier / nom
        bruts[nom] = _valider(chemin) if chemin.is_file() else None
    ids = _ids_joueurs(bruts["match.json"][1]) if bruts["match.json"] else None
    resultat: dict[str, list[Probleme] | None] = {}
    for nom, brut in bruts.items():
        if brut is None:
            resultat[nom] = None
            continue
        problemes, donnees = brut
        if ids is not None and nom != "match.json":
            problemes = problemes + _croiser(nom, donnees, ids)
        resultat[nom] = _limiter(nom, problemes)
    return resultat
