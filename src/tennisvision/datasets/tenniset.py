"""Annotations TenniSet : points (TSV), résolution vers les vidéos, conversion en `events.json`."""

import json
from pathlib import Path

import pandas as pd

from tennisvision.config import points_file, tenniset_video
from tennisvision.io.video import probe

COLONNES = ["point_id", "video", "start", "end", "number"]


def load_points(path: str | Path) -> pd.DataFrame:
    """Charge les points annotés : point_id, video, start, end, number."""
    return pd.read_csv(
        path,
        sep="	",
        header=None,
        names=COLONNES,
        dtype={"point_id": str, "video": str, "number": str},
    )


def get_point(point_id: str, path: str | Path) -> tuple[str, int, int]:
    """Retourne (video, frame_debut, frame_fin) d'un point. KeyError si inconnu."""
    points = load_points(path)
    ligne = points[points["point_id"] == point_id]
    if ligne.empty:
        raise KeyError(f"Point inconnu : {point_id}")
    row = ligne.iloc[0]
    return str(row["video"]), int(row["start"]), int(row["end"])


def resoudre_point(point_id: str, dossier_tenniset: str | Path) -> tuple[Path, int, int]:
    """Retourne (chemin_video, frame_debut, frame_fin) d'un point d'un dossier TenniSet.

    KeyError si le point est inconnu, FileNotFoundError si le fichier d'annotations ou la
    vidéo est absent.
    """
    dossier = Path(dossier_tenniset)
    fichier = points_file(dossier)
    if not fichier.is_file():
        raise FileNotFoundError(
            f"Fichier d'annotations introuvable : {fichier} (dossier TenniSet : {dossier})"
        )
    nom, debut, fin = get_point(point_id, fichier)
    video = tenniset_video(dossier, nom)
    if not video.is_file():
        raise FileNotFoundError(f"Vidéo du point {point_id} introuvable : {video}")
    return video, debut, fin


# --- Conversion des annotations complètes (`annotations/V0xx.json`) en `events.json` ---------

FPS_TENNISET = 25.0  # vidéos à cadence fixe
HAUTEUR_DEFAUT = 720
_CLES_SPLIT = {"Train": "train", "Val": "val", "Test": "test"}
_SPINS = {"Topspin": "topspin", "Flat": "flat", "Slice": "slice", "Unsure": "unknown"}


def _trouver_joueurs(classes: dict) -> dict[str, str]:
    """{nom: "P1"/"P2"} ; P1 = joueur du premier service. ValueError sans exactement 2 joueurs."""
    services = sorted(classes.get("Serve", []), key=lambda e: e["start"])
    if not services:
        raise ValueError("Aucun service annoté : impossible de désigner P1")
    premier = services[0]["custom"]["Player"]
    # Seuls Serve et Hit fournissent des noms fiables : Winner / Server contiennent parfois un
    # score ou « Tiebreak » (erreurs d'annotation), qui donnent alors un identifiant null
    noms = {premier}
    for type_ in ("Serve", "Hit"):
        noms.update(e["custom"].get("Player") for e in classes.get(type_, []))
    noms.discard("")
    noms.discard(None)
    if len(noms) != 2:
        raise ValueError(f"Deux joueurs attendus, trouvé {len(noms)} : {sorted(noms)}")
    autre = next(n for n in sorted(noms) if n != premier)
    return {premier: "P1", autre: "P2"}


def _stroke_spin(custom: dict) -> tuple[str, str | None]:
    """Type de coup du contrat et effet, d'après `Type` et `Side` de TenniSet."""
    type_, cote = custom.get("Type"), custom.get("Side")
    if type_ == "Smash":
        return "smash", None
    if type_ == "Volley":
        volees = {"Forehand": "forehand_volley", "Backhand": "backhand_volley"}
        return volees.get(cote, "uncertain"), None
    cotes = {"Forehand": "forehand", "Backhand": "backhand"}
    return cotes.get(cote, "uncertain"), _SPINS.get(type_)


def charger_evenements_tenniset(
    chemin_json: str | Path, fps: float = FPS_TENNISET
) -> tuple[dict, dict[str, str]]:
    """Convertit un fichier `annotations/V0xx.json` en document `events.json` (contrat 1.0).

    Retourne (document, {nom du joueur: "P1"/"P2"}). P1 est le joueur du premier service.
    `frame_idx` = début arrondi de l'élément TenniSet, `end_frame_idx` = fin arrondie, `t_ms` =
    frame_idx * 1000 / fps (vidéos à cadence fixe, 25 fps) ; `source` = « annotation ».

    Conventions : `Match`, `USE` et `SPLITS` sont ignorés. Les `Serve` deviennent des événements
    `serve` (`serve_number` déduit : 1 pour le premier service d'un point, 2 après une faute, un
    let garde le numéro courant ; 1 hors de tout point) mais **aucun `hit` n'est inventé** pour
    eux : TenniSet ne les annote pas comme des frappes. `Game` et `Set` n'ont pas de
    `score_after` : l'ordre des scores TenniSet (« 1-0 ») est ambigu. Un `Winner` vide donne
    `winner_id` null.

    Les joueurs sont les noms des `Serve` et `Hit` ; un `Winner` ou `Server` inconnu (score ou
    « Tiebreak » dans certains fichiers) donne un identifiant null. ValueError s'il n'y a pas
    exactement deux joueurs.
    """
    brut = json.loads(Path(chemin_json).read_text(encoding="utf-8"))
    classes = brut["classes"]
    joueurs = _trouver_joueurs(classes)
    video = Path(str(brut.get("video", "")).replace("\\", "/")).name

    def joueur(nom: str | None) -> str | None:
        return joueurs.get(nom) if nom else None

    points = sorted(classes.get("Point", []), key=lambda e: e["start"])
    services = sorted(classes.get("Serve", []), key=lambda e: e["start"])

    # Serveur de chaque point et numéro de chaque service
    serveur_point: dict[int, str] = {}
    numeros: dict[str, int] = {}
    courant: dict[int, int] = {}
    for sv in services:
        i = next(
            (k for k, pt in enumerate(points) if pt["start"] <= sv["start"] <= pt["end"]), None
        )
        numeros[sv["name"]] = courant.get(i, 1) if i is not None else 1
        if i is not None:
            serveur_point.setdefault(i, joueurs[sv["custom"]["Player"]])
            if sv["custom"]["Result"].lower() == "fault":
                courant[i] = 2

    evenements: list[dict] = []
    ids: set[str] = set()

    def ajouter(type_: str, element: dict, **champs) -> None:
        debut, fin = round(element["start"]), round(element["end"])
        ident = base = f"{type_}_{element['name']}"
        suffixe = 2
        while ident in ids:
            ident = f"{base}_{suffixe}"
            suffixe += 1
        ids.add(ident)
        ev = {
            "id": ident,
            "type": type_,
            "t_ms": debut * 1000 / fps,
            "frame_idx": debut,
            "end_t_ms": fin * 1000 / fps,
            "end_frame_idx": fin,
            **champs,
            "confidence": None,
            "source": "annotation",
        }
        if element.get("desc"):
            ev["desc"] = element["desc"]
        evenements.append(ev)

    for i, pt in enumerate(points):
        champs = {"winner_id": joueur(pt["custom"].get("Winner"))}
        if i in serveur_point:
            champs["server_id"] = serveur_point[i]
        ajouter("point", pt, **champs)
    for jeu in classes.get("Game", []):
        champs = {"winner_id": joueur(jeu["custom"].get("Winner"))}
        if joueur(jeu["custom"].get("Server")):
            champs["server_id"] = joueur(jeu["custom"]["Server"])
        ajouter("game", jeu, **champs)
    for manche in classes.get("Set", []):
        ajouter("set", manche, winner_id=joueur(manche["custom"].get("Winner")))
    for sv in services:
        ajouter(
            "serve",
            sv,
            player_id=joueurs[sv["custom"]["Player"]],
            serve_number=numeros[sv["name"]],
            result=sv["custom"]["Result"].lower(),
        )
    for hit in classes.get("Hit", []):
        stroke, spin = _stroke_spin(hit["custom"])
        ajouter("hit", hit, player_id=joueurs[hit["custom"]["Player"]], stroke=stroke, spin=spin)

    evenements.sort(key=lambda e: e["frame_idx"])
    return {"schema_version": "1.0", "video": video, "events": evenements}, joueurs


def intervalles_split_tenniset(chemin_json: str | Path) -> dict[str, list[tuple[int, int]]]:
    """Intervalles de frames (bornes incluses) des splits « train », « val » et « test »."""
    classes = json.loads(Path(chemin_json).read_text(encoding="utf-8"))["classes"]
    resultat: dict[str, list[tuple[int, int]]] = {"train": [], "val": [], "test": []}
    for element in classes.get("SPLITS", []):
        cle = _CLES_SPLIT.get(element["custom"].get("Type"))
        if cle:
            resultat[cle].append((round(element["start"]), round(element["end"])))
    return resultat


def filtrer_split(doc: dict, intervalles: list[tuple[int, int]]) -> dict:
    """Copie de `doc` limitée aux événements dont `frame_idx` tombe dans un intervalle."""
    gardes = [e for e in doc["events"] if any(a <= e["frame_idx"] <= b for a, b in intervalles)]
    return {**doc, "events": gardes}


def _hauteur_video(dossier: Path, nom: str) -> int:
    video = tenniset_video(dossier, nom)
    if video.is_file():
        return probe(video).height
    return HAUTEUR_DEFAUT


def charger_verites_tenniset(
    dossier_data: str | Path, split: str = "test", fps: float = FPS_TENNISET
) -> dict[str, dict]:
    """Vérités terrain de chaque `annotations/V0*.json` du dossier, au format du banc.

    `split` : « train », « val », « test » ou « all » (aucun filtre). La hauteur d'image est lue
    sur la vidéo `videos/<V0xx>.mp4` si elle existe, sinon 720. Seuls les événements sont
    renseignés (`ball`, `court`, `players` = None) ; `intervalles` porte le split, que le banc
    applique aussi aux prédictions. FileNotFoundError s'il n'y a aucune annotation.
    """
    dossier = Path(dossier_data)
    fichiers = sorted((dossier / "annotations").glob("V[0-9]*.json"))
    if not fichiers:
        raise FileNotFoundError(f"Aucune annotation V0xx.json dans {dossier / 'annotations'}")
    verites = {}
    for fichier in fichiers:
        doc, _ = charger_evenements_tenniset(fichier, fps)
        intervalles = None
        if split != "all":
            intervalles = intervalles_split_tenniset(fichier)[split]
            doc = filtrer_split(doc, intervalles)
        verites[fichier.stem] = {
            "events": doc,
            "ball": None,
            "court": None,
            "players": None,
            "hauteur_image": _hauteur_video(dossier, fichier.stem),
            "intervalles": intervalles,
        }
    return verites
