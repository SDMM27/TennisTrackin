"""Étiquettes faibles TenniSet : extraction par règles des `desc` de points, échantillon de `Side`.

Les `desc` sont des phrases anglaises de commentaire (« Serena hits a forehand winner down the
line »). Les règles sont des expressions régulières tolérantes aux fautes de frappe observées
(« forehandhand », « out-side ») ; le nom de la joueuse n'est pas utilisé.
"""

import csv
import json
import random
import re
from pathlib import Path

from tennisvision.config import tenniset_video
from tennisvision.io.video import iter_frames

COLONNES_ETIQUETTES = [
    "video",
    "point",
    "frame_debut",
    "frame_fin",
    "desc",
    "fin",
    "dernier_coup",
    "retour",
    "direction",
    "lob",
    "volley",
    "smash",
    "slice",
]
COLONNES_ECHANTILLON = [
    "video",
    "frame_debut",
    "frame_fin",
    "joueur",
    "side",
    "type",
    "verif_coup",
    "verif_cote_image",
]

_COUP = re.compile(r"(forehand|backhand)")
_RETOUR = re.compile(r"(forehand|backhand)\w*\s+return")
_DOUBLE_FAUTE = re.compile(r"double[\s-]*fault")
_ACE = re.compile(r"\bace\b")
# Service que la relanceuse ne renvoie pas (ace en l'absence de retour, gagnant sinon)
_NON_RETOURNE = re.compile(
    r"unable to return|no answer to it|(?:fails?|struggles?) to (?:put|get|return) it(?: back)?|unable to get it back|faces? difficulty in "
    r"returning|only (?:reaches|manages)|struggles with it\.?$"
)
_GAGNANT = re.compile(r"winner")
_FILET = re.compile(r"\bnets?\b|\bnetted\b|\bnet\b")
_DEHORS = re.compile(
    r"\blong\b|\bwide\b|\bout\b|out-?\s?side|outside|out of|fails? to (?:land|keep)|"
    r"struggles? to keep|misses the (?:court|target)|(?:doesn't|does not) land|beyond|"
    r"into the (?:fence|tarpaulin)"
)
# Approche du filet : ce n'est pas une faute de filet
_APPROCHE_FILET = re.compile(r"\b(?:to|at|approach\w*)\s+(?:the\s+)?net\b")
_CROISE = re.compile(r"cross[\s-]*court")
_LIGNE = re.compile(r"down[\s-]*the[\s-]*line")


def _dernier_segment(desc: str) -> str:
    """Dernier segment de la phrase (après la dernière virgule) : le coup qui finit le point."""
    segments = [s.strip() for s in desc.lower().split(",") if s.strip()]
    return segments[-1] if segments else ""


def analyser_desc(desc: str) -> dict:
    """Étiquettes d'une phrase de point : `fin`, `dernier_coup`, `retour`, `direction` et
    booléens `lob`, `volley`, `smash`, `slice` (sur le dernier segment)."""
    texte = (desc or "").lower()
    dernier = _dernier_segment(texte)
    echange = _RETOUR.search(texte) or "rally" in texte  # le service a été retourné
    if _DOUBLE_FAUTE.search(texte):
        fin = "double_faute"
    elif _ACE.search(texte) or (_NON_RETOURNE.search(texte) and not echange):
        fin = "ace"
    elif _GAGNANT.search(dernier) or "unable to return" in dernier:
        fin = "gagnant"
    elif _FILET.search(_APPROCHE_FILET.sub("", dernier)):
        fin = "faute_filet"
    elif _DEHORS.search(dernier):
        fin = "faute_out"
    else:
        fin = "inconnu"
    coup = _COUP.search(dernier)
    retour = _RETOUR.search(texte)
    direction = "cross-court" if _CROISE.search(dernier) else None
    if _LIGNE.search(dernier):
        direction = "down-the-line"
    return {
        "fin": fin,
        "dernier_coup": coup.group(1) if coup else None,
        "retour": retour.group(1) if retour else None,
        "direction": direction,
        "lob": "lob" in dernier,
        "volley": "volley" in dernier,
        "smash": "smash" in dernier,
        "slice": "slice" in dernier,
    }


def _classes(chemin: Path) -> dict:
    try:
        brut = json.loads(chemin.read_text(encoding="utf-8"))
        classes = brut["classes"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{chemin} : annotation TenniSet illisible ({exc!r})") from exc
    if not isinstance(classes, dict):
        raise ValueError(f"{chemin} : clé « classes » invalide")  # noqa: TRY004
    return classes


def _fichiers(dossier_data: Path) -> list[Path]:
    fichiers = sorted((dossier_data / "annotations").glob("V[0-9]*.json"))
    if not fichiers:
        raise FileNotFoundError(f"Aucune annotation V0xx.json dans {dossier_data / 'annotations'}")
    return fichiers


def etiquettes_faibles(dossier_data: str | Path) -> list[dict]:
    """Une ligne par point annoté de tous les `annotations/V0xx.json`. FileNotFoundError s'il n'y
    a aucune annotation."""
    lignes = []
    for fichier in _fichiers(Path(dossier_data)):
        for point in _classes(fichier).get("Point", []):
            desc = point.get("desc", "")
            lignes.append(
                {
                    "video": fichier.stem,
                    "point": point.get("name", ""),
                    "frame_debut": round(point["start"]),
                    "frame_fin": round(point["end"]),
                    "desc": desc,
                    **analyser_desc(desc),
                }
            )
    return lignes


def couverture(lignes: list[dict]) -> float | None:
    """Part des points dont `fin` est connue (≠ inconnu) ; None sans point."""
    if not lignes:
        return None
    return sum(ligne["fin"] != "inconnu" for ligne in lignes) / len(lignes)


def _ecrire_csv(chemin: Path, colonnes: list[str], lignes: list[dict]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8", newline="") as f:
        ecrivain = csv.DictWriter(f, fieldnames=colonnes, lineterminator="\n")
        ecrivain.writeheader()
        for ligne in lignes:
            ecrivain.writerow(
                {c: "" if ligne.get(c) is None else ligne.get(c, "") for c in colonnes}
            )


def ecrire_etiquettes(chemin: str | Path, lignes: list[dict]) -> None:
    """Écrit le CSV des étiquettes faibles (booléens en 0/1, vide = non déterminé)."""
    _ecrire_csv(
        Path(chemin),
        COLONNES_ETIQUETTES,
        [
            {**ligne, **{c: int(ligne[c]) for c in ("lob", "volley", "smash", "slice")}}
            for ligne in lignes
        ],
    )


def echantillon_cotes(dossier_data: str | Path, n: int = 50, graine: int = 0) -> list[dict]:
    """Échantillon aléatoire (reproductible) de frappes `Hit` de toutes les vidéos, à vérifier à
    la main : `Side` désigne-t-il coup droit/revers ou le côté de l'image ?"""
    frappes = []
    for fichier in _fichiers(Path(dossier_data)):
        for hit in _classes(fichier).get("Hit", []):
            custom = hit.get("custom", {})
            frappes.append(
                {
                    "video": fichier.stem,
                    "frame_debut": round(hit["start"]),
                    "frame_fin": round(hit["end"]),
                    "joueur": custom.get("Player", ""),
                    "side": custom.get("Side", ""),
                    "type": custom.get("Type", ""),
                    "verif_coup": "",
                    "verif_cote_image": "",
                }
            )
    frappes.sort(key=lambda h: (h["video"], h["frame_debut"], h["frame_fin"]))
    choisies = random.Random(graine).sample(frappes, min(n, len(frappes)))
    return sorted(choisies, key=lambda h: (h["video"], h["frame_debut"]))


def ecrire_echantillon(chemin: str | Path, lignes: list[dict]) -> None:
    """Écrit le CSV de l'échantillon (colonnes de vérification vides)."""
    _ecrire_csv(Path(chemin), COLONNES_ECHANTILLON, lignes)


def sauver_images(
    lignes: list[dict], dossier_data: str | Path, dossier_images: str | Path
) -> tuple[int, list[str]]:
    """Enregistre en JPEG la frame médiane de chaque frappe dont la vidéo existe.

    Retourne (nombre d'images, avertissements : vidéos absentes ou frames illisibles).
    """
    import cv2

    dossier_data, dossier_images = Path(dossier_data), Path(dossier_images)
    dossier_images.mkdir(parents=True, exist_ok=True)
    avertissements: list[str] = []
    absentes: set[str] = set()
    ecrites = 0
    for ligne in lignes:
        video = tenniset_video(dossier_data, ligne["video"])
        if not video.is_file():
            if ligne["video"] not in absentes:
                absentes.add(ligne["video"])
                avertissements.append(f"vidéo {ligne['video']} absente, images ignorées")
            continue
        milieu = (ligne["frame_debut"] + ligne["frame_fin"]) // 2
        nom = f"{ligne['video']}_{ligne['frame_debut']}_{ligne['frame_fin']}.jpg"
        try:
            image = next(iter_frames(video, milieu, milieu)).image
        except (StopIteration, ValueError) as exc:
            avertissements.append(f"{ligne['video']} frame {milieu} illisible : {exc}")
            continue
        if cv2.imwrite(str(dossier_images / nom), image):
            ecrites += 1
        else:
            avertissements.append(f"{nom} : écriture impossible")
    return ecrites, avertissements
