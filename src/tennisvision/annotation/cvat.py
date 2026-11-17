"""Lecture d'un export « CVAT for video 1.1 » (XML) : pistes de points et tags.

Aucune dépendance à CVAT : seul le XML exporté est lu (bibliothèque standard).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PointCVAT:
    """Un point d'une piste ou d'une forme : position, clé d'interpolation, attributs."""

    frame: int
    x: float
    y: float
    keyframe: bool
    outside: bool
    occluded: bool = False
    attributs: dict[str, str] = field(default_factory=dict)


@dataclass
class Piste:
    """Une piste (ou une forme isolée) d'étiquette `label`, points triés par frame."""

    identifiant: str
    label: str
    points: list[PointCVAT]

    def attribut(self, nom: str) -> str | None:
        """Première valeur de l'attribut sur la piste (les attributs fixes ne sont écrits qu'une
        fois par CVAT)."""
        for point in self.points:
            if nom in point.attributs:
                return point.attributs[nom]
        return None


@dataclass
class Tag:
    """Une étiquette de frame (sans position)."""

    frame: int
    label: str
    attributs: dict[str, str]


@dataclass
class ExportCVAT:
    """Contenu utile d'un export : taille d'image, pistes de points et tags."""

    taille_image: tuple[int, int] | None
    pistes: list[Piste]
    tags: list[Tag]


def _attributs(element: ET.Element) -> dict[str, str]:
    return {a.get("name", ""): (a.text or "").strip() for a in element.findall("attribute")}


def _entier(valeur: str | None, nom: str, balise: str) -> int:
    try:
        return int(valeur)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"attribut « {nom} » invalide dans <{balise}> : {valeur!r}") from None


def _point(element: ET.Element, frame: int | None = None) -> PointCVAT:
    """Premier point de l'attribut `points="x,y;x,y"` d'un élément <points>."""
    brut = (element.get("points") or "").split(";")[0]
    try:
        x, y = (float(v) for v in brut.split(","))
    except ValueError:
        raise ValueError(f"coordonnées invalides dans <points> : {brut!r}") from None
    return PointCVAT(
        frame=_entier(element.get("frame"), "frame", "points") if frame is None else frame,
        x=x,
        y=y,
        keyframe=element.get("keyframe", "1") == "1",
        outside=element.get("outside", "0") == "1",
        occluded=element.get("occluded", "0") == "1",
        attributs=_attributs(element),
    )


def lire_export(chemin: str | Path) -> ExportCVAT:
    """Lit un export CVAT for video 1.1. ValueError si le fichier est illisible ou d'un autre
    format (CVAT for images, par exemple)."""
    chemin = Path(chemin)
    try:
        racine = ET.parse(chemin).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{chemin.name} : XML illisible : {exc}") from exc
    if racine.tag != "annotations":
        raise ValueError(f"{chemin.name} : export CVAT inattendu (racine <{racine.tag}>)")
    if racine.find("image") is not None and racine.find("track") is None:
        raise ValueError(
            f"{chemin.name} : export au format « CVAT for images » ; "
            "exporter au format « CVAT for video 1.1 »"
        )

    debut = (racine.findtext(".//task/start_frame") or "0").strip()
    filtre = (racine.findtext(".//task/frame_filter") or "").strip()
    if debut not in ("", "0") or filtre:
        raise ValueError(
            f"{chemin.name} : l'export ne couvre pas toute la vidéo (start_frame = {debut}, "
            f"frame_filter = {filtre!r}) ; les numéros de frame ne correspondraient plus à "
            "frames.parquet. Exporter la vidéo entière, sans pas de frames."
        )

    try:
        taille = None
        largeur, hauteur = (
            racine.find(".//original_size/width"),
            racine.find(".//original_size/height"),
        )
        if largeur is not None and hauteur is not None:
            taille = (int(float(largeur.text or "")), int(float(hauteur.text or "")))

        pistes = []
        for element in racine.findall("track"):
            points = [_point(p) for p in element.findall("points")]
            points.sort(key=lambda p: p.frame)
            pistes.append(Piste(element.get("id", "?"), element.get("label", ""), points))
        for i, element in enumerate(racine.findall("points")):  # formes isolées
            pistes.append(Piste(f"forme{i}", element.get("label", ""), [_point(element)]))
        tags = [
            Tag(_entier(e.get("frame"), "frame", "tag"), e.get("label", ""), _attributs(e))
            for e in racine.findall("tag")
        ]
    except ValueError as exc:
        raise ValueError(f"{chemin.name} : {exc}") from exc
    tags.sort(key=lambda t: t.frame)
    return ExportCVAT(taille, pistes, tags)
