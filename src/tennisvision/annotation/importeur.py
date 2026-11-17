"""Import d'un export CVAT vers les contrats de données (`ball.csv`, `court.json`, `events.json`).

Le temps (`t_ms`) vient de `frames.parquet`, jamais de `frame_idx / fps`. Les incohérences
d'annotation ne font pas échouer l'import : elles sont signalées (avertissements, rapport). Les
fichiers produits sont validés contre les contrats avant toute écriture.
"""

import csv
import itertools
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tennisvision.annotation import labels as L
from tennisvision.annotation.cvat import ExportCVAT, Piste, Tag, lire_export
from tennisvision.schemas import COURT_KEYPOINTS_M, charger_schema
from tennisvision.validate import valider_dossier

MARGE_SERVICE_FRAMES = 2  # écart toléré entre un tag `service` et la frappe correspondante
NOM_RAPPORT = "import_report.json"


@dataclass
class ResultatImport:
    """Bilan d'un import : fichiers écrits, niveau d'annotation, comptes, avertissements."""

    niveau: int
    fichiers: list[str]
    comptes: dict[str, int]
    avertissements: list[str] = field(default_factory=list)
    supprimes: list[str] = field(default_factory=list)


# --- Temps ---------------------------------------------------------------------------------


def lire_temps(chemin: Path) -> list[float]:
    """`t_ms` de chaque frame d'un `frames.parquet` (indexé par frame_idx)."""
    if not chemin.is_file():
        raise ValueError(
            f"{chemin} introuvable : ce fichier donne le temps réel de chaque frame. Le produire "
            "avec `tennisvision analyze <video> --output-dir <dossier du match>` (qui écrit "
            "frames.parquet) ou avec tennisvision.io.write_frames_parquet."
        )
    try:
        table = pq.read_table(chemin, columns=["frame_idx", "t_ms"])
    except (pa.ArrowException, OSError) as exc:
        raise ValueError(
            f"{chemin} : parquet illisible ou colonnes frame_idx/t_ms absentes : {exc}"
        ) from exc
    idx, t_ms = table["frame_idx"].to_pylist(), table["t_ms"].to_pylist()
    if idx != list(range(len(idx))):
        raise ValueError(f"{chemin} : frame_idx doit valoir 0..n-1 sans trou")
    return [float(t) for t in t_ms]


# --- Balle ---------------------------------------------------------------------------------


def _entrees_balle(pistes: list[Piste], avert: list[str]) -> dict[int, tuple]:
    """{frame: (x, y, visible, keyframe)} fusionné sur les pistes `balle`."""
    entrees: dict[int, tuple] = {}
    for piste in pistes:
        visible_courant = True
        for p in piste.points:
            if L.ATTR_VISIBLE in p.attributs:
                visible_courant = p.attributs[L.ATTR_VISIBLE].lower() != "false"
            if p.frame in entrees:
                avert.append(
                    f"balle : frame {p.frame} couverte par deux pistes, la première est gardée"
                )
                continue
            visible = visible_courant and not p.outside and not p.occluded
            entrees[p.frame] = (p.x, p.y, visible, p.keyframe)
    return entrees


def construire_balle(pistes: list[Piste], temps: list[float], avert: list[str]) -> list[dict]:
    """Lignes de `ball.csv` : une par frame, de la première à la dernière de la piste."""
    entrees = _entrees_balle(pistes, avert)
    if not entrees:
        return []
    frames = sorted(entrees)
    lignes = []
    hors = 0
    for f in range(frames[0], frames[-1] + 1):
        if f in entrees:
            x, y, visible, cle = entrees[f]
            source = "annotation" if cle else "interpolated"
        else:  # frame absente de l'export : interpolation linéaire entre les voisines
            avant = max(g for g in frames if g < f)
            apres = min(g for g in frames if g > f)
            (xa, ya, va, _), (xb, yb, vb, _) = entrees[avant], entrees[apres]
            visible, source = va and vb, "interpolated"
            if visible:
                k = (f - avant) / (apres - avant)
                x, y = xa + k * (xb - xa), ya + k * (yb - ya)
        if f >= len(temps):
            hors += 1
            continue
        lignes.append(
            {
                "frame_idx": f,
                "t_ms": temps[f],
                "x": x if visible else None,
                "y": y if visible else None,
                "visible": 1 if visible else 0,
                "score": None,
                "source": source,
            }
        )
    if hors:
        avert.append(f"balle : {hors} frames au-delà de frames.parquet ignorées")
    return lignes


# --- Court ---------------------------------------------------------------------------------


def _projeter(matrice: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    px, py, w = matrice @ np.array([x, y, 1.0])
    if abs(w) < 1e-12:
        return None
    return float(px / w), float(py / w)


def construire_court(
    pistes: list[Piste], taille: tuple[int, int] | None, avert: list[str]
) -> dict | None:
    """Document `court.json` : un segment par frame portant des points `court`."""
    par_frame: dict[int, list[float | None]] = {}
    for piste in pistes:
        cle = piste.attribut(L.ATTR_KP)
        if cle not in L.KEYPOINTS:
            avert.append(f"court : piste {piste.identifiant} sans attribut kp valide ({cle!r})")
            continue
        kp = int(cle)
        for p in piste.points:
            if p.outside or not p.keyframe:
                continue
            points = par_frame.setdefault(p.frame, [None] * 14)
            if points[kp] is not None:
                avert.append(f"court : point {kp} annoté deux fois à la frame {p.frame}")
                continue
            points[kp] = [p.x, p.y]
    if not par_frame:
        return None
    if taille is None:
        raise ValueError("taille de l'image introuvable dans l'export (balise original_size)")

    segments = []
    for frame in sorted(par_frame):
        kps = par_frame[frame]
        presents = [k for k, pt in enumerate(kps) if pt is not None]
        if len(presents) < 4:
            avert.append(f"court : frame {frame} : {len(presents)} points (4 au minimum), ignorée")
            continue
        src = np.array([kps[k] for k in presents], dtype=np.float64)
        dst = np.array([COURT_KEYPOINTS_M[k] for k in presents], dtype=np.float64)
        h, _ = cv2.findHomography(src, dst, 0)
        if h is None or abs(np.linalg.det(h)) < 1e-12:
            avert.append(f"court : frame {frame} : homographie impossible (points dégénérés)")
            continue
        inverse = np.linalg.inv(h)
        ecarts = []
        for k in presents:
            retour = _projeter(inverse, *COURT_KEYPOINTS_M[k])
            if retour is not None:
                ecarts.append(float(np.hypot(retour[0] - kps[k][0], retour[1] - kps[k][1])))
        segments.append(
            {
                "valid_from_frame": frame,
                "valid_to_frame": None,
                "homography": h.tolist(),
                "keypoints_px": kps,
                "reprojection_error_px": float(np.mean(ecarts)) if ecarts else 0.0,
                "confidence": None,
                "source": "annotation",
            }
        )
    if not segments:
        return None
    for courant, suivant in itertools.pairwise(segments):
        courant["valid_to_frame"] = suivant["valid_from_frame"] - 1
    # Caméra supposée fixe jusqu'au premier déplacement : le premier jeu vaut dès la frame 0
    segments[0]["valid_from_frame"] = 0
    return {"schema_version": "1.0", "image_size": list(taille), "segments": segments}


def _segment(court: dict | None, frame: int) -> dict | None:
    for seg in (court or {}).get("segments", []):
        fin = seg["valid_to_frame"]
        if seg["valid_from_frame"] <= frame and (fin is None or frame <= fin):
            return seg
    return None


# --- Événements ----------------------------------------------------------------------------


class _Evenements:
    """Construit `events.json` à partir des tags et des rebonds."""

    def __init__(self, temps: list[float], court: dict | None, avert: list[str]):
        self.temps, self.court, self.avert = temps, court, avert
        self.evenements: list[dict] = []
        self.perdus: list[int] = []  # rebonds sans calibration
        self.compteurs: dict[str, int] = {}

    def _t(self, frame: int, contexte: str) -> float | None:
        if 0 <= frame < len(self.temps):
            return self.temps[frame]
        self.avert.append(f"{contexte} : frame {frame} hors de frames.parquet, ignoré")
        return None

    def _nouveau(self, type_: str, frame: int, contexte: str, **champs) -> dict | None:
        t = self._t(frame, contexte)
        if t is None:
            return None
        self.compteurs[type_] = self.compteurs.get(type_, 0) + 1
        ev = {
            "id": f"{type_}_{self.compteurs[type_]:04d}",
            "type": type_,
            "t_ms": t,
            "frame_idx": frame,
            **champs,
            "confidence": None,
            "source": "annotation",
        }
        self.evenements.append(ev)
        return ev

    def _joueur(self, valeur: str | None, contexte: str) -> str | None:
        if valeur in L.JOUEURS:
            return valeur
        self.avert.append(f"{contexte} : joueur invalide ou absent ({valeur!r})")
        return None

    def frappes(self, tags: list[Tag]) -> list[dict]:
        for tag in tags:
            contexte = f"frappe (frame {tag.frame})"
            joueur = self._joueur(tag.attributs.get(L.ATTR_JOUEUR), contexte)
            if joueur is None:
                continue
            coup = tag.attributs.get(L.ATTR_COUP) or None
            if coup is not None and coup not in L.COUPS:
                self.avert.append(f"{contexte} : coup inconnu {coup!r}, ignoré")
                coup = None
            issue = tag.attributs.get(L.ATTR_ISSUE) or None
            if issue is not None and issue not in L.ISSUES:
                self.avert.append(f"{contexte} : issue inconnue {issue!r}, ignorée")
                issue = None
            self._nouveau("hit", tag.frame, contexte, player_id=joueur, stroke=coup, outcome=issue)
        return [e for e in self.evenements if e["type"] == "hit"]

    def _frappe_precedente(self, frame: int) -> str | None:
        precedentes = [e for e in self.evenements if e["type"] == "hit" and e["frame_idx"] <= frame]
        return max(precedentes, key=lambda e: e["frame_idx"])["id"] if precedentes else None

    def services(self, tags: list[Tag], a_des_frappes: bool) -> None:
        hits = [e for e in self.evenements if e["type"] == "hit"]
        for tag in tags:
            contexte = f"service (frame {tag.frame})"
            a = tag.attributs
            joueur = self._joueur(a.get(L.ATTR_SERVEUR), contexte)
            numero, resultat = a.get(L.ATTR_NUMERO), a.get(L.ATTR_RESULTAT)
            if joueur is None:
                continue
            if numero not in L.NUMEROS_SERVICE or resultat not in L.RESULTATS_SERVICE:
                self.avert.append(
                    f"{contexte} : numéro ({numero!r}) ou résultat ({resultat!r}) invalide"
                )
                continue
            champs = {"player_id": joueur, "serve_number": int(numero), "result": resultat}
            if resultat == "fault" and a.get(L.ATTR_FAUTE) in L.FAUTES:
                champs["fault_type"] = a[L.ATTR_FAUTE]
            proches = [
                e
                for e in hits
                if abs(e["frame_idx"] - tag.frame) <= MARGE_SERVICE_FRAMES
                and e["player_id"] == joueur
            ]
            if proches:
                frappe = min(proches, key=lambda e: abs(e["frame_idx"] - tag.frame))
                champs["hit_id"] = frappe["id"]
            elif a_des_frappes:
                self.avert.append(
                    f"{contexte} : aucune frappe de {joueur} à ±{MARGE_SERVICE_FRAMES} frames"
                )
            self._nouveau("serve", tag.frame, contexte, **champs)

    def rebonds(self, pistes: list[Piste]) -> None:
        for piste in pistes:
            in_out = piste.attribut(L.ATTR_IN_OUT)
            for p in piste.points:
                if p.outside or not p.keyframe:
                    continue
                contexte = f"rebond (frame {p.frame})"
                seg = _segment(self.court, p.frame)
                if seg is None:
                    self.perdus.append(p.frame)
                    self.avert.append(
                        f"{contexte} : aucun segment de court ne couvre cette frame ; le contrat "
                        "exige court_xy, rebond ignoré"
                    )
                    continue
                projete = _projeter(np.array(seg["homography"]), p.x, p.y)
                if projete is None:
                    self.perdus.append(p.frame)
                    self.avert.append(f"{contexte} : projection impossible, rebond ignoré")
                    continue
                valeur = in_out if in_out in L.IN_OUT else "too_close"
                if in_out not in L.IN_OUT:
                    self.avert.append(
                        f"{contexte} : in_out absent ou invalide, « too_close » retenu"
                    )
                champs = {"court_xy": list(projete), "image_xy": [p.x, p.y], "in_out": valeur}
                precedente = self._frappe_precedente(p.frame)
                if precedente:
                    champs["hit_id"] = precedente
                self._nouveau("bounce", p.frame, contexte, **champs)

    def filets(self, tags: list[Tag]) -> None:
        for tag in tags:
            champs = {}
            precedente = self._frappe_precedente(tag.frame)
            if precedente:
                champs["hit_id"] = precedente
            self._nouveau("net", tag.frame, f"filet (frame {tag.frame})", **champs)

    # --- points, jeux et score

    def _score(self, tag: Tag) -> dict | None:
        """`score_after` d'un tag `score` (jeux « P1-P2 », sets « 6-4 3-6 »)."""

        def paire(texte: str) -> list[int] | None:
            morceaux = texte.split("-")
            if len(morceaux) == 2 and all(m.strip().isdigit() for m in morceaux):
                return [int(m) for m in morceaux]
            return None

        jeux = paire(tag.attributs.get(L.ATTR_JEUX, ""))
        sets = [paire(s) for s in tag.attributs.get(L.ATTR_SETS, "").split()]
        if jeux is None or None in sets:
            self.avert.append(
                f"score (frame {tag.frame}) : jeux ({tag.attributs.get(L.ATTR_JEUX)!r}) ou "
                f"sets ({tag.attributs.get(L.ATTR_SETS)!r}) illisibles, tag ignoré"
            )
            return None
        return {"sets": sets, "games": jeux, "points": ["0", "0"]}

    def points_et_jeux(
        self, debuts: list[Tag], fins: list[Tag], services: list[Tag], scores: list[Tag]
    ):
        # Appariement début -> fin, dans l'ordre des frames
        evts = sorted(
            [(t.frame, 0, t) for t in debuts] + [(t.frame, 1, t) for t in fins],
            key=lambda e: (e[0], e[1]),
        )
        ouvert: Tag | None = None
        paires: list[tuple[Tag, Tag]] = []
        for frame, genre, tag in evts:
            if genre == 0:
                if ouvert is not None:
                    self.avert.append(
                        f"point : deux debut_point de suite (frames {ouvert.frame} et {frame}), "
                        "le premier est ignoré"
                    )
                ouvert = tag
            elif ouvert is None:
                self.avert.append(f"point : fin_point (frame {frame}) sans debut_point, ignoré")
            else:
                paires.append((ouvert, tag))
                ouvert = None
        if ouvert is not None:
            self.avert.append(f"point : debut_point (frame {ouvert.frame}) sans fin_point, ignoré")

        hits = [e for e in self.evenements if e["type"] == "hit"]
        points: list[dict] = []
        for debut, fin in paires:
            contexte = f"point (frames {debut.frame}-{fin.frame})"
            gagnant = fin.attributs.get(L.ATTR_GAGNANT)
            if gagnant not in L.JOUEURS:
                self.avert.append(f"{contexte} : gagnant absent ou invalide ({gagnant!r})")
                gagnant = None
            serveur = fin.attributs.get(L.ATTR_SERVEUR)
            if serveur not in L.JOUEURS:
                serveur = next(
                    (
                        s.attributs.get(L.ATTR_SERVEUR)
                        for s in services
                        if debut.frame <= s.frame <= fin.frame
                        and s.attributs.get(L.ATTR_SERVEUR) in L.JOUEURS
                    ),
                    None,
                )
            champs = {
                "end_t_ms": self._t(fin.frame, contexte),
                "end_frame_idx": fin.frame,
                "winner_id": gagnant,
                "rally_length": sum(debut.frame <= h["frame_idx"] <= fin.frame for h in hits),
            }
            if serveur:
                champs["server_id"] = serveur
            if champs["end_t_ms"] is None:
                continue
            raison = fin.attributs.get(L.ATTR_RAISON)
            if raison:
                champs["desc"] = raison
            ev = self._nouveau("point", debut.frame, contexte, **champs)
            if ev is not None:
                points.append(ev)

        # Score : sur le point qui contient le tag, sinon sur le dernier point terminé avant
        scores_lus = []
        for tag in scores:
            score = self._score(tag)
            if score is None:
                continue
            scores_lus.append((tag, score))
            cible = next(
                (p for p in points if p["frame_idx"] <= tag.frame <= p["end_frame_idx"]), None
            )
            if cible is None:
                avant = [p for p in points if p["end_frame_idx"] <= tag.frame]
                cible = max(avant, key=lambda p: p["end_frame_idx"]) if avant else None
            if cible is None:
                self.avert.append(f"score (frame {tag.frame}) : aucun point auquel le rattacher")
            elif "score_after" in cible:
                self.avert.append(f"score (frame {tag.frame}) : point déjà muni d'un score")
            else:
                cible["score_after"] = score
        self._jeux(scores_lus, points)

    def _jeux(self, scores: list[tuple[Tag, dict]], points: list[dict]) -> None:
        """Un événement `game` par tag `score` (fin = tag, début = premier point suivant le
        jeu précédent)."""
        precedent = {"sets": [], "games": [0, 0]}
        fin_precedente = -1
        for tag, score in scores:
            debut = next(
                (p["frame_idx"] for p in points if fin_precedente < p["frame_idx"] <= tag.frame),
                tag.frame,
            )
            if len(score["sets"]) > len(precedent["sets"]):  # le jeu a clos un set
                avant, apres = precedent["games"], score["sets"][len(precedent["sets"])]
            else:
                avant, apres = precedent["games"], score["games"]
            delta = [apres[0] - avant[0], apres[1] - avant[1]]
            gagnant = "P1" if delta == [1, 0] else "P2" if delta == [0, 1] else None
            if gagnant is None:
                self.avert.append(
                    f"jeu (frame {tag.frame}) : gagnant indéterminé (score {avant} -> {apres})"
                )
            serveur = next(
                (
                    p["server_id"]
                    for p in points
                    if debut <= p["frame_idx"] <= tag.frame and "server_id" in p
                ),
                None,
            )
            champs = {"end_t_ms": self._t(tag.frame, "jeu"), "end_frame_idx": tag.frame}
            champs["winner_id"] = gagnant
            if serveur:
                champs["server_id"] = serveur
            champs["score_after"] = score
            if champs["end_t_ms"] is not None:
                self._nouveau("game", debut, f"jeu (frame {tag.frame})", **champs)
            precedent, fin_precedente = score, tag.frame


# --- Import --------------------------------------------------------------------------------


def _tags(export: ExportCVAT, label: str) -> list[Tag]:
    return [t for t in export.tags if t.label == label]


def _pistes(export: ExportCVAT, label: str) -> list[Piste]:
    return [p for p in export.pistes if p.label == label]


def _ecrire_json(chemin: Path, doc: dict) -> None:
    chemin.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def _ecrire_ball(chemin: Path, lignes: list[dict]) -> None:
    colonnes = charger_schema("ball.row.schema.json")["x-columns"]
    with chemin.open("w", encoding="utf-8", newline="") as f:
        ecrivain = csv.writer(f, lineterminator="\n")
        ecrivain.writerow(colonnes)
        for ligne in lignes:
            ecrivain.writerow(["" if ligne[c] is None else ligne[c] for c in colonnes])


def importer_cvat(
    export: str | Path,
    dossier_match: str | Path,
    frames: str | Path | None = None,
    annotateur: str | None = None,
) -> ResultatImport:
    """Convertit un export CVAT for video 1.1 en fichiers de contrat dans `dossier_match`.

    Écrit `ball.csv`, `court.json`, `events.json` (selon ce que l'export contient) et
    `import_report.json`. ValueError si l'export, `frames.parquet` ou les fichiers produits sont
    invalides : dans ce cas rien n'est écrit.
    """
    dossier = Path(dossier_match)
    if not dossier.is_dir():
        raise ValueError(f"Dossier du match introuvable : {dossier}")
    temps = lire_temps(Path(frames) if frames else dossier / "frames.parquet")
    donnees = lire_export(export)
    avert: list[str] = []

    lignes_balle = construire_balle(_pistes(donnees, L.BALLE), temps, avert)
    court = construire_court(_pistes(donnees, L.COURT), donnees.taille_image, avert)
    evts = _Evenements(temps, court, avert)
    frappes = evts.frappes(_tags(donnees, L.FRAPPE))
    tags_service = _tags(donnees, L.SERVICE)
    evts.services(tags_service, bool(frappes))
    pistes_rebonds = _pistes(donnees, L.REBOND)
    if court is None and any(
        not p.outside and p.keyframe for piste in pistes_rebonds for p in piste.points
    ):
        raise ValueError(
            "l'export contient des rebonds mais aucun repère de court exploitable : placer "
            "d'abord les 14 repères `court` (4 au minimum), le contrat exige la position au sol"
        )
    evts.rebonds(pistes_rebonds)
    if evts.perdus:
        frames_perdues = ", ".join(str(f) for f in sorted(evts.perdus))
        avert.append(
            f"{len(evts.perdus)} rebonds perdus (pas de calibration du court) : "
            f"frames {frames_perdues}"
        )
    evts.filets(_tags(donnees, L.FILET))
    evts.points_et_jeux(
        _tags(donnees, L.DEBUT_POINT),
        _tags(donnees, L.FIN_POINT),
        tags_service,
        _tags(donnees, L.SCORE),
    )
    evts.evenements.sort(key=lambda e: (e["frame_idx"], e["id"]))
    comptes = {
        t: evts.compteurs.get(t, 0) for t in ("point", "serve", "hit", "bounce", "net", "game")
    }
    comptes["balle_frames"] = len(lignes_balle)
    comptes["court_segments"] = len(court["segments"]) if court else 0

    if lignes_balle:
        niveau = 3
    elif comptes["hit"] or comptes["bounce"]:
        niveau = 2
    elif comptes["point"] or comptes["serve"] or comptes["game"]:
        niveau = 1
    else:
        raise ValueError("aucune annotation exploitable dans l'export (niveau 0)")

    # Validation dans un dossier temporaire : rien n'est écrit dans le match si un fichier est invalide
    produits: dict[str, object] = {}
    if lignes_balle:
        produits["ball.csv"] = lignes_balle
    if court:
        produits["court.json"] = court
    if evts.evenements:
        produits["events.json"] = {"schema_version": "1.0", "events": evts.evenements}
    with tempfile.TemporaryDirectory() as tmp:
        provisoire = Path(tmp)
        for nom, contenu in produits.items():
            if nom == "ball.csv":
                _ecrire_ball(provisoire / nom, contenu)  # type: ignore[arg-type]
            else:
                _ecrire_json(provisoire / nom, contenu)  # type: ignore[arg-type]
        if (dossier / "match.json").is_file():
            shutil.copyfile(dossier / "match.json", provisoire / "match.json")
        invalides = [
            f"{nom} : {pb.emplacement + ' : ' if pb.emplacement else ''}{pb.message}"
            for nom, problemes in valider_dossier(provisoire).items()
            if nom in produits or nom == "match.json"
            for pb in (problemes or [])
        ]
        if invalides:
            raise ValueError(
                "fichiers produits invalides, rien n'est écrit :\n  - " + "\n  - ".join(invalides)
            )
        for nom in produits:
            shutil.copyfile(provisoire / nom, dossier / nom)
        # Ré-import : un fichier d'un import précédent que cet export ne produit plus serait périmé
        supprimes = [
            n
            for n in ("ball.csv", "court.json", "events.json")
            if n not in produits and (dossier / n).is_file()
        ]
        for nom in supprimes:
            (dossier / nom).unlink()
            avert.append(f"{nom} d'un import précédent supprimé (absent de cet export)")

    rapport = {
        "export": Path(export).name,
        "annotateur": annotateur,
        "niveau_annotation": niveau,
        "fichiers": sorted(produits),
        "comptes": comptes,
        "fichiers_supprimes": supprimes,
        "rebonds_perdus": sorted(evts.perdus),
        "avertissements": avert,
    }
    _ecrire_json(dossier / NOM_RAPPORT, rapport)
    return ResultatImport(niveau, sorted(produits), comptes, avert, supprimes)
