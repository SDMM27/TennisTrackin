"""Pipeline d'analyse : ingestion -> personnes + balle -> exports et vidéo annotée."""

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Protocol

from tennisvision import __version__, config
from tennisvision.balle.detecteur import Candidat, DetecteurBalle
from tennisvision.balle.masque import MasqueCourt
from tennisvision.balle.trajectoire import (
    ParamsTrajectoire,
    construire_ball_csv_diagnostic,
    ecrire_ball_csv,
)
from tennisvision.config import DEFAULT_CONF, DEFAULT_IMGSZ
from tennisvision.io.video import (
    VideoWriter,
    frame_index,
    iter_frames,
    probe,
    resolve_interval,
    write_frames_parquet,
)
from tennisvision.models import decrire_modele
from tennisvision.tracking.detector import Detection, Tracker
from tennisvision.tracking.draw import BallTrail, draw_frame
from tennisvision.tracking.export import detections_to_dataframe, summarize
from tennisvision.validate import valider_fichier

logger = logging.getLogger(__name__)

PAS_LOG = 250  # fréquence du log de progression (en frames)
MODELE_BALLE_DEFAUT = "wasb_tennis"
SANS_BALLE = "aucun"  # valeur de `modele_balle` qui désactive la balle


class SuiviObjets(Protocol):
    """Ce que le pipeline attend d'un tracker (permet d'en injecter un faux dans les tests)."""

    def update(self, frame_idx: int, image) -> list[Detection]:
        """Détecte et suit les objets d'une frame."""


class DetecteurDeBalle(Protocol):
    """Ce que le pipeline attend d'un détecteur de balle (un faux est injectable dans les tests)."""

    modele_decrit: dict

    def pousser(self, frame_idx: int, image) -> list[Candidat]:
        """Ajoute une image ; renvoie les candidats des images devenues prédites."""

    def terminer(self) -> list[Candidat]:
        """Prédit les images restantes."""


def _verifier_match(match: str | Path) -> Path:
    """Valide un `match.json` ; ValueError listant `emplacement : message` s'il est invalide."""
    chemin = Path(match)
    if not chemin.is_file():
        raise FileNotFoundError(f"Fichier match introuvable : {chemin}")
    # La validation se fait d'après le nom du fichier : on la passe sous le nom `match.json`
    with tempfile.TemporaryDirectory() as tmp:
        copie = Path(tmp) / "match.json"
        shutil.copyfile(chemin, copie)
        problemes = valider_fichier(copie)
    if problemes:
        details = "\n".join(
            f"  - {pb.emplacement} : {pb.message}" if pb.emplacement else f"  - {pb.message}"
            for pb in problemes
        )
        raise ValueError(f"match.json invalide ({chemin}) :\n{details}")
    return chemin


def _charger_masque(
    court: str | Path | None, masque: str | Path | None
) -> tuple[MasqueCourt | None, str | None]:
    """(masque, nom de sa source) ; `(None, None)` sans masque. `court` et `masque` : ValueError."""
    if court is not None and masque is not None:
        raise ValueError("`court` et `masque` s'excluent : donnez l'un ou l'autre")
    if court is not None:
        if not Path(court).is_file():
            raise FileNotFoundError(f"Fichier court introuvable : {court}")
        return MasqueCourt.depuis_court_json(court), "court.json"
    if masque is not None:
        if not Path(masque).is_file():
            raise FileNotFoundError(f"Fichier masque introuvable : {masque}")
        return MasqueCourt.depuis_polygone(masque), "polygone"
    return None, None


def _balle_desactivee(modele_balle: str | None) -> bool:
    return modele_balle is None or modele_balle.lower() == SANS_BALLE


def _ecrire_video_annote(
    info,
    start: int,
    end: int,
    chemin: Path,
    dets_par_frame: dict[int, list[Detection]],
    positions_balle: dict[int, tuple[float, float, bool]],
) -> None:
    """Seconde passe de décodage : personnes (détections stockées) et traînée de la balle.

    `positions_balle` : `frame_idx` -> (x, y, interpolée) pour les frames où la balle est visible.
    """
    sillage = BallTrail()
    writer = VideoWriter(chemin, (info.width, info.height), info.fps_nominal)
    frames = iter_frames(info.path, start, end)
    try:
        for idx, t_ms, image in frames:
            position = positions_balle.get(idx)
            if position is None:
                sillage.update(idx, None)
            else:
                sillage.update(idx, position[:2], interpole=position[2])
            writer.write(draw_frame(image, dets_par_frame.get(idx, ()), sillage, idx, t_ms), t_ms)
    finally:
        frames.close()
        writer.release()


def analyze(
    video: str | Path,
    output_dir: str | Path | None = None,
    start: int = 0,
    end: int | None = None,
    model: str | None = None,
    conf: float = DEFAULT_CONF,
    device: str | None = None,
    imgsz: int = DEFAULT_IMGSZ,
    write_video: bool = True,
    match: str | Path | None = None,
    tracker: SuiviObjets | None = None,
    modele_balle: str | None = MODELE_BALLE_DEFAUT,
    seuil_balle: float = 0.5,
    court: str | Path | None = None,
    masque: str | Path | None = None,
    detecteur_balle: DetecteurDeBalle | None = None,
    params_trajectoire: ParamsTrajectoire | None = None,
    personnes: bool = True,
    architecture_balle: str | None = None,
) -> dict:
    """Analyse un intervalle de frames d'une vidéo et écrit les résultats.

    Produit dans `output_dir` : frames.parquet (index complet de la vidéo), tracks.csv (personnes
    seulement), ball.csv (balle), summary.json, annotated.mp4 (optionnel) et match.json (copie,
    si `match` est donné). Le dossier est créé après validation des bornes, du fichier `match`,
    du masque et chargement des modèles : un modèle absent lève `ModeleAbsent` avant toute
    écriture.

    Une seule passe de décodage alimente le tracker de personnes et le détecteur de balle ; la
    trajectoire de la balle est ensuite construite hors-ligne sur tout l'intervalle (`ball.csv`),
    puis la vidéo annotée est écrite en seconde passe.

    `tracker` est un objet avec `update(frame_idx, image) -> list[Detection]` ; None construit
    le `Tracker` YOLO. `personnes=False` saute le suivi des personnes (ni YOLO, ni tracks.csv).
    `modele_balle` est un nom du registre, un chemin de poids ou `None` / « aucun » (pas de
    balle) ; `detecteur_balle` (objet `pousser` / `terminer` / `modele_decrit`) remplace le
    `DetecteurBalle` construit d'après `modele_balle` (`architecture_balle` : `hrnet` ou `unet`,
    requise pour un fichier de poids local, par exemple issu de `ball train`). `court` (court.json) ou `masque`
    (polygone manuel) restreignent les candidats balle ; les deux ensemble sont refusés.

    Retourne le contenu de summary.json (chemins relatifs au dossier de sortie) plus la clé
    `dossier` (chemin absolu, pour l'affichage uniquement).
    """
    info = probe(video)
    start, end = resolve_interval(info, start, end)
    if match is not None:
        match = _verifier_match(match)
    masque_court, source_masque = _charger_masque(court, masque)
    avec_balle = detecteur_balle is not None or not _balle_desactivee(modele_balle)
    if not personnes and not avec_balle:
        raise ValueError("rien à analyser : ni personnes, ni balle")
    if avec_balle and detecteur_balle is None:
        detecteur_balle = DetecteurBalle(
            modele_balle,
            device=device,
            seuil_score=seuil_balle,
            fps=info.fps_nominal,
            architecture=architecture_balle,
        )
    modele = None
    if personnes:
        if tracker is None:
            tracker = Tracker(model_path=model, conf=conf, device=device, imgsz=imgsz)
        modele = decrire_modele(model)

    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else config.output_root() / f"{info.path.stem}_{start}-{end}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    chemin_parquet = write_frames_parquet(info.path, out_dir / "frames.parquet")
    if match is not None:
        shutil.copyfile(match, out_dir / "match.json")

    toutes: list[Detection] = []
    candidats: list[Candidat] = []
    temps: dict[int, float] = {}
    total = end - start + 1
    frames = iter_frames(info.path, start, end)
    t0 = time.perf_counter()
    n_traitees = 0
    try:
        for idx, t_ms, image in frames:
            if personnes:
                toutes.extend(tracker.update(idx, image))
            if avec_balle:
                candidats.extend(detecteur_balle.pousser(idx, image))
            temps[idx] = t_ms
            n_traitees += 1
            if n_traitees % PAS_LOG == 0:
                cadence = n_traitees / (time.perf_counter() - t0)
                logger.info("%d/%d frames (%.1f fps)", n_traitees, total, cadence)
    finally:
        frames.close()

    table_balle = None
    bloc_balle = None
    chemin_ball = out_dir / "ball.csv"
    if avec_balle:
        candidats.extend(detecteur_balle.terminer())
        table_balle, diagnostic = construire_ball_csv_diagnostic(
            candidats, temps, info.height, masque_court, params_trajectoire
        )
        ecrire_ball_csv(table_balle, chemin_ball)
        bloc_balle = {
            "modele": getattr(detecteur_balle, "modele_decrit", None),
            "pas_temporel": getattr(detecteur_balle, "pas_temporel", 1),
            "seuil_score": seuil_balle,
            "taux_visible": int(table_balle["visible"].sum()) / n_traitees if n_traitees else 0.0,
            "n_detectes": diagnostic["n_detectes"],
            "n_interpoles": diagnostic["n_interpoles"],
            "diagnostic": diagnostic,
            "masque": source_masque,
        }

    chemin_video = out_dir / "annotated.mp4"
    if write_video:
        dets_par_frame: dict[int, list[Detection]] = {}
        for det in toutes:
            dets_par_frame.setdefault(det.frame_idx, []).append(det)
        positions: dict[int, tuple[float, float, bool]] = {}
        if table_balle is not None:
            for ligne in table_balle[table_balle["visible"] == 1].itertuples():
                positions[int(ligne.frame_idx)] = (ligne.x, ligne.y, ligne.source == "interpolated")
        _ecrire_video_annote(info, start, end, chemin_video, dets_par_frame, positions)
    duree = time.perf_counter() - t0  # de bout en bout : passe unique, post-traitement, vidéo

    df = detections_to_dataframe(toutes, temps)
    chemin_csv = out_dir / "tracks.csv"
    if personnes:
        df.to_csv(chemin_csv, index=False)

    # Bornes temporelles lues dans l'index (qui fait foi), pas parmi les frames décodées
    index = frame_index(info.path)
    resume = summarize(df, info, (start, end), n_traitees, (index[start], index[end]))
    if not personnes:  # pas de suivi de personnes : aucune statistique d'identités
        resume.pop("n_identites")
        resume.pop("top_identites")
    resume["modele"] = modele
    resume["balle"] = bloc_balle
    resume["version_tennisvision"] = __version__
    resume["duree_traitement_s"] = round(duree, 2)
    resume["fps_traitement"] = round(n_traitees / duree, 2) if duree > 0 else 0.0
    resume["sorties"] = {
        "annotated": chemin_video.name if write_video else None,
        "tracks": chemin_csv.name if personnes else None,
        "ball": chemin_ball.name if avec_balle else None,
        "frames": chemin_parquet.name,
        "summary": "summary.json",
        "match": "match.json" if match is not None else None,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)
    return {**resume, "dossier": str(out_dir.resolve())}
