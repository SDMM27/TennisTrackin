"""Pipeline d'analyse : ingestion -> détection/suivi -> vidéo annotée + exports."""

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Protocol

from tennisvision import __version__, config
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


class SuiviObjets(Protocol):
    """Ce que le pipeline attend d'un tracker (permet d'en injecter un faux dans les tests)."""

    def update(self, frame_idx: int, image) -> list[Detection]:
        """Détecte et suit les objets d'une frame."""


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
) -> dict:
    """Analyse un intervalle de frames d'une vidéo et écrit les résultats.

    Produit dans `output_dir` : frames.parquet (index complet de la vidéo), tracks.csv,
    summary.json, annotated.mp4 (optionnel) et match.json (copie, si `match` est donné). Le
    dossier est créé après validation des bornes et du fichier `match`.

    `tracker` est un objet avec `update(frame_idx, image) -> list[Detection]` ; None construit
    le `Tracker` YOLO. Retourne le contenu de summary.json (chemins relatifs au dossier de
    sortie) plus la clé `dossier` (chemin absolu, pour l'affichage uniquement).
    """
    info = probe(video)
    start, end = resolve_interval(info, start, end)
    if match is not None:
        match = _verifier_match(match)
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
    temps: dict[int, float] = {}
    sillage = BallTrail()
    chemin_video = out_dir / "annotated.mp4"
    writer = (
        VideoWriter(chemin_video, (info.width, info.height), info.fps_nominal)
        if write_video
        else None
    )
    total = end - start + 1
    frames = iter_frames(info.path, start, end)
    t0 = time.perf_counter()
    n_traitees = 0
    try:
        for idx, t_ms, image in frames:
            dets = tracker.update(idx, image)
            toutes.extend(dets)
            temps[idx] = t_ms
            if writer is not None:
                balles = [d for d in dets if d.label == "ball"]
                meilleure = max(balles, key=lambda d: d.conf) if balles else None
                sillage.update(idx, None if meilleure is None else (meilleure.cx, meilleure.cy))
                writer.write(draw_frame(image, dets, sillage, idx, t_ms), t_ms)
            n_traitees += 1
            if n_traitees % PAS_LOG == 0:
                cadence = n_traitees / (time.perf_counter() - t0)
                logger.info("%d/%d frames (%.1f fps)", n_traitees, total, cadence)
    finally:
        frames.close()
        if writer is not None:
            writer.release()
    duree = time.perf_counter() - t0

    df = detections_to_dataframe(toutes, temps)
    chemin_csv = out_dir / "tracks.csv"
    df.to_csv(chemin_csv, index=False)

    # Bornes temporelles lues dans l'index (qui fait foi), pas parmi les frames décodées
    index = frame_index(info.path)
    resume = summarize(df, info, (start, end), n_traitees, (index[start], index[end]))
    resume["modele"] = modele
    resume["version_tennisvision"] = __version__
    resume["duree_traitement_s"] = round(duree, 2)
    resume["fps_traitement"] = round(n_traitees / duree, 2) if duree > 0 else 0.0
    resume["sorties"] = {
        "annotated": chemin_video.name if write_video else None,
        "tracks": chemin_csv.name,
        "frames": chemin_parquet.name,
        "summary": "summary.json",
        "match": "match.json" if match is not None else None,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)
    return {**resume, "dossier": str(out_dir.resolve())}
