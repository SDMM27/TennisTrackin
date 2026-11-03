"""Pipeline d'analyse : ingestion -> détection/suivi -> vidéo annotée + exports."""

import json
import logging
import time
from collections import deque
from pathlib import Path

from tennisvision.config import DEFAULT_CONF, DEFAULT_IMGSZ, REPORTS_DIR
from tennisvision.io.video import VideoWriter, iter_frames, probe
from tennisvision.tracking.detector import Detection, Tracker
from tennisvision.tracking.draw import draw_frame
from tennisvision.tracking.export import detections_to_dataframe, summarize

logger = logging.getLogger(__name__)

PAS_LOG = 250  # fréquence du log de progression (en frames)


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
    trail: int = 15,
) -> dict:
    """Analyse un intervalle de frames d'une vidéo et écrit les résultats.

    Produit dans `output_dir` : annotated.mp4 (optionnel), tracks.csv et summary.json.
    Retourne le résumé (avec les chemins de sortie).
    """
    info = probe(video)
    if end is None or end >= info.n_frames:
        end = info.n_frames - 1
    out_dir = (
        Path(output_dir)
        if output_dir is not None
        else REPORTS_DIR / f"{Path(video).stem}_{start}-{end}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Générateur paresseux : les bornes incohérentes lèvent ValueError à la première frame
    frames = iter_frames(video, start, end)
    tracker = Tracker(model_path=model, conf=conf, device=device, imgsz=imgsz)

    toutes: list[Detection] = []
    sillage: deque[tuple[int, int]] = deque(maxlen=max(trail, 1))
    chemin_video = out_dir / "annotated.mp4"
    writer = VideoWriter(chemin_video, info.fps, (info.width, info.height)) if write_video else None
    total = end - start + 1
    t0 = time.perf_counter()
    n_traitees = 0
    try:
        for idx, frame in frames:
            dets = tracker.update(idx, frame)
            toutes.extend(dets)
            if writer is not None:
                writer.write(draw_frame(frame, dets, sillage, frame_idx=idx, fps=info.fps))
            n_traitees += 1
            if n_traitees % PAS_LOG == 0:
                cadence = n_traitees / (time.perf_counter() - t0)
                logger.info("%d/%d frames (%.1f fps)", n_traitees, total, cadence)
    finally:
        frames.close()
        if writer is not None:
            writer.release()
    duree = time.perf_counter() - t0

    df = detections_to_dataframe(toutes)
    chemin_csv = out_dir / "tracks.csv"
    df.to_csv(chemin_csv, index=False)

    resume = summarize(df, info, (start, end))
    resume["n_frames_traitees"] = n_traitees
    if n_traitees:
        frames_balle = df.loc[df["label"] == "ball", "frame_idx"].nunique()
        resume["taux_detection_balle"] = frames_balle / n_traitees
    resume["duree_traitement_s"] = round(duree, 2)
    resume["fps_traitement"] = round(n_traitees / duree, 2) if duree > 0 else 0.0
    resume["sorties"] = {
        "dossier": str(out_dir),
        "video_annotee": str(chemin_video) if write_video else None,
        "tracks": str(chemin_csv),
        "resume": str(out_dir / "summary.json"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(resume, f, indent=2, ensure_ascii=False)
    return resume
