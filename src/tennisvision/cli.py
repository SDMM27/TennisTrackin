"""Interface en ligne de commande : `tennisvision analyze|clip`."""

import argparse
import logging
import sys
from pathlib import Path

from tennisvision.config import DEFAULT_CONF, DEFAULT_IMGSZ
from tennisvision.datasets.tenniset import get_point
from tennisvision.io.video import cut_clip
from tennisvision.pipeline import analyze


class ErreurUsage(Exception):
    """Erreur d'argument ou de fichier : message affiché proprement, code retour 2."""


def _ajouter_intervalle(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=int, default=None, help="frame de début (défaut : 0)")
    parser.add_argument("--end", type=int, default=None, help="frame de fin incluse")
    parser.add_argument(
        "--point", help="identifiant d'un point annoté (ex. P00000001) : fixe start et end"
    )


def build_parser() -> argparse.ArgumentParser:
    """Construit le parseur d'arguments."""
    parser = argparse.ArgumentParser(
        prog="tennisvision", description="Analyse de vidéos de tennis (joueurs et balle)."
    )
    sub = parser.add_subparsers(dest="commande", required=True)

    p_an = sub.add_parser("analyze", help="détecte et suit joueurs et balle, produit les sorties")
    p_an.add_argument("video", type=Path, help="chemin de la vidéo")
    _ajouter_intervalle(p_an)
    p_an.add_argument("--output-dir", type=Path, help="dossier de sortie (défaut : reports/...)")
    p_an.add_argument("--model", help="poids YOLO (défaut : models/yolo11n.pt)")
    p_an.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"seuil de confiance (défaut : {DEFAULT_CONF})",
    )
    p_an.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"taille d'inférence YOLO, en px (défaut : {DEFAULT_IMGSZ})",
    )
    p_an.add_argument("--device", help="périphérique de calcul, ex. cpu ou 0")
    p_an.add_argument("--no-video", action="store_true", help="ne pas écrire la vidéo annotée")
    p_an.add_argument("-v", "--verbose", action="store_true", help="logs détaillés")

    p_clip = sub.add_parser("clip", help="extrait un intervalle de frames dans un nouveau clip")
    p_clip.add_argument("video", type=Path, help="chemin de la vidéo")
    _ajouter_intervalle(p_clip)
    p_clip.add_argument("-o", "--output", type=Path, required=True, help="clip de sortie (.mp4)")
    p_clip.add_argument("-v", "--verbose", action="store_true", help="logs détaillés")
    return parser


def _resoudre_intervalle(args: argparse.Namespace) -> tuple[int, int | None]:
    """Détermine (start, end) à partir de --point ou de --start/--end."""
    if args.point is None:
        return (args.start or 0), args.end
    if args.start is not None or args.end is not None:
        raise ErreurUsage("--point est incompatible avec --start/--end")
    try:
        video, debut, fin = get_point(args.point)
    except KeyError as exc:
        raise ErreurUsage(str(exc.args[0])) from exc
    except FileNotFoundError as exc:
        raise ErreurUsage(f"Fichier d'annotations introuvable : {exc.filename}") from exc
    if video != args.video.stem:
        raise ErreurUsage(
            f"Le point {args.point} appartient à la vidéo {video}, pas à {args.video.stem}"
        )
    return debut, fin


def _afficher_resume(resume: dict) -> None:
    debut, fin = resume["intervalle"]
    largeur, hauteur = resume["resolution"]
    print(f"Vidéo : {resume['video']} ({largeur}x{hauteur} @ {resume['fps']:.1f} fps)")
    print(f"Intervalle : frames {debut} à {fin}")
    print(f"Frames traitées : {resume['n_frames_traitees']}")
    # Pas encore de filtre « sur le court » : arbitres, ramasseurs et public sont comptés
    print(f"Identités « personne » (joueurs, arbitres, public…) : {resume['n_identites']}")
    for ident in resume["top_identites"]:
        print(f"  - ID {ident['track_id']} : {ident['n_frames']} frames")
    print(
        f"Balle (brut, faux positifs inclus) : {resume['taux_detection_balle'] * 100:.1f} % "
        f"des frames ({resume['n_detections_balle']} détections)"
    )
    print(f"Vitesse de traitement : {resume['fps_traitement']:.1f} fps")
    sorties = resume["sorties"]
    print(f"Sorties : {sorties['dossier']}")
    if sorties["video_annotee"]:
        print(f"  - vidéo annotée : {sorties['video_annotee']}")
    print(f"  - détections : {sorties['tracks']}")
    print(f"  - résumé : {sorties['resume']}")


def _executer(args: argparse.Namespace) -> None:
    start, end = _resoudre_intervalle(args)
    if args.commande == "analyze":
        resume = analyze(
            args.video,
            output_dir=args.output_dir,
            start=start,
            end=end,
            model=args.model,
            conf=args.conf,
            device=args.device,
            imgsz=args.imgsz,
            write_video=not args.no_video,
        )
        _afficher_resume(resume)
    else:
        if end is None:
            raise ErreurUsage("clip nécessite --end (ou --point)")
        dst = cut_clip(args.video, args.output, start, end)
        print(f"Clip écrit : {dst}")


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée. Codes retour : 0 succès, 2 erreur d'argument ou de fichier."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse : --help (0) ou erreur d'usage (2)
        return int(exc.code or 0)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        _executer(args)
    except (ErreurUsage, FileNotFoundError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
    return 0
