"""Interface en ligne de commande : `tennisvision analyze|clip|tenniset|models|validate|eval`."""

import argparse
import json
import logging
import sys
from pathlib import Path

from tennisvision import config
from tennisvision.config import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_MODEL
from tennisvision.datasets.tenniset import charger_verites_tenniset, resoudre_point
from tennisvision.eval.banc import (
    charger_verites_amateur,
    ecrire_rapport,
    evaluer,
    f1_principaux,
)
from tennisvision.io.video import cut_clip
from tennisvision.models import chemin_modele, importer, registre, telecharger
from tennisvision.pipeline import analyze
from tennisvision.validate import valider_dossier, valider_fichier


class ErreurUsage(Exception):
    """Erreur d'argument ou de fichier : message affiché proprement, code retour 2."""


def _ajouter_intervalle(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=int, default=None, help="frame de début (défaut : 0)")
    parser.add_argument("--end", type=int, default=None, help="frame de fin incluse")


def _ajouter_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="logs détaillés")


def _ajouter_options_analyse(parser: argparse.ArgumentParser) -> None:
    """Options communes à `analyze` et `tenniset analyze`."""
    parser.add_argument(
        "--match",
        type=Path,
        help="fichier match.json (validé, puis copié dans le dossier de sortie)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="dossier de sortie (défaut : ./tennisvision-sorties/<video>_<debut>-<fin>)",
    )
    parser.add_argument(
        "--model", help=f"nom du registre ou chemin de poids YOLO (défaut : {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"seuil de confiance (défaut : {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"taille d'inférence YOLO, en px (défaut : {DEFAULT_IMGSZ})",
    )
    parser.add_argument("--device", help="périphérique de calcul, ex. cpu ou 0")
    parser.add_argument("--no-video", action="store_true", help="ne pas écrire la vidéo annotée")
    _ajouter_verbose(parser)


def _ajouter_data_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="dossier TenniSet (défaut : $TENNISVISION_TENNISET_DIR, sinon ./data)",
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
    _ajouter_options_analyse(p_an)

    p_clip = sub.add_parser("clip", help="extrait un intervalle de frames dans un nouveau clip")
    p_clip.add_argument("video", type=Path, help="chemin de la vidéo")
    _ajouter_intervalle(p_clip)
    p_clip.add_argument("-o", "--output", type=Path, required=True, help="clip de sortie (.mp4)")
    _ajouter_verbose(p_clip)

    p_ten = sub.add_parser("tenniset", help="commandes sur les points annotés du jeu TenniSet")
    sub_ten = p_ten.add_subparsers(dest="action", required=True)
    p_ten_an = sub_ten.add_parser("analyze", help="analyse un point annoté (vidéo et bornes lues)")
    p_ten_an.add_argument("point_id", help="identifiant du point (ex. P00000001)")
    _ajouter_data_dir(p_ten_an)
    _ajouter_options_analyse(p_ten_an)
    p_ten_clip = sub_ten.add_parser("clip", help="extrait un point annoté dans un clip")
    p_ten_clip.add_argument("point_id", help="identifiant du point (ex. P00000001)")
    _ajouter_data_dir(p_ten_clip)
    p_ten_clip.add_argument("-o", "--output", type=Path, required=True, help="clip de sortie")
    _ajouter_verbose(p_ten_clip)

    p_exp = sub_ten.add_parser(
        "export-events", help="convertit la vérité TenniSet en events.json (un dossier par vidéo)"
    )
    _ajouter_data_dir(p_exp)
    p_exp.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="split exporté (défaut : all)",
    )
    p_exp.add_argument("-o", "--output", type=Path, required=True, help="dossier de sortie")
    _ajouter_verbose(p_exp)

    p_mod = sub.add_parser("models", help="gère les modèles du registre (cache utilisateur)")
    sub_mod = p_mod.add_subparsers(dest="action", required=True)
    commun = argparse.ArgumentParser(add_help=False)
    commun.add_argument(
        "--models-dir",
        type=Path,
        help="dossier du cache (défaut : $TENNISVISION_MODELS_DIR, sinon cache utilisateur)",
    )
    _ajouter_verbose(commun)
    sub_mod.add_parser("list", parents=[commun], help="liste les modèles du registre")
    p_pull = sub_mod.add_parser(
        "pull", parents=[commun], help="télécharge des modèles dans le cache"
    )
    p_pull.add_argument("noms", nargs="*", metavar="NOM", help=f"défaut : {DEFAULT_MODEL}")
    p_imp = sub_mod.add_parser(
        "import", parents=[commun], help="importe un fichier de poids local dans le cache"
    )
    p_imp.add_argument("nom", metavar="NOM", help="nom du modèle dans le registre")
    p_imp.add_argument("fichier", type=Path, help="fichier de poids à importer")

    p_val = sub.add_parser(
        "validate", help="vérifie des fichiers de sortie contre les contrats de données"
    )
    p_val.add_argument("chemin", type=Path, help="dossier de sortie ou fichier unique")
    _ajouter_verbose(p_val)

    p_eval = sub.add_parser("eval", help="évalue des prédictions contre une vérité terrain")
    p_eval.add_argument("--dataset", choices=["tenniset", "amateur"], required=True)
    p_eval.add_argument(
        "--predictions", type=Path, required=True, help="dossier <match>/ des sorties du pipeline"
    )
    p_eval.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help="split évalué (défaut : test)",
    )
    _ajouter_data_dir(p_eval)
    p_eval.add_argument("--gt-dir", type=Path, help="dossier de vérité terrain (amateur)")
    p_eval.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "eval",
        help="dossier des rapports (défaut : ./reports/eval)",
    )
    _ajouter_verbose(p_eval)
    return parser


def _afficher_resume(resume: dict) -> None:
    debut, fin = resume["intervalle"]
    largeur, hauteur = resume["resolution"]
    print(
        f"Vidéo : {resume['video']} ({largeur}x{hauteur}, {resume['fps_nominal']:.1f} fps "
        f"nominal, rotation {resume['rotation']}°)"
    )
    print(f"Intervalle : frames {debut} à {fin} ({resume['duree_ms'] / 1000:.2f} s)")
    print(f"Frames traitées : {resume['n_frames_traitees']}")
    # Pas encore de filtre « sur le court » : arbitres, ramasseurs et public sont comptés
    print(f"Identités « personne » (joueurs, arbitres, public…) : {resume['n_identites']}")
    for ident in resume["top_identites"]:
        print(f"  - ID {ident['track_id']} : {ident['n_frames']} frames")
    print(
        f"Balle (brut, faux positifs inclus) : {resume['taux_detection_balle'] * 100:.1f} % "
        f"des frames ({resume['n_detections_balle']} détections)"
    )
    modele = resume["modele"]
    version = f" v{modele['version']}" if modele["version"] else ""
    print(f"Modèle : {modele['nom']}{version} (sha256 {modele['sha256'][:12]}...)")
    print(f"Vitesse de traitement : {resume['fps_traitement']:.1f} fps")
    print(f"Sorties : {resume['dossier']}")
    for fichier in resume["sorties"].values():
        if fichier:
            print(f"  - {fichier}")


def _valider(args: argparse.Namespace) -> int:
    """Affiche le verdict par fichier ; renvoie 0 (tout valide) ou 1 (au moins un invalide)."""
    chemin = args.chemin
    if not chemin.exists():
        raise ErreurUsage(f"Chemin introuvable : {chemin}")
    if chemin.is_dir():
        resultats = valider_dossier(chemin)
        if all(problemes is None for problemes in resultats.values()):
            raise ErreurUsage(f"Aucun fichier de contrat trouvé dans {chemin}")
    else:
        resultats = {chemin.name: valider_fichier(chemin)}
    invalide = False
    for nom, problemes in resultats.items():
        if problemes is None:
            print(f"ABSENT  {nom}")
        elif not problemes:
            print(f"OK  {nom}")
        else:
            invalide = True
            print(f"INVALIDE  {nom} ({len(problemes)} problèmes)")
            for pb in problemes:
                print(
                    f"  - {pb.emplacement} : {pb.message}"
                    if pb.emplacement
                    else f"  - {pb.message}"
                )
    return 1 if invalide else 0


def _analyser(args: argparse.Namespace, video: Path, start: int, end: int | None) -> None:
    resume = analyze(
        video,
        output_dir=args.output_dir,
        start=start,
        end=end,
        model=args.model,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
        write_video=not args.no_video,
        match=args.match,
    )
    _afficher_resume(resume)


def _decouper(video: Path, sortie: Path, start: int, end: int | None) -> None:
    if end is None:
        raise ErreurUsage("clip nécessite --end")
    print(f"Clip écrit : {cut_clip(video, sortie, start, end)}")


def _resoudre_point(args: argparse.Namespace) -> tuple[Path, int, int]:
    """Vidéo et bornes d'un point TenniSet ; point inconnu -> ErreurUsage."""
    try:
        return resoudre_point(args.point_id, config.tenniset_dir(args.data_dir))
    except KeyError as exc:
        raise ErreurUsage(str(exc.args[0])) from exc


def _models(args: argparse.Namespace) -> None:
    if args.action == "list":
        print(f"Cache : {config.models_dir(args.models_dir)}")
        for nom, ref in registre().items():
            etat = "présent" if chemin_modele(nom, args.models_dir).is_file() else "absent"
            print(f"{nom}  v{ref.version}  {ref.licence}  {etat}  {ref.description}")
    elif args.action == "pull":
        for nom in args.noms or [DEFAULT_MODEL]:
            print(f"{nom} : {telecharger(nom, args.models_dir)}")
    else:
        print(f"{args.nom} : {importer(args.nom, args.fichier, args.models_dir)}")


def _exporter_evenements(args: argparse.Namespace) -> None:
    """Écrit `<sortie>/<V0xx>/events.json` pour chaque annotation TenniSet."""
    verites = charger_verites_tenniset(config.tenniset_dir(args.data_dir), args.split)
    for nom, verite in verites.items():
        dossier = args.output / nom
        dossier.mkdir(parents=True, exist_ok=True)
        fichier = dossier / "events.json"
        contenu = json.dumps(verite["events"], indent=2, ensure_ascii=False)
        fichier.write_text(contenu + "\n", encoding="utf-8", newline="\n")
        print(f"{fichier} ({len(verite['events']['events'])} événements)")


def _evaluer(args: argparse.Namespace) -> None:
    if not args.predictions.is_dir():
        raise ErreurUsage(f"Dossier de prédictions introuvable : {args.predictions}")
    if args.dataset == "amateur":
        if args.gt_dir is None:
            raise ErreurUsage("--gt-dir est obligatoire avec --dataset amateur")
        if not args.gt_dir.is_dir():
            raise ErreurUsage(f"Dossier de vérité terrain introuvable : {args.gt_dir}")
        verites = charger_verites_amateur(args.gt_dir, args.split)
    else:
        verites = charger_verites_tenniset(config.tenniset_dir(args.data_dir), args.split)
    rapport = evaluer(verites, args.predictions, dataset=args.dataset, split=args.split)
    chemin = ecrire_rapport(rapport, args.output_dir)
    print(f"Rapport : {chemin}")
    print(f"Matchs évalués : {rapport['agrege']['n_matchs']}")
    for nom, f1 in f1_principaux(rapport["agrege"]).items():
        print(f"  F1 {nom} : {'n/a' if f1 is None else f'{f1:.3f}'}")
    for avertissement in rapport["avertissements"]:
        print(f"Avertissement : {avertissement}")


def _executer(args: argparse.Namespace) -> int:
    if args.commande == "validate":
        return _valider(args)
    if args.commande == "eval":
        _evaluer(args)
        return 0
    if args.commande == "models":
        _models(args)
    elif args.commande == "tenniset" and args.action == "export-events":
        _exporter_evenements(args)
    elif args.commande == "tenniset":
        video, debut, fin = _resoudre_point(args)
        if args.action == "analyze":
            _analyser(args, video, debut, fin)
        else:
            _decouper(video, args.output, debut, fin)
    elif args.commande == "analyze":
        _analyser(args, args.video, args.start or 0, args.end)
    else:
        _decouper(args.video, args.output, args.start or 0, args.end)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée.

    Codes retour : 0 succès (pour `validate` : tous les fichiers présents sont valides ; pour
    `eval` : rapport écrit, quels que soient les scores), 1 au moins un fichier invalide
    (`validate`), 2 erreur d'argument ou de fichier (chemin introuvable, `--gt-dir` manquant ou
    split absent de `splits.json` (`eval`), aucune annotation TenniSet,
    bornes ou `match.json` invalides, point TenniSet inconnu, modèle absent du cache — le message
    indique `models pull` —, téléchargement ou import de modèle en échec, dossier sans fichier de
    contrat…).
    """
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse : --help (0) ou erreur d'usage (2)
        return int(exc.code or 0)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return _executer(args)
    except (ErreurUsage, OSError, ValueError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 2
