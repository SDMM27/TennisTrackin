"""Interface en ligne de commande.

Commandes : `analyze`, `clip`, `tenniset`, `models`, `ball`, `validate`, `eval`, `annotation`,
`dataset`.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from tennisvision import config
from tennisvision.annotation.importeur import importer_cvat
from tennisvision.config import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_MODEL
from tennisvision.datasets import etiquettes_faibles
from tennisvision.datasets.tenniset import charger_verites_tenniset, resoudre_point
from tennisvision.eval.banc import (
    charger_verites_amateur,
    ecrire_rapport,
    evaluer,
    f1_principaux,
)
from tennisvision.io.video import cut_clip
from tennisvision.jeu_amateur import accord, controle, decoupage, fiche
from tennisvision.models import chemin_modele, importer, registre, telecharger
from tennisvision.pipeline import MODELE_BALLE_DEFAUT, SANS_BALLE, analyze
from tennisvision.validate import valider_dossier, valider_fichier

ARCHITECTURES = ("hrnet", "unet")  # comme tennisvision.balle.reseaux (sans importer torch ici)


class ErreurUsage(Exception):
    """Erreur d'argument ou de fichier : message affiché proprement, code retour 2."""


def _ajouter_intervalle(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=int, default=None, help="frame de début (défaut : 0)")
    parser.add_argument("--end", type=int, default=None, help="frame de fin incluse")


def _ajouter_verbose(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-v", "--verbose", action="store_true", help="logs détaillés")


def _ajouter_architecture_balle(parser: argparse.ArgumentParser, option: str) -> None:
    parser.add_argument(
        option,
        dest="architecture_balle",
        choices=ARCHITECTURES,
        help="architecture du modèle de balle ; obligatoire pour un fichier de poids local",
    )


def _ajouter_options_masque(parser: argparse.ArgumentParser, option_seuil: str) -> None:
    """Seuil de score de la balle, `--court` et `--mask` (communs à `analyze` et `ball track`)."""
    parser.add_argument(
        option_seuil,
        dest="seuil_balle",
        type=float,
        default=0.5,
        help="seuil de score des candidats balle, de 0 à 1 (défaut : 0.5)",
    )
    parser.add_argument(
        "--court",
        type=Path,
        help="court.json : ignore les candidats balle hors du court étendu (exclusif de --mask)",
    )
    parser.add_argument(
        "--mask",
        type=Path,
        help='polygone manuel {"polygone": [[x, y], ...]} en pixels (exclusif de --court)',
    )


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
    parser.add_argument(
        "--ball-model",
        default=MODELE_BALLE_DEFAUT,
        help=(
            f"nom du registre ou chemin de poids du modèle de balle (défaut : "
            f"{MODELE_BALLE_DEFAUT} ; « {SANS_BALLE} » désactive la balle)"
        ),
    )
    _ajouter_architecture_balle(parser, "--ball-architecture")
    _ajouter_options_masque(parser, "--ball-threshold")
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

    p_weak = sub_ten.add_parser(
        "weak-labels", help="extrait des étiquettes faibles (fin de point, coups) des desc"
    )
    _ajouter_data_dir(p_weak)
    p_weak.add_argument(
        "--sortie",
        type=Path,
        default=Path("weak_labels.csv"),
        help="CSV (défaut : weak_labels.csv)",
    )
    _ajouter_verbose(p_weak)
    p_side = sub_ten.add_parser(
        "side-sample", help="échantillon de frappes pour vérifier à la main la convention de Side"
    )
    _ajouter_data_dir(p_side)
    p_side.add_argument("--n", type=int, default=50, help="nombre de frappes (défaut : 50)")
    p_side.add_argument("--seed", type=int, default=0, help="graine du tirage (défaut : 0)")
    p_side.add_argument(
        "--sortie",
        type=Path,
        default=Path("side_sample.csv"),
        help="CSV (défaut : side_sample.csv)",
    )
    p_side.add_argument(
        "--images", type=Path, help="dossier où enregistrer la frame médiane de chaque frappe"
    )
    _ajouter_verbose(p_side)

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
    p_pull.add_argument(
        "noms",
        nargs="*",
        metavar="NOM",
        help=f"défaut : les modèles d'analyze ({DEFAULT_MODEL}, {MODELE_BALLE_DEFAUT})",
    )
    p_imp = sub_mod.add_parser(
        "import", parents=[commun], help="importe un fichier de poids local dans le cache"
    )
    p_imp.add_argument("nom", metavar="NOM", help="nom du modèle dans le registre")
    p_imp.add_argument("fichier", type=Path, help="fichier de poids à importer")

    p_ball = sub.add_parser("ball", help="suivi de la balle : trajectoire, débit, export ONNX")
    sub_ball = p_ball.add_subparsers(dest="action", required=True)
    p_track = sub_ball.add_parser(
        "track", help="suit la balle seule (ball.csv et summary.json, sans YOLO)"
    )
    p_track.add_argument("video", type=Path, help="chemin de la vidéo")
    _ajouter_intervalle(p_track)
    p_track.add_argument(
        "--output-dir", type=Path, help="dossier de sortie (défaut : comme analyze)"
    )
    p_track.add_argument(
        "--model",
        default=MODELE_BALLE_DEFAUT,
        help=f"nom du registre ou chemin de poids (défaut : {MODELE_BALLE_DEFAUT})",
    )
    p_track.add_argument("--device", help="périphérique de calcul, ex. cpu ou 0")
    _ajouter_architecture_balle(p_track, "--ball-architecture")
    _ajouter_options_masque(p_track, "--threshold")
    _ajouter_verbose(p_track)
    p_bench = sub_ball.add_parser("bench", help="mesure le débit du détecteur (JSON)")
    p_bench.add_argument(
        "--model", default=MODELE_BALLE_DEFAUT, help=f"modèle (défaut : {MODELE_BALLE_DEFAUT})"
    )
    p_bench.add_argument("--device", help="périphérique de calcul, ex. cpu ou 0")
    _ajouter_architecture_balle(p_bench, "--architecture")
    p_bench.add_argument("--batch", type=int, default=4, help="triplets par lot (défaut : 4)")
    p_bench.add_argument("--triplets", type=int, default=8, help="triplets mesurés (défaut : 8)")
    _ajouter_verbose(p_bench)
    p_train = sub_ball.add_parser(
        "train", help="affine un modèle de balle (best.pth et entrainement.json)"
    )
    p_train.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="FORMAT:CHEMIN",
        help="données : tracknet:<racine> ou amateur:<gt_dir> (répétable)",
    )
    p_train.add_argument("--output-dir", type=Path, required=True, help="dossier de sortie")
    p_train.add_argument(
        "--architecture", choices=ARCHITECTURES, default="hrnet", help="défaut : hrnet"
    )
    p_train.add_argument(
        "--init",
        default="wasb_tennis",
        help="poids initiaux : nom du registre ou chemin (défaut : wasb_tennis ; « aucun » : aléatoire)",
    )
    p_train.add_argument("--epochs", type=int, default=10, help="époques (défaut : 10)")
    p_train.add_argument("--batch", type=int, default=8, help="triplets par lot (défaut : 8)")
    p_train.add_argument(
        "--lr", type=float, default=1e-3, help="taux d'apprentissage (défaut : 0.001)"
    )
    p_train.add_argument("--device", help="périphérique de calcul, ex. cpu ou 0")
    p_train.add_argument("--seed", type=int, default=0, help="graine (défaut : 0)")
    p_train.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="part de validation des sources TrackNet (défaut : 0.1) ; le jeu amateur suit splits.json",
    )
    p_train.add_argument(
        "--temporal-step",
        type=int,
        default=1,
        help="pas entre les images d'un triplet : 2 pour du 60 fps (défaut : 1)",
    )
    p_train.add_argument(
        "--workers", type=int, default=0, help="processus de chargement (défaut : 0)"
    )
    p_train.add_argument(
        "--cache-dir",
        type=Path,
        help="cache des images décodées (défaut : <output-dir>/cache_frames)",
    )
    p_train.add_argument(
        "--videos-dir",
        type=Path,
        help="dossier des vidéos du jeu amateur (défaut : dossier du match)",
    )
    p_train.add_argument(
        "--no-interpolated",
        action="store_true",
        help="écarte les positions interpolées de ball.csv (jeu amateur)",
    )
    p_train.add_argument(
        "--no-augmentation", action="store_true", help="désactive les augmentations"
    )
    p_train.add_argument("--max-batches", type=int, help="lots par époque au plus (essais rapides)")
    _ajouter_verbose(p_train)
    p_onnx = sub_ball.add_parser("export-onnx", help="exporte un modèle de balle en ONNX")
    p_onnx.add_argument("output", metavar="OUTPUT", type=Path, help="fichier .onnx de sortie")
    p_onnx.add_argument(
        "--model", default=MODELE_BALLE_DEFAUT, help=f"modèle (défaut : {MODELE_BALLE_DEFAUT})"
    )
    _ajouter_architecture_balle(p_onnx, "--architecture")
    p_onnx.add_argument("--opset", type=int, default=17, help="version d'opset ONNX (défaut : 17)")
    _ajouter_verbose(p_onnx)

    p_ann = sub.add_parser("annotation", help="outils d'annotation de la vérité terrain")
    sub_ann = p_ann.add_subparsers(dest="action", required=True)
    p_imp_cvat = sub_ann.add_parser(
        "import-cvat", help="convertit un export CVAT for video 1.1 en fichiers de contrat"
    )
    p_imp_cvat.add_argument("export", type=Path, help="export CVAT (XML)")
    p_imp_cvat.add_argument(
        "--match", type=Path, required=True, help="dossier du match (sorties écrites ici)"
    )
    p_imp_cvat.add_argument(
        "--frames", type=Path, help="frames.parquet (défaut : <match>/frames.parquet)"
    )
    p_imp_cvat.add_argument("--annotateur", help="identifiant de l'annotateur (noté au rapport)")
    _ajouter_verbose(p_imp_cvat)

    p_data = sub.add_parser("dataset", help="gère le jeu de données amateur (vérité terrain)")
    sub_data = p_data.add_subparsers(dest="action", required=True)
    p_chk = sub_data.add_parser("check", help="contrôle le dossier et les critères du Lot 2")
    p_chk.add_argument("gt_dir", type=Path, help="dossier de vérité terrain (registre.csv, matchs)")
    p_chk.add_argument("--json", type=Path, help="écrit le résumé dans ce fichier JSON")
    _ajouter_verbose(p_chk)
    p_spl = sub_data.add_parser("split", help="écrit splits.json (par match, test gelé)")
    p_spl.add_argument("gt_dir", type=Path, help="dossier de vérité terrain")
    p_spl.add_argument("--test-ratio", type=float, default=0.2, help="part du test (défaut : 0.2)")
    p_spl.add_argument("--val-ratio", type=float, default=0.1, help="part de val (défaut : 0.1)")
    p_spl.add_argument("--seed", type=int, default=0, help="graine du tirage (défaut : 0)")
    p_spl.add_argument("--refaire-test", action="store_true", help="refait le test gelé")
    p_spl.add_argument(
        "--je-sais-ce-que-je-fais", action="store_true", help="confirme --refaire-test"
    )
    _ajouter_verbose(p_spl)
    p_acc = sub_data.add_parser("accord", help="accord inter-annotateurs de deux annotations")
    p_acc.add_argument("dir_a", type=Path, help="annotation A (un match, ou dossier de matchs)")
    p_acc.add_argument("dir_b", type=Path, help="annotation B (même structure)")
    p_acc.add_argument("--json", type=Path, help="écrit le rapport dans ce fichier JSON")
    _ajouter_verbose(p_acc)
    p_card = sub_data.add_parser("card", help="régénère le bloc de statistiques de la fiche")
    p_card.add_argument("gt_dir", type=Path, help="dossier de vérité terrain")
    p_card.add_argument(
        "--sortie",
        type=Path,
        default=Path("docs") / "dataset_card.md",
        help="fiche à mettre à jour (défaut : docs/dataset_card.md)",
    )
    _ajouter_verbose(p_card)

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


def _afficher_balle(balle: dict | None) -> None:
    if balle is None:
        print("Balle : désactivée")
        return
    modele = balle["modele"]
    nom = "inconnu" if modele is None else modele["nom"]
    print(
        f"Balle ({nom}, seuil {balle['seuil_score']}) : visible sur "
        f"{balle['taux_visible'] * 100:.1f} % des frames "
        f"({balle['n_detectes']} détectées, {balle['n_interpoles']} interpolées)"
    )


def _afficher_resume(resume: dict) -> None:
    debut, fin = resume["intervalle"]
    largeur, hauteur = resume["resolution"]
    print(
        f"Vidéo : {resume['video']} ({largeur}x{hauteur}, {resume['fps_nominal']:.1f} fps "
        f"nominal, rotation {resume['rotation']}°)"
    )
    print(f"Intervalle : frames {debut} à {fin} ({resume['duree_ms'] / 1000:.2f} s)")
    print(f"Frames traitées : {resume['n_frames_traitees']}")
    modele = resume["modele"]
    if modele is not None:  # absent pour `ball track` (pas de suivi des personnes)
        # Pas encore de filtre « sur le court » : arbitres, ramasseurs et public sont comptés
        print(f"Identités « personne » (joueurs, arbitres, public…) : {resume['n_identites']}")
        for ident in resume["top_identites"]:
            print(f"  - ID {ident['track_id']} : {ident['n_frames']} frames")
    _afficher_balle(resume["balle"])
    if modele is not None:
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
        modele_balle=args.ball_model,
        seuil_balle=args.seuil_balle,
        court=args.court,
        masque=args.mask,
        architecture_balle=args.architecture_balle,
    )
    _afficher_resume(resume)


def _balle(args: argparse.Namespace) -> None:
    if args.action == "track":
        resume = analyze(
            args.video,
            output_dir=args.output_dir,
            start=args.start or 0,
            end=args.end,
            device=args.device,
            write_video=False,
            modele_balle=args.model,
            seuil_balle=args.seuil_balle,
            court=args.court,
            masque=args.mask,
            personnes=False,
            architecture_balle=args.architecture_balle,
        )
        _afficher_resume(resume)
        return
    if args.action == "train":
        _entrainer(args)
        return
    from tennisvision.balle import export

    try:
        if args.action == "bench":
            resultat = export.mesurer_debit(
                args.model,
                args.device,
                n_triplets=args.triplets,
                taille_lot=args.batch,
                architecture=args.architecture_balle,
            )
            print(json.dumps(resultat, indent=2, ensure_ascii=False))
        else:
            chemin = export.exporter_onnx(
                args.model, args.output, architecture=args.architecture_balle, opset=args.opset
            )
            print(f"Modèle exporté : {chemin}")
    except ImportError as exc:  # dépendances optionnelles manquantes
        raise ErreurUsage(str(exc)) from exc


def _entrainer(args: argparse.Namespace) -> None:
    from tennisvision.balle.augmentations import ParametresAugmentation
    from tennisvision.balle.entrainement import ConfigEntrainement, entrainer

    augmentation = ParametresAugmentation()
    if args.no_augmentation:
        augmentation = augmentation.desactivees()
    config_entrainement = ConfigEntrainement(
        sources=tuple(args.source),
        dossier_sortie=args.output_dir,
        architecture=args.architecture,
        poids_initiaux=None if args.init.lower() == SANS_BALLE else args.init,
        epochs=args.epochs,
        taille_lot=args.batch,
        lr=args.lr,
        device=args.device,
        graine=args.seed,
        fraction_validation=args.val_fraction,
        pas_temporel=args.temporal_step,
        nb_workers=args.workers,
        max_lots=args.max_batches,
        dossier_cache=args.cache_dir,
        videos_dir=args.videos_dir,
        interpolees=not args.no_interpolated,
        augmentation=augmentation,
    )
    print(f"Meilleur modèle : {entrainer(config_entrainement)}")


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
        for nom in args.noms or [DEFAULT_MODEL, MODELE_BALLE_DEFAUT]:
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


def _etiquettes_faibles(args: argparse.Namespace) -> None:
    lignes = etiquettes_faibles.etiquettes_faibles(config.tenniset_dir(args.data_dir))
    etiquettes_faibles.ecrire_etiquettes(args.sortie, lignes)
    part = etiquettes_faibles.couverture(lignes)
    print(f"{args.sortie} ({len(lignes)} points)")
    print(f"Couverture : {'n/a' if part is None else f'{part * 100:.1f} %'} des points ont une fin")


def _echantillon_cotes(args: argparse.Namespace) -> None:
    dossier = config.tenniset_dir(args.data_dir)
    lignes = etiquettes_faibles.echantillon_cotes(dossier, args.n, args.seed)
    etiquettes_faibles.ecrire_echantillon(args.sortie, lignes)
    print(f"{args.sortie} ({len(lignes)} frappes)")
    if args.images is not None:
        n, avertissements = etiquettes_faibles.sauver_images(lignes, dossier, args.images)
        print(f"{n} images dans {args.images}")
        for avertissement in avertissements:
            print(f"Avertissement : {avertissement}")


def _importer_cvat(args: argparse.Namespace) -> None:
    if not args.export.is_file():
        raise ErreurUsage(f"Export CVAT introuvable : {args.export}")
    resultat = importer_cvat(args.export, args.match, args.frames, args.annotateur)
    print(f"Niveau d'annotation détecté : {resultat.niveau}")
    for fichier in resultat.fichiers:
        print(f"  - {args.match / fichier}")
    comptes = ", ".join(f"{cle} : {n}" for cle, n in resultat.comptes.items())
    print(f"Contenu : {comptes}")
    for avertissement in resultat.avertissements:
        print(f"Avertissement : {avertissement}")
    print(f"Rapport : {args.match / 'import_report.json'}")


def _ecrire_json(chemin: Path, doc: dict) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    texte = json.dumps(doc, indent=2, ensure_ascii=False)
    chemin.write_text(texte + "\n", encoding="utf-8", newline="\n")


def _dataset(args: argparse.Namespace) -> None:
    if args.action == "accord":
        for dossier in (args.dir_a, args.dir_b):
            if not dossier.is_dir():
                raise ErreurUsage(f"Dossier introuvable : {dossier}")
        rapport = accord.comparer(args.dir_a, args.dir_b)
        print("\n".join(accord.formater_accord(rapport)))
        if args.json:
            _ecrire_json(args.json, rapport)
        return
    if not args.gt_dir.is_dir():
        raise ErreurUsage(f"Dossier de vérité terrain introuvable : {args.gt_dir}")
    if args.action == "check":
        rapport = controle.controler(args.gt_dir)
        print("\n".join(controle.formater_controle(rapport)))
        if args.json:
            _ecrire_json(args.json, rapport)
    elif args.action == "split":
        splits = decoupage.decouper(
            args.gt_dir,
            args.test_ratio,
            args.val_ratio,
            args.seed,
            args.refaire_test,
            args.je_sais_ce_que_je_fais,
        )
        print(args.gt_dir / decoupage.NOM_SPLITS)
        for nom, matchs in splits.items():
            print(f"  {nom} : {len(matchs)} ({', '.join(matchs)})")
    else:
        print(f"Fiche mise à jour : {fiche.mettre_a_jour_fiche(args.gt_dir, args.sortie)}")


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
    try:
        rapport = evaluer(verites, args.predictions, dataset=args.dataset, split=args.split)
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ErreurUsage(f"fichier d'entrée de structure inattendue ({exc!r})") from exc
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
    if args.commande == "annotation":
        _importer_cvat(args)
        return 0
    if args.commande == "dataset":
        _dataset(args)
        return 0
    if args.commande == "ball":
        _balle(args)
        return 0
    if args.commande == "models":
        _models(args)
    elif args.commande == "tenniset" and args.action == "export-events":
        _exporter_evenements(args)
    elif args.commande == "tenniset" and args.action == "weak-labels":
        _etiquettes_faibles(args)
    elif args.commande == "tenniset" and args.action == "side-sample":
        _echantillon_cotes(args)
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
    `eval` : rapport écrit, quels que soient les scores ; pour `dataset check` : contrôle
    imprimé, même si des critères ne sont pas atteints), 1 au moins un fichier invalide
    (`validate`), 2 erreur d'argument ou de fichier (chemin introuvable, JSON ou CSV illisible ou
    de structure invalide — le message nomme le fichier —, `--gt-dir` manquant ou
    split absent de `splits.json` (`eval`), aucune annotation TenniSet,
    bornes ou `match.json` invalides, `--court` et `--mask` donnés ensemble, point TenniSet inconnu,
    modèle (YOLO ou balle) absent du cache — le message
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
