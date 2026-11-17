"""Bloc de statistiques automatiques de la fiche descriptive du jeu (`docs/dataset_card.md`)."""

from collections import Counter
from pathlib import Path

from tennisvision.jeu_amateur.controle import stats_match
from tennisvision.jeu_amateur.registre import lire_registre, liste

MARQUEUR_DEBUT = "<!-- debut-stats-auto -->"
MARQUEUR_FIN = "<!-- fin-stats-auto -->"
ENTETE_DEFAUT = "# Fiche descriptive du jeu de données amateur\n\n"


def _repartition(valeurs: list[str]) -> str:
    comptes = Counter(v or "non renseigné" for v in valeurs)
    return ", ".join(f"{cle} : {n}" for cle, n in sorted(comptes.items())) or "aucun"


def bloc_statistiques(gt_dir: str | Path) -> str:
    """Texte Markdown du bloc (sans les marqueurs). Aucun nom de joueur : pseudonymes comptés
    seulement. ValueError si le registre est absent ou illisible."""
    gt_dir = Path(gt_dir)
    lignes = lire_registre(gt_dir)
    if lignes is None:
        raise ValueError(f"registre.csv introuvable dans {gt_dir}")
    totaux: Counter = Counter()
    for ligne in lignes:
        dossier = gt_dir / ligne["match"]
        if dossier.is_dir():
            stats = stats_match(dossier)
            totaux.update({k: stats[k] for k in ("hits", "rebonds", "points", "services", "jeux")})
            totaux["balle_frames"] += stats["balle_frames"]
    joueurs = {p for ligne in lignes for p in liste(ligne["joueurs"])}
    doubles = sum(ligne["double_annotation"] == "oui" for ligne in lignes)
    return "\n".join(
        [
            "_Bloc généré par `tennisvision dataset card` : ne pas modifier à la main._",
            "",
            f"- Matchs : {len(lignes)}",
            f"- Joueurs distincts (pseudonymes) : {len(joueurs)}",
            f"- Surfaces : {_repartition([ligne['surface'] for ligne in lignes])}",
            f"- Cadences nominales (fps) : {_repartition([ligne['fps_nominal'] for ligne in lignes])}",
            f"- Modèles de téléphone : {_repartition([ligne['telephone'] for ligne in lignes])}",
            f"- Niveaux de jeu : {_repartition([ligne['niveau_jeu'] for ligne in lignes])}",
            "- Niveaux d'annotation : "
            + _repartition([ligne["niveau_annotation"] for ligne in lignes]),
            f"- Licences : {_repartition([ligne['licence'] for ligne in lignes])}",
            f"- Matchs doublement annotés : {doubles}",
            f"- Points annotés : {totaux['points']}",
            f"- Services annotés : {totaux['services']}",
            f"- Frappes annotées : {totaux['hits']}",
            f"- Rebonds annotés : {totaux['rebonds']}",
            f"- Frames de balle annotées : {totaux['balle_frames']}",
        ]
    )


def mettre_a_jour_fiche(gt_dir: str | Path, sortie: str | Path) -> Path:
    """Remplace le bloc entre les marqueurs de `sortie` (créé avec un squelette s'il n'existe
    pas). ValueError si le fichier existe sans marqueurs."""
    sortie = Path(sortie)
    bloc = f"{MARQUEUR_DEBUT}\n{bloc_statistiques(gt_dir)}\n{MARQUEUR_FIN}"
    if not sortie.is_file():
        sortie.parent.mkdir(parents=True, exist_ok=True)
        texte = f"{ENTETE_DEFAUT}## Statistiques\n\n{bloc}\n"
    else:
        texte = sortie.read_text(encoding="utf-8")
        debut, fin = texte.find(MARQUEUR_DEBUT), texte.find(MARQUEUR_FIN)
        if debut == -1 or fin == -1 or fin < debut:
            raise ValueError(
                f"{sortie} : marqueurs {MARQUEUR_DEBUT} / {MARQUEUR_FIN} absents ou dans le désordre"
            )
        texte = texte[:debut] + bloc + texte[fin + len(MARQUEUR_FIN) :]
    sortie.write_text(texte, encoding="utf-8", newline="\n")
    return sortie
