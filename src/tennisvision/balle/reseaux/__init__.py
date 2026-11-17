"""Architectures de réseaux de détection de balle (cartes de chaleur sur 3 images) et chargement."""

from pathlib import Path

import torch

from tennisvision.models import decrire_modele, resoudre_modele

LARGEUR_ENTREE = 512
HAUTEUR_ENTREE = 288
NB_IMAGES = 3
ARCHITECTURES = ("hrnet", "unet")


def architecture_du_registre(nom: str) -> str:
    """Architecture associée à un nom du registre (`wasb*` : HRNet, `tracknetv2*` : U-Net)."""
    if nom.startswith("wasb"):
        return "hrnet"
    if nom.startswith("tracknetv2"):
        return "unet"
    raise ValueError(
        f"architecture inconnue pour le modèle {nom} : précisez `architecture` "
        f"({', '.join(ARCHITECTURES)})"
    )


def construire_reseau(nom: str) -> torch.nn.Module:
    """Réseau aux poids aléatoires : `nom` est une architecture ou un nom du registre."""
    # Import différé : les modules d'architecture ne sont utiles qu'à la construction.
    if nom not in ARCHITECTURES:
        nom = architecture_du_registre(nom)
    if nom == "hrnet":
        from tennisvision.balle.reseaux.hrnet import construire_hrnet_wasb

        return construire_hrnet_wasb()
    from tennisvision.balle.reseaux.unet import TrackNetV2

    return TrackNetV2()


def _etat(chemin: Path) -> dict:
    """Charge un point de contrôle (toujours `weights_only=True`) et en extrait les poids."""
    contenu = torch.load(chemin, map_location="cpu", weights_only=True)
    etat = contenu.get("model_state_dict", contenu)
    return {(k.removeprefix("module.")): v for k, v in etat.items()}


def charger_reseau(
    nom_ou_chemin: str, device: str | torch.device = "cpu", architecture: str | None = None
) -> tuple[torch.nn.Module, dict]:
    """Charge un réseau en mode évaluation, avec la description du modèle (nom, version, sha256).

    Un nom du registre détermine l'architecture ; un chemin de fichier local exige `architecture`.
    """
    chemin = resoudre_modele(nom_ou_chemin)
    if architecture is None:
        if Path(nom_ou_chemin).is_file():
            raise ValueError(
                f"architecture requise pour un fichier local ({', '.join(ARCHITECTURES)})"
            )
        architecture = architecture_du_registre(nom_ou_chemin)
    reseau = construire_reseau(architecture)
    reseau.load_state_dict(_etat(chemin), strict=True)
    return reseau.to(device).eval(), decrire_modele(nom_ou_chemin)
