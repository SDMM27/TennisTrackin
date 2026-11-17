"""Export ONNX des réseaux de balle et mesure de débit (images par seconde)."""

import time
import warnings
from pathlib import Path

import numpy as np

from tennisvision.balle.reseaux import HAUTEUR_ENTREE, LARGEUR_ENTREE, NB_IMAGES

_MESSAGE_ONNX = (
    'l\'export ONNX nécessite des dépendances optionnelles : pip install "tennisvision[onnx]"'
)


def exporter_onnx(
    modele: str, chemin_sortie: str | Path, architecture: str | None = None, opset: int = 17
) -> Path:
    """Exporte un réseau en ONNX (entrée `images` (N, 9, 288, 512), sortie `cartes` (N, 3, 288, 512)).

    L'axe du lot est dynamique ; la taille d'image est fixe. `onnx` et `onnxscript` sont requis.
    """
    import torch

    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except ImportError as erreur:
        raise ImportError(_MESSAGE_ONNX) from erreur

    from tennisvision.balle.reseaux import charger_reseau

    reseau, _ = charger_reseau(modele, "cpu", architecture)
    chemin = Path(chemin_sortie)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    exemple = torch.zeros(1, 3 * NB_IMAGES, HAUTEUR_ENTREE, LARGEUR_ENTREE)
    # Exporteur historique (TorchScript) : l'exporteur `dynamo` échoue sous une console cp1252.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.onnx.export(
            reseau,
            (exemple,),
            str(chemin),
            input_names=["images"],
            output_names=["cartes"],
            dynamic_axes={"images": {0: "lot"}, "cartes": {0: "lot"}},
            opset_version=opset,
            dynamo=False,
        )
    return chemin


def mesurer_debit(
    modele: str,
    device: str | None = None,
    n_triplets: int = 8,
    taille_lot: int = 4,
    architecture: str | None = None,
) -> dict:
    """Débit en images par seconde : réseau seul, puis prétraitement + réseau + post-traitement.

    La mesure porte sur `n_triplets` triplets (soit 3 x `n_triplets` images) en entrée aléatoire
    (réseau seul) ou en images synthétiques 1920x1080 (bout en bout), après une passe d'échauffement.
    """
    import torch

    from tennisvision.balle.detecteur import DetecteurBalle, resoudre_device
    from tennisvision.balle.reseaux import charger_reseau

    cible = resoudre_device(device)
    reseau, _ = charger_reseau(modele, cible, architecture)

    def synchroniser():
        if cible.startswith("cuda"):
            torch.cuda.synchronize()

    entree = torch.rand(taille_lot, 3 * NB_IMAGES, HAUTEUR_ENTREE, LARGEUR_ENTREE, device=cible)
    with torch.inference_mode():
        reseau(entree)  # échauffement
        synchroniser()
        debut = time.perf_counter()
        n_lots = max(1, -(-n_triplets // taille_lot))
        for _ in range(n_lots):
            reseau(entree)
        synchroniser()
    fps_reseau = n_lots * taille_lot * NB_IMAGES / (time.perf_counter() - debut)

    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8) for _ in range(NB_IMAGES)]
    n_images = n_triplets * NB_IMAGES

    def passe(n: int) -> None:
        detecteur = DetecteurBalle(
            modele, device=cible, taille_lot=taille_lot, architecture=architecture, reseau=reseau
        )
        for i in range(n):
            detecteur.pousser(i, images[i % NB_IMAGES])
        detecteur.terminer()

    passe(NB_IMAGES * taille_lot)  # échauffement du chemin complet
    synchroniser()
    debut = time.perf_counter()
    passe(n_images)
    synchroniser()
    fps_bout_en_bout = n_images / (time.perf_counter() - debut)

    return {
        "modele": modele,
        "device": cible,
        "fps_reseau": fps_reseau,
        "fps_bout_en_bout": fps_bout_en_bout,
        "taille_lot": taille_lot,
        "n_triplets": n_triplets,
        "resolution_video": "1920x1080",
    }
