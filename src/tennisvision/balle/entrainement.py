"""Affinage du détecteur de balle (WASB HRNet / TrackNetV2) sur des triplets d'images annotés.

Reprend la recette d'évaluation et de données de WASB-SBDT (https://github.com/nttcom/WASB-SBDT,
licence MIT, Copyright (c) 2023 NTT Communications Corporation), dont le dépôt ne publie pas le
code d'entraînement :
- données : `src/dataloaders/dataset_loader.py` (`ImageDataset`) et `src/datasets/tennis.py` ; images
  ramenées à 512x288 par la même transformation affine qu'à l'inférence, normalisation ImageNet ;
- cible : carte binaire (`gen_binary_map` de `src/utils/heatmap.py`, rayon 2,5 px,
  `configs/dataloader/default.yaml`, `binary_fixed_size`) centrée sur la position entière de la
  balle dans l'espace modèle, vide si la balle n'est pas visible ;
- perte : `WBCELoss` de `src/losses/wbce.py` (`configs/loss/hm_wbce.yaml`), sur des probabilités
  bornées à [1e-4, 1 - 1e-4] (`_sigmoid` de `src/utils/utils.py`) ;
- optimiseur : Adam (`configs/optimizer/adam_multistep.yaml`, lr 1e-3) ; la décroissance par
  paliers y est remplacée ici par une décroissance en cosinus, facultative.

Deux formats de données : l'arborescence TrackNet tennis (`<racine>/<partie>/<Clip>/` avec les
images et `Label.csv`) et le jeu amateur du Lot 2 (`<gt_dir>/<match>/ball.csv` + la vidéo).
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tennisvision import __version__
from tennisvision.balle.augmentations import (
    Augmenteur,
    Echantillon,
    ParametresAugmentation,
)
from tennisvision.balle.detecteur import (
    ECART_TYPE_IMAGENET,
    MOYENNE_IMAGENET,
    appliquer_affine,
    extraire_candidats,
    resoudre_device,
    transformations,
)
from tennisvision.balle.reseaux import (
    ARCHITECTURES,
    HAUTEUR_ENTREE,
    LARGEUR_ENTREE,
    NB_IMAGES,
    charger_reseau,
    construire_reseau,
)
from tennisvision.eval.banc import lire_splits
from tennisvision.io.video import iter_frames, probe

logger = logging.getLogger(__name__)

RAYON_CIBLE = 2.5  # px, espace modèle (WASB : `sigmas: [2.5]`, carte binaire)
DRAPEAUX_VISIBLES_TRACKNET = (1, 2)  # visibilité 0 : absente, 3 : masquée -> non visible
COLONNES_TRACKNET = ("file name", "visibility", "x-coordinate", "y-coordinate")
EXTENSIONS_VIDEO = (".mp4", ".mov", ".m4v", ".mkv", ".avi")
TOLERANCE_PX = 4.0  # F1 de validation, en pixels de l'espace modèle
QUALITE_CACHE = 95
IDENTITE = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# --- Triplets -----------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Triplet:
    """3 images (fichiers) et la balle de chacune dans l'espace modèle.

    `aller` : transformation affine image -> modèle à appliquer à la lecture (`None` quand le fichier
    est déjà à la taille du modèle, comme dans le cache du jeu amateur).
    """

    images: tuple[Path, ...]
    positions: tuple[tuple[float, float], ...]
    visibles: tuple[bool, ...]
    aller: np.ndarray | None = field(default=None, compare=False)


def _fenetres(
    annotations: dict[int, tuple[Path, float, float, bool]],
    pas_temporel: int,
    aller: np.ndarray | None,
) -> list[Triplet]:
    """Triplets (t, t+k, t+2k) dont les 3 images sont annotées ; positions déjà dans l'espace modèle."""
    triplets = []
    for t in sorted(annotations):
        indices = [t + i * pas_temporel for i in range(NB_IMAGES)]
        if not all(i in annotations for i in indices):
            continue
        lignes = [annotations[i] for i in indices]
        triplets.append(
            Triplet(
                tuple(ligne[0] for ligne in lignes),
                tuple((ligne[1], ligne[2]) for ligne in lignes),
                tuple(ligne[3] for ligne in lignes),
                aller,
            )
        )
    return triplets


def triplets_tracknet(racine: str | Path, pas_temporel: int = 1) -> dict[str, list[Triplet]]:
    """Triplets de chaque clip `<racine>/<partie>/<Clip>/` (format TrackNet tennis de WASB).

    `Label.csv` : `file name, visibility, x-coordinate, y-coordinate, status`. Visibilité 1 et 2 :
    balle visible ; 0 (absente) et 3 (masquée) : non visible. Clés : `partie/Clip`.
    """
    racine = Path(racine)
    if not racine.is_dir():
        raise FileNotFoundError(f"Dossier TrackNet introuvable : {racine}")
    clips: dict[str, list[Triplet]] = {}
    for partie in sorted(p for p in racine.iterdir() if p.is_dir()):
        for dossier in sorted(c for c in partie.iterdir() if (c / "Label.csv").is_file()):
            table = pd.read_csv(dossier / "Label.csv")
            manquantes = [c for c in COLONNES_TRACKNET if c not in table.columns]
            if manquantes:
                raise ValueError(f"{dossier / 'Label.csv'} : colonnes manquantes {manquantes}")
            images = {
                int(Path(nom).stem): dossier / nom
                for nom in table["file name"].astype(str)
                if (dossier / nom).is_file()
            }
            if not images:
                continue
            premiere = cv2.imread(str(next(iter(images.values()))), cv2.IMREAD_COLOR)
            aller, _ = transformations(*premiere.shape[:2])
            annotations = {}
            for nom, visibilite, x, y in zip(
                table["file name"].astype(str),
                table["visibility"],
                table["x-coordinate"],
                table["y-coordinate"],
                strict=True,
            ):
                fid = int(Path(nom).stem)
                if fid not in images:
                    continue
                visible = int(visibilite) in DRAPEAUX_VISIBLES_TRACKNET and not np.isnan(x + y)
                position = (
                    appliquer_affine((float(x), float(y)), aller) if visible else (-1.0, -1.0)
                )
                annotations[fid] = (images[fid], *position, visible)
            triplets = _fenetres(annotations, pas_temporel, aller)
            if triplets:
                clips[f"{partie.name}/{dossier.name}"] = triplets
    return clips


def trouver_video(dossier_match: Path, videos_dir: Path | None = None) -> Path:
    """Vidéo d'un match : `<videos_dir>/<match>.*`, puis `<match>/<match>.*`, `match.*`, `video.*`,
    puis l'unique fichier vidéo du dossier du match."""
    nom = dossier_match.name
    for dossier, racine in [(videos_dir, nom), (dossier_match, nom), (dossier_match, "match"),
                            (dossier_match, "video")]:  # fmt: skip
        if dossier is None:
            continue
        for extension in EXTENSIONS_VIDEO:
            if (dossier / f"{racine}{extension}").is_file():
                return dossier / f"{racine}{extension}"
    candidats = [p for p in dossier_match.glob("*") if p.suffix.lower() in EXTENSIONS_VIDEO]
    if len(candidats) == 1:
        return candidats[0]
    raise FileNotFoundError(
        f"vidéo du match {nom} introuvable (cherchée dans {videos_dir or dossier_match})"
    )


def preparer_cache(
    video: Path, dossier: Path, frames_utiles: set[int], aller: np.ndarray
) -> dict[int, Path]:
    """Décode une fois les `frames_utiles` de la vidéo, ramenées à 512x288, en JPEG (qualité 95).

    Les passages suivants relisent le cache sans décoder la vidéo. Le cache est invalidé quand la
    vidéo change (taille ou date de modification).
    """
    dossier.mkdir(parents=True, exist_ok=True)
    stat = video.stat()
    signature = {
        "video": video.name,
        "taille": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "qualite": QUALITE_CACHE,
    }
    fichier_meta = dossier / "meta.json"
    if not fichier_meta.is_file() or json.loads(fichier_meta.read_text()) != signature:
        for ancien in dossier.glob("*.jpg"):
            ancien.unlink()
        fichier_meta.write_text(json.dumps(signature))
    chemins = {i: dossier / f"{i:07d}.jpg" for i in frames_utiles}
    a_faire = {i for i, c in chemins.items() if not c.is_file()}
    if a_faire:
        logger.info("cache de %d images pour %s", len(a_faire), video.name)
        for frame in iter_frames(video, min(a_faire), max(a_faire)):
            if frame.idx in a_faire:
                reduite = cv2.warpAffine(
                    frame.image, aller, (LARGEUR_ENTREE, HAUTEUR_ENTREE), flags=cv2.INTER_LINEAR
                )
                cv2.imwrite(
                    str(chemins[frame.idx]), reduite, [cv2.IMWRITE_JPEG_QUALITY, QUALITE_CACHE]
                )
    return chemins


def triplets_amateur(
    gt_dir: str | Path,
    matchs: list[str],
    dossier_cache: str | Path,
    pas_temporel: int = 1,
    videos_dir: str | Path | None = None,
    interpolees: bool = True,
) -> dict[str, list[Triplet]]:
    """Triplets des matchs du jeu amateur (`<gt_dir>/<match>/ball.csv` + vidéo), clés = matchs.

    Seules les frames présentes dans `ball.csv` sont des cibles (`visible = 0` : balle absente) ;
    un triplet est valide si ses 3 images le sont. `interpolees=False` écarte les lignes de source
    `interpolated` (positions interpolées entre deux points clés plutôt que posées à la main).
    """
    gt_dir = Path(gt_dir)
    resultat: dict[str, list[Triplet]] = {}
    for match in matchs:
        dossier_match = gt_dir / match
        csv = dossier_match / "ball.csv"
        if not csv.is_file():
            raise FileNotFoundError(f"ball.csv introuvable : {csv}")
        table = pd.read_csv(csv)
        if not interpolees and "source" in table.columns:
            table = table[table["source"] != "interpolated"]
        video = trouver_video(dossier_match, Path(videos_dir) if videos_dir else None)
        info = probe(video)
        aller, _ = transformations(info.height, info.width)
        table = table[table["frame_idx"] < info.n_frames]
        utiles = set(table["frame_idx"].astype(int))  # le cache couvre toutes les frames annotées
        chemins = preparer_cache(video, Path(dossier_cache) / match, utiles, aller)
        annotations = {}
        for i, x, y, visible in zip(
            table["frame_idx"].astype(int),
            table["x"],
            table["y"],
            table["visible"],
            strict=True,
        ):
            est_visible = int(visible) == 1 and not np.isnan(x + y)
            position = (
                appliquer_affine((float(x), float(y)), aller) if est_visible else (-1.0, -1.0)
            )
            annotations[i] = (chemins[i], *position, est_visible)
        triplets = _fenetres(annotations, pas_temporel, None)
        if triplets:
            resultat[match] = triplets
    return resultat


# --- Jeu de données -------------------------------------------------------------------------------


def cible_binaire(positions: np.ndarray, visibles: np.ndarray) -> np.ndarray:
    """Cibles (3, H, W) : disque de rayon 2,5 px centré sur la position entière (WASB `gen_binary_map`).

    Une balle non visible, ou hors de l'image, donne une carte vide.
    """
    cible = np.zeros((len(positions), HAUTEUR_ENTREE, LARGEUR_ENTREE), np.float32)
    marge = int(np.ceil(RAYON_CIBLE))
    for i, ((x, y), visible) in enumerate(zip(positions, visibles, strict=True)):
        cx, cy = int(x), int(y)  # troncature, comme `astype(np.int32)` de WASB
        if not visible or not (0 <= cx < LARGEUR_ENTREE and 0 <= cy < HAUTEUR_ENTREE):
            continue
        x0, x1 = max(0, cx - marge), min(LARGEUR_ENTREE, cx + marge + 1)
        y0, y1 = max(0, cy - marge), min(HAUTEUR_ENTREE, cy + marge + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cible[i, y0:y1, x0:x1] = ((yy - cy) ** 2 + (xx - cx) ** 2 <= RAYON_CIBLE**2).astype(
            np.float32
        )
    return cible


def _lire_image(chemin: Path, aller: np.ndarray | None) -> np.ndarray:
    """Image RVB uint8 de 288x512 (transformation affine de WASB si le fichier n'y est pas déjà)."""
    bgr = cv2.imread(str(chemin), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"image illisible : {chemin}")
    if aller is not None:
        bgr = cv2.warpAffine(bgr, aller, (LARGEUR_ENTREE, HAUTEUR_ENTREE), flags=cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


_MOYENNE = np.array(MOYENNE_IMAGENET, np.float32)
_ECART = np.array(ECART_TYPE_IMAGENET, np.float32)


def normaliser(frames: tuple[np.ndarray, ...]) -> np.ndarray:
    """3 images RVB uint8 (H, W, 3) -> (9, H, W) float32 normalisé ImageNet (comme l'inférence)."""
    x = (np.stack(frames).astype(np.float32) / 255.0 - _MOYENNE) / _ECART  # (3, H, W, 3)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2).reshape(NB_IMAGES * 3, *x.shape[1:3]))


class DatasetBalle:
    """Dataset PyTorch de triplets : `(images (9,H,W), cible (3,H,W), visibles (3,), positions (3,2))`.

    `positions` sont en pixels de l'espace modèle (après augmentation). Avec `augmentation`, chaque
    échantillon est tiré avec un générateur dérivé de `(graine, epoch, index)` : le résultat ne
    dépend ni du nombre de workers ni de l'ordre des lots ; `set_epoch` renouvelle les tirages.
    """

    def __init__(
        self,
        triplets: list[Triplet],
        augmentation: ParametresAugmentation | None = None,
        graine: int = 0,
    ):
        self.triplets = triplets
        self.augmenteur = Augmenteur(augmentation) if augmentation is not None else None
        self.graine = graine
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.triplets)

    def echantillon(self, index: int) -> Echantillon:
        """Le triplet `index` en images RVB uint8 de l'espace modèle, augmenté s'il y a lieu."""
        t = self.triplets[index]
        ech = Echantillon(
            tuple(_lire_image(chemin, t.aller) for chemin in t.images),
            np.array(t.positions, dtype=np.float64),
            np.array(t.visibles, dtype=bool),
        )
        if self.augmenteur is not None:
            ech = self.augmenteur(ech, np.random.default_rng([self.graine, self.epoch, index]))
        return ech

    def __getitem__(self, index: int):
        import torch

        ech = self.echantillon(index)
        return (
            torch.from_numpy(normaliser(ech.frames)),
            torch.from_numpy(cible_binaire(ech.points, ech.visibles)),
            torch.from_numpy(ech.visibles.copy()),
            torch.from_numpy(ech.points.astype(np.float32)),
        )


# --- Perte et métriques -----------------------------------------------------------------------


def perte_wbce(logits, cible):
    """Entropie croisée pondérée de WASB (`WBCELoss`, équivalente à la perte focale de gamma 2).

    `logits` (B, 3, H, W) ; la sigmoïde est bornée à [1e-4, 1 - 1e-4] comme `_sigmoid` upstream.
    """
    import torch

    p = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1 - 1e-4)
    return torch.mean(
        -((1 - p) ** 2 * cible * torch.log(p) + p**2 * (1 - cible) * torch.log(1 - p))
    )


def compter_detections(
    meilleurs: list[tuple[float, float] | None],
    positions: np.ndarray,
    visibles: np.ndarray,
    tolerance: float = TOLERANCE_PX,
) -> tuple[int, int, int]:
    """(vrais positifs, faux positifs, faux négatifs) : protocole TrackNet de `eval/balle.py`.

    Une détection trop éloignée d'une balle visible compte à la fois comme FP et comme FN.
    """
    tp = fp = fn = 0
    for meilleur, position, visible in zip(meilleurs, positions, visibles, strict=True):
        juste = (
            meilleur is not None
            and bool(visible)
            and float(np.hypot(meilleur[0] - position[0], meilleur[1] - position[1])) <= tolerance
        )
        tp += juste
        fp += meilleur is not None and not juste
        fn += bool(visible) and not juste
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    rappel = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0
    return {"precision": precision, "rappel": rappel, "f1": f1}


# --- Configuration et boucle -------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigEntrainement:
    """Réglages de l'affinage ; ne contient que des valeurs simples (sérialisable en JSON).

    `sources` : chaînes `tracknet:<racine>` (arborescence TrackNet tennis) ou `amateur:<gt_dir>`
    (jeu amateur, découpé par `splits.json` : `train` et `val`, jamais `test`). `poids_initiaux` :
    nom du registre, chemin de fichier, ou `None` (poids aléatoires).
    """

    sources: tuple[str, ...]
    dossier_sortie: Path
    architecture: str = "hrnet"
    poids_initiaux: str | None = "wasb_tennis"
    epochs: int = 10
    taille_lot: int = 8
    lr: float = 1e-3
    cosinus: bool = True
    device: str | None = None
    graine: int = 0
    fraction_validation: float = 0.1  # sources TrackNet ; le jeu amateur suit `splits.json`
    pas_temporel: int = 1  # 2 pour des vidéos à 60 fps (triplets t, t+2, t+4)
    nb_workers: int = 0
    seuil_score: float = 0.5
    max_lots: int | None = None  # lots par époque (essais rapides)
    dossier_cache: Path | None = None  # défaut : <dossier_sortie>/cache_frames
    videos_dir: Path | None = None
    interpolees: bool = True
    augmentation: ParametresAugmentation = field(default_factory=ParametresAugmentation)

    def __post_init__(self):
        sources = (self.sources,) if isinstance(self.sources, str) else tuple(self.sources)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "dossier_sortie", Path(self.dossier_sortie))
        for nom in ("dossier_cache", "videos_dir"):
            if getattr(self, nom) is not None:
                object.__setattr__(self, nom, Path(getattr(self, nom)))
        if not sources or any(s.partition(":")[0] not in ("tracknet", "amateur") for s in sources):
            raise ValueError("sources : « tracknet:<racine> » ou « amateur:<gt_dir> » attendu")
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"architecture inconnue : {self.architecture} ({ARCHITECTURES})")
        if self.epochs < 1 or self.taille_lot < 1 or self.pas_temporel < 1:
            raise ValueError("epochs, taille_lot et pas_temporel doivent être >= 1")
        if not 0.0 <= self.fraction_validation < 1.0:
            raise ValueError("fraction_validation doit être dans [0, 1[")


def _separer(
    clips: dict[str, list[Triplet]], fraction: float, graine: int
) -> tuple[dict[str, list[Triplet]], dict[str, list[Triplet]]]:
    """Découpe TrackNet : par partie (match) s'il y en a plusieurs, sinon par clip."""
    parties = sorted({cle.split("/")[0] for cle in clips})
    unites = (
        {cle: cle.split("/")[0] for cle in clips} if len(parties) > 1 else {c: c for c in clips}
    )
    noms = sorted(set(unites.values()))
    n_val = round(fraction * len(noms))
    if fraction > 0 and len(noms) >= 2:
        n_val = max(1, n_val)
    ordre = np.random.default_rng(graine).permutation(len(noms))
    val = {noms[i] for i in ordre[:n_val]}
    return (
        {c: t for c, t in clips.items() if unites[c] not in val},
        {c: t for c, t in clips.items() if unites[c] in val},
    )


def construire_jeux(config: ConfigEntrainement) -> tuple[list[Triplet], list[Triplet]]:
    """Triplets d'entraînement et de validation de toutes les sources.

    La validation prend un triplet sur trois (images prédites une seule fois, comme à l'inférence).
    """
    train: list[Triplet] = []
    val: list[Triplet] = []
    cache = config.dossier_cache or config.dossier_sortie / "cache_frames"
    for source in config.sources:
        format_, _, chemin = source.partition(":")
        if format_ == "tracknet":
            clips = triplets_tracknet(chemin, config.pas_temporel)
            clips_train, clips_val = _separer(clips, config.fraction_validation, config.graine)
        else:
            splits = lire_splits(Path(chemin))
            options = {
                "dossier_cache": cache,
                "pas_temporel": config.pas_temporel,
                "videos_dir": config.videos_dir,
                "interpolees": config.interpolees,
            }
            clips_train = triplets_amateur(chemin, splits.get("train", []), **options)
            clips_val = triplets_amateur(chemin, splits.get("val", []), **options)
        for triplets in clips_train.values():
            train.extend(triplets)
        for triplets in clips_val.values():
            val.extend(triplets[::NB_IMAGES])
    return train, val


def _meilleurs_candidats(cartes: np.ndarray, seuil: float) -> list[tuple[float, float] | None]:
    """Meilleur candidat (score maximal) de chaque carte (N, H, W), en pixels de l'espace modèle."""
    sortie = []
    for i, carte in enumerate(cartes):
        candidats = extraire_candidats(carte, IDENTITE, seuil, i)
        meilleur = max(candidats, key=lambda c: c.score, default=None)
        sortie.append(None if meilleur is None else (meilleur.x, meilleur.y))
    return sortie


def evaluer(reseau, chargeur, device: str, seuil: float, max_lots: int | None = None) -> dict:
    """Perte moyenne et précision / rappel / F1 à 4 px (espace modèle, meilleur candidat par image)."""
    import torch

    reseau.eval()
    pertes, tp, fp, fn = [], 0, 0, 0
    with torch.inference_mode():
        for n, (x, cible, visibles, positions) in enumerate(chargeur):
            if max_lots is not None and n >= max_lots:
                break
            logits = reseau(x.to(device))
            pertes.append(float(perte_wbce(logits, cible.to(device))))
            cartes = torch.sigmoid(logits).cpu().numpy().reshape(-1, HAUTEUR_ENTREE, LARGEUR_ENTREE)
            c = compter_detections(
                _meilleurs_candidats(cartes, seuil),
                positions.numpy().reshape(-1, 2),
                visibles.numpy().reshape(-1),
            )
            tp, fp, fn = tp + c[0], fp + c[1], fn + c[2]
    return {"perte": float(np.mean(pertes)) if pertes else float("nan"), **_f1(tp, fp, fn)}


def _sha256(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        while bloc := f.read(1 << 20):
            h.update(bloc)
    return h.hexdigest()


def _json_defaut(objet):
    if isinstance(objet, Path):
        return str(objet)
    raise TypeError(f"non sérialisable : {type(objet)}")


def entrainer(config: ConfigEntrainement, reseau=None) -> Path:
    """Affine le réseau ; écrit `best.pth` (state_dict seul) et `entrainement.json` ; renvoie `best.pth`.

    `best.pth` est la meilleure époque : F1 de validation maximal (à égalité, perte de validation
    minimale) ; sans validation, perte d'entraînement minimale. `reseau` permet d'injecter un module
    déjà construit (tests). Déterministe sur CPU pour une graine donnée.
    """
    import torch
    from torch.utils.data import DataLoader

    device = resoudre_device(config.device)
    torch.manual_seed(config.graine)
    sortie = config.dossier_sortie
    sortie.mkdir(parents=True, exist_ok=True)

    train, val = construire_jeux(config)
    if not train:
        raise ValueError("aucun triplet d'entraînement (3 images annotées consécutives requises)")
    logger.info("%d triplets d'entraînement, %d de validation", len(train), len(val))

    poids = {"nom": None, "version": None, "sha256": None}
    if reseau is None:
        if config.poids_initiaux is None:
            reseau = construire_reseau(config.architecture)
        else:
            reseau, poids = charger_reseau(config.poids_initiaux, device, config.architecture)
    reseau = reseau.to(device)

    jeu_train = DatasetBalle(train, config.augmentation, config.graine)
    jeu_val = DatasetBalle(val)
    chargeur_val = DataLoader(jeu_val, config.taille_lot, num_workers=config.nb_workers)
    optimiseur = torch.optim.Adam(reseau.parameters(), lr=config.lr)
    planning = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimiseur, T_max=config.epochs)
        if config.cosinus
        else None
    )

    meilleur = sortie / "best.pth"
    historique: list[dict] = []
    meilleure_cle: tuple | None = None
    meilleure_epoch = None

    def ecrire_rapport() -> None:
        rapport = {
            "tennisvision": __version__,
            "torch": torch.__version__,
            "graine": config.graine,
            "device": device,
            "config": asdict(config),
            "poids_initiaux": poids,
            "triplets": {"entrainement": len(train), "validation": len(val)},
            "meilleure_epoch": meilleure_epoch,
            "sha256_best": _sha256(meilleur) if meilleur.is_file() else None,
            "epochs": historique,
        }
        (sortie / "entrainement.json").write_text(
            json.dumps(rapport, indent=2, ensure_ascii=False, default=_json_defaut),
            encoding="utf-8",
        )

    for epoch in range(config.epochs):
        debut = time.monotonic()
        jeu_train.set_epoch(epoch)
        generateur = torch.Generator().manual_seed(config.graine + epoch)
        chargeur = DataLoader(
            jeu_train,
            config.taille_lot,
            shuffle=True,
            generator=generateur,
            num_workers=config.nb_workers,
        )
        reseau.train()
        pertes = []
        for n, (x, cible, _visibles, _positions) in enumerate(chargeur):
            if config.max_lots is not None and n >= config.max_lots:
                break
            perte = perte_wbce(reseau(x.to(device)), cible.to(device))
            if not torch.isfinite(perte):
                raise RuntimeError(f"perte non finie à l'époque {epoch + 1}, lot {n + 1}")
            optimiseur.zero_grad()
            perte.backward()
            optimiseur.step()
            pertes.append(float(perte.detach()))
        ligne = {
            "epoch": epoch + 1,
            "lr": optimiseur.param_groups[0]["lr"],
            "perte_entrainement": float(np.mean(pertes)),
        }
        if planning is not None:
            planning.step()
        if len(jeu_val):
            validation = evaluer(reseau, chargeur_val, device, config.seuil_score, config.max_lots)
            ligne.update({f"{k}_validation": v for k, v in validation.items()})
            cle = (validation["f1"], -validation["perte"])
        else:
            cle = (-ligne["perte_entrainement"],)
        ligne["duree_s"] = round(time.monotonic() - debut, 2)
        historique.append(ligne)
        if meilleure_cle is None or cle > meilleure_cle:
            meilleure_cle, meilleure_epoch = cle, epoch + 1
            torch.save({k: v.detach().cpu() for k, v in reseau.state_dict().items()}, meilleur)
        ecrire_rapport()
        logger.info("époque %d : %s", epoch + 1, ligne)
    return meilleur
