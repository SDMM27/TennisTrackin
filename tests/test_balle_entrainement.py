"""Tests de l'affinage : jeux de données (TrackNet, amateur), perte, métriques, boucle."""

import json
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from tennisvision.balle import entrainement as ent
from tennisvision.balle.augmentations import ParametresAugmentation
from tennisvision.balle.detecteur import appliquer_affine, transformations
from tennisvision.balle.reseaux import HAUTEUR_ENTREE, LARGEUR_ENTREE, construire_reseau
from tennisvision.io.video import VideoWriter

LARGEUR, HAUTEUR = 160, 90  # 16:9, agrandi 3,2 fois dans l'espace modèle
# position de la balle (x, y) dans l'image d'origine, par frame
BALLE = [(40 + 6 * i, 30 + 3 * i) for i in range(8)]
SANS_AUGMENTATION = ParametresAugmentation().desactivees()


def image_avec_balle(position, visible=True):
    image = np.full((HAUTEUR, LARGEUR, 3), 70, np.uint8)
    if visible:
        cv2.circle(image, position, 3, (255, 255, 255), -1)
    return image


def barycentre(carte):
    ys, xs = np.nonzero(carte > 0.5)
    return xs.mean(), ys.mean()


def ecrire_tracknet(racine, parties=("game1",), n=8, visibilites=None):
    visibilites = visibilites or {}
    for partie in parties:
        dossier = racine / partie / "Clip1"
        dossier.mkdir(parents=True)
        lignes = []
        for i in range(n):
            visibilite = visibilites.get(i, 1)
            cv2.imwrite(
                str(dossier / f"{i:04d}.jpg"),
                image_avec_balle(BALLE[i], visibilite in (1, 2)),
                [cv2.IMWRITE_JPEG_QUALITY, 98],
            )
            lignes.append(
                {
                    "file name": f"{i:04d}.jpg",
                    "visibility": visibilite,
                    "x-coordinate": np.nan if visibilite == 0 else BALLE[i][0],
                    "y-coordinate": np.nan if visibilite == 0 else BALLE[i][1],
                    "status": 0 if visibilite in (0, 3) else 1,
                }
            )
        pd.DataFrame(lignes).to_csv(dossier / "Label.csv", index=False)
    return racine


def ecrire_match_amateur(gt_dir, nom="m01", n=8, invisibles=(3,)):
    dossier = gt_dir / nom
    dossier.mkdir(parents=True)
    with VideoWriter(dossier / "match.mp4", (LARGEUR, HAUTEUR), 30.0) as ecrivain:
        for i in range(n):
            ecrivain.write(image_avec_balle(BALLE[i], i not in invisibles), i * 33.0)
    lignes = [
        {
            "frame_idx": i,
            "t_ms": i * 33.0,
            "x": np.nan if i in invisibles else BALLE[i][0],
            "y": np.nan if i in invisibles else BALLE[i][1],
            "visible": 0 if i in invisibles else 1,
            "score": np.nan,
            "source": "annotation",
        }
        for i in range(n)
    ]
    pd.DataFrame(lignes).to_csv(dossier / "ball.csv", index=False)
    (gt_dir / "splits.json").write_text(
        json.dumps({"train": [nom], "val": [], "test": []}), encoding="utf-8"
    )
    return gt_dir


def attendu(i):
    aller, _ = transformations(HAUTEUR, LARGEUR)
    return appliquer_affine(BALLE[i], aller)


# --- Données TrackNet -------------------------------------------------------------------------


def test_tracknet_longueur_formes_et_pic_de_la_cible(tmp_path):
    racine = ecrire_tracknet(tmp_path / "tennis")
    clips = ent.triplets_tracknet(racine)
    assert list(clips) == ["game1/Clip1"]
    assert len(clips["game1/Clip1"]) == 6  # 8 images, fenêtres de 3 consécutives
    jeu = ent.DatasetBalle(clips["game1/Clip1"])
    x, cible, visibles, positions = jeu[0]
    assert x.shape == (9, HAUTEUR_ENTREE, LARGEUR_ENTREE) and x.dtype == torch.float32
    assert cible.shape == (3, HAUTEUR_ENTREE, LARGEUR_ENTREE)
    assert visibles.tolist() == [True, True, True]
    for i in range(3):
        assert barycentre(cible[i].numpy()) == pytest.approx(attendu(i), abs=1.0)
        assert tuple(positions[i]) == pytest.approx(attendu(i), abs=1e-3)
        assert cible[i].sum() == 21  # disque de rayon 2,5 px
    # la balle de l'image est bien là où la cible la place (même transformation affine)
    image = (x[:3].permute(1, 2, 0).numpy() * ent._ECART + ent._MOYENNE).clip(0, 1)
    assert barycentre(image.min(axis=2) > 0.9) == pytest.approx(attendu(0), abs=1.5)


def test_tracknet_balle_invisible_donne_une_carte_vide(tmp_path):
    racine = ecrire_tracknet(tmp_path / "t", visibilites={1: 0, 2: 3})
    jeu = ent.DatasetBalle(ent.triplets_tracknet(racine)["game1/Clip1"])
    _, cible, visibles, _ = jeu[0]  # images 0, 1 (absente), 2 (masquée)
    assert visibles.tolist() == [True, False, False]
    assert cible[1].sum() == 0 and cible[2].sum() == 0 and cible[0].sum() > 0


def test_tracknet_pas_temporel_et_colonnes_obligatoires(tmp_path):
    racine = ecrire_tracknet(tmp_path / "t", n=8)
    assert len(ent.triplets_tracknet(racine, pas_temporel=2)["game1/Clip1"]) == 4
    csv = racine / "game1" / "Clip1" / "Label.csv"
    pd.read_csv(csv).drop(columns=["visibility"]).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="visibility"):
        ent.triplets_tracknet(racine)


# --- Données amateur --------------------------------------------------------------------------


def test_amateur_triplets_cache_et_cibles(tmp_path, monkeypatch):
    gt = ecrire_match_amateur(tmp_path / "gt")
    cache = tmp_path / "cache"
    triplets = ent.triplets_amateur(gt, ["m01"], cache)["m01"]
    assert len(triplets) == 6
    assert len(list((cache / "m01").glob("*.jpg"))) == 8
    jeu = ent.DatasetBalle(triplets)
    x, cible, visibles, _ = jeu[2]  # images 2, 3 (non visible), 4
    assert x.shape == (9, HAUTEUR_ENTREE, LARGEUR_ENTREE)
    assert visibles.tolist() == [True, False, True]
    assert cible[1].sum() == 0
    for i, frame in ((0, 2), (2, 4)):
        assert barycentre(cible[i].numpy()) == pytest.approx(attendu(frame), abs=1.0)

    # 2e passage : plus aucun décodage de la vidéo
    def interdit(*args, **kwargs):
        raise AssertionError("la vidéo ne doit pas être relue")

    monkeypatch.setattr(ent, "iter_frames", interdit)
    assert len(ent.triplets_amateur(gt, ["m01"], cache)["m01"]) == 6


def test_amateur_pas_temporel_et_frames_non_annotees(tmp_path):
    gt = ecrire_match_amateur(tmp_path / "gt")
    assert len(ent.triplets_amateur(gt, ["m01"], tmp_path / "c", pas_temporel=2)["m01"]) == 4
    # frame 5 absente de ball.csv : les triplets qui la contiennent disparaissent
    csv = gt / "m01" / "ball.csv"
    table = pd.read_csv(csv)
    table[table["frame_idx"] != 5].to_csv(csv, index=False)
    assert len(ent.triplets_amateur(gt, ["m01"], tmp_path / "c2")["m01"]) == 3  # 0-2, 1-3, 2-4


def test_amateur_n_utilise_pas_le_split_test(tmp_path):
    gt = ecrire_match_amateur(tmp_path / "gt")
    (gt / "splits.json").write_text(json.dumps({"train": [], "val": [], "test": ["m01"]}))
    config = ent.ConfigEntrainement((f"amateur:{gt}",), tmp_path / "out")
    assert ent.construire_jeux(config) == ([], [])


# --- Perte, métriques, configuration ---------------------------------------------------------


def test_perte_wbce_formule_et_signe():
    cible = torch.zeros(1, 3, 4, 4)
    cible[0, 0, 1, 1] = 1.0
    parfait = torch.where(cible > 0, 30.0, -30.0)
    assert ent.perte_wbce(parfait, cible) < 1e-3 < ent.perte_wbce(-parfait, cible)
    logits = torch.zeros(1, 3, 4, 4)  # p = 0,5 : (1-p)^2 t log p + p^2 (1-t) log(1-p), en moyenne
    attendue = -0.25 * np.log(0.5)
    assert float(ent.perte_wbce(logits, cible)) == pytest.approx(attendue, rel=1e-5)


def test_compter_detections_protocole_tracknet():
    positions = np.array([[10.0, 10.0], [50.0, 50.0], [0.0, 0.0], [30.0, 30.0]])
    visibles = np.array([True, True, False, False])
    meilleurs = [(12.0, 10.0), (60.0, 50.0), None, (30.0, 30.0)]
    # 0 : TP ; 1 : trop loin (FP + FN) ; 2 : rien ; 3 : fausse alerte (FP)
    assert ent.compter_detections(meilleurs, positions, visibles) == (1, 2, 1)


def test_config_validation_et_valeurs_simples(tmp_path):
    config = ent.ConfigEntrainement(sources=["tracknet:/x"], dossier_sortie=str(tmp_path))
    assert config.sources == ("tracknet:/x",) and config.dossier_sortie == tmp_path
    assert config.poids_initiaux == "wasb_tennis" and config.architecture == "hrnet"
    with pytest.raises(ValueError):
        ent.ConfigEntrainement(sources=("x",), dossier_sortie=tmp_path)
    with pytest.raises(ValueError):
        ent.ConfigEntrainement(sources=("tracknet:/x",), dossier_sortie=tmp_path, epochs=0)


# --- Boucle -----------------------------------------------------------------------------------


def petit_reseau():
    return torch.nn.Sequential(
        torch.nn.Conv2d(9, 8, 3, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(8, 3, 3, padding=1)
    )


def test_entrainement_hrnet_aleatoire_deux_lots(tmp_path):
    racine = ecrire_tracknet(tmp_path / "t", parties=("game1", "game2"), n=5)
    config = ent.ConfigEntrainement(
        sources=(f"tracknet:{racine}",),
        dossier_sortie=tmp_path / "sortie",
        poids_initiaux=None,
        epochs=1,
        taille_lot=1,
        max_lots=2,
        device="cpu",
        fraction_validation=0.5,
        graine=1,
    )
    meilleur = ent.entrainer(config)
    assert meilleur == tmp_path / "sortie" / "best.pth"
    etat = torch.load(meilleur, map_location="cpu", weights_only=True)
    construire_reseau("hrnet").load_state_dict(etat, strict=True)
    rapport = json.loads((tmp_path / "sortie" / "entrainement.json").read_text(encoding="utf-8"))
    assert rapport["graine"] == 1 and rapport["triplets"] == {"entrainement": 3, "validation": 1}
    assert rapport["poids_initiaux"]["sha256"] is None
    assert rapport["config"]["epochs"] == 1 and rapport["tennisvision"]
    ligne = rapport["epochs"][0]
    assert np.isfinite(ligne["perte_entrainement"]) and np.isfinite(ligne["perte_validation"])
    assert 0.0 <= ligne["f1_validation"] <= 1.0
    assert rapport["meilleure_epoch"] == 1 and rapport["sha256_best"]


def test_surapprentissage_d_un_triplet_et_determinisme(tmp_path):
    racine = ecrire_tracknet(tmp_path / "t", n=3)

    def lancer(nom):
        config = ent.ConfigEntrainement(
            sources=(f"tracknet:{racine}",),
            dossier_sortie=tmp_path / nom,
            poids_initiaux=None,
            epochs=20,
            taille_lot=1,
            lr=1e-2,
            cosinus=False,
            device="cpu",
            fraction_validation=0.0,
            augmentation=SANS_AUGMENTATION,
        )
        torch.manual_seed(0)
        ent.entrainer(config, reseau=petit_reseau())
        rapport = json.loads((tmp_path / nom / "entrainement.json").read_text(encoding="utf-8"))
        return [e["perte_entrainement"] for e in rapport["epochs"]], tmp_path / nom / "best.pth"

    pertes, chemin = lancer("a")
    assert pertes[-1] < 0.5 * pertes[0]
    pertes_bis, chemin_bis = lancer("b")
    assert pertes == pertes_bis  # déterministe sur CPU pour une graine
    a = torch.load(chemin, weights_only=True)
    b = torch.load(chemin_bis, weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_l_epoch_renouvelle_les_augmentations(tmp_path):
    racine = ecrire_tracknet(tmp_path / "t", n=3)
    triplets = ent.triplets_tracknet(racine)["game1/Clip1"]
    jeu = ent.DatasetBalle(triplets, replace(ParametresAugmentation(), p_photometrie=1.0), graine=0)
    a = jeu[0][0]
    assert torch.equal(a, jeu[0][0])  # même (graine, époque, index) : même tirage
    jeu.set_epoch(1)
    assert not torch.equal(a, jeu[0][0])


def test_amateur_validation_suit_splits_json(tmp_path):
    gt = ecrire_match_amateur(tmp_path / "gt")
    (gt / "splits.json").write_text(json.dumps({"train": ["m01"], "val": ["m01"], "test": []}))
    config = ent.ConfigEntrainement((f"amateur:{gt}",), tmp_path / "out")
    train, val = ent.construire_jeux(config)
    assert len(train) == 6 and len(val) == 2  # validation : un triplet sur trois
    assert (tmp_path / "out" / "cache_frames" / "m01").is_dir()
