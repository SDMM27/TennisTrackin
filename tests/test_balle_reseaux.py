"""Tests des architectures de balle (poids aléatoires ; poids réels si présents dans le cache)."""

import pytest
import torch

from tennisvision.balle.reseaux import (
    _etat,
    architecture_du_registre,
    charger_reseau,
    construire_reseau,
)
from tennisvision.models import ModeleAbsent, resoudre_modele

NOMS = ["wasb_tennis", "tracknetv2_tennis"]


def poids_ou_skip(nom: str):
    try:
        return resoudre_modele(nom)
    except ModeleAbsent:
        pytest.skip(f"poids {nom} absents du cache")


@pytest.mark.parametrize("nom", NOMS)
def test_forme_sortie_poids_aleatoires(nom):
    reseau = construire_reseau(nom).eval()
    with torch.no_grad():
        sortie = reseau(torch.rand(2, 9, 288, 512))
    assert sortie.shape == (2, 3, 288, 512)


def test_architecture_du_registre():
    assert architecture_du_registre("wasb_tennis") == "hrnet"
    assert architecture_du_registre("tracknetv2_tennis") == "unet"
    with pytest.raises(ValueError, match="architecture"):
        architecture_du_registre("yolo11n")


def test_fichier_local_exige_architecture(tmp_path):
    fichier = tmp_path / "poids.pt"
    torch.save(construire_reseau("unet").state_dict(), fichier)
    with pytest.raises(ValueError, match="architecture requise"):
        charger_reseau(str(fichier))
    reseau, description = charger_reseau(str(fichier), architecture="unet")
    assert not reseau.training
    assert description["nom"] == "poids.pt"


def test_etat_extrait_model_state_dict_et_prefixe_module(tmp_path):
    etat = construire_reseau("unet").state_dict()
    fichier = tmp_path / "ckpt.pth.tar"
    torch.save({"model_state_dict": {f"module.{k}": v for k, v in etat.items()}}, fichier)
    assert set(_etat(fichier)) == set(etat)


@pytest.mark.parametrize("nom", NOMS)
def test_poids_reels_compatibles_strict(nom):
    chemin = poids_ou_skip(nom)
    reseau = construire_reseau(nom)
    attendu = reseau.state_dict()
    etat = _etat(chemin)
    assert set(etat) == set(attendu)
    assert all(etat[k].shape == attendu[k].shape for k in attendu)
    reseau.load_state_dict(etat, strict=True)
