"""Tests de l'export ONNX et de la mesure de débit (dépendances et poids optionnels)."""

import numpy as np
import pytest
import torch

from tennisvision.balle.export import exporter_onnx, mesurer_debit
from tennisvision.balle.reseaux import charger_reseau
from tennisvision.models import ModeleAbsent, resoudre_modele


def poids_ou_skip(nom):
    try:
        return resoudre_modele(nom)
    except ModeleAbsent:
        pytest.skip(f"poids {nom} absents du cache")


def test_export_onnx_parite_avec_torch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")
    ort = pytest.importorskip("onnxruntime")
    poids_ou_skip("wasb_tennis")
    chemin = exporter_onnx("wasb_tennis", tmp_path / "sous" / "wasb.onnx")
    assert chemin.is_file()
    x = torch.rand(2, 9, 288, 512)  # lot de 2 : axe du lot dynamique
    reseau, _ = charger_reseau("wasb_tennis")
    with torch.no_grad():
        attendu = reseau(x).numpy()
    obtenu = ort.InferenceSession(str(chemin)).run(None, {"images": x.numpy()})[0]
    assert obtenu.shape == attendu.shape
    assert np.abs(obtenu - attendu).max() < 1e-3


def test_export_onnx_sans_dependances(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "onnx", None)
    with pytest.raises(ImportError, match=r"tennisvision\[onnx\]"):
        exporter_onnx("wasb_tennis", tmp_path / "x.onnx")


def test_mesurer_debit():
    poids_ou_skip("wasb_tennis")
    mesure = mesurer_debit("wasb_tennis", device="cpu", n_triplets=1, taille_lot=1)
    assert mesure["modele"] == "wasb_tennis"
    assert mesure["device"] == "cpu"
    assert mesure["taille_lot"] == 1
    assert mesure["fps_reseau"] > 0
    assert mesure["fps_bout_en_bout"] > 0
