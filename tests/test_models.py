"""Tests du registre de modèles (aucun accès réseau : URL file:// uniquement)."""

import hashlib
import re

import pytest

from tennisvision import models
from tennisvision.models import ModeleAbsent, ModeleRef

CONTENU = b"poids factices" * 1000
SOMME = hashlib.sha256(CONTENU).hexdigest()


@pytest.fixture
def source(tmp_path):
    chemin = tmp_path / "source" / "test.pt"
    chemin.parent.mkdir()
    chemin.write_bytes(CONTENU)
    return chemin


@pytest.fixture
def registre_test(monkeypatch, source):
    ref = ModeleRef(
        nom="test",
        version="1",
        fichier="test.pt",
        url=source.as_uri(),
        sha256=SOMME,
        licence="MIT",
        description="modèle de test",
    )
    monkeypatch.setattr(models, "registre", lambda: {"test": ref})
    return ref


@pytest.fixture
def cache(tmp_path):
    return tmp_path / "cache"


def test_telecharger_depuis_file_url(registre_test, cache):
    chemin = models.telecharger("test", cache)
    assert chemin == cache / "test.pt"
    assert chemin.read_bytes() == CONTENU
    assert models.verifier("test", cache)
    assert not list(cache.glob("*.part"))


def test_telecharger_somme_fausse(monkeypatch, registre_test, cache):
    faux = ModeleRef(**{**registre_test.__dict__, "sha256": "0" * 64})
    monkeypatch.setattr(models, "registre", lambda: {"test": faux})
    with pytest.raises(ValueError, match="attendue") as exc:
        models.telecharger("test", cache)
    assert SOMME in str(exc.value)
    assert not (cache / "test.pt").exists()
    assert not list(cache.glob("*.part"))


def test_telecharger_deja_present_ne_relit_pas(monkeypatch, registre_test, cache):
    appels = []
    reel = models.urllib.request.urlopen
    monkeypatch.setattr(
        models.urllib.request, "urlopen", lambda *a, **k: appels.append(a) or reel(*a, **k)
    )
    models.telecharger("test", cache)
    models.telecharger("test", cache)
    assert len(appels) == 1
    models.telecharger("test", cache, force=True)
    assert len(appels) == 2


def test_resoudre_chemin_existant(source):
    assert models.resoudre_modele(str(source)) == source


def test_resoudre_nom_present(registre_test, cache):
    models.telecharger("test", cache)
    assert models.resoudre_modele("test", cache) == cache / "test.pt"


def test_resoudre_nom_absent(registre_test, cache):
    with pytest.raises(ModeleAbsent, match="models pull test"):
        models.resoudre_modele("test", cache)
    assert issubclass(ModeleAbsent, FileNotFoundError)


def test_resoudre_nom_inconnu(registre_test, cache):
    with pytest.raises(ValueError, match="modèle inconnu") as exc:
        models.resoudre_modele("inexistant", cache)
    assert "test" in str(exc.value)


def test_resoudre_defaut_est_yolo11n(cache):
    with pytest.raises(ModeleAbsent, match="yolo11n"):
        models.resoudre_modele(None, cache)


def test_importer(registre_test, source, cache):
    chemin = models.importer("test", source, cache)
    assert chemin == cache / "test.pt"
    assert models.verifier("test", cache)


def test_importer_mauvaise_somme(registre_test, tmp_path, cache):
    mauvais = tmp_path / "mauvais.pt"
    mauvais.write_bytes(b"autre chose")
    with pytest.raises(ValueError, match="SHA-256"):
        models.importer("test", mauvais, cache)
    assert not (cache / "test.pt").exists()


def test_verifier_absent_ou_corrompu(registre_test, cache):
    assert not models.verifier("test", cache)
    cache.mkdir()
    (cache / "test.pt").write_bytes(b"corrompu")
    assert not models.verifier("test", cache)


def test_vrai_registre():
    ref = models.registre()["yolo11n"]
    assert re.fullmatch(r"[0-9a-f]{64}", ref.sha256)
    assert ref.fichier == "yolo11n.pt"
