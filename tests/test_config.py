"""Tests de la résolution des chemins (explicite -> environnement -> défaut)."""

import inspect
from pathlib import Path

import pytest

from tennisvision import config


@pytest.fixture(autouse=True)
def _sans_variables(monkeypatch):
    for variable in (config.ENV_MODELS_DIR, config.ENV_OUTPUT_DIR, config.ENV_TENNISET_DIR):
        monkeypatch.delenv(variable, raising=False)


def test_models_dir_ordre_de_resolution(monkeypatch, tmp_path):
    defaut = config.models_dir()
    assert defaut.name == "models"
    assert "tennisvision" in defaut.parts

    monkeypatch.setenv(config.ENV_MODELS_DIR, str(tmp_path / "env"))
    assert config.models_dir() == tmp_path / "env"
    assert config.models_dir(tmp_path / "explicite") == tmp_path / "explicite"


def test_output_root_ordre_de_resolution(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert config.output_root() == tmp_path / "tennisvision-sorties"

    monkeypatch.setenv(config.ENV_OUTPUT_DIR, str(tmp_path / "env"))
    assert config.output_root() == tmp_path / "env"
    assert config.output_root(tmp_path / "explicite") == tmp_path / "explicite"


def test_tenniset_dir_ordre_de_resolution(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert config.tenniset_dir() == tmp_path / "data"

    monkeypatch.setenv(config.ENV_TENNISET_DIR, str(tmp_path / "env"))
    assert config.tenniset_dir() == tmp_path / "env"
    assert config.tenniset_dir(tmp_path / "explicite") == tmp_path / "explicite"


def test_defauts_suivent_le_dossier_courant(monkeypatch, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    sortie_a, tenniset_a = config.output_root(), config.tenniset_dir()
    monkeypatch.chdir(tmp_path / "b")
    assert config.output_root() != sortie_a
    assert config.tenniset_dir() != tenniset_a


def test_fichiers_derives_de_tenniset(tmp_path):
    assert config.points_file(tmp_path) == tmp_path / "annotations" / "points.txt"
    assert config.tenniset_video(tmp_path, "V006") == tmp_path / "videos" / "V006.mp4"


def test_aucun_chemin_vers_le_depot():
    source = inspect.getsource(config)
    assert "parents[" not in source
    assert "__file__" not in source
    assert not hasattr(config, "ROOT_DIR")
    assert Path(config.__file__).parent not in config.models_dir().parents
