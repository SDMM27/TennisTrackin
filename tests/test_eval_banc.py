"""Tests du banc d'évaluation (orchestration, agrégation, rapport)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import tennisvision
from tennisvision.eval import banc
from tennisvision.eval.banc import (
    charger_verites_amateur,
    ecrire_rapport,
    evaluer,
    f1_principaux,
    plis_leave_one_out,
)
from tennisvision.validate import valider_fichier

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
AMATEUR = FIXTURES / "amateur"


def test_fixtures_valides_au_schema():
    for dossier in (AMATEUR / "M1",):
        for fichier in ("events.json", "court.json", "ball.csv", "players.csv"):
            assert valider_fichier(dossier / fichier) == [], fichier


def test_plis_leave_one_out():
    assert plis_leave_one_out(["b", "a", "c"]) == [
        (["b", "c"], "a"),
        (["a", "c"], "b"),
        (["a", "b"], "c"),
    ]
    assert plis_leave_one_out(["a"]) == [([], "a")]
    assert plis_leave_one_out([]) == []


@pytest.fixture
def verites():
    return charger_verites_amateur(AMATEUR, "test")


def test_chargement_amateur(verites):
    v = verites["M1"]
    assert v["hauteur_image"] == 720  # lu dans court.json
    assert len(v["events"]["events"]) == 10
    assert len(v["ball"]) == 20
    assert len(v["players"]) == 10
    assert v["court"]["image_size"] == [1280, 720]


def test_chargement_amateur_erreurs(tmp_path):
    with pytest.raises(FileNotFoundError, match="splits.json"):
        charger_verites_amateur(tmp_path, "test")
    (tmp_path / "splits.json").write_text('{"test": ["M9"], "val": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="absent"):
        charger_verites_amateur(tmp_path, "train")
    with pytest.raises(ValueError, match="Aucun match"):
        charger_verites_amateur(tmp_path, "val")
    with pytest.raises(FileNotFoundError, match="M9"):
        charger_verites_amateur(tmp_path, "test")


def test_prediction_egale_verite(verites):
    rapport = evaluer(verites, AMATEUR, dataset="amateur", split="test")
    assert rapport["avertissements"] == []
    ev = rapport["agrege"]["evenements"]
    assert ev["frappes"]["tolerances"]["2"]["f1"] == 1.0
    assert ev["frappes"]["tolerances"]["5"]["f1"] == 1.0
    assert ev["services"]["f1"] == 1.0
    assert ev["rebonds"]["tolerances"]["5"]["f1"] == 1.0
    assert ev["rebonds"]["erreur_position_cm"]["moyenne"] == 0
    assert ev["points"]["f1"] == 1.0
    assert ev["points"]["jeux"]["pct_jeux_score_juste"]["valeur"] == 1.0
    assert rapport["agrege"]["balle"]["seuils"]["4"]["f1"] == 1.0
    assert rapport["agrege"]["court"]["erreur_cm"]["max"] == pytest.approx(0, abs=1e-6)
    assert rapport["agrege"]["joueurs"]["idf1"] == 1.0
    assert all(i["erreur_abs_max"] in (0, None) for i in rapport["agrege"]["indicateurs"])
    assert all(f1 == 1.0 for k, f1 in f1_principaux(rapport["agrege"]).items())


def test_prediction_vide(verites, tmp_path):
    (tmp_path / "M1").mkdir()
    rapport = evaluer(verites, tmp_path)
    agr = rapport["agrege"]
    assert agr["evenements"]["frappes"]["tolerances"]["5"]["rappel"] == 0
    assert agr["evenements"]["services"]["rappel"] == 0
    assert agr["balle"]["seuils"]["10"]["rappel"] == 0
    assert agr["joueurs"]["pct_frames_deux_joueurs"]["valeur"] == 0
    assert agr["court"]["n_segments_manques"] == 1
    assert sorted(rapport["avertissements"]) == [
        "M1 : ball.csv absent, prédiction vide",
        "M1 : court.json absent, prédiction vide",
        "M1 : events.json absent, prédiction vide",
        "M1 : players.csv absent, prédiction vide",
    ]


def test_rapport_reproductible(verites):
    a = evaluer(verites, AMATEUR)
    b = evaluer(verites, AMATEUR)
    assert a["par_match"] == b["par_match"]
    assert a["agrege"] == b["agrege"]


def test_agregation_micro(verites, evenement, doc_events, tmp_path):
    """Un match parfait et un match vide : somme des effectifs, pas moyenne des F1."""
    events = verites["M1"]["events"]
    deux = {"A": dict(verites["M1"]), "B": dict(verites["M1"])}
    for v in deux.values():
        v.update(ball=None, court=None, players=None, events=events)
    shutil.copytree(AMATEUR / "M1", tmp_path / "A")
    (tmp_path / "B").mkdir()
    rapport = evaluer(deux, tmp_path)
    t5 = rapport["agrege"]["evenements"]["frappes"]["tolerances"]["5"]
    n = rapport["par_match"]["A"]["evenements"]["frappes"]["tolerances"]["5"]["tp"]
    assert (t5["tp"], t5["n_pred"], t5["n_gt"]) == (n, n, 2 * n)
    assert t5["precision"] == 1.0
    assert t5["rappel"] == 0.5
    assert t5["f1"] == pytest.approx(2 / 3)
    assert rapport["agrege"]["n_matchs"] == 2
    indicateurs = {(i["id"], i["player_id"]): i for i in rapport["agrege"]["indicateurs"]}
    assert indicateurs[("forehands", "P2")]["pire_match"] == "B"


def test_split_interne_applique_aux_predictions(evenement, doc_events, tmp_path):
    """Les événements prédits hors des intervalles du split ne comptent pas comme FP."""
    gt = doc_events(evenement("hit", 100, player_id="P1", stroke="forehand"))
    pred = doc_events(
        evenement("hit", 100, player_id="P1", stroke="forehand"),
        evenement("hit", 900, player_id="P1", stroke="forehand"),
    )
    (tmp_path / "V1").mkdir()
    (tmp_path / "V1" / "events.json").write_text(json.dumps(pred), encoding="utf-8")
    verite = {"events": gt, "ball": None, "court": None, "players": None, "hauteur_image": 720}
    sans = evaluer({"V1": verite}, tmp_path)
    avec = evaluer({"V1": verite | {"intervalles": [(0, 500)]}}, tmp_path)
    assert sans["agrege"]["evenements"]["frappes"]["tolerances"]["5"]["precision"] == 0.5
    assert avec["agrege"]["evenements"]["frappes"]["tolerances"]["5"]["precision"] == 1.0


def test_meta(verites, tmp_path):
    shutil.copytree(AMATEUR / "M1", tmp_path / "M1")
    (tmp_path / "M1" / "summary.json").write_text(
        json.dumps(
            {
                "modele": {"nom": "yolo11n", "version": "11", "sha256": "x"},
                "version_tennisvision": "0.1.0",
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "M1" / "stats.json").write_text(
        json.dumps({"generated_with": {"tennisvision_version": "0.1.0", "models": {"balle": "2"}}}),
        encoding="utf-8",
    )
    meta = evaluer(verites, tmp_path, dataset="amateur", split="test")["meta"]
    assert meta["tennisvision_version"] == tennisvision.__version__
    assert meta["dataset"] == "amateur"
    assert meta["split"] == "test"
    assert meta["modeles"] == {"balle": "2", "tennisvision": "0.1.0", "yolo11n": "11"}
    assert meta["tolerances"]["frappe_frames"] == [2, 5]
    assert meta["tolerances"]["service_frames"] == 5
    assert meta["tolerances"]["marge_ligne_m"] == 0.3
    assert meta["date"].endswith("+00:00")
    assert evaluer(verites, AMATEUR)["meta"]["modeles"] == {}


def test_commit_indisponible(monkeypatch):
    def echec(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(banc.subprocess, "run", echec)
    assert banc._commit() is None
    monkeypatch.setattr(
        banc.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(128, "git")),
    )
    assert banc._commit() is None


def test_ecriture_du_rapport(verites, tmp_path, monkeypatch):
    monkeypatch.setattr(banc, "_commit", lambda: "abc1234")
    rapport = evaluer(verites, AMATEUR, dataset="<script>amateur</script>", split="test")
    sortie = tmp_path / "sortie"
    chemin = ecrire_rapport(rapport, sortie)
    assert chemin.name == "rapport.json"
    horodatage = chemin.parent.name
    assert horodatage.endswith("_abc1234")
    assert len(horodatage.split("_")[0]) == len("20260101-120000")
    assert json.loads(chemin.read_text(encoding="utf-8")) == rapport
    texte = chemin.read_text(encoding="utf-8")
    assert str(tmp_path) not in texte
    assert str(AMATEUR) not in texte

    html = (chemin.parent / "rapport.html").read_text(encoding="utf-8")
    assert "<script" not in html
    assert "&lt;script&gt;amateur" in html
    assert "http" not in html
    assert "M1" in html and "frappes" in html and "forehands" in html

    ecrire_rapport(rapport, sortie)
    lignes = (sortie / "historique.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lignes) == 2
    ligne = json.loads(lignes[0])
    assert ligne["commit"] == "abc1234"
    assert ligne["split"] == "test"
    assert set(ligne["f1"]) == {"frappes_pm5", "services", "rebonds_pm5", "points", "balle_10px"}
    assert ligne["f1"]["services"] == 1.0


def test_sans_commit(verites, tmp_path, monkeypatch):
    monkeypatch.setattr(banc, "_commit", lambda: None)
    chemin = ecrire_rapport(evaluer(verites, AMATEUR), tmp_path)
    assert chemin.parent.name.endswith("_sans-commit")


def test_note_tenniset_dans_le_html(tmp_path):
    rapport = evaluer({}, tmp_path, dataset="tenniset", split="test")
    chemin = ecrire_rapport(rapport, tmp_path / "sortie")
    assert "ADR-003" in (chemin.parent / "rapport.html").read_text(encoding="utf-8")
