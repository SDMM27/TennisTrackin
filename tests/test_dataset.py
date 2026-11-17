"""Tests des commandes `dataset check|split|accord|card`."""

import csv
import json
import shutil
from pathlib import Path

import pytest

from tennisvision import cli
from tennisvision.jeu_amateur import accord, decoupage, fiche
from tennisvision.jeu_amateur.registre import COLONNES, lire_registre

MODELE = Path(__file__).parent / "fixtures" / "eval" / "amateur" / "M1"


def ligne(match: str, **champs) -> dict:
    base = {
        "match": match,
        "date": "2026-05-01",
        "surface": "dur",
        "lieu_type": "club",
        "telephone": "PhoneA",
        "fps_nominal": "30",
        "resolution": "1920x1080",
        "joueurs": "J01;J02",
        "niveau_jeu": "debutant",
        "niveau_annotation": "3",
        "consentements": "C01;C02",
        "consentements_archives": "oui",
        "licence": "usage interne",
        "annotateurs": "A1",
        "double_annotation": "non",
        "notes": "",
    }
    return {**base, **champs}


def ecrire_registre(gt: Path, lignes: list[dict], colonnes: list[str] = COLONNES) -> None:
    with (gt / "registre.csv").open("w", encoding="utf-8", newline="") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(colonnes)
        for ligne_ in lignes:
            ecrivain.writerow([ligne_.get(c, "") for c in colonnes])


def creer_match(gt: Path, nom: str) -> Path:
    shutil.copytree(MODELE, gt / nom)
    return gt / nom


@pytest.fixture
def gt(tmp_path) -> Path:
    dossier = tmp_path / "gt"
    dossier.mkdir()
    return dossier


@pytest.fixture
def gt_complet(gt) -> Path:
    """10 matchs, 6 joueurs, 3 surfaces, 30 et 60 fps, 2 téléphones, niveau 3 partout."""
    surfaces = ["terre", "dur", "interieur"]
    lignes = []
    for i in range(10):
        nom = f"M{i + 1:02d}"
        creer_match(gt, nom)
        lignes.append(
            ligne(
                nom,
                surface=surfaces[i % 3],
                fps_nominal="60" if i % 2 else "30",
                telephone="PhoneA" if i < 5 else "PhoneB",
                joueurs=f"J{i % 6 + 1:02d};J{(i + 1) % 6 + 1:02d}",
                double_annotation="oui" if i == 0 else "non",
            )
        )
    ecrire_registre(gt, lignes)
    return gt


# --- check ---------------------------------------------------------------------------------


def test_check_dossier_inexistant(tmp_path, capsys):
    assert cli.main(["dataset", "check", str(tmp_path / "absent")]) == 2
    assert "Erreur" in capsys.readouterr().err


def test_check_registre_colonnes_inattendues(gt, capsys):
    ecrire_registre(gt, [], colonnes=["match", "date"])
    assert cli.main(["dataset", "check", str(gt)]) == 2
    assert "colonnes inattendues" in capsys.readouterr().err


def test_check_sans_registre_reste_lisible(gt, capsys):
    creer_match(gt, "M1")
    assert cli.main(["dataset", "check", str(gt)]) == 0
    sortie = capsys.readouterr().out
    assert "registre.csv absent" in sortie and "dossier M1/ absent du registre" in sortie


def test_check_jeu_complet_atteint_les_criteres(gt_complet, tmp_path, capsys):
    resume = tmp_path / "resume.json"
    assert cli.main(["dataset", "check", str(gt_complet), "--json", str(resume)]) == 0
    sortie = capsys.readouterr().out
    assert "[OK] ≥ 10 matchs" in sortie
    doc = json.loads(resume.read_text(encoding="utf-8"))
    criteres = {c["id"]: c for c in doc["criteres"]}
    # 18 frames de balle par match : le critère des 3000 frames n'est pas atteint
    assert criteres["niveau3"]["ok"] is False
    assert all(c["ok"] for k, c in criteres.items() if k != "niveau3"), criteres
    assert criteres["niveau3"]["observe"]["matchs"] == 10
    assert all(not m["erreurs"] for m in doc["matchs"])


def test_check_signale_les_ecarts(gt, capsys):
    creer_match(gt, "M1")
    creer_match(gt, "M2")
    (gt / "M2" / "ball.csv").unlink()
    (gt / "M2" / "court.json").unlink()
    ecrire_registre(
        gt,
        [
            ligne("M1", surface="sable", niveau_annotation="2"),
            ligne("M2"),  # niveau 3 déclaré sans ball.csv ni court.json
            ligne("M3"),  # dossier absent
        ],
    )
    assert cli.main(["dataset", "check", str(gt)]) == 0
    sortie = capsys.readouterr().out
    assert "surface : « sable »" in sortie
    assert "niveau constaté 3 > niveau déclaré 2" in sortie
    assert "niveau 3 déclaré mais ball.csv absent" in sortie
    assert "dossier M3/ introuvable" in sortie
    assert "[--] ≥ 10 matchs : 3" in sortie


def test_check_fichier_invalide(gt, capsys):
    creer_match(gt, "M1")
    (gt / "M1" / "events.json").write_text('{"events": 5}', encoding="utf-8")
    ecrire_registre(gt, [ligne("M1")])
    cli.main(["dataset", "check", str(gt)])
    assert "ERREUR  M1" in capsys.readouterr().out


# --- split ---------------------------------------------------------------------------------


def splits(gt: Path) -> dict:
    return json.loads((gt / "splits.json").read_text(encoding="utf-8"))


def test_split_par_match_deterministe(gt_complet, capsys):
    assert cli.main(["dataset", "split", str(gt_complet), "--seed", "3"]) == 0
    premier = splits(gt_complet)
    assert len(premier["test"]) == 2 and len(premier["val"]) == 1 and len(premier["train"]) == 7
    tous = premier["train"] + premier["val"] + premier["test"]
    assert sorted(tous) == [f"M{i:02d}" for i in range(1, 11)]  # chaque match une seule fois
    (gt_complet / "splits.json").unlink()
    cli.main(["dataset", "split", str(gt_complet), "--seed", "3"])
    assert splits(gt_complet) == premier
    # test stratifié : au moins deux surfaces différentes dans le test
    lignes = {r["match"]: r["surface"] for r in lire_registre(gt_complet)}
    assert len({lignes[m] for m in premier["test"]}) == 2


def test_split_test_gele(gt_complet, capsys):
    decoupage.decouper(gt_complet, seed=0)
    avant = splits(gt_complet)
    for i in (11, 12):
        creer_match(gt_complet, f"M{i}")
    resultat = decoupage.decouper(gt_complet, seed=99)  # autre graine : le test ne bouge pas
    assert resultat["test"] == avant["test"]
    assert set(avant["train"]) <= set(resultat["train"])
    assert set(avant["val"]) <= set(resultat["val"])
    nouveaux = {"M11", "M12"}
    assert nouveaux <= set(resultat["train"]) | set(resultat["val"])
    assert not nouveaux & set(resultat["test"])


def test_split_match_de_test_supprime(gt_complet, capsys):
    decoupage.decouper(gt_complet)
    (gt_complet / splits(gt_complet)["test"][0]).rename(gt_complet.parent / "retire")
    assert cli.main(["dataset", "split", str(gt_complet)]) == 2
    assert "absent(s) du disque" in capsys.readouterr().err


def test_split_refaire_test_exige_la_confirmation(gt_complet, capsys):
    decoupage.decouper(gt_complet, seed=0)
    avant = splits(gt_complet)
    assert cli.main(["dataset", "split", str(gt_complet), "--refaire-test"]) == 2
    assert "--je-sais-ce-que-je-fais" in capsys.readouterr().err
    assert splits(gt_complet) == avant
    code = cli.main(
        ["dataset", "split", str(gt_complet), "--refaire-test", "--je-sais-ce-que-je-fais",
         "--seed", "7"]
    )  # fmt: skip
    assert code == 0
    assert len(splits(gt_complet)["test"]) == 2


def test_split_ratios_invalides(gt_complet):
    with pytest.raises(ValueError, match="ratio"):
        decoupage.decouper(gt_complet, test_ratio=0.8, val_ratio=0.5)


def test_split_lisible_par_le_banc(gt_complet):
    from tennisvision.eval.banc import charger_verites_amateur

    decoupage.decouper(gt_complet)
    assert len(charger_verites_amateur(gt_complet, "test")) == 2


# --- accord --------------------------------------------------------------------------------


def evenement(type_: str, frame: int, **champs) -> dict:
    return {
        "id": f"{type_}_{frame}",
        "type": type_,
        "t_ms": frame * 40.0,
        "frame_idx": frame,
        "confidence": None,
        "source": "annotation",
        **champs,
    }


def ecrire_events(dossier: Path, evenements: list[dict]) -> None:
    dossier.mkdir(parents=True, exist_ok=True)
    doc = {"schema_version": "1.0", "events": evenements}
    (dossier / "events.json").write_text(json.dumps(doc), encoding="utf-8")


def frappes(frames: list[int], coups: list[str]) -> list[dict]:
    return [
        evenement("hit", f, player_id="P1", stroke=c) for f, c in zip(frames, coups, strict=True)
    ]


def test_kappa_valeurs_connues():
    parfait = {"forehand": {"forehand": 10}, "backhand": {"backhand": 10}}
    assert accord.kappa_cohen(parfait) == 1.0
    # 2 classes, accord observé 0,8, marges 50/50 -> attendu 0,5 -> kappa 0,6
    matrice = {
        "forehand": {"forehand": 4, "backhand": 1},
        "backhand": {"forehand": 1, "backhand": 4},
    }
    assert accord.kappa_cohen(matrice) == pytest.approx(0.6)
    # accord au niveau du hasard
    hasard = {"a": {"a": 1, "b": 1}, "b": {"a": 1, "b": 1}}
    assert accord.kappa_cohen(hasard) == pytest.approx(0.0)
    assert accord.kappa_cohen({}) is None


def test_accord_un_match(tmp_path, capsys):
    a, b = tmp_path / "annotateur_a" / "M1", tmp_path / "annotateur_b" / "M1"
    ecrire_events(
        a,
        [
            *frappes([10, 30, 50, 70], ["forehand", "backhand", "forehand", "smash"]),
            evenement("bounce", 20, court_xy=[0, 0], in_out="in"),
            evenement("bounce", 40, court_xy=[0, 0], in_out="out"),
            evenement("point", 5, end_frame_idx=80, winner_id="P1", end_t_ms=3200.0),
            evenement("serve", 10, player_id="P1", serve_number=1, result="in"),
        ],
    )
    ecrire_events(
        b,
        [
            # 70 -> 80 : hors tolérance ; 30 -> 31 : dans la tolérance
            *frappes([11, 31, 51, 80], ["forehand", "forehand", "forehand", "smash"]),
            evenement("bounce", 21, court_xy=[0, 0], in_out="in"),
            evenement("bounce", 39, court_xy=[0, 0], in_out="in"),
            evenement("point", 6, end_frame_idx=79, winner_id="P2", end_t_ms=3200.0),
            evenement("serve", 12, player_id="P1", serve_number=1, result="fault"),
        ],
    )
    rapport = accord.comparer(a, b)
    agrege = rapport["agrege"]
    assert agrege["frappes"]["tp"] == 3 and agrege["frappes"]["pct_a_dans_b"] == 0.75
    assert agrege["frappes"]["f1"] == pytest.approx(0.75)
    # coups appariés : (forehand, forehand), (backhand, forehand), (forehand, forehand)
    assert agrege["matrice_coups"] == {
        "backhand": {"forehand": 1},
        "forehand": {"forehand": 2},
    }
    assert agrege["kappa_coup"] == pytest.approx(0.0)
    assert agrege["in_out"]["valeur"] == 0.5 and agrege["gagnant_point"]["valeur"] == 0.0
    assert agrege["resultat_service"]["valeur"] == 0.0
    echecs = {o["id"]: o["ok"] for o in rapport["objectifs"]}
    assert echecs == {"frappes_f1": False, "kappa_coup": False, "in_out": False}

    sortie = tmp_path / "accord.json"
    code = cli.main(["dataset", "accord", str(a), str(b), "--json", str(sortie)])
    assert code == 0
    assert "[ÉCHEC] instant de frappe" in capsys.readouterr().out
    assert json.loads(sortie.read_text(encoding="utf-8"))["n_matchs"] == 1


def test_accord_identique_atteint_les_objectifs(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    for dossier in (a, b):
        ecrire_events(
            dossier / "M1",
            [
                *frappes([10, 30], ["forehand", "backhand"]),
                evenement("bounce", 20, court_xy=[0, 0], in_out="in"),
            ],
        )
    ecrire_events(a / "M2", frappes([5], ["forehand"]))  # M2 n'existe que chez A : ignoré
    assert cli.main(["dataset", "accord", str(a), str(b)]) == 0
    sortie = capsys.readouterr().out
    assert "Accord" in sortie and "M2" not in sortie
    rapport = accord.comparer(a, b)
    assert rapport["n_matchs"] == 1
    assert all(o["ok"] for o in rapport["objectifs"])


def test_accord_agrege_plusieurs_matchs(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for nom in ("M1", "M2"):
        ecrire_events(a / nom, frappes([10], ["forehand"]))
    ecrire_events(b / "M1", frappes([10], ["forehand"]))
    ecrire_events(b / "M2", frappes([60], ["forehand"]))  # non apparié
    rapport = accord.comparer(a, b)
    assert rapport["n_matchs"] == 2
    assert rapport["par_match"]["M1"]["frappes"]["f1"] == 1.0
    assert rapport["par_match"]["M2"]["frappes"]["f1"] == 0.0
    assert rapport["agrege"]["frappes"]["f1"] == pytest.approx(0.5)


def test_accord_erreurs(tmp_path, capsys):
    vide_a, vide_b = tmp_path / "va", tmp_path / "vb"
    vide_a.mkdir()
    vide_b.mkdir()
    assert cli.main(["dataset", "accord", str(vide_a), str(vide_b)]) == 2
    assert cli.main(["dataset", "accord", str(vide_a), str(tmp_path / "absent")]) == 2
    (vide_a / "events.json").write_text('{"events": 5}', encoding="utf-8")
    (vide_b / "events.json").write_text("{", encoding="utf-8")
    assert cli.main(["dataset", "accord", str(vide_a), str(vide_b)]) == 2
    err = capsys.readouterr().err
    assert "Aucun events.json commun" in err and "events.json" in err


# --- card ----------------------------------------------------------------------------------


def test_card_remplace_le_bloc_sans_toucher_le_reste(gt_complet, tmp_path):
    carte = tmp_path / "fiche.md"
    carte.write_text(
        f"# Fiche\n\nTexte écrit à la main.\n\n{fiche.MARQUEUR_DEBUT}\nancien\n{fiche.MARQUEUR_FIN}\n\n"
        "## Biais\n\nÀ remplir.\n",
        encoding="utf-8",
    )
    assert cli.main(["dataset", "card", str(gt_complet), "--sortie", str(carte)]) == 0
    texte = carte.read_text(encoding="utf-8")
    assert "ancien" not in texte
    assert texte.startswith("# Fiche\n\nTexte écrit à la main.\n\n")
    assert texte.endswith("## Biais\n\nÀ remplir.\n")
    assert "- Matchs : 10" in texte and "- Joueurs distincts (pseudonymes) : 6" in texte
    assert "- Surfaces : dur : 3, interieur : 3, terre : 4" in texte
    assert "- Frappes annotées : 40" in texte and "- Frames de balle annotées : 180" in texte
    assert "J01" not in texte  # jamais de pseudonyme ni de nom
    avant = texte
    cli.main(["dataset", "card", str(gt_complet), "--sortie", str(carte)])
    assert carte.read_text(encoding="utf-8") == avant  # idempotent


def test_card_cree_le_fichier_et_refuse_sans_marqueurs(gt_complet, tmp_path, capsys):
    nouvelle = tmp_path / "docs" / "fiche.md"
    assert cli.main(["dataset", "card", str(gt_complet), "--sortie", str(nouvelle)]) == 0
    assert fiche.MARQUEUR_DEBUT in nouvelle.read_text(encoding="utf-8")
    sans = tmp_path / "sans.md"
    sans.write_text("# rien\n", encoding="utf-8")
    assert cli.main(["dataset", "card", str(gt_complet), "--sortie", str(sans)]) == 2
    assert "marqueurs" in capsys.readouterr().err
    assert sans.read_text(encoding="utf-8") == "# rien\n"


def test_modele_de_registre_livre_dans_la_doc():
    modele = Path(__file__).parents[1] / "docs" / "dataset" / "registre_modele.csv"
    if not modele.is_file():
        pytest.skip("docs/ absent (paquet installé hors du dépôt)")
    assert modele.read_text(encoding="utf-8-sig").splitlines()[0] == ",".join(COLONNES)
