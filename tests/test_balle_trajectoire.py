"""Tests du post-traitement hors-ligne des candidats balle (données construites à la main)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from tennisvision.balle.masque import MasqueCourt
from tennisvision.balle.trajectoire import (
    ParamsTrajectoire,
    ajuster_segment,
    construire_ball_csv,
    construire_ball_csv_diagnostic,
    ecrire_ball_csv,
    filtrer_masque,
    interpoler_segment,
    normaliser_candidats,
    rejeter_aberrants,
    retirer_statiques,
    segmenter,
    selectionner_viterbi,
)
from tennisvision.validate import valider_fichier

H = 1080
COLONNES = ["frame_idx", "t_ms", "x", "y", "visible", "score", "source"]


@dataclass
class Candidat:
    frame_idx: int
    x: float
    y: float
    score: float


def temps(n: int, fps: float = 30.0) -> dict[int, float]:
    return {i: i * 1000.0 / fps for i in range(n)}


def balle(t_ms_s) -> tuple[np.ndarray, np.ndarray]:
    """Parabole de référence (pixels) en fonction du temps en ms."""
    tau = np.asarray(t_ms_s, dtype=float) / 1000.0
    return 200 + 700 * tau, 700 - 450 * tau + 250 * tau**2


def candidats_balle(t: dict[int, float], frames=None, score=0.8, bruit=0.0, graine=0):
    generateur = np.random.default_rng(graine)
    frames = list(t) if frames is None else list(frames)
    x, y = balle([t[f] for f in frames])
    return [
        Candidat(f, xi + generateur.normal(0, bruit), yi + generateur.normal(0, bruit), score)
        for f, xi, yi in zip(frames, x, y, strict=True)
    ]


def ecart_a_la_verite(table: pd.DataFrame, frames) -> np.ndarray:
    """Distance (px) entre la sortie et la parabole de référence sur `frames` ; NaN si absent."""
    ligne = table.set_index("frame_idx").loc[list(frames)]
    x, y = balle(ligne["t_ms"].to_numpy())
    return np.hypot(ligne["x"].to_numpy() - x, ligne["y"].to_numpy() - y)


def verifier_contrat(table: pd.DataFrame, t, tmp_path):
    assert list(table.columns) == COLONNES
    assert table["frame_idx"].tolist() == sorted(t)
    assert len(table) == len(t)
    chemin = tmp_path / "ball.csv"
    ecrire_ball_csv(table, chemin)
    assert valider_fichier(chemin) == []


# ----------------------------------------------------------------------- bout en bout


def test_balle_simple_detectee_et_valide(tmp_path):
    t = temps(54)
    table = construire_ball_csv(candidats_balle(t, bruit=0.5), t, H)
    verifier_contrat(table, t, tmp_path)
    assert (table["visible"] == 1).all()
    assert (table["source"] == "detected").all()
    assert (ecart_a_la_verite(table, t) < 2).all()


def test_candidats_vides(tmp_path):
    t = temps(40)
    for vide in ([], pd.DataFrame(columns=["frame_idx", "x", "y", "score"])):
        table = construire_ball_csv(vide, t, H)
        verifier_contrat(table, t, tmp_path)
        assert (table["visible"] == 0).all()
        assert table[["x", "y", "score", "source"]].isna().all().all()


def test_objet_statique_jamais_retenu():
    t = temps(150)  # 5 s
    generateur = np.random.default_rng(1)
    candidats = candidats_balle(t, frames=range(20, 100), score=0.6, bruit=0.4)
    # Roue de caméra : score élevé, présente sur 80 % des frames, légère gigue
    for f in t:
        if f % 5 != 0:
            gigue = generateur.normal(0, 0.7, 2)
            candidats.append(Candidat(f, 900 + gigue[0], 200 + gigue[1], 0.95))
    table = construire_ball_csv(candidats, t, H)
    vues = table[table["visible"] == 1]
    assert (np.hypot(vues["x"] - 900, vues["y"] - 200) > 10).all()
    reel = table[(table["frame_idx"] >= 20) & (table["frame_idx"] < 100)]
    assert (reel["visible"] == 1).mean() >= 0.95
    assert (ecart_a_la_verite(table, range(25, 95)) < 2).all()


def test_balle_d_un_court_voisin_hors_masque():
    t = temps(54)
    masque = MasqueCourt.depuis_polygone({"polygone": [[0, 0], [1000, 0], [1000, 1080], [0, 1080]]})
    # Parabole décalée de 1200 px : toute dans le court voisin (x > 1000)
    voisine = [Candidat(c.frame_idx, c.x + 1200, c.y, 0.99) for c in candidats_balle(t)]
    table, diag = construire_ball_csv_diagnostic(voisine, t, H, masque=masque)
    assert (table["visible"] == 0).all()
    assert diag["n_hors_masque"] == 54
    # Avec la vraie balle en plus, seule la vraie est retenue
    vraie = candidats_balle(t, frames=range(30), score=0.6)  # reste à gauche de x = 1000
    table = construire_ball_csv(voisine + vraie, t, H, masque=masque)
    assert (table["x"].dropna() < 1000).all()
    assert (ecart_a_la_verite(table, range(30)) < 2).all()


def test_selection_suit_la_vraie_balle_parmi_des_distracteurs():
    t = temps(54)
    generateur = np.random.default_rng(3)
    candidats = candidats_balle(t, score=0.7, bruit=0.5)
    for f in t:
        x, y, score = (
            generateur.uniform(0, 1900),
            generateur.uniform(0, 1000),
            generateur.uniform(0.4, 0.95),
        )
        candidats.append(Candidat(f, x, y, score))
    table = construire_ball_csv(candidats, t, H)
    proche = ecart_a_la_verite(table, t) <= 2
    assert proche.mean() >= 0.95


def test_point_aberrant_teleporte_retire():
    t = temps(54)
    candidats = candidats_balle(t, score=0.7, bruit=0.3)
    candidats.append(Candidat(30, 1700, 100, 0.99))
    table = construire_ball_csv(candidats, t, H)
    assert (np.hypot(table["x"] - 1700, table["y"] - 100).dropna() > 50).all()
    assert (ecart_a_la_verite(table, t) <= 2).sum() >= 52


def test_trou_de_6_frames_interpole(tmp_path):
    t = temps(54)
    frames = [f for f in t if not 20 <= f <= 25]
    table = construire_ball_csv(candidats_balle(t, frames=frames), t, H)
    verifier_contrat(table, t, tmp_path)
    trou = table[table["frame_idx"].between(20, 25)]
    assert (trou["source"] == "interpolated").all()
    assert (trou["visible"] == 1).all()
    assert trou["score"].isna().all()
    assert (ecart_a_la_verite(table, range(20, 26)) < 2).all()
    assert (table[~table["frame_idx"].between(20, 25)]["source"] == "detected").all()


def test_trou_de_12_frames_non_comble():
    t = temps(70)
    frames = [f for f in t if not 30 <= f <= 41]
    table = construire_ball_csv(candidats_balle(t, frames=frames), t, H)
    trou = table[table["frame_idx"].between(30, 41)]
    assert (trou["visible"] == 0).all()
    assert trou[["x", "y", "score", "source"]].isna().all().all()
    assert (table["source"] != "interpolated").all()


def rebond(frames) -> list[Candidat]:
    """Balle qui descend (y croît) puis remonte : changement de signe de vy en frame 25."""
    return [Candidat(f, 200 + 20.0 * f, 900 - 30.0 * abs(f - 25), 0.8) for f in frames]


def test_trou_a_un_changement_de_direction_non_comble():
    t = temps(50)
    frames = [f for f in t if not 23 <= f <= 27]  # le point de rebond manque
    table = construire_ball_csv(rebond(frames), t, H)
    trou = table[table["frame_idx"].between(23, 27)]
    assert (trou["visible"] == 0).all()
    assert (table["source"] != "interpolated").all()
    assert (table["visible"] == 1).sum() == len(frames)  # les deux côtés sont conservés
    # Sans trou : le rebond ne casse rien et tout est détecté
    complet = construire_ball_csv(rebond(t), t, H)
    assert (complet["source"] == "detected").all()


def test_cadence_variable():
    generateur = np.random.default_rng(5)
    valeurs = np.concatenate([[0.0], np.cumsum(generateur.uniform(20, 50, 60))])
    t = {i: float(v) for i, v in enumerate(valeurs)}
    frames = [f for f in t if not 25 <= f <= 29]
    table = construire_ball_csv(candidats_balle(t, frames=frames, bruit=0.4), t, H)
    assert len(table) == len(t)
    assert (ecart_a_la_verite(table, t) < 2).all()
    assert (table[table["frame_idx"].between(25, 29)]["source"] == "interpolated").all()
    # Les t_ms de sortie sont ceux fournis, y compris sous forme de Series
    table_serie = construire_ball_csv(candidats_balle(t, bruit=0.4), pd.Series(t), H)
    assert table_serie["t_ms"].tolist() == pytest.approx(list(t.values()))


def test_soixante_fps():
    t = temps(110, fps=60.0)
    frames = [f for f in t if not 50 <= f <= 57]
    table = construire_ball_csv(candidats_balle(t, frames=frames, bruit=0.4), t, H)
    assert (ecart_a_la_verite(table, t) < 2).all()
    assert (table[table["frame_idx"].between(50, 57)]["source"] == "interpolated").all()


def test_candidats_en_dataframe_et_frames_hors_intervalle():
    t = temps(54)
    candidats = pd.DataFrame(
        [(c.frame_idx, c.x, c.y, c.score) for c in candidats_balle(t)] + [(999, 500.0, 500.0, 0.9)],
        columns=["frame_idx", "x", "y", "score"],
    )
    table, diag = construire_ball_csv_diagnostic(candidats, t, H)
    assert len(table) == 54
    assert diag["n_hors_intervalle"] == 1
    assert diag["n_detectes"] == 54
    assert diag["n_interpoles"] == 0
    assert set(diag) >= {"n_candidats", "n_hors_masque", "n_statiques", "n_isoles"}


def test_trace_isolee_ou_trop_courte_ecartee():
    t = temps(60)
    table = construire_ball_csv([Candidat(30, 500, 500, 0.9)], t, H)
    assert (table["visible"] == 0).all()
    deux = [Candidat(30, 500, 500, 0.9), Candidat(31, 520, 490, 0.9)]
    assert (construire_ball_csv(deux, t, H)["visible"] == 0).all()


# ----------------------------------------------------------------------- étapes isolées


def _cands(lignes) -> pd.DataFrame:
    table = pd.DataFrame(lignes, columns=["frame_idx", "x", "y", "score"])
    table["pos"] = table["frame_idx"]
    return table


def test_normaliser_candidats():
    table = normaliser_candidats([Candidat(2, 1.0, 2.0, 0.5)])
    assert table.to_dict("records") == [{"frame_idx": 2, "x": 1.0, "y": 2.0, "score": 0.5}]
    assert normaliser_candidats([]).empty


def test_filtrer_masque():
    masque = MasqueCourt.depuis_polygone({"polygone": [[0, 0], [100, 0], [100, 100], [0, 100]]})
    cands = _cands([(0, 50, 50, 0.9), (1, 500, 50, 0.9)])
    assert filtrer_masque(cands, masque)["frame_idx"].tolist() == [0]
    assert len(filtrer_masque(cands, None)) == 2


def test_retirer_statiques():
    t = np.arange(120) * 1000.0 / 30  # 4 s
    statique = [(f, 500.0, 300.0, 0.9) for f in range(120) if f % 2 == 0]  # 50 % des frames
    court = [(f, 800.0, 300.0, 0.9) for f in range(10)]  # fixe mais 0,3 s seulement
    mobile = [(f, 100.0 + 15 * f, 600.0, 0.9) for f in range(120)]
    restants = retirer_statiques(_cands(statique + court + mobile), t, rayon_px=6.5)
    assert not ((restants["x"] == 500.0) & (restants["y"] == 300.0)).any()
    assert len(restants[restants["x"] == 800.0]) == 10
    assert len(restants[restants["y"] == 600.0]) == 120
    # Trop clairsemé (20 % des frames) : conservé
    clairseme = [(f, 500.0, 300.0, 0.9) for f in range(120) if f % 5 == 0]
    assert len(retirer_statiques(_cands(clairseme), t, rayon_px=6.5)) == len(clairseme)


def test_retirer_statiques_fin_de_video():
    t = np.arange(120) * 1000.0 / 30
    fin = [(f, 500.0, 300.0, 0.9) for f in range(50, 120)]  # statique jusqu'à la dernière frame
    assert retirer_statiques(_cands(fin), t, rayon_px=6.5).empty


def test_selectionner_viterbi_un_candidat_par_frame():
    t = np.arange(30) * 1000.0 / 30
    params = ParamsTrajectoire().resoudre(H)
    lignes = [(f, 100.0 + 15 * f, 500.0, 0.6) for f in range(30)]
    lignes += [(f, 100.0 + 15 * f, 900.0, 0.9) for f in range(0, 30, 3)]  # forts mais épars
    choisis = selectionner_viterbi(_cands(lignes), t, H, params)
    assert choisis["pos"].is_unique
    assert (choisis["y"] == 500.0).sum() >= 27
    assert selectionner_viterbi(_cands([]), t, H, params).empty


def test_selectionner_viterbi_interdit_les_vitesses_impossibles():
    t = np.arange(40) * 1000.0 / 30
    params = ParamsTrajectoire().resoudre(H)
    # Dix points puis un saut de ~1500 px en une frame (45 000 px/s >> 4 320 px/s)
    lignes = [(f, 100.0 + 15 * f, 500.0, 0.8) for f in range(10)] + [(10, 1700.0, 100.0, 0.99)]
    choisis = selectionner_viterbi(_cands(lignes), t, H, params)
    assert not ((choisis["pos"] == 10) & (choisis["x"] == 1700.0)).any() or (
        9 not in choisis["pos"].tolist()
    )
    assert len(choisis) >= 9


def test_rejeter_aberrants_vitesse_et_isolement():
    t = np.arange(40) * 1000.0 / 30
    points = pd.DataFrame(
        {
            "pos": list(range(10)) + [20],  # le dernier est isolé
            "x": [100.0 + 15 * f for f in range(10)] + [900.0],
            "y": [500.0] * 11,
            "score": [0.8] * 11,
        }
    )
    points.loc[4, ["x", "y"]] = [1800.0, 50.0]  # téléportation
    restants, n_vitesse, n_isoles = rejeter_aberrants(points, t, 4320.0, 3)
    assert 4 not in restants["pos"].tolist()
    assert 20 not in restants["pos"].tolist()
    assert (n_vitesse, n_isoles) == (1, 1)
    assert len(restants) == 9


def test_segmenter_trous_ruptures_et_segments_courts():
    t = np.arange(120) * 1000.0 / 30
    params = ParamsTrajectoire().resoudre(H)
    # Rebond en frame 20 (vy change de signe), puis trou de 15 frames, puis segment de 2 points
    pos = list(range(41)) + list(range(56, 60)) + [90, 91]
    y = [900 - 30.0 * abs(f - 20) for f in range(41)] + [500.0] * 4 + [300.0] * 2
    points = pd.DataFrame({"pos": pos, "x": [200 + 20.0 * f for f in pos], "y": y, "score": 0.8})
    segments, n_courts = segmenter(points, t, H, params)
    bornes = [(int(s["pos"].iloc[0]), int(s["pos"].iloc[-1])) for s in segments]
    assert len(segments) == 3
    # Rupture adjacente au rebond (entre 19|20 ou 20|21) puis coupure au trou
    assert bornes[0][0] == 0 and 19 <= bornes[0][1] <= 20
    assert 20 <= bornes[1][0] <= 21 and bornes[1][1] == 40
    assert bornes[2] == (56, 59)
    assert n_courts == 2


def test_segmenter_ignore_le_bruit():
    t = np.arange(60) * 1000.0 / 30
    params = ParamsTrajectoire().resoudre(H)
    generateur = np.random.default_rng(2)
    x, y = balle(t[:54])
    points = pd.DataFrame(
        {
            "pos": np.arange(54),
            "x": x + generateur.normal(0, 1.5, 54),
            "y": y + generateur.normal(0, 1.5, 54),
            "score": 0.8,
        }
    )
    segments, _ = segmenter(points, t, H, params)
    assert len(segments) == 1


def test_ajuster_segment_rejette_les_aberrants():
    t = np.arange(30) * 1000.0 / 30
    x, y = (v.copy() for v in balle(t))
    y[12] += 40
    x[20] -= 25
    inliers, courbe = ajuster_segment(t, x, y, residu_max_px=0.008 * H)
    assert not inliers[12] and not inliers[20]
    assert inliers.sum() == 28
    xc, yc = courbe.evaluer(t)
    attendu_x, attendu_y = balle(t)
    assert np.abs(xc - attendu_x).max() < 0.1 and np.abs(yc - attendu_y).max() < 0.1


def test_ajuster_segment_petit_nombre_de_points():
    t = np.arange(4) * 1000.0 / 30
    x, y = (v.copy() for v in balle(t))
    y[2] += 50
    inliers, courbe = ajuster_segment(t, x, y, residu_max_px=8.0)
    assert inliers.tolist() == [True, True, False, True]
    assert courbe is not None


def test_interpoler_segment_ne_comble_que_les_trous_courts():
    pos = np.array([0, 1, 5, 6, 20, 21])  # trous de 3 puis 13 frames
    assert interpoler_segment(pos, 10).tolist() == [2, 3, 4]
    assert interpoler_segment(np.array([3]), 10).size == 0
