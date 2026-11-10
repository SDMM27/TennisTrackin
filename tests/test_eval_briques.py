"""Tests des métriques de la balle, du court et des joueurs."""

import numpy as np
import pandas as pd
import pytest

from tennisvision.eval.balle import evaluer_balle, lire_ball_csv
from tennisvision.eval.court import evaluer_court
from tennisvision.eval.joueurs import evaluer_joueurs, lire_players_csv
from tennisvision.schemas import COURT_KEYPOINTS_M

# --- Balle ---------------------------------------------------------------------------------


def table_balle(lignes: dict[int, tuple[float, float] | None]) -> pd.DataFrame:
    """{frame: (x, y)} ; None = balle non visible."""
    return pd.DataFrame(
        [
            {
                "frame_idx": f,
                "x": p[0] if p else np.nan,
                "y": p[1] if p else np.nan,
                "visible": 1 if p else 0,
            }
            for f, p in lignes.items()
        ]
    )


def test_balle_tp_fp_fn_et_trous():
    gt = table_balle(
        {0: (100, 100), 1: (100, 100), 2: (100, 100), 3: None}
        | {f: (100, 100) for f in range(4, 12)}
    )
    pred = table_balle(
        {
            0: (103, 100),  # à 3 px : TP aux deux seuils
            1: (108, 100),  # à 8 px : FP + FN à 4 px, TP à 10 px
            3: (50, 50),  # gt non visible : FP
            11: (100, 100),  # TP
            50: (0, 0),  # hors gt : ignorée
        }
    )
    r = evaluer_balle(pred, gt, hauteur_image=1080)
    s4, s10 = r["seuils"]["4"], r["seuils"]["10"]
    # gt visible : 11 frames (0, 1, 2, 4 à 11)
    assert (s4["n_tp"], s4["n_fp"], s4["n_fn"]) == (2, 2, 9)
    assert s4["precision"] == 0.5
    assert s4["rappel"] == pytest.approx(2 / 11)
    assert s4["f1"] == pytest.approx(4 / 15)
    assert (s10["n_tp"], s10["n_fp"], s10["n_fn"]) == (3, 1, 8)
    assert s10["precision"] == 0.75
    assert s10["f1"] == pytest.approx(0.4)
    # trous gt visible / pred absent : [2] (longueur 1) et [4..10] (longueur 7 > 5)
    assert r["n_trous_longs"] == 1
    assert r["n_frames_trous_longs"] == 7
    assert r["taux_trous_longs"] == pytest.approx(7 / 11)


def test_balle_mise_a_l_echelle_720p():
    gt = table_balle({0: (100, 100)})
    # À 720p les seuils valent 4 x 2/3 = 2,67 px et 10 x 2/3 = 6,67 px
    proche = evaluer_balle(table_balle({0: (100, 106)}), gt, hauteur_image=720)
    assert proche["seuils"]["4"]["n_tp"] == 0
    assert proche["seuils"]["10"]["n_tp"] == 1
    loin = evaluer_balle(table_balle({0: (100, 107)}), gt, hauteur_image=720)
    assert loin["seuils"]["10"]["n_tp"] == 0
    # Une détection trop éloignée compte FP et FN
    assert (loin["seuils"]["10"]["n_fp"], loin["seuils"]["10"]["n_fn"]) == (1, 1)
    # Même écart à 1080p : dans le seuil de 10 px
    assert evaluer_balle(table_balle({0: (100, 107)}), gt, 1080)["seuils"]["10"]["n_tp"] == 1


def test_balle_prediction_vide():
    gt = table_balle({0: (100, 100), 1: (100, 100)})
    r = evaluer_balle(pd.DataFrame(), gt, hauteur_image=1080)
    assert r["seuils"]["10"]["rappel"] == 0
    assert r["seuils"]["10"]["precision"] is None
    assert r["seuils"]["10"]["f1"] == 0


def test_balle_trous_non_contigus_non_fusionnes():
    frames = [0, 1, 2, 10, 11, 12]
    gt = table_balle({f: (100, 100) for f in frames})
    r = evaluer_balle(table_balle({}), gt, hauteur_image=1080)
    # deux trous de 3 frames, séparés par une rupture de frame_idx : aucun n'est > 5
    assert r["n_trous_longs"] == 0
    assert r["taux_trous_longs"] == 0


def test_balle_parfaite():
    gt = table_balle({0: (1, 2), 1: None, 2: (3, 4)})
    r = evaluer_balle(gt, gt, hauteur_image=1080)
    assert r["seuils"]["4"]["f1"] == 1.0
    assert r["taux_trous_longs"] == 0


def test_lire_ball_csv(tmp_path):
    chemin = tmp_path / "ball.csv"
    chemin.write_text(
        "frame_idx,t_ms,x,y,visible,score,source\n0,0,1.5,2.5,1,,annotation\n1,40,,,0,,\n",
        encoding="utf-8",
    )
    table = lire_ball_csv(chemin)
    assert table["x"].isna().tolist() == [False, True]
    assert table["visible"].tolist() == [1, 0]


# --- Court ---------------------------------------------------------------------------------

H_VRAIE = [[0.02, 0.0, -12.8], [0.0, 0.04, -14.4], [0.0, 0.0, 1.0]]  # px (640+50x, 360+25y) -> m


def points_px() -> list[list[float]]:
    return [[640 + 50 * x, 360 + 25 * y] for x, y in COURT_KEYPOINTS_M]


def segment(debut=0, fin=None, homographie=H_VRAIE, points=None):
    return {
        "valid_from_frame": debut,
        "valid_to_frame": fin,
        "homography": homographie,
        "keypoints_px": points if points is not None else points_px(),
    }


def test_court_parfait():
    doc = {"segments": [segment()]}
    r = evaluer_court(doc, doc)
    assert r["erreur_cm"]["moyenne"] == pytest.approx(0, abs=1e-9)
    assert r["erreur_px"]["max"] == pytest.approx(0, abs=1e-9)
    assert (r["n_segments"], r["n_segments_manques"]) == (1, 0)


def test_court_homographie_a_une_echelle_pres():
    # Une homographie est définie à un facteur près : 2 x H projette pareil
    gt = {"segments": [segment()]}
    pred = {"segments": [segment(homographie=[[2 * v for v in ligne] for ligne in H_VRAIE])]}
    assert evaluer_court(pred, gt)["erreur_cm"]["max"] == pytest.approx(0, abs=1e-9)


def test_court_decalage_connu():
    # Décalage de 0,1 m en x : 10 cm partout
    decalee = [[0.02, 0.0, -12.8 + 0.1], [0.0, 0.04, -14.4], [0.0, 0.0, 1.0]]
    r = evaluer_court({"segments": [segment(homographie=decalee)]}, {"segments": [segment()]})
    assert r["erreur_cm"]["moyenne"] == pytest.approx(10.0)
    assert r["erreur_cm"]["max"] == pytest.approx(10.0)
    assert r["erreur_cm"]["n"] == 14


def test_court_erreur_pixels_et_points_nuls():
    gt_pts = points_px()
    gt_pts[3] = None  # hors image côté gt : ignoré
    pred_pts = [[x + 3, y + 4] for x, y in points_px()]  # 5 px partout
    pred_pts[5] = None  # absent côté pred : ignoré en px
    r = evaluer_court(
        {"segments": [segment(points=pred_pts)]}, {"segments": [segment(points=gt_pts)]}
    )
    assert r["erreur_px"]["moyenne"] == pytest.approx(5.0)
    assert r["erreur_px"]["n"] == 12


def test_court_segments_manques():
    gt = {"segments": [segment(0, 99), segment(100, None)]}
    pred = {"segments": [segment(0, 50)]}  # rien de valide à la frame 100
    r = evaluer_court(pred, gt)
    assert (r["n_segments"], r["n_segments_manques"]) == (2, 1)
    # segment pred valide à valid_from_frame, même s'il a commencé plus tôt et court jusqu'au bout
    pred_long = {"segments": [segment(0, None)]}
    assert evaluer_court(pred_long, gt)["n_segments_manques"] == 0
    assert evaluer_court({"segments": []}, gt)["n_segments_manques"] == 2


# --- Joueurs -------------------------------------------------------------------------------


def ligne(frame, joueur, boite):
    return dict(zip(["frame_idx", "player_id"], [frame, joueur], strict=True)) | dict(
        zip(["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"], boite, strict=True)
    )


P1 = (0, 0, 10, 10)
P2 = (100, 0, 110, 10)


def test_joueurs_idf1_et_echange():
    gt = pd.DataFrame([ligne(f, j, b) for f in range(4) for j, b in (("P1", P1), ("P2", P2))])
    pred = pd.DataFrame(
        [
            ligne(0, "P1", P1),
            ligne(0, "P2", P2),
            ligne(1, "P1", (2, 0, 12, 10)),  # IoU 80/120 = 2/3 : compte
            ligne(2, "P1", P2),  # identités échangées
            ligne(2, "P2", P1),
            ligne(9, "P1", P1),  # hors des frames du gt : ignorées
            ligne(9, "P2", P2),
        ]
    )
    r = evaluer_joueurs(pred, gt)
    # frames de référence : 0 à 3 ; les deux joueurs sont présents aux frames 0 et 2
    assert r["pct_frames_deux_joueurs"] == {"n_ok": 2, "n": 4, "valeur": 0.5}
    # IDTP = 3 (frame 0 : 2, frame 1 : 1) ; 8 lignes gt, 5 lignes pred sur les frames du gt
    assert r["idf1"] == pytest.approx(6 / 13)
    assert r["n_echanges_identite"] == 1


def test_joueurs_iou_insuffisant():
    gt = pd.DataFrame([ligne(0, "P1", P1), ligne(0, "P2", P2)])
    pred = pd.DataFrame([ligne(0, "P1", (6, 0, 16, 10)), ligne(0, "P2", P2)])  # IoU 40/160
    assert evaluer_joueurs(pred, gt)["idf1"] == pytest.approx(2 / 4)


def test_joueurs_prediction_vide():
    gt = pd.DataFrame([ligne(0, "P1", P1), ligne(0, "P2", P2)])
    r = evaluer_joueurs(pd.DataFrame(), gt)
    assert r["pct_frames_deux_joueurs"]["valeur"] == 0
    assert r["idf1"] == 0
    assert r["n_echanges_identite"] == 0


def test_joueurs_parfait():
    gt = pd.DataFrame([ligne(f, j, b) for f in range(3) for j, b in (("P1", P1), ("P2", P2))])
    r = evaluer_joueurs(gt, gt)
    assert r["idf1"] == 1.0
    assert r["pct_frames_deux_joueurs"]["valeur"] == 1.0
    assert r["n_echanges_identite"] == 0


def test_lire_players_csv(tmp_path):
    chemin = tmp_path / "players.csv"
    chemin.write_text(
        "frame_idx,t_ms,player_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2\n0,0,P1,1,2,3,4\n",
        encoding="utf-8",
    )
    assert lire_players_csv(chemin)["player_id"].tolist() == ["P1"]
