"""Post-traitement hors-ligne des candidats balle : du réseau à un `ball.csv` propre.

Entrée : des candidats `(frame_idx, x, y, score)` en pixels (zéro, un ou plusieurs par frame).
Sortie : une ligne par frame de l'intervalle analysé, au format du contrat `ball.csv`.

Étapes, chacune étant une fonction testée séparément (priorité à la précision : un faux
positif coûte plus cher qu'un trou) :

1. masque de court : candidats hors du masque écartés ;
2. objets statiques : un amas spatial qui reste dans `rayon_statique_px` plus de
   `duree_statique_ms`, même détecté par intermittence, est écarté (roue de caméra, balle
   posée au sol, lumière) ;
3. sélection : au plus un candidat par frame, par programmation dynamique (Viterbi) avec un
   état « pas de balle » explicite ;
4. points aberrants : vitesse impossible entre points retenus, points isolés ;
5. segmentation : coupure aux trous trop longs et aux changements de direction brusques
   (frappes, rebonds) ; les segments trop courts sont écartés ;
6. ajustement robuste d'une parabole par segment ; les points trop éloignés de la courbe sont
   rejetés. Les points détectés gardent leur position MESURÉE ;
7. interpolation des trous courts à l'intérieur d'un segment, depuis la courbe ajustée
   (jamais entre deux segments, jamais d'extrapolation).

Les durées se calculent toujours avec les vrais `t_ms` (cadence variable possible).
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from tennisvision.balle.masque import MasqueCourt

COLONNES_CANDIDATS = ["frame_idx", "x", "y", "score"]
COLONNES_BALL = ["frame_idx", "t_ms", "x", "y", "visible", "score", "source"]

# Valeurs par défaut relatives à la hauteur de l'image (1080p : 0,006 -> ~6,5 px)
RAYON_STATIQUE_REL = 0.006
RESIDU_MAX_REL = 0.008
VITESSE_MAX_REL = 4.0  # hauteurs d'image par seconde


@dataclass(frozen=True)
class ParamsTrajectoire:
    """Paramètres du post-traitement. Les valeurs `None` se déduisent de la hauteur d'image."""

    # Étape 2 : objets statiques
    rayon_statique_px: float | None = None  # défaut 0,006 x hauteur
    duree_statique_ms: float = 1000.0
    fraction_presence_statique: float = 0.5  # part des frames de la fenêtre avec une détection
    # Étape 3 : sélection (Viterbi)
    k_candidats: int = 5  # candidats conservés par frame (meilleurs scores)
    vitesse_max: float | None = None  # px/s ; défaut 4,0 x hauteur par seconde
    cout_absence: float = 0.7  # coût d'une frame sans balle ; un candidat coûte 1 - score
    cout_debut: float = 0.3  # coût d'ouverture d'une trace après une absence
    poids_deviation: float = 0.1  # pénalité par `deviation_ref` de déviation à la vitesse constante
    deviation_ref_rel: float = 0.01  # x hauteur : déviation de référence
    deviation_max_ratio: float = 6.0  # plafond de la pénalité, en multiples de la référence
    poids_deplacement: float = 0.5  # pénalité de déplacement (vitesse / vitesse_max)
    # Étape 4 : aberrants
    isolement_frames: int = 3
    # Étape 5 : segmentation
    trou_max_frames: int = 10
    angle_rupture_deg: float = 40.0
    vitesse_min_rupture_rel: float = 0.1  # x hauteur par seconde : en dessous, angle ignoré
    fenetre_rupture: int = 3  # points de part et d'autre pour estimer la vitesse
    segment_min_points: int = 3  # sans exception : les segments plus courts sont du bruit
    # Étape 6 : ajustement
    residu_max_px: float | None = None  # défaut 0,008 x hauteur

    def resoudre(self, hauteur_image: int) -> "ParamsTrajectoire":
        """Copie dont les valeurs relatives à la hauteur d'image sont chiffrées en pixels."""
        return replace(
            self,
            rayon_statique_px=self.rayon_statique_px or RAYON_STATIQUE_REL * hauteur_image,
            vitesse_max=self.vitesse_max or VITESSE_MAX_REL * hauteur_image,
            residu_max_px=self.residu_max_px or RESIDU_MAX_REL * hauteur_image,
        )


# --------------------------------------------------------------------------- entrées


def normaliser_candidats(candidats) -> pd.DataFrame:
    """Candidats (DataFrame `frame_idx,x,y,score` ou objets ayant ces attributs) -> DataFrame."""
    if isinstance(candidats, pd.DataFrame):
        table = candidats.loc[:, COLONNES_CANDIDATS].copy()
    else:
        lignes = [(c.frame_idx, c.x, c.y, c.score) for c in candidats]
        table = pd.DataFrame(lignes, columns=COLONNES_CANDIDATS)
    table = table.astype({"frame_idx": "int64", "x": "float64", "y": "float64", "score": "float64"})
    table = table.dropna()
    return table.reset_index(drop=True)


def _temps_par_frame(t_ms: Mapping[int, float] | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(frames triées, t_ms correspondants)."""
    serie = pd.Series(dict(t_ms) if isinstance(t_ms, Mapping) else t_ms.to_dict(), dtype=float)
    serie = serie.sort_index()
    return serie.index.to_numpy(dtype="int64"), serie.to_numpy(dtype=float)


# --------------------------------------------------------------------------- étape 1


def filtrer_masque(candidats: pd.DataFrame, masque: MasqueCourt | None) -> pd.DataFrame:
    """Écarte les candidats hors du masque de court (sans masque : tout est accepté)."""
    if masque is None or candidats.empty:
        return candidats
    dedans = masque.contient_vec(
        candidats["frame_idx"].to_numpy(), candidats["x"].to_numpy(), candidats["y"].to_numpy()
    )
    return candidats[dedans]


# --------------------------------------------------------------------------- étape 2


def _amas_spatiaux(x: np.ndarray, y: np.ndarray, rayon: float) -> np.ndarray:
    """Étiquette d'amas par candidat : un candidat rejoint le centre existant le plus proche
    à moins de `rayon`, sinon il devient un nouveau centre (grille de hachage, une passe)."""
    etiquettes = np.empty(len(x), dtype=int)
    centres: list[tuple[float, float]] = []
    grille: dict[tuple[int, int], list[int]] = {}
    for i in range(len(x)):
        ci, cj = int(np.floor(x[i] / rayon)), int(np.floor(y[i] / rayon))
        meilleur, meilleure_distance = -1, rayon
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for c in grille.get((ci + di, cj + dj), ()):
                    distance = np.hypot(x[i] - centres[c][0], y[i] - centres[c][1])
                    if distance <= meilleure_distance:
                        meilleur, meilleure_distance = c, distance
        if meilleur < 0:
            meilleur = len(centres)
            centres.append((x[i], y[i]))
            grille.setdefault((ci, cj), []).append(meilleur)
        etiquettes[i] = meilleur
    return etiquettes


def _frames_couvertes_statiques(
    positions: np.ndarray, t: np.ndarray, duree_ms: float, fraction: float, min_presence: int
) -> np.ndarray:
    """Pour un amas présent aux positions de frame `positions` (uniques, triées) : booléen par
    frame, vrai dans toute fenêtre de `duree_ms` (ancrée sur une détection, débutant ou
    finissant par elle) où l'amas est présent dans au moins `fraction` des frames."""
    n = len(t)
    t_amas = t[positions]
    rang = np.arange(len(positions))
    diff = np.zeros(n + 1, dtype=int)

    # Fenêtres [t_k, t_k + duree] (entièrement dans la vidéo)
    fin_t = t_amas + duree_ms
    fin_pos = np.searchsorted(t, fin_t, side="right")
    n_frames = fin_pos - positions
    presents = np.searchsorted(t_amas, fin_t, side="right") - rang
    bon = (fin_t <= t[-1]) & (presents >= fraction * n_frames) & (presents >= min_presence)
    np.add.at(diff, positions[bon], 1)
    np.add.at(diff, fin_pos[bon], -1)

    # Fenêtres [t_k - duree, t_k] (couvrent la fin de la vidéo)
    debut_t = t_amas - duree_ms
    debut_pos = np.searchsorted(t, debut_t, side="left")
    n_frames = positions + 1 - debut_pos
    presents = rang + 1 - np.searchsorted(t_amas, debut_t, side="left")
    bon = (debut_t >= t[0]) & (presents >= fraction * n_frames) & (presents >= min_presence)
    np.add.at(diff, debut_pos[bon], 1)
    np.add.at(diff, positions[bon] + 1, -1)
    return np.cumsum(diff[:n]) > 0


def retirer_statiques(
    candidats: pd.DataFrame,
    t: np.ndarray,
    rayon_px: float,
    duree_ms: float = 1000.0,
    fraction_presence: float = 0.5,
    min_presence: int = 3,
) -> pd.DataFrame:
    """Écarte les candidats d'un objet immobile.

    `candidats` porte une colonne `pos` (position de la frame dans `t`, les `t_ms` triés).
    Les candidats sont regroupés en amas de rayon `rayon_px` ; un amas est statique sur toute
    fenêtre de `duree_ms` où il est présent dans au moins `fraction_presence` des frames
    (détections intermittentes comprises). Une balle en vol ne reste jamais aussi longtemps
    dans un si petit rayon.
    """
    if candidats.empty or len(t) == 0:
        return candidats
    etiquettes = _amas_spatiaux(candidats["x"].to_numpy(), candidats["y"].to_numpy(), rayon_px)
    positions = candidats["pos"].to_numpy()
    statique = np.zeros(len(candidats), dtype=bool)
    ordre = np.argsort(etiquettes, kind="stable")
    bornes = np.flatnonzero(np.diff(etiquettes[ordre])) + 1
    for groupe in np.split(ordre, bornes):
        if len(groupe) < min_presence:
            continue
        uniques = np.unique(positions[groupe])
        couvertes = _frames_couvertes_statiques(
            uniques, t, duree_ms, fraction_presence, min_presence
        )
        statique[groupe] = couvertes[positions[groupe]]
    return candidats[~statique]


# --------------------------------------------------------------------------- étape 3


def selectionner_viterbi(
    candidats: pd.DataFrame, t: np.ndarray, hauteur_image: int, params: ParamsTrajectoire
) -> pd.DataFrame:
    """Au plus un candidat par frame, par Viterbi sur toutes les frames.

    Chaque frame a un état « pas de balle » (coût `cout_absence`) et au plus `k_candidats`
    états candidats (coût `1 - score`). Transition vers un candidat :

    - depuis une balle (frame précédente) ou une absence de moins de `trou_max_frames` frames :
      interdite si la vitesse (avec les vrais `t_ms`) dépasse `vitesse_max` ; sinon pénalité
      proportionnelle à l'écart à la prédiction à vitesse constante (plafonnée), quand le point
      précédent a lui-même un prédécesseur, ou à la vitesse relative sinon ;
    - après une absence, ouvrir une nouvelle trace coûte `cout_debut` (retenu s'il est moins
      cher que la continuité) ; elle est la seule façon de « sauter » d'un objet à un autre.

    Le prédécesseur du point précédent (nécessaire à la vitesse constante) est celui du meilleur
    chemin qui y arrive : approximation d'ordre 2 qui garde O(frames x candidats²).
    """
    if candidats.empty:
        return candidats
    k = max(1, params.k_candidats)
    tries = candidats.sort_values(["pos", "score"], ascending=[True, False], kind="stable")
    tries = tries.groupby("pos", sort=True).head(k)
    cx = tries["x"].to_numpy()
    cy = tries["y"].to_numpy()
    cp = tries["pos"].to_numpy()
    cscore = tries["score"].to_numpy()
    debuts = np.searchsorted(cp, np.arange(len(t) + 1), side="left")

    vmax = params.vitesse_max
    d0 = params.deviation_ref_rel * hauteur_image
    trou = params.trou_max_frames

    def penalite(dernier: int, avant: int, cible: int) -> float | None:
        """Pénalité de continuer de `dernier` à `cible` (indices candidats) ; None = interdit."""
        dt = max(t[cp[cible]] - t[cp[dernier]], 1e-3) / 1000.0
        dx, dy = cx[cible] - cx[dernier], cy[cible] - cy[dernier]
        vitesse = np.hypot(dx, dy) / dt
        if vitesse > vmax:
            return None
        manquantes = cp[cible] - cp[dernier] - 1
        if avant >= 0 and cp[dernier] - cp[avant] - 1 <= trou:
            dt_av = max(t[cp[dernier]] - t[cp[avant]], 1e-3) / 1000.0
            px = cx[dernier] + (cx[dernier] - cx[avant]) / dt_av * dt
            py = cy[dernier] + (cy[dernier] - cy[avant]) / dt_av * dt
            ecart = np.hypot(cx[cible] - px, cy[cible] - py) / (d0 * (1 + 0.25 * manquantes))
            return params.poids_deviation * min(ecart, params.deviation_max_ratio)
        return params.poids_deplacement * vitesse / vmax

    # Par frame : coût cumulé, prédécesseur, dernière balle, balle précédente, nb d'absences
    # (état 0 = pas de balle ; état s >= 1 = candidat `debuts[f] + s - 1`)
    ids0 = list(range(debuts[0], debuts[1]))
    couts = [np.array([params.cout_absence] + [params.cout_debut + 1.0 - cscore[i] for i in ids0])]
    retours: list[np.ndarray] = [np.zeros(len(ids0) + 1, dtype=int)]
    derniere: list[list[int]] = [[-1, *ids0]]
    avant_derniere: list[list[int]] = [[-1] * (len(ids0) + 1)]
    absences: list[list[int]] = [[1] + [0] * len(ids0)]
    for f in range(1, len(t)):
        ids = list(range(debuts[f], debuts[f + 1]))
        prec_ids = list(range(debuts[f - 1], debuts[f]))
        n_prec = len(couts[-1])
        cout = np.empty(len(ids) + 1)
        retour = np.zeros(len(ids) + 1, dtype=int)
        dern = [-1] * (len(ids) + 1)
        avant = [-1] * (len(ids) + 1)
        abs_ = [0] * (len(ids) + 1)

        # État « pas de balle » : le plus bas coût précédent, en mémorisant la dernière balle
        s_min = int(np.argmin(couts[-1]))
        cout[0] = couts[-1][s_min] + params.cout_absence
        retour[0] = s_min
        if s_min == 0:
            dern[0], avant[0], abs_[0] = (
                derniere[-1][0],
                avant_derniere[-1][0],
                absences[-1][0] + 1,
            )
        else:
            dern[0], avant[0], abs_[0] = prec_ids[s_min - 1], avant_derniere[-1][s_min], 1
        if abs_[0] > trou:
            dern[0] = avant[0] = -1

        for j, cid in enumerate(ids, start=1):
            meilleur, meilleur_s = np.inf, -1
            for s in range(n_prec):
                if s == 0:
                    transition = params.cout_debut
                    if derniere[-1][0] >= 0:
                        pen = penalite(derniere[-1][0], avant_derniere[-1][0], cid)
                        if pen is not None:
                            transition = min(transition, pen)
                else:
                    pen = penalite(prec_ids[s - 1], avant_derniere[-1][s], cid)
                    if pen is None:
                        continue
                    transition = pen
                total = couts[-1][s] + transition
                if total < meilleur:
                    meilleur, meilleur_s = total, s
            if meilleur_s < 0:  # aucune transition possible : repartir de la meilleure absence
                meilleur_s = 0
                meilleur = couts[-1][0] + params.cout_debut
            cout[j] = meilleur + 1.0 - cscore[cid]
            retour[j] = meilleur_s
            # la « dernière balle » avant ce candidat sert à la vitesse constante suivante
            if meilleur_s == 0:
                avant[j] = derniere[-1][0]
            else:
                avant[j] = prec_ids[meilleur_s - 1]
            dern[j] = cid
            abs_[j] = 0
        couts.append(cout)
        retours.append(retour)
        derniere.append(dern)
        avant_derniere.append(avant)
        absences.append(abs_)

    # Remontée
    etat = int(np.argmin(couts[-1]))
    choisis: list[int] = []
    for f in range(len(t) - 1, -1, -1):
        if etat > 0:
            choisis.append(debuts[f] + etat - 1)
        etat = int(retours[f][etat])
    return tries.iloc[sorted(choisis)]


# --------------------------------------------------------------------------- étape 4


def rejeter_aberrants(
    points: pd.DataFrame, t: np.ndarray, vitesse_max: float, isolement_frames: int = 3
) -> tuple[pd.DataFrame, int, int]:
    """Retire les points à vitesse impossible puis les points isolés.

    `points` : colonnes `pos, x, y, score`, une ligne par frame. Tant que deux points
    consécutifs impliquent une vitesse supérieure à `vitesse_max`, on retire celui impliqué
    dans le plus de violations (à égalité, le moins bien noté). Un point isolé n'a aucun autre
    point retenu à moins de `isolement_frames` frames.

    Renvoie `(points restants, n retirés pour vitesse, n retirés pour isolement)`.
    """
    points = points.sort_values("pos")
    n_vitesse = 0
    while len(points) >= 2:
        pos = points["pos"].to_numpy()
        x, y = points["x"].to_numpy(), points["y"].to_numpy()
        dt = np.maximum(np.diff(t[pos]), 1e-3) / 1000.0
        viole = np.hypot(np.diff(x), np.diff(y)) / dt > vitesse_max
        if not viole.any():
            break
        violations = np.zeros(len(points), dtype=int)
        violations[:-1] += viole
        violations[1:] += viole
        score = points["score"].to_numpy()
        pire = max(np.flatnonzero(violations), key=lambda i: (violations[i], -score[i]))
        points = points.iloc[[i for i in range(len(points)) if i != pire]]
        n_vitesse += 1
    if len(points) == 0:
        return points, n_vitesse, 0
    pos = points["pos"].to_numpy()
    gauche = np.r_[np.inf, np.diff(pos)]
    droite = np.r_[np.diff(pos), np.inf]
    isole = (gauche > isolement_frames) & (droite > isolement_frames)
    return points[~isole], n_vitesse, int(isole.sum())


# --------------------------------------------------------------------------- étape 5


def _ruptures(
    x: np.ndarray,
    y: np.ndarray,
    tt: np.ndarray,
    angle_deg: float,
    vitesse_min: float,
    fenetre: int,
) -> list[int]:
    """Indices `i` où la trajectoire casse entre le point `i` et le point `i + 1`.

    Vitesse avant : sur au plus `fenetre` points se terminant en `i` ; après : sur au plus
    `fenetre` points commençant en `i + 1` (au moins un intervalle de chaque côté). Il y a
    rupture si l'angle entre les deux vitesses dépasse `angle_deg` et que les deux vitesses
    dépassent `vitesse_min` (px/s ; en dessous, par ex. au sommet d'un lob, la direction n'a
    pas de sens). Parmi des ruptures adjacentes, seule la plus marquée est gardée.
    """
    n = len(x)
    angles = np.zeros(max(n - 1, 0))
    for i in range(n - 1):
        a, b = max(0, i - fenetre + 1), min(n - 1, i + fenetre)
        if i - a < 1 or b - (i + 1) < 1:
            continue
        v1 = np.array([x[i] - x[a], y[i] - y[a]]) / (tt[i] - tt[a])
        v2 = np.array([x[b] - x[i + 1], y[b] - y[i + 1]]) / (tt[b] - tt[i + 1])
        if np.hypot(*v1) < vitesse_min or np.hypot(*v2) < vitesse_min:
            continue
        angles[i] = abs(np.degrees(np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1 @ v2)))
    marquees = angles > angle_deg
    ruptures: list[int] = []
    i = 0
    while i < len(marquees):
        if not marquees[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(marquees) and marquees[j + 1]:
            j += 1
        ruptures.append(i + int(np.argmax(angles[i : j + 1])))
        i = j + 1
    return ruptures


def segmenter(
    points: pd.DataFrame, t: np.ndarray, hauteur_image: int, params: ParamsTrajectoire
) -> tuple[list[pd.DataFrame], int]:
    """Découpe la trace en segments (trous > `trou_max_frames`, changements de direction).

    Renvoie `(segments d'au moins segment_min_points points, nombre de points des segments
    trop courts écartés)`. Un segment court est du bruit, sans exception.
    """
    points = points.sort_values("pos")
    if points.empty:
        return [], 0
    pos = points["pos"].to_numpy()
    coupures = np.flatnonzero(np.diff(pos) - 1 > params.trou_max_frames) + 1
    vitesse_min = params.vitesse_min_rupture_rel * hauteur_image
    segments: list[pd.DataFrame] = []
    for ligne in np.split(np.arange(len(points)), coupures):
        suite = points.iloc[ligne]
        tt = t[suite["pos"].to_numpy()] / 1000.0
        ruptures = _ruptures(
            suite["x"].to_numpy(), suite["y"].to_numpy(), tt, params.angle_rupture_deg,
            vitesse_min, params.fenetre_rupture,
        )  # fmt: skip
        for morceau in np.split(np.arange(len(suite)), np.array(ruptures, dtype=int) + 1):
            segments.append(suite.iloc[morceau])
    longs = [s for s in segments if len(s) >= params.segment_min_points]
    return longs, sum(len(s) for s in segments) - sum(len(s) for s in longs)


# --------------------------------------------------------------------------- étape 6


@dataclass(frozen=True)
class Courbe:
    """Courbe x(t), y(t) polynomiale (t en secondes, centré sur `t0`)."""

    t0: float
    coef_x: np.ndarray
    coef_y: np.ndarray

    def evaluer(self, t_ms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tau = (np.asarray(t_ms, dtype=float) - self.t0) / 1000.0
        return np.polyval(self.coef_x, tau), np.polyval(self.coef_y, tau)


def _ajuster(tau: np.ndarray, x: np.ndarray, y: np.ndarray, degre: int, masque: np.ndarray):
    return (
        np.polyfit(tau[masque], x[masque], degre),
        np.polyfit(tau[masque], y[masque], degre),
    )


def _residus(tau, x, y, coef_x, coef_y) -> np.ndarray:
    return np.hypot(np.polyval(coef_x, tau) - x, np.polyval(coef_y, tau) - y)


def ajuster_segment(
    t_ms: np.ndarray, x: np.ndarray, y: np.ndarray, residu_max_px: float
) -> tuple[np.ndarray, Courbe | None]:
    """Ajustement robuste (type RANSAC déterministe) d'une parabole x(t), y(t) à un segment.

    Degré 2 à partir de 5 points, 1 en dessous (sinon l'ajustement est exact et ne détecte
    rien). On retient le jeu d'inliers le plus nombreux (résidu <= `residu_max_px`, à égalité
    la somme des résidus la plus faible), puis on ré-ajuste par moindres carrés sur les
    inliers. Renvoie `(masque des inliers, courbe)` ; courbe `None` si moins de 2 inliers.
    """
    n = len(t_ms)
    t0 = float(np.mean(t_ms))
    tau = (np.asarray(t_ms, dtype=float) - t0) / 1000.0
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    degre = 2 if n >= 5 else 1
    taille = degre + 1
    if n <= taille:
        return np.ones(n, dtype=bool), Courbe(t0, *_ajuster(tau, x, y, 1, np.ones(n, bool)))
    if n <= 10:
        echantillons = list(combinations(range(n), taille))
    else:
        generateur = np.random.default_rng(0)
        echantillons = [generateur.choice(n, taille, replace=False) for _ in range(150)]
    meilleur: tuple[int, float] = (-1, 0.0)
    meilleurs_inliers = np.zeros(n, dtype=bool)
    for echantillon in echantillons:
        choix = np.zeros(n, dtype=bool)
        choix[list(echantillon)] = True
        coef_x, coef_y = _ajuster(tau, x, y, degre, choix)
        residus = _residus(tau, x, y, coef_x, coef_y)
        inliers = residus <= residu_max_px
        cle = (int(inliers.sum()), -float(residus[inliers].sum()))
        if cle > meilleur:
            meilleur, meilleurs_inliers = cle, inliers
    inliers = meilleurs_inliers
    for _ in range(3):  # raffinement : moindres carrés sur les inliers, puis réévaluation
        if inliers.sum() < taille:
            break
        coef_x, coef_y = _ajuster(tau, x, y, degre, inliers)
        nouveaux = _residus(tau, x, y, coef_x, coef_y) <= residu_max_px
        if np.array_equal(nouveaux, inliers):
            break
        inliers = nouveaux
    if inliers.sum() < 2:
        return inliers, None
    coef_x, coef_y = _ajuster(tau, x, y, min(degre, int(inliers.sum()) - 1), inliers)
    return inliers, Courbe(t0, coef_x, coef_y)


# --------------------------------------------------------------------------- étape 7


def interpoler_segment(pos_inliers: np.ndarray, trou_max_frames: int) -> np.ndarray:
    """Positions de frame à combler dans un segment (valeurs données ensuite par la courbe) : trous de 1 à `trou_max_frames` frames
    entre deux points retenus consécutifs. Jamais avant le premier ni après le dernier."""
    manquantes: list[np.ndarray] = []
    pos = np.sort(pos_inliers)
    for a, b in pairwise(pos):
        if 1 <= b - a - 1 <= trou_max_frames:
            manquantes.append(np.arange(a + 1, b))
    return np.concatenate(manquantes) if manquantes else np.array([], dtype=int)


# --------------------------------------------------------------------------- assemblage


def construire_ball_csv_diagnostic(
    candidats,
    t_ms: Mapping[int, float] | pd.Series,
    hauteur_image: int,
    masque: MasqueCourt | None = None,
    params: ParamsTrajectoire | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Comme `construire_ball_csv`, avec en plus le décompte des candidats écartés à chaque
    étape (pour `summary.json`)."""
    params = (params or ParamsTrajectoire()).resoudre(hauteur_image)
    frames, t = _temps_par_frame(t_ms)
    cands = normaliser_candidats(candidats)
    diag: dict[str, int] = {"n_candidats": len(cands)}

    index = pd.Series(np.arange(len(frames)), index=frames)
    cands["pos"] = cands["frame_idx"].map(index)
    hors = cands["pos"].isna()
    diag["n_hors_intervalle"] = int(hors.sum())
    cands = cands[~hors].astype({"pos": "int64"})

    avant = len(cands)
    cands = filtrer_masque(cands, masque)
    diag["n_hors_masque"] = avant - len(cands)

    avant = len(cands)
    cands = retirer_statiques(
        cands, t, params.rayon_statique_px, params.duree_statique_ms,
        params.fraction_presence_statique,
    )  # fmt: skip
    diag["n_statiques"] = avant - len(cands)

    avant = len(cands)
    points = selectionner_viterbi(cands, t, hauteur_image, params)
    diag["n_non_retenus_selection"] = avant - len(points)

    points, n_vitesse, n_isoles = rejeter_aberrants(
        points, t, params.vitesse_max, params.isolement_frames
    )
    diag["n_vitesse_impossible"] = n_vitesse
    diag["n_isoles"] = n_isoles

    segments, n_courts = segmenter(points, t, hauteur_image, params)
    diag["n_segments_courts"] = n_courts

    n = len(frames)
    x_out = np.full(n, np.nan)
    y_out = np.full(n, np.nan)
    score_out = np.full(n, np.nan)
    source_out = np.full(n, None, dtype=object)
    n_rejetes = 0
    for segment in segments:
        pos = segment["pos"].to_numpy()
        inliers, courbe = ajuster_segment(
            t[pos], segment["x"].to_numpy(), segment["y"].to_numpy(), params.residu_max_px
        )
        n_rejetes += int((~inliers).sum())
        if courbe is None or inliers.sum() < params.segment_min_points:
            n_rejetes += int(inliers.sum())
            continue
        retenus = segment[inliers]
        p = retenus["pos"].to_numpy()
        x_out[p], y_out[p] = retenus["x"].to_numpy(), retenus["y"].to_numpy()
        score_out[p] = retenus["score"].clip(0, 1).to_numpy()
        source_out[p] = "detected"
        a_combler = interpoler_segment(p, params.trou_max_frames)
        if len(a_combler):
            x_out[a_combler], y_out[a_combler] = courbe.evaluer(t[a_combler])
            source_out[a_combler] = "interpolated"
    diag["n_rejetes_ajustement"] = n_rejetes
    visible = pd.notna(source_out)
    diag["n_detectes"] = int((source_out == "detected").sum())
    diag["n_interpoles"] = int((source_out == "interpolated").sum())

    table = pd.DataFrame(
        {
            "frame_idx": frames,
            "t_ms": t,
            "x": x_out,
            "y": y_out,
            "visible": visible.astype("int64"),
            "score": score_out,
            "source": source_out,
        },
        columns=COLONNES_BALL,
    )
    return table, diag


def construire_ball_csv(
    candidats,
    t_ms: Mapping[int, float] | pd.Series,
    hauteur_image: int,
    masque: MasqueCourt | None = None,
    params: ParamsTrajectoire | None = None,
) -> pd.DataFrame:
    """Candidats du réseau -> table `ball.csv` (une ligne par frame de `t_ms`, dans l'ordre).

    `candidats` : DataFrame `frame_idx, x, y, score` ou objets ayant ces attributs.
    `t_ms` : horodatage réel par `frame_idx` (toutes les frames de l'intervalle analysé).
    Frames sans balle : `visible = 0`, `x`, `y`, `score` et `source` vides.
    """
    return construire_ball_csv_diagnostic(candidats, t_ms, hauteur_image, masque, params)[0]


def ecrire_ball_csv(table: pd.DataFrame, chemin: str | Path) -> None:
    """Écrit la table au format CSV du contrat (cellule vide = null)."""
    table.to_csv(chemin, index=False, float_format="%.4f")
