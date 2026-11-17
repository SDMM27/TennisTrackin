"""Masque de court : ignore les détections de balle hors du court étendu.

Deux sources possibles :

- un `court.json` (contrat du Lot 0) : pour chaque segment valable sur un intervalle de frames,
  le double du court est agrandi de marges (4 m sur les côtés, 6 m derrière chaque fond de
  court par défaut), puis projeté dans l'image avec l'inverse de l'homographie (image px →
  court m, origine au centre du filet). Une balle en l'air apparaît AU-DESSUS de sa position au
  sol : les deux coins du côté loin sont donc aussi reliés au bord haut de l'image (même x), ce
  qui inclut la région au-dessus de la moitié lointaine du court ;
- un polygone manuel, fichier JSON `{"polygone": [[x, y], ...]}` en pixels de l'image (au
  moins 3 sommets, même repère que le contrat : origine en haut à gauche, après rotation),
  valable pour toutes les frames.

Une frame hors de tous les segments de `court.json` utilise le segment le plus proche. Sans
masque (`None` côté appelant), tout est accepté.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from tennisvision.schemas import L, Wd

MARGE_LATERALE_M = 4.0
MARGE_FOND_M = 6.0


def _dans_polygone(px: np.ndarray, py: np.ndarray, polygone: np.ndarray) -> np.ndarray:
    """Appartenance de points à un polygone (lancer de rayon, vectorisé)."""
    dedans = np.zeros(px.shape, dtype=bool)
    n = len(polygone)
    xj, yj = polygone[-1]
    for i in range(n):
        xi, yi = polygone[i]
        franchit = (yi > py) != (yj > py)
        denominateur = (yj - yi) if yj != yi else 1.0
        x_inter = (xj - xi) * (py - yi) / denominateur + xi
        dedans ^= franchit & (px < x_inter)
        xj, yj = xi, yi
    return dedans


def polygone_court_etendu(
    homographie: Sequence[Sequence[float]],
    image_size: tuple[int, int],
    marge_laterale_m: float = MARGE_LATERALE_M,
    marge_fond_m: float = MARGE_FOND_M,
) -> np.ndarray:
    """Polygone image (N×2) du court étendu, prolongé vers le haut côté loin.

    `homographie` : image px → court m. `image_size` : (largeur, hauteur) en pixels.
    """
    largeur, hauteur = image_size
    h_inverse = np.linalg.inv(np.asarray(homographie, dtype=float))
    demi_largeur = Wd + marge_laterale_m
    demi_longueur = L + marge_fond_m
    # Ordre : loin gauche, loin droit, proche droit, proche gauche (y court < 0 = côté loin)
    coins_m = np.array(
        [
            [-demi_largeur, -demi_longueur, 1.0],
            [demi_largeur, -demi_longueur, 1.0],
            [demi_largeur, demi_longueur, 1.0],
            [-demi_largeur, demi_longueur, 1.0],
        ]
    )
    projetes = coins_m @ h_inverse.T
    if np.any(projetes[:, 2] <= 0):
        raise ValueError("Marges trop grandes : un coin du court étendu passe derrière la caméra")
    coins = projetes[:, :2] / projetes[:, 2:3]
    loin_gauche, loin_droit, proche_droit, proche_gauche = coins
    polygone = np.array(
        [
            [loin_gauche[0], 0.0],
            loin_gauche,
            proche_gauche,
            proche_droit,
            loin_droit,
            [loin_droit[0], 0.0],
        ]
    )
    polygone[:, 0] = np.clip(polygone[:, 0], 0, largeur)
    polygone[:, 1] = np.clip(polygone[:, 1], 0, hauteur)
    return polygone


class MasqueCourt:
    """Région de l'image où une balle est plausible, par frame.

    `segments` : liste de `(frame_debut, frame_fin | None, polygone N×2)` ; `frame_fin` est
    incluse, `None` = jusqu'à la fin.
    """

    def __init__(self, segments: Sequence[tuple[int, int | None, np.ndarray]]):
        if not segments:
            raise ValueError("Un masque de court demande au moins un segment")
        self._segments = sorted(
            (
                (int(debut), float("inf") if fin is None else int(fin), np.asarray(poly, float))
                for debut, fin, poly in segments
            ),
            key=lambda segment: segment[0],
        )
        for _, _, poly in self._segments:
            if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
                raise ValueError("Un polygone de masque demande au moins 3 sommets (x, y)")

    @classmethod
    def depuis_court_json(
        cls,
        court: str | Path | Mapping,
        marge_laterale_m: float = MARGE_LATERALE_M,
        marge_fond_m: float = MARGE_FOND_M,
    ) -> "MasqueCourt":
        """Construit le masque d'un `court.json` (chemin ou document déjà chargé)."""
        if not isinstance(court, Mapping):
            court = json.loads(Path(court).read_text(encoding="utf-8"))
        try:
            image_size = (float(court["image_size"][0]), float(court["image_size"][1]))
            segments = [
                (
                    segment["valid_from_frame"],
                    segment["valid_to_frame"],
                    polygone_court_etendu(
                        segment["homography"], image_size, marge_laterale_m, marge_fond_m
                    ),
                )
                for segment in court["segments"]
            ]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"court.json inexploitable pour le masque : {exc!r}") from exc
        return cls(segments)

    @classmethod
    def depuis_polygone(cls, polygone: str | Path | Mapping) -> "MasqueCourt":
        """Construit le masque d'un polygone manuel `{"polygone": [[x, y], ...]}` en pixels."""
        if not isinstance(polygone, Mapping):
            polygone = json.loads(Path(polygone).read_text(encoding="utf-8"))
        try:
            sommets = np.asarray(polygone["polygone"], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Polygone manuel inexploitable : {exc!r}") from exc
        return cls([(0, None, sommets)])

    def _indice_segment(self, frame_idx: int) -> int:
        meilleur, meilleure_distance = 0, float("inf")
        for i, (debut, fin, _) in enumerate(self._segments):
            distance = max(debut - frame_idx, frame_idx - fin, 0)
            if distance < meilleure_distance:
                meilleur, meilleure_distance = i, distance
        return meilleur

    def contient(self, frame_idx: int, x: float, y: float) -> bool:
        """Vrai si le point (x, y) est dans le masque valable pour cette frame."""
        polygone = self._segments[self._indice_segment(frame_idx)][2]
        return bool(_dans_polygone(np.array([float(x)]), np.array([float(y)]), polygone)[0])

    def contient_vec(self, frame_idx, x, y) -> np.ndarray:
        """Version vectorisée de `contient` : tableaux de même longueur, renvoie un booléen
        par point."""
        frame_idx = np.asarray(frame_idx)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        resultat = np.zeros(x.shape, dtype=bool)
        if x.size == 0:
            return resultat
        indices = np.empty(x.shape, dtype=int)
        for frame in np.unique(frame_idx):
            indices[frame_idx == frame] = self._indice_segment(int(frame))
        for i, (_, _, polygone) in enumerate(self._segments):
            choisis = indices == i
            if choisis.any():
                resultat[choisis] = _dans_polygone(x[choisis], y[choisis], polygone)
        return resultat
