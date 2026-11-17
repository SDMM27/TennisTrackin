"""Tests des augmentations de triplets : cohérence des coordonnées, photométrie, déterminisme."""

import cv2
import numpy as np
import pytest

from tennisvision.balle.augmentations import (
    Augmenteur,
    Echantillon,
    ParametresAugmentation,
    distracteurs,
    flou_de_bouge,
    grille,
    photometrie,
    recadrer,
    recompression_jpeg,
    retourner_horizontalement,
)

H, W = 288, 512
POINTS = np.array([[100.0, 60.0], [112.0, 66.0], [124.0, 72.0]])


def dessiner(points, visibles):
    """3 images grises avec un disque blanc de rayon 4 aux points visibles."""
    fond = np.random.default_rng(0).integers(60, 90, (H, W, 1), dtype=np.uint8)
    frames = []
    for (x, y), visible in zip(points, visibles, strict=True):
        image = np.repeat(fond, 3, axis=2).copy()
        if visible:
            cv2.circle(image, (round(x), round(y)), 4, (255, 255, 255), -1)
        frames.append(image)
    return tuple(frames)


def echantillon(visibles=(True, True, True)):
    visibles = np.array(visibles)
    return Echantillon(dessiner(POINTS, visibles), POINTS.copy(), visibles)


def barycentre_blanc(image):
    ys, xs = np.nonzero(image.min(axis=2) > 200)
    return xs.mean(), ys.mean()


def test_retournement_x_devient_largeur_moins_un_moins_x():
    ech = echantillon()
    sortie = retourner_horizontalement(ech, None, ParametresAugmentation())
    assert sortie.points[:, 0] == pytest.approx(W - 1 - POINTS[:, 0])
    assert sortie.points[:, 1] == pytest.approx(POINTS[:, 1])
    for image, point in zip(sortie.frames, sortie.points, strict=True):
        assert barycentre_blanc(image) == pytest.approx(tuple(point), abs=0.6)


def test_recadrage_deplace_la_cible_avec_l_image():
    rng = np.random.default_rng(3)
    sortie = recadrer(echantillon(), rng, ParametresAugmentation(recadrage_max=0.125))
    assert sortie.frames[0].shape == (H, W, 3)
    for image, point, visible in zip(sortie.frames, sortie.points, sortie.visibles, strict=True):
        assert visible
        assert barycentre_blanc(image) == pytest.approx(tuple(point), abs=1.0)


def test_recadrage_balle_sortie_du_cadre_devient_invisible():
    ech = echantillon()
    points = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    ech = Echantillon(dessiner(points, [True] * 3), points, np.array([True] * 3))
    sorties = [
        recadrer(ech, np.random.default_rng(s), ParametresAugmentation(recadrage_max=0.5))
        for s in range(8)
    ]
    assert any(not s.visibles.any() for s in sorties)  # coin haut-gauche rogné


@pytest.mark.parametrize("fonction", [flou_de_bouge, recompression_jpeg, photometrie, grille])
def test_augmentations_photometriques_gardent_les_coordonnees(fonction):
    ech = echantillon((True, False, True))
    sortie = fonction(ech, np.random.default_rng(1), ParametresAugmentation())
    assert np.array_equal(sortie.points, ech.points)
    assert np.array_equal(sortie.visibles, ech.visibles)
    assert len(sortie.frames) == 3
    assert all(f.shape == (H, W, 3) and f.dtype == np.uint8 for f in sortie.frames)
    assert any(not np.array_equal(a, b) for a, b in zip(sortie.frames, ech.frames, strict=True))


def test_meme_transformation_sur_les_trois_images():
    # Trois images identiques en entrée : elles le restent, quelle que soit l'augmentation
    # (flou, JPEG, photométrie et grille sont fixes d'une image à l'autre).
    image = echantillon().frames[0]
    ech = Echantillon((image, image.copy(), image.copy()), POINTS.copy(), np.array([True] * 3))
    for fonction in (flou_de_bouge, recompression_jpeg, photometrie, grille, recadrer):
        sortie = fonction(ech, np.random.default_rng(5), ParametresAugmentation())
        assert np.array_equal(sortie.frames[0], sortie.frames[1])
        assert np.array_equal(sortie.frames[0], sortie.frames[2])


def test_photometrie_et_flou_respectent_les_bornes():
    ech = echantillon()
    sombre = photometrie(
        ech,
        np.random.default_rng(2),
        ParametresAugmentation(photometrie_force=0.0, gamma_force=0.0),
    )
    assert all(np.array_equal(a, b) for a, b in zip(sombre.frames, ech.frames, strict=True))
    flou = flou_de_bouge(ech, np.random.default_rng(2), ParametresAugmentation())
    # un flou moyenne : l'amplitude du blanc au centre de la balle diminue ou reste, jamais ne monte
    assert flou.frames[0].max() <= ech.frames[0].max()


def test_distracteurs_n_ajoutent_pas_de_cible_et_epargnent_la_vraie_balle():
    ech = echantillon((True, False, True))
    params = ParametresAugmentation(distracteurs_max=3)
    modifie = False
    for graine in range(5):
        sortie = distracteurs(ech, np.random.default_rng(graine), params)
        assert np.array_equal(sortie.points, ech.points)
        assert np.array_equal(sortie.visibles, ech.visibles)
        for image, original, point, visible in zip(
            sortie.frames, ech.frames, POINTS, ech.visibles, strict=True
        ):
            if visible:  # voisinage de la vraie balle intact
                x, y = round(point[0]), round(point[1])
                assert np.array_equal(
                    image[y - 15 : y + 16, x - 15 : x + 16],
                    original[y - 15 : y + 16, x - 15 : x + 16],
                )
        modifie |= any(not np.array_equal(a, b) for a, b in zip(sortie.frames, ech.frames))
    assert modifie


def test_distracteurs_jaune_vert():
    ech = echantillon((False, False, False))
    sortie = distracteurs(ech, np.random.default_rng(0), ParametresAugmentation())
    difference = sortie.frames[0].astype(int) - ech.frames[0].astype(int)
    r, g, b = (difference[:, :, c].max() for c in range(3))
    assert g > 80 and r > 60 and b < g  # teinte jaune-vert, plus claire que le fond


def test_determinisme_par_graine():
    params = ParametresAugmentation(
        p_retournement=1, p_recadrage=1, p_distracteurs=1, p_grille=1, p_flou=1,
        p_photometrie=1, p_jpeg=1,
    )  # fmt: skip
    augmenteur = Augmenteur(params)
    a = augmenteur(echantillon(), np.random.default_rng(42))
    b = augmenteur(echantillon(), np.random.default_rng(42))
    c = augmenteur(echantillon(), np.random.default_rng(43))
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames, strict=True))
    assert np.array_equal(a.points, b.points)
    assert not all(np.array_equal(x, y) for x, y in zip(a.frames, c.frames, strict=True))


def test_probabilite_zero_est_l_identite():
    ech = echantillon()
    sortie = Augmenteur(ParametresAugmentation().desactivees())(ech, np.random.default_rng(0))
    assert all(np.array_equal(x, y) for x, y in zip(sortie.frames, ech.frames, strict=True))
    assert np.array_equal(sortie.points, ech.points)
