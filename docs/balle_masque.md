# Masque de court et post-traitement de la balle (Lot 4)

Code : `src/tennisvision/balle/masque.py` et `src/tennisvision/balle/trajectoire.py`.

## Masque de court

Les candidats balle hors du masque sont écartés (balles des courts voisins). Deux sources :

- **`court.json`** : le double est agrandi de 4 m sur les côtés et de 6 m derrière chaque fond
  de court (paramétrable), projeté dans l'image avec l'inverse de l'homographie, puis prolongé
  vers le haut côté loin (une balle en l'air est au-dessus de sa position au sol). Un segment
  par intervalle `valid_from_frame` / `valid_to_frame` ; une frame hors segment prend le
  segment le plus proche.
- **Polygone manuel**, fichier JSON valable pour toute la vidéo, en pixels de l'image (origine
  en haut à gauche, après rotation), au moins 3 sommets :

```json
{"polygone": [[120, 80], [1800, 80], [1900, 1060], [20, 1060]]}
```

Sans masque, tout est accepté.

## Post-traitement hors-ligne

`construire_ball_csv(candidats, t_ms, hauteur_image, masque, params)` enchaîne : masque, retrait
des objets statiques (amas dans ~0,6 % de la hauteur plus de 1 s, même détectés par
intermittence), sélection par Viterbi (un candidat au plus par frame, état « pas de balle »),
rejet des points aberrants, segmentation (trous > 10 frames, changements de direction),
ajustement parabolique robuste par segment, interpolation des trous de 10 frames au plus
(`source = interpolated`). `construire_ball_csv_diagnostic` renvoie en plus le décompte des
candidats écartés à chaque étape, pour `summary.json`.
