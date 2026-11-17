# Suivi de la balle (Lot 4)

Code : `src/tennisvision/balle/`. Décision de modèle : `docs/adr/004-modele-balle.md` ; licences :
`docs/adr/005-licences.md`. Masque de court et détail du post-traitement :
[`docs/balle_masque.md`](balle_masque.md).

La balle est suivie par son propre réseau, pas par YOLO : ByteTrack ne traite plus que la
classe « personne » (`tracks.csv` ne contient que des personnes). La trajectoire est écrite
dans `ball.csv` (contrat : `docs/contracts.md`, une ligne par frame traitée).

## Chaîne de traitement

1. **Réseau** (`detecteur.py`) : trois images consécutives en entrée, une carte de chaleur par
   image en sortie (WASB-SBDT HRNet par défaut ; TrackNetV2 du même dépôt en comparaison). Chaque
   image est prédite une seule fois (triplets disjoints). Les cartes sont réduites en
   **candidats** `(frame_idx, x, y, score)` : composantes connexes au-dessus de `--ball-threshold`,
   centre pondéré par la carte, ramené en pixels de l'image d'origine.
2. **Masque de court** : les candidats hors du court étendu sont écartés (balles des courts
   voisins). Source : `court.json` (`--court`) ou polygone manuel (`--mask`) ; sans masque, tout
   est accepté.
3. **Objets statiques** : un amas de candidats immobile (~0,6 % de la hauteur d'image) pendant
   plus de 1 s est retiré, même détecté par intermittence (roue de caméra, marque sur le court).
4. **Viterbi** : au plus un candidat par frame, avec un état « pas de balle », sous contrainte de
   vitesse maximale et de régularité du mouvement.
5. **Points aberrants** : vitesse physiquement impossible, points isolés.
6. **Segments** : coupure aux trous de plus de 10 frames et aux changements de direction
   (rebonds, frappes) ; les segments de moins de 3 points sont du bruit.
7. **Ajustement** : parabole robuste par segment (les points trop éloignés de la courbe sont
   rejetés).
8. **Interpolation** : les trous de 10 frames au plus à l'intérieur d'un segment sont comblés
   par la courbe et marqués `source = interpolated` (les autres frames visibles sont
   `detected`, `visible = 0` sinon).

Toutes les distances de seuil sont relatives à la hauteur de l'image (la même chaîne sert en 720p
et en 1080p) ; `ParamsTrajectoire` expose les paramètres.

## Modèles et licences

| Nom du registre | Origine | Remarque |
|---|---|---|
| `wasb_tennis` (défaut) | WASB-SBDT (BMVC 2023), HRNet | code sous licence MIT |
| `tracknetv2_tennis` | TracknetV2 (U-Net) du même banc | code sous licence MIT |

Le code d'origine est celui de WASB-SBDT (licence MIT, Copyright (c) 2023 NTT Communications
Corporation). Les **poids** ont été entraînés sur le jeu de données TrackNet tennis, dont la
**licence n'a pas été vérifiée** : tant qu'elle ne l'est pas, ils servent à la référence et à la
recherche, pas à une distribution (ADR-005). Téléchargement explicite :
`tennisvision models pull wasb_tennis`.

**TrackNetV3** n'est pas comparé : il n'existe pas de poids publics pour le tennis (seulement
pour le badminton), et réentraîner pour une simple comparaison coûterait plus que ce qu'elle
apporterait. La référence se fait donc entre WASB et TrackNetV2, qui ont des poids tennis.

## 30 et 60 fps

Les réseaux sont entraînés à environ 30 fps. Le détecteur reçoit `fps = fps_nominal` de la
vidéo et en déduit un `pas_temporel = round(fps / 30)` (au moins 1) : à 60 fps, les triplets
sont formés d'images espacées de 2 (t, t+2, t+4), ce qui rend au réseau le mouvement entre
images qu'il a appris ; chaque image reste prédite une fois. Le `pas_temporel` est écrit dans
`summary.json`. Le post-traitement travaille sur les `t_ms` réels, donc sans dépendre de la
cadence.

## Utilisation

```bash
# Analyse complète : personnes (YOLO) + balle (réseau dédié) -> tracks.csv, ball.csv, ...
tennisvision analyze match.mp4 --court court.json
tennisvision analyze match.mp4 --mask masque.json --ball-threshold 0.4
tennisvision analyze match.mp4 --ball-model aucun        # sans la balle

# La balle seule : ball.csv et summary.json, sans YOLO ni vidéo annotée
tennisvision ball track match.mp4 --start 250 --end 500 --court court.json

# Débit (JSON) et export ONNX
tennisvision ball bench --model wasb_tennis --device cpu --batch 4 --triplets 8
pip install "tennisvision[onnx]"
tennisvision ball export-onnx wasb.onnx --model wasb_tennis --opset 17
```

`--court` et `--mask` s'excluent. Les poids absents du cache arrêtent la commande avant toute
écriture, avec l'invite `tennisvision models pull wasb_tennis`. Même options pour
`tennisvision tenniset analyze`.

La vidéo annotée est écrite après le post-traitement, en seconde passe de décodage : traînée
jaune, cercle plein pour une position détectée, cercle creux pour une position interpolée.

## Affinage

`tennisvision ball train` affine un réseau (WASB HRNet par défaut) sur des triplets d'images
annotés. Aucun entraînement réel n'a encore été lancé : le code est testé sur des données
synthétiques.

```bash
tennisvision ball train --source amateur:data/gt --output-dir runs/amateur1 \
    --init wasb_tennis --epochs 10 --batch 8 --device 0
tennisvision ball train --source tracknet:/donnees/tennis --source amateur:data/gt \
    --output-dir runs/mixte --temporal-step 2      # vidéos à 60 fps
```

**Sources** (`--source`, répétable) :

- `tracknet:<racine>` : arborescence TrackNet tennis `<racine>/<partie>/<Clip>/` avec les images
  (`0000.jpg`…) et `Label.csv` (`file name, visibility, x-coordinate, y-coordinate, status`).
  Visibilité 1 et 2 : balle visible ; 0 et 3 : non visible (carte cible vide). La validation prend
  une partie (match) entière, ou un clip s'il n'y a qu'une partie (`--val-fraction`).
- `amateur:<gt_dir>` : jeu amateur du Lot 2 (`<gt_dir>/<match>/ball.csv` et la vidéo, cherchée
  dans `--videos-dir` ou dans le dossier du match). Les splits `train` et `val` de `splits.json` sont
  utilisés ; le split `test` n'est **jamais** lu. Seules les frames présentes dans `ball.csv` sont
  des cibles, un triplet est valide si ses 3 images le sont. Les images décodées (512x288, JPEG
  qualité 95) sont mises en cache dans `--cache-dir` (défaut : `<output-dir>/cache_frames`).
  `--no-interpolated` écarte les positions interpolées.

**Données et perte** : mêmes prétraitement et transformation affine qu'à l'inférence ; cible =
disque binaire de rayon 2,5 px (WASB, `gen_binary_map`) ; perte = entropie croisée pondérée de WASB
(`WBCELoss`) ; Adam, décroissance en cosinus. Validation à chaque époque : F1 à 4 px (espace
modèle, meilleur candidat par image).

**Augmentations** (désactivables par `--no-augmentation`), identiques sur les 3 images d'un
triplet : miroir, recadrage/agrandissement (WASB), balles leurres jaune-vert hors de la vraie balle
(sans cible), grillage ou filet, flou de mouvement, recompression JPEG, luminosité/contraste/gamma.
Réglages dans `ParametresAugmentation`.

**Sorties** dans `--output-dir` : `best.pth` (meilleure époque, `state_dict` seul, chargeable avec
`weights_only=True`) et `entrainement.json` (configuration, graine, métriques par époque, somme
SHA-256 des poids initiaux, version de `tennisvision`). Sur CPU, deux exécutions de même graine
donnent les mêmes résultats.

**Utiliser les poids affinés** : un fichier local exige l'architecture.

```bash
tennisvision analyze match.mp4 --ball-model runs/amateur1/best.pth --ball-architecture hrnet
tennisvision ball track match.mp4 --model runs/amateur1/best.pth --ball-architecture hrnet
```

Évaluer ensuite avec `tennisvision eval` sur le split `test` gelé et comparer au modèle d'origine.

## Mesure de référence (poids publics, sans affinage)

Mesurée le 2026-11-17 sur le point P00000001 de V006 (TenniSet, frames 25740 à 25984, 245
frames, 1280×720, 25 fps nominal), CPU seul (torch 2.15, Windows), commande
`tennisvision ball track ... --model <modèle>`, seuil 0,5, sans masque de court.

| Modèle | Frames avec balle | Détectées | Interpolées | Débit bout en bout (CPU) |
|---|---|---|---|---|
| `wasb_tennis` | 67,8 % | 150 | 16 | 6,5 fps |
| `tracknetv2_tennis` | 67,3 % | 156 | 9 | 4,4 fps |

Constats :

- **Roue de caméra** : elle n'est retenue sur aucune frame par aucun des deux modèles (toutes
  les positions retenues ont x ≤ 887 px ; la roue est vers x ≈ 1220 px). Avec YOLO COCO, elle
  gagnait sur la majorité des frames (constat C3).
- Les positions retenues, superposées à l'image, suivent le lancer de service et les échanges ;
  le contrôle est **visuel**, faute de vérité terrain balle sur TenniSet.
- WASB est 1,5 fois plus rapide que TrackNetV2 sur CPU et donne le même taux de présence : il
  reste le modèle par défaut.

Ce qui **n'est pas encore mesuré** :

- précision, rappel et F1 à 4 et 10 px : il faut la vérité terrain balle du jeu amateur
  (Lot 2, niveau 3), puis `tennisvision eval` ;
- le débit GPU : l'environnement de mesure n'a qu'un torch CPU (objectif : ≥ 30 fps GPU) ;
- l'objectif CPU (≥ 5 fps) est atteint par WASB, pas par TrackNetV2 ;
- l'affinage n'a pas été lancé (pas encore de frames amateures annotées).

## `summary.json` : bloc `balle`

```json
"balle": {
  "modele": {"nom": "wasb_tennis", "version": "bmvc2023", "sha256": "..."},
  "pas_temporel": 1,
  "seuil_score": 0.5,
  "taux_visible": 0.28,
  "n_detectes": 17,
  "n_interpoles": 0,
  "diagnostic": {"n_candidats": 19, "n_hors_masque": 0, "n_statiques": 0, "...": 0},
  "masque": null
}
```

- `taux_visible` : frames avec `visible = 1` sur frames traitées (détectées et interpolées) ;
- `diagnostic` : candidats écartés à chaque étape (masque, statiques, sélection, vitesse,
  isolés, segments courts, ajustement) ;
- `masque` : `"court.json"`, `"polygone"` ou `null` ;
- `balle` vaut `null` avec `--ball-model aucun`, et `sorties.ball` est alors `null` ;
- `sorties.ball = "ball.csv"` sinon.

Avec `ball track`, `modele` (YOLO) vaut `null`, `n_identites` et `top_identites` sont absents et
`sorties.tracks` est `null`. Les anciennes clés `n_detections_balle` et `taux_detection_balle`
n'existent plus : elles comptaient les détections brutes de YOLO, faux positifs compris.
`fps_traitement` est mesuré de bout en bout (passe de décodage, post-traitement, vidéo annotée).
