# Feuille de route — TennisTrackin

> **Note (2026-11-10)** : cette feuille de route est remplacée par le plan révisé
> [docs/PLAN_AMELIORATION.md](docs/PLAN_AMELIORATION.md) (§ 3.6 : roadmap révisée en phases A–G,
> lots 0–12). Elle change notamment parce que les vidéos amateurs passent de l'étape 6 au début
> (jeu de référence, [ADR-003](docs/adr/003-jeu-amateur-reference.md)).

Chaque étape produit un **livrable démontrable seul** et une **métrique chiffrée** calculée
contre les annotations TenniSet (ou un petit jeu annoté à la main quand TenniSet ne suffit pas).
Les évaluations vivront dans `src/tennisvision/eval/` et écriront leurs résultats dans `reports/`,
pour pouvoir suivre la progression d'une version à l'autre.

## Ce que contiennent vraiment les annotations TenniSet

À garder en tête, car ça conditionne ce qu'on peut mesurer :

| Événement | Champs | Exemple (V006) |
|---|---|---|
| `Point` | début/fin (frames), `Winner`, `Score`, `desc` (texte libre) | 81 points |
| `Serve` | début/fin, `Player` (Near/Far), `Result` (In/…) | 112 services |
| `Hit` | début/fin, `Player` (Near/Far), `Side` (Left/Right), `Type` (Topspin/Slice/Flat/Volley/Smash/Unsure) | 220 frappes |
| `Game`, `Set` | `Winner`, `Score`, `Server` / `Near` | 13 jeux, 2 sets |
| `SPLITS` | intervalles Train / Val / Test par vidéo | — |

Points importants :
- **Pas de boîtes englobantes de balle ni de lignes de court** : le fine-tuning balle et la
  calibration demandent des labels externes ou annotés par nous.
- **Pas de label « coup droit / revers »** direct : `Side` (Left/Right) doit être converti
  avec la latéralité du joueur et la convention de côté. À vérifier sur un échantillon avant
  de s'en servir comme vérité terrain.
- **Pas de label « coup gagnant / faute »** structuré : seulement la `desc` en texte libre.
- Ce sont des images de **diffusion TV** (plans de coupe, ralentis) alors que tes vidéos seront
  en caméra fixe : prévoir un petit jeu annoté sur **tes propres matchs** pour mesurer l'écart
  de domaine.

---

## Étape 0 — Ingestion + détection/tracking ✅ (cette livraison)

- **Livrable** : `python -m tennisvision analyze <video>` → `annotated.mp4`, `tracks.csv`,
  `summary.json` ; `python -m tennisvision clip` pour extraire un rallye.
- **Qualité** : taux de frames où la balle est détectée pendant les points (baseline YOLO COCO,
  attendu bas), stabilité des ID joueurs (nombre d'identités « personne » par point, idéal = 2),
  vitesse de traitement (frames/s CPU).
- **Mesuré** (point `P00000001`, V006, CPU) : ~10 fps à 640 px ; 31 identités « personne »
  (public et arbitres compris) ; vraie balle sur < 2 % des frames. À 1280 px, le taux brut
  monte à 69 % mais 169 détections sur 173 sont une roue de caméra immobile : d'où la nécessité
  d'une vérité terrain balle avant tout chiffre de « détection ».

## Étape 1 — Suivi de balle fiable

- **Approche** : fine-tune d'un détecteur balle (YOLO petite résolution d'objet, ou TrackNet qui
  exploite 3 frames consécutives), puis filtre de Kalman + interpolation des trous.
  Labels : dataset public de balle de tennis + quelques centaines de frames TenniSet annotées.
- **Livrable** : vidéo avec trajectoire de balle continue ; `ball.csv` (x, y, visible, interpolé).
- **Qualité** : précision / rappel / F1 de détection avec tolérance ≤ 10 px sur des frames
  annotées tenues à l'écart ; taux de détection pendant les points avant/après fine-tune.
  Contrôle indirect : les changements de direction de la balle tombent-ils sur les `Hit` TenniSet ?

## Étape 2 — Calibration du court + vue de dessus

- **Approche** : détection de lignes (Hough / segmentation) → intersections → homographie vers
  le modèle de court ITF (23,77 m × 10,97 m). Caméra fixe ⇒ une homographie par plan, avec
  détection des changements de plan pour la diffusion TV.
- **Livrable** : minimap animée (joueurs + balle en vue de dessus) incrustée dans la vidéo.
- **Qualité** : erreur de reprojection (px et cm) sur ~20 frames par vidéo où l'on a cliqué les
  coins du court à la main ; % de frames où la calibration réussit.

## Étape 3 — Détection et classification des coups

- **3a. Instant de frappe** : changement de direction de la balle + proximité d'un joueur.
  Qualité : précision / rappel / F1 contre les débuts `Hit` + `Serve`, tolérance ±5 frames (0,2 s).
- **3b. Type de coup** (service / coup droit / revers) : séquences de poses (YOLO-pose) autour de
  l'instant de frappe → petit classifieur (GRU/1D-CNN ou gradient boosting sur features).
  Qualité : accuracy et F1 par classe + matrice de confusion sur le split **Test** de TenniSet
  (entraînement sur Train, réglages sur Val, jamais l'inverse).
- **Livrable** : vidéo avec étiquette du coup au moment de la frappe ; `shots.csv`.

## Étape 4 — Logique de jeu

- **Approche** : segmentation des points (début = service, fin = balle morte / inactivité),
  attribution du point, machine à états du score (jeux, sets, tie-break).
- **Livrable** : `points.csv` (début, fin, nb de frappes, serveur, gagnant estimé) + score courant.
- **Qualité** : % de points TenniSet retrouvés (IoU temporel ≥ 0,5), erreur moyenne sur la
  longueur d'échange, exactitude du gagnant du point (`Point.Winner`) et de `Serve.Result`,
  % de jeux dont le score final est juste.
- **Exploratoire assumé** : coup gagnant vs faute directe. Nécessite d'étiqueter à la main un
  sous-ensemble à partir des `desc` ; on publiera une matrice de confusion, pas une promesse.

## Étape 5 — Rapport de statistiques

- **Livrable** : rapport HTML + CSV par match : services (et % de premières), coups droits /
  revers par joueur, longueur moyenne et distribution des échanges, heatmaps de zones de frappe
  et de rebond sur la minimap, points gagnés.
- **Qualité** : pour chaque statistique, écart entre la valeur calculée par le pipeline et la
  même statistique calculée depuis les annotations TenniSet (erreur absolue et relative),
  affiché dans le rapport lui-même.

## Étape 6 — Passage à tes propres vidéos

- **Livrable** : le pipeline complet sur un de tes matchs filmés.
- **Qualité** : annoter à la main ~10 points d'un de tes matchs (mêmes champs que TenniSet) et
  recalculer toutes les métriques ci-dessus pour mesurer la perte due au changement de domaine.
