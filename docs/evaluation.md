# Banc d'évaluation

> Lot 3 de [PLAN_AMELIORATION.md](PLAN_AMELIORATION.md). Une commande calcule toutes les
> métriques du projet en comparant des **prédictions** à une **vérité terrain**, deux jeux de
> fichiers de même format ([contracts.md](contracts.md)).

## Commande

```bash
# Vérité TenniSet convertie en events.json (un dossier par vidéo) : sert aussi d'oracle
tennisvision tenniset export-events --data-dir data --split all -o sortie_oracle/

# Évaluation sur TenniSet (split « test » par défaut)
tennisvision eval --dataset tenniset --data-dir data --predictions sortie_oracle/

# Évaluation sur le jeu amateur (Lot 2)
tennisvision eval --dataset amateur --gt-dir verite/ --predictions sorties/ --split test
```

Options : `--split {train,val,test,all}` (défaut `test`), `--output-dir` (défaut
`./reports/eval`), `--data-dir` (TenniSet ; défaut `$TENNISVISION_TENNISET_DIR`, sinon `./data`).
Codes retour : 0 rapport écrit (quels que soient les scores), 2 erreur d'usage (dossier
introuvable, `--gt-dir` manquant, split absent de `splits.json` ou sans match, aucune annotation
TenniSet).

Une prédiction égale à la vérité donne F1 = 1 partout et des erreurs d'indicateurs nulles ;
`eval --predictions` appliqué à la sortie de `tenniset export-events` doit donc rendre F1 = 1 (les
événements prédits situés hors du split évalué sont écartés, comme les événements gt).

## Formats

### Dossier de vérité terrain (jeu amateur)

```
<gt_dir>/
├── splits.json            {"train": [...], "val": [...], "test": [...]}  (noms de matchs)
└── <match>/
    ├── events.json        } tout ou partie de ces fichiers, au format des contrats
    ├── ball.csv           }
    ├── court.json         }
    ├── players.csv        }
    └── match.json         facultatif
```

Le découpage se fait **par match**, jamais à l'intérieur d'un match (ADR-003). Une brique n'est
évaluée que si la vérité terrain a le fichier correspondant. La hauteur d'image, qui met les
seuils de la balle à l'échelle, vient de la clé `image_size` de `match.json` si elle existe, sinon
de `court.json.image_size`, sinon 1080.

### Dossier de prédictions

```
<pred_dir>/<match>/        mêmes noms de fichiers : sorties du pipeline
```

Un fichier absent alors que la vérité terrain l'a vaut **prédiction vide** (aucun événement, table
vide, aucun segment de court) ; le match est alors listé dans `avertissements` du rapport.
`summary.json` et `stats.json` sont lus, s'ils existent, pour enregistrer les modèles utilisés.

### Découpage leave-one-match-out

`tennisvision.eval.banc.plis_leave_one_out(matchs)` donne les plis (matchs d'entraînement, match
de test) dans l'ordre trié : un pli par match.

## Métriques

Les tolérances sont des constantes de `tennisvision.eval.evenements` / `balle` / `joueurs`,
recopiées dans `meta.tolerances` du rapport pour que deux rapports soient comparables.

### Appariement

Un-pour-un, optimal (algorithme hongrois, `scipy.optimize.linear_sum_assignment`) : on maximise
d'abord le nombre de paires valides, puis on minimise l'écart total. Instants : paire valide si
|Δframe| ≤ tolérance (l'égalité est incluse). Intervalles : paire valide si IoU temporel ≥ 0,5
(bornes incluses, longueur = fin − début + 1). `precision = tp / n_pred`, `rappel = tp / n_gt`,
`f1 = 2·tp / (n_pred + n_gt)` ; `None` si le dénominateur est nul. Une prédiction vide face à une
vérité non vide donne donc rappel = 0 et F1 = 0.

### Événements (`events.json`, appariement sur `frame_idx`)

| Bloc | Métriques |
|---|---|
| `frappes` (`hit`) | scores et `ecart_moyen_frames` à ±2 et ±5 frames ; sur les paires à ±5 : `exactitude_joueur`, matrice de confusion `{classe_gt: {classe_pred: n}}` et `f1_par_classe` (un-contre-tous) sur les paires où les deux `stroke` sont non nuls (`n_paires_coup`) |
| `services` (`serve`) | scores à ±5 frames ; exactitude de `result`, `serve_number`, `player_id` sur les paires |
| `rebonds` (`bounce`) | scores à ±2 et ±5 ; sur les paires à ±5 : `erreur_position_cm` (moyenne, max de la distance euclidienne de `court_xy` × 100), `exactitude_in_out`, et `exactitude_in_out_pres_ligne` pour les rebonds gt à moins de 0,30 m d'une ligne du court de simple (avec son effectif) |
| `points` (`point`) | scores sur l'IoU des intervalles `[frame_idx, end_frame_idx]`, `iou_moyen`, `exactitude_gagnant` (paires où le gt connaît le gagnant) ; sous-bloc `jeux` (`game`, même appariement) avec `pct_jeux_score_juste` : part des jeux gt ayant un `score_after` dont le jeu apparié a le même |

Un bloc dont le gt n'a aucun événement du type vaut `null`.

### Balle (`ball.csv`, protocole TrackNet)

Seuils de 4 et 10 px **à 1080p**, multipliés par `hauteur_image / 1080`. Évaluation sur les frames
du gt ; une frame absente de la prédiction est « non visible ». Gt visible, pred visible et
distance ≤ seuil : TP. Pred visible sans gt visible, ou trop éloignée : FP. Gt visible sans pred
visible, ou trop éloignée : FN (une détection trop loin compte donc FP **et** FN). Le taux de
trous longs est la part des frames gt visibles situées dans un trou de plus de 5 frames (suite de
frames consécutives où le gt est visible et la prédiction non).

### Court (`court.json`)

Chaque segment gt est comparé au segment prédit valide à son `valid_from_frame` ; sans segment, il
est compté dans `n_segments_manques`. `erreur_px` : distance entre points clés prédits et vrais
(points non nuls des deux côtés). `erreur_cm` : points clés gt projetés par l'homographie prédite
(coordonnées homogènes) puis comparés à `COURT_KEYPOINTS_M`.

### Joueurs (`players.csv`)

`pct_frames_deux_joueurs` : sur les frames où le gt a P1 et P2, part de celles où la prédiction a
aussi les deux. Les identités P1/P2 étant fixes par contrat, `idf1 = 2·IDTP / (lignes gt + lignes
pred)` sur les frames du gt, IDTP = lignes gt dont la ligne prédite de même `player_id` a une IoU
de boîte ≥ 0,5. `n_echanges_identite` : frames où la prédiction de P1 recouvre (IoU ≥ 0,5) le gt
de P2 et celle de P2 le gt de P1.

### Indicateurs (statistiques vues par l'utilisateur)

`tennisvision.eval.indicateurs` calcule depuis un `events.json` : `first_serve_pct`,
`second_serve_pct` (lets exclus, pourcentages sur 0-100), `double_faults`, `aces`, `points_won`,
`forehands`, `backhands`, `volleys`, `smashes`, `winners`, `unforced_errors` (par joueur), puis
`rally_length_mean` et `bounce_in_pct` (match). C'est l'implémentation de référence du banc, que
le Lot 10 réutilisera ou remplacera. Pour chaque match : `erreur_abs = |pred − gt|`,
`erreur_rel = erreur_abs / |gt|` (`null` si gt vaut 0 ou est indéfini). Agrégé par statistique :
moyenne et maximum des erreurs, nombre de matchs, `pire_match`.

### Agrégation

Pour les scores de détection, somme des `tp` / `n_pred` / `n_gt` sur les matchs puis précision,
rappel et F1 globaux (micro-moyenne). Les exactitudes sont sommées (`n_ok`, `n`), les erreurs
moyennes pondérées par leur effectif (maximum des maxima). La liste par match est toujours
conservée dans `par_match`.

## Conventions TenniSet

Source : `data/annotations/V0xx.json` (pas le sous-dossier `generalised/`).

- **P1 est le joueur du premier `Serve`** (par `start`), P2 l'autre. Les joueurs sont les noms
  des `Serve` et `Hit` ; un `Winner` ou `Server` inconnu (score ou « Tiebreak » dans certains
  fichiers) donne un identifiant null.
- `frame_idx` = début arrondi, `end_frame_idx` = fin arrondie, `t_ms = frame_idx × 1000 / 25`
  (vidéos à cadence fixe). L'instant d'impact exact reste à estimer.
- **Pas de `score_after`** : l'ordre des scores TenniSet (« 1-0 ») est ambigu, donc
  `pct_jeux_score_juste` n'est pas calculable sur TenniSet.
- **Les services ne sont pas comptés comme frappes** : TenniSet ne les annote pas comme des `Hit`
  et le chargeur n'en invente pas. Un `Serve` donne un événement `serve` (`serve_number` : 1 pour
  le premier service d'un point, 2 après une faute, un let garde le numéro, 1 hors de tout
  point).
- `Hit` : `Type` Smash donne `stroke = smash` ; Volley donne `forehand_volley` / `backhand_volley`
  selon `Side` ; sinon `Side` donne `forehand` / `backhand`. `spin` : Topspin / Slice / Flat,
  Unsure donne `unknown`. Pas d'`outcome`.
- Pas de rebonds, de balle, de court ni de joueurs dans TenniSet : seules les briques
  d'événements et les indicateurs liés sont évalués.
- Les splits sont les `SPLITS` (intervalles de frames d'une même vidéo) ; l'évaluation
  filtre vérité **et** prédictions sur le split demandé.
- **Les scores TenniSet sont optimistes** (ADR-003) : Train et Test partagent joueuses, court,
  caméra et jour. Seul le jeu amateur, découpé par match, fait foi.

## Lire le rapport

`<output-dir>/<AAAAMMJJ-HHMMSS>_<commit>/` contient :

- `rapport.json` : `meta` (version, commit, date UTC, jeu, split, tolérances, modèles lus dans
  les `summary.json` / `stats.json` des prédictions), `avertissements`, `par_match`, `agrege`
  (dont `indicateurs`). Aucun chemin absolu.
- `rapport.html` : le même contenu en tableaux, statique et autonome (ni script ni ressource
  externe).

`<output-dir>/historique.jsonl` reçoit une ligne par évaluation : date, commit, jeu, split et
quelques F1 agrégés (`frappes_pm5`, `services`, `rebonds_pm5`, `points`, `balle_10px`), pour
suivre l'évolution d'un commit à l'autre. Pour comparer deux versions, utiliser le même jeu, le
même split et les mêmes tolérances (`meta.tolerances`), et vérifier `meta.modeles`.
