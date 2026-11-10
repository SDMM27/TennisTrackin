# Contrats de données — version 1.0

> Lot 0 de [PLAN_AMELIORATION.md](PLAN_AMELIORATION.md) (§ 5). Les modules communiquent
> **uniquement** par ces fichiers. Les schémas qui font foi sont dans
> [`src/tennisvision/schemas/`](../src/tennisvision/schemas/) (JSON Schema 2020-12) ; ce
> document en explique les conventions. Des exemples valides sont dans
> `tests/fixtures/contrats/valide/`.
>
> Vérifier un dossier de sortie : `tennisvision validate <dossier>` (codes retour : 0 valide,
> 1 au moins un fichier invalide, 2 chemin introuvable ou aucun fichier contrat).

## Vue d'ensemble

| Fichier | Format | Produit par | Schéma |
|---|---|---|---|
| `match.json` | JSON | utilisateur (formulaire) | `match.schema.json` |
| `frames.parquet` | Parquet | ingestion (Lot 1) | `frames.row.schema.json` |
| `ball.csv` | CSV | balle (Lot 4) | `ball.row.schema.json` |
| `court.json` | JSON | court (Lot 5) | `court.schema.json` |
| `players.csv` | CSV | joueurs (Lot 5) | `players.row.schema.json` |
| `events.json` | JSON | Lots 6 à 9, et annotations de vérité terrain | `events.schema.json` |
| `stats.json` | JSON | statistiques (Lot 10) | `stats.schema.json` |

Pour les tables, le schéma décrit **une ligne** ; l'extension `x-columns` donne la liste
**exacte et ordonnée** des colonnes. Dans un CSV, une cellule vide vaut `null`.

## Conventions communes

- **Version** : chaque document JSON porte `"schema_version": "1.0"`. Tout changement
  incompatible incrémente la version majeure et passe par une nouvelle ADR.
- **Temps** : `t_ms` = horodatage de présentation réel de la frame, en millisecondes depuis le
  début de la vidéo, lu dans `frames.parquet`. Les durées se calculent **toujours** par
  différence de `t_ms`, jamais par `frame_idx / fps` (vidéos de téléphone en cadence variable).
- **Frames** : `frame_idx` commence à 0 et suit l'ordre de décodage de la vidéo source ; les
  bornes de fin (`end_frame_idx`, `valid_to_frame`) sont **incluses**.
- **Pixels** : coordonnées dans l'image **après application de la rotation** de la vidéo,
  origine en haut à gauche, x vers la droite, y vers le bas, à la résolution native (pas celle
  d'inférence).
- **Joueurs** : identifiants `P1` et `P2`, définis dans `match.json`, stables sur tout le
  match quel que soit le côté du court. « Près / loin » n'est jamais un identifiant.
- **Provenance** (`source`) : `auto` = prédit par le pipeline, `user` = corrigé par
  l'utilisateur, `annotation` = vérité terrain d'un jeu de données. La vérité terrain utilise
  le **même format** que les prédictions : le banc d'évaluation (Lot 3) compare deux fichiers
  de même schéma.
- **Confiance** : `confidence` dans [0, 1], `null` quand elle n'a pas de sens (saisie
  utilisateur, annotation).
- **Chemins** : aucun chemin absolu dans les fichiers ; les références sont relatives au
  dossier de sortie.

## Repère du court

Vue de la caméra, placée derrière la ligne de fond **proche**. Unités : mètres.

- Origine : centre du filet, au sol.
- `x` : vers la **droite** de l'image (latéral).
- `y` : vers la **caméra** (côté proche positif, côté loin négatif).

Dimensions ITF : demi-longueur `L = 11,885`, demi-largeur double `Wd = 5,485`, demi-largeur
simple `Ws = 4,115`, distance filet–ligne de service `S = 6,40`.

### Les 14 points clés (`court.json` → `keypoints_px`)

| # | Point | (x, y) en m |
|---|---|---|
| 0 | fond loin × couloir gauche | (−Wd, −L) |
| 1 | fond loin × couloir droit | (Wd, −L) |
| 2 | fond proche × couloir gauche | (−Wd, L) |
| 3 | fond proche × couloir droit | (Wd, L) |
| 4 | fond loin × ligne de simple gauche | (−Ws, −L) |
| 5 | fond proche × ligne de simple gauche | (−Ws, L) |
| 6 | fond loin × ligne de simple droite | (Ws, −L) |
| 7 | fond proche × ligne de simple droite | (Ws, L) |
| 8 | service loin × simple gauche | (−Ws, −S) |
| 9 | service loin × simple droite | (Ws, −S) |
| 10 | service proche × simple gauche | (−Ws, S) |
| 11 | service proche × simple droite | (Ws, S) |
| 12 | service loin × ligne médiane | (0, −S) |
| 13 | service proche × ligne médiane | (0, S) |

Ces valeurs sont exposées en Python par `tennisvision.schemas.COURT_KEYPOINTS_M`. Un point
hors de l'image vaut `null`.

## Fichier par fichier

### `match.json`

Joueurs (`id`, `name`, `handedness` : `right` / `left` / `unknown`), `first_server`,
`format` (`sets_to_win`, `games_per_set`, `tiebreak_at` — `null` = pas de jeu décisif —,
`final_set` : `tiebreak` / `super_tiebreak` / `advantage`, `deciding_point` = point décisif
sans avantage), `surface`, `camera` (`fps_nominal`, `position`, `height_m`).
`court_corners` est le **repli manuel** de calibration (Lot 5) : 4 coins en pixels dans
l'ordre loin-gauche, loin-droite, près-droite, près-gauche, en précisant s'il s'agit du
rectangle de simple ou de double.

### `frames.parquet`

`frame_idx` (entier, 0 à n−1, contigu), `t_ms` (réel, strictement croissant). C'est la seule
source de temps du pipeline.

### `ball.csv`

Une ligne par frame traitée : `frame_idx, t_ms, x, y, visible, score, source`.
`visible = 1` impose `x`, `y` et une `source` (`detected`, `interpolated`, `annotation`) ;
`visible = 0` impose `x` et `y` vides. `score` = sortie du détecteur (vide pour une frame
interpolée ou annotée). `frame_idx` strictement croissant.

### `court.json`

`image_size` puis une liste de `segments`, chacun valable sur un intervalle de frames
(`valid_to_frame = null` : jusqu'à la fin). Plusieurs segments quand le téléphone a bougé ;
ils ne se chevauchent pas. Chaque segment : `homography` (3×3, image px → court m, par
lignes), les 14 `keypoints_px`, `reprojection_error_px` (et `_cm` si mesurée), `confidence`,
`source`.

### `players.csv`

Au plus une ligne par joueur et par frame, seulement pour les 2 joueurs retenus :
`frame_idx, t_ms, player_id, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, foot_x_m,
foot_y_m, confidence`, puis 17 points clés COCO `kp{i}_x, kp{i}_y, kp{i}_c` (i = 0…16).
`foot_*_m` : milieu des pieds projeté sur le court (vide sans calibration). Points clés vides
quand la pose n'a pas été calculée sur cette frame. `track_id` = identifiant du suivi brut,
pour le débogage.

### `events.json`

`{"schema_version", "video", "events": [...]}`. Champs communs : `id` (unique dans le
fichier), `type`, `t_ms`, `frame_idx`, `player_id`, `confidence`, `source`, `desc` (texte
libre, optionnel). Les événements à durée ajoutent `end_t_ms` et `end_frame_idx`.

| `type` | Instant (`t_ms`) | Attributs propres |
|---|---|---|
| `serve` | impact | `serve_number` (1/2), `result` (`in`/`fault`/`let`), `fault_type` (`net`/`long`/`wide`/`foot`/`unknown`), `hit_id` |
| `hit` | impact | `stroke` (`serve`, `forehand`, `backhand`, `forehand_volley`, `backhand_volley`, `smash`, `uncertain`, ou `null` tant que non classé), `spin`, `outcome` (`in_play`, `winner`, `unforced_error`, `forced_error`, `unknown`) |
| `bounce` | contact au sol | `court_xy` (m), `image_xy` (px), `uncertainty_cm`, `in_out` (`in`/`out`/`too_close`), `zone`, `hit_id` (frappe précédente) |
| `net` | balle arrêtée par le filet | `hit_id` |
| `point` | début → fin | `winner_id`, `server_id`, `score_after`, `end_reason`, `rally_length` |
| `game`, `set` | début → fin | `winner_id`, `server_id`, `score_after` |

**Service** : chaque impact de raquette est un `hit` (le Lot 6 détecte les impacts sans
savoir si c'est un service ; le Lot 7 lui donne `stroke = "serve"`). L'événement `serve`
(Lot 9) porte en plus l'issue du service et pointe vers ce `hit` par `hit_id` (absent quand la
frappe n'est pas connue, par exemple dans TenniSet). Les services se comptent sur le type
`serve`, la longueur d'échange (`rally_length`) sur les `hit`, service compris.

`score_after` : `sets` (jeux de chaque set terminé, `[P1, P2]`), `games` (`[P1, P2]` dans le
set en cours), `points` (`["0"…"40", "AD"]`, ou entiers en texte pendant un jeu décisif).

`in_out = "too_close"` est l'issue honnête quand la balle est à moins de la marge
d'incertitude d'une ligne ([ADR-006](adr/006-limites-monocamera.md)).

**Correspondance TenniSet** (chargeur du Lot 3) : `Point`/`Game`/`Set` → mêmes types,
`Winner`/`Server` → `winner_id`/`server_id` via le nom de joueur ; `Serve.Result` →
`result` ; `Hit.Side` (`Forehand`/`Backhand` dans les fichiers `data/annotations/V0xx.json`)
→ `stroke` ; `Hit.Type` (`Topspin`/`Slice`/`Flat`/`Volley`/`Smash`) → `spin` ou `stroke`
selon le cas. Les bornes TenniSet sont des intervalles : `t_ms`/`frame_idx` reçoivent le
début, et l'instant d'impact reste à estimer.

### `stats.json`

`generated_with` (version du paquet, versions des modèles, rapport d'évaluation de
référence), puis `stats`, chacune avec : `id` (ex. `first_serve_pct`), `player_id`, `scope`
(`set`, `serve_number`), `value`, `unit`, `n` (effectif), `confidence`, `validated`,
`error` (erreur mesurée par le banc), `source_events` (ids des événements qui la justifient).
Une stat `validated = false` est masquée ou marquée « expérimental » dans le rapport.

## Évolution des contrats

1. Ajout d'un champ **optionnel** : version mineure (`1.1`) ; le schéma 1.1 accepte
   `schema_version` `"1.0"` et `"1.1"`, donc les fichiers existants restent valides.
2. Champ renommé, supprimé, rendu obligatoire, ou changement d'unité / de repère : version
   majeure, nouvelle ADR, migration des fixtures et des jeux annotés.
