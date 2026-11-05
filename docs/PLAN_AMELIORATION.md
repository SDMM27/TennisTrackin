# TennisTrackin — Revue et plan d'amélioration

> Document de référence, autonome : chaque **lot** (section 6) peut être copié seul dans une
> nouvelle conversation. Les sections 1 à 5 donnent le contexte commun ; un lot qui en dépend
> le dit explicitement.
>
> Revue faite le 2026-11-05 sur le commit `8aa2f62` (branche `main`).

## Sommaire

1. [Cible produit](#1-cible-produit)
2. [État des lieux](#2-état-des-lieux)
3. [Critique de la roadmap et des choix structurants](#3-critique-de-la-roadmap-et-des-choix-structurants)
4. [Hypothèses « usage personnel » à lever](#4-hypothèses-usage-personnel-à-lever)
5. [Architecture cible et contrats de données](#5-architecture-cible-et-contrats-de-données)
6. [Lots de travail](#6-lots-de-travail)
7. [Ordonnancement](#7-ordonnancement)
8. [Ce qui n'a pas pu être vérifié](#8-ce-qui-na-pas-pu-être-vérifié)
9. [Références](#9-références)

Échelles utilisées :
- **Priorité** : P0 bloquant (sans lui, le produit ne peut pas exister ou ses chiffres sont
  faux) · P1 cœur de valeur · P2 important pour l'usage · P3 bonus.
- **Effort** (1 développeur à plein temps) : **S** ≤ 2 j · **M** 3–7 j · **L** 2–4 semaines ·
  **XL** > 1 mois.

---

## 1. Cible produit

Un joueur amateur pose son téléphone derrière le fond de court, filme son match en simple, et
obtient automatiquement :

- des **statistiques fiables** : services (1re/2de, % de réussite, doubles fautes, placement),
  coups droits / revers / volées / smashs par joueur, fautes (filet / long / large) et coups
  gagnants, zones de rebond et de frappe, longueur des échanges, score ;
- des **moments forts** (plus longs échanges, points gagnants) et un **match condensé** sans
  temps morts ;
- des **conseils concrets**, chacun appuyé par des chiffres et des extraits vidéo.

Le critère qui prime sur tous les autres : **chaque statistique affichée est mesurée contre une
vérité terrain, avec son erreur connue et publiée**. Une statistique non validée n'est pas
affichée, ou l'est avec la mention « expérimental ».

### Les statistiques qui font vraiment progresser un amateur

Point de vue d'entraîneur, par ordre d'utilité pour un joueur de club (et donc ordre de priorité
de validation) :

| Rang | Statistique | Pourquoi elle compte | Ce qu'il faut détecter |
|---|---|---|---|
| 1 | Fautes directes par coup et par type (filet / long / large) | C'est ce qui décide la majorité des points chez les amateurs | coup + joueur + rebond hors court ou balle dans le filet |
| 2 | % de 1res balles, points gagnés sur 1re / 2de, doubles fautes | Le service est le seul coup totalement maîtrisé par le joueur | services + rebond dans/hors carré + gagnant du point |
| 3 | Longueur des échanges et % de points gagnés selon la longueur (1–4, 5–8, 9+) | Révèle si le joueur doit construire ou abréger | frappes par point + gagnant |
| 4 | Profondeur des balles (rebond au-delà de la ligne de service) | Indicateur n°1 de qualité du fond de court en club | rebonds projetés sur le court |
| 5 | Répartition coup droit / revers et taux de faute de chacun | Montre le coup que l'adversaire attaque et celui à travailler | type de coup + issue |
| 6 | Placement du service (extérieur / corps / T) par côté | Prévisibilité au service | rebond du service dans le carré |
| 7 | Position de frappe (dans / derrière la ligne de fond) | Recul excessif = jeu subi | pieds du joueur projetés sur le court au moment de la frappe |
| 8 | Vitesse de balle | Motivante mais **peu actionnable** et **imprécise en monocaméra** | — à repousser en P3 |

Conséquence : la **chaîne critique** n'est pas « la balle partout, tout le temps », mais :
**instant de frappe + joueur + type de coup + rebond projeté sur le court + segmentation des
points + gagnant du point**.

---

## 2. État des lieux

### 2.1 Ce qui existe (vérifié dans le code)

Le dépôt contient un squelette « étape 0 » de ~600 lignes de Python :

| Fichier | Rôle | État |
|---|---|---|
| `src/tennisvision/io/video.py` | `probe`, `iter_frames`, `VideoWriter`, `cut_clip` | Propre, testé, gère les erreurs |
| `src/tennisvision/tracking/detector.py` | YOLO11n COCO + ByteTrack via `model.track(persist=True)`, classes `person` et `sports ball` | Fonctionne, mais modèle inadapté à la balle |
| `src/tennisvision/tracking/draw.py` | Boîtes joueurs, traînée de balle, bandeau | Fonctionne, défauts mineurs (2.3) |
| `src/tennisvision/tracking/export.py` | `tracks.csv` + résumé | Testé |
| `src/tennisvision/pipeline.py` | Enchaînement lecture → tracking → sorties | Pas de test |
| `src/tennisvision/datasets/tenniset.py` | Lecture de `points.txt` (bornes des points) | Ne lit **que** `points.txt`, pas les événements JSON |
| `src/tennisvision/cli.py` | `analyze` et `clip` | Testé (sans YOLO) |
| `notebooks/01_palier1_tracking.ipynb` | Exploration, duplique la logique du pipeline | Numérotation « palier » incohérente avec la ROADMAP |
| `tests/` | 31 tests, vidéos synthétiques | **31/31 passent**, `ruff check` propre (vérifié) |

### 2.2 Ce qui marche (vérifié)

- Installation et tests : `pytest -q` → 31 passed ; `ruff check src tests` → aucun problème.
- Environnement : Python 3.14.6, torch 2.15, ultralytics 8.4.201, opencv 5.0, pandas 3.1.
- Lecture vidéo : le saut direct à une frame (`cap.set(CAP_PROP_POS_FRAMES)`) renvoie
  **exactement** la même image qu'une lecture séquentielle sur V006 (testé aux frames 997 et
  3001). L'alignement avec les annotations TenniSet est donc fiable **pour ces vidéos à cadence
  constante** (25 fps).
- Sortie sur disque `reports/V006_25740-25984/` : 245 frames traitées à 12,5 fps (CPU,
  640 px), joueurs suivis sur tout le point.
- Le README et la ROADMAP sont **honnêtes** sur les limites : le faux positif « roue de
  caméra » à 1280 px est documenté, la nécessité d'une vérité terrain est affirmée. C'est la
  bonne culture de projet, à conserver.
- Les chiffres d'annotation cités dans la ROADMAP pour V006 (81 points, 112 services,
  220 frappes, 13 jeux) sont exacts (recomptés dans `generalised/V006.json`).

### 2.3 Ce qui est cassé ou fragile (vérifié dans le code)

| # | Constat | Où | Gravité |
|---|---|---|---|
| C1 | **Balle quasi jamais détectée** : 0 détection sur le point P00000001 à 640 px (`summary.json`), < 2 % de vraie balle selon le README | `detector.py` | Bloquant |
| C2 | La balle passe dans **ByteTrack** avec les personnes : un filtre de Kalman à vitesse constante conçu pour des piétons, inadapté à un objet de 3–8 px qui rebondit et change de direction à chaque frappe ; les ID de balle n'ont pas de sens | `detector.py:65-74` | Majeur |
| C3 | Choix de « la » balle par **confiance maximale**, sans cohérence spatiale ou temporelle (c'est précisément ce qui laisse gagner la roue de caméra) | `draw.py:52` | Majeur |
| C4 | La traînée de balle **ne s'efface jamais** quand la balle disparaît, et relie des détections éloignées dans le temps (un faux positif et la vraie balle sont reliés par un trait) | `draw.py:50-57` | Mineur |
| C5 | `n_identites` = 31 personnes sur un point : aucun filtrage des 2 joueurs (arbitres, ramasseurs, public) | `export.py`, `summarize` | Majeur pour toute stat par joueur |
| C6 | Le dossier de sortie est créé **avant** la validation des bornes : une erreur `start > end` laisse un dossier vide au nom incohérent | `pipeline.py:45-48` | Mineur |
| C7 | `n_frames_traitees` et `taux_detection_balle` calculés dans `summarize` puis **écrasés** dans `pipeline` : deux définitions du même indicateur | `export.py:54,58`, `pipeline.py:79-82` | Mineur |
| C8 | `summary.json` contient des **chemins absolus** de la machine (`C:\Users\...`) : non portable, fuite d'information si partagé | `pipeline.py:85-90` | Mineur (majeur en multi-utilisateur) |
| C9 | Vidéo de sortie en **`mp4v`** (MPEG-4 Part 2) : non lisible dans la plupart des navigateurs et sur iOS ; inutilisable pour des extraits à partager | `video.py:102` | Majeur pour le produit |
| C10 | Lecture vidéo **sans horodatage** : les téléphones filment en cadence variable (VFR) ; `CAP_PROP_FPS` et `CAP_PROP_FRAME_COUNT` deviennent approximatifs et la durée des échanges serait fausse | `video.py` | Majeur pour les vidéos réelles |
| C11 | Aucune gestion de l'**orientation** (vidéo portrait, métadonnée de rotation) | `video.py` | Majeur pour les vidéos réelles (non testé) |
| C12 | `pipeline.py`, `detector.py`, `draw.py` **sans aucun test** (pas même avec un faux tracker) | `tests/` | Moyen |
| C13 | Le module `src/tennisvision/eval/` annoncé par la ROADMAP **n'existe pas** : aucune métrique n'est calculée automatiquement, les chiffres du README sont des mesures manuelles non reproductibles | — | Bloquant pour la promesse « mesuré et validé » |
| C14 | Performance : 12,5 fps CPU au modèle nano. Un match d'1 h 30 filmé en 30 fps = 162 000 frames ≈ **3,6 h** de calcul, avant même balle, pose et court | — | Majeur |
| C15 | La vidéo **V009 est absente** de `data/videos/` alors que ses annotations sont présentes : elle porte **366 des 746 points** (49 %) du jeu | `data/` | Majeur pour l'évaluation |

### 2.4 Ce qui manque (par rapport à la cible)

Tout ce qui produit de la valeur pour le joueur reste à faire : suivi de balle, calibration du
court, sélection et identité des 2 joueurs, détection des frappes, classification des coups,
détection des rebonds et in/out, segmentation des points, score, statistiques, rapport, moments
forts, conseils, interface utilisateur, et surtout **le jeu de données de vidéos amateurs** et
**le banc d'évaluation**.

Pas de dossier `docs/` ni de spécifications avant ce document : seuls le README et la ROADMAP
décrivent le projet.

---

## 3. Critique de la roadmap et des choix structurants

La ROADMAP actuelle est **bien construite sur la méthode** (un livrable + une métrique par
étape, séparation Train/Val/Test). Elle pèche sur cinq choix structurants qui, laissés en
l'état, mènent à un produit qui marche sur TenniSet et pas chez l'utilisateur.

### 3.1 Jeu de données : TenniSet n'est pas le domaine cible

**Constat.** TenniSet, ce sont 5 matchs professionnels féminins en **diffusion TV** (plans de
coupe, ralentis, caméra haute et centrée, 720p à 25 fps, éclairage de stade). V006 est
Williams–Sharapova (vu dans les `desc`). La cible, c'est un téléphone posé par un amateur :
caméra plus basse, souvent décentrée, grillage au premier plan, courts voisins avec d'autres
joueurs et d'autres balles, terre battue / dur / intérieur, contre-jour, 30 ou 60 fps en
cadence variable, joueurs de niveaux très hétérogènes.

La ROADMAP repousse les « vidéos réelles » à l'**étape 6**, la dernière. C'est l'inverse de ce
qu'il faut : le changement de domaine est **le** risque principal du projet, il doit être
mesuré dès le départ.

**Ce que TenniSet apporte vraiment :**
- un **schéma d'événements** éprouvé (Point, Serve, Hit avec joueur/côté/type, Game, Set) à
  reprendre tel quel pour nos annotations ;
- un banc pour la **logique de jeu** (segmentation des points, score), qui dépend peu de
  l'image ;
- des descriptions en texte libre (`desc`) qui mentionnent *forehand*, *backhand*, *out*,
  *winner* : source d'étiquettes faibles pour coup droit/revers et fautes/gagnants.

**Ce qu'il n'apporte pas :** aucune position de balle, aucune ligne de court, une géométrie de
caméra qui n'est pas la nôtre.

**Défaut méthodologique.** Les découpages `SPLITS` de TenniSet sont des **intervalles dans la
même vidéo** (V006 : Train 25529–87022, Val 87032–94849, Test ensuite). Train et Test partagent
les mêmes joueuses, le même court, la même caméra, le même jour : les scores obtenus seront
**optimistes**. Il faut évaluer en **« un match tenu à l'écart »** (leave-one-match-out), et
pour le produit, sur des matchs amateurs jamais vus.

**Décision proposée :**
1. Constituer **dès maintenant** un jeu « amateur » : au moins 10 matchs, au moins 6 joueurs
   différents, 3 surfaces, 2 cadences (30 et 60 fps), avec un protocole de tournage écrit
   (Lot 2). C'est le jeu de **validation qui fait foi**.
2. Garder TenniSet comme jeu secondaire (pré-entraînement des modèles d'événements, tests de
   la logique de score).
3. Utiliser le jeu de données **TrackNet** (balle annotée image par image, diffusion TV) pour
   amorcer le détecteur de balle, puis l'affiner sur des frames amateurs annotées.

### 3.2 Téléphone ou serveur : commencer côté serveur

| Critère | Traitement sur téléphone (façon SwingVision) | Traitement sur serveur / PC (batch) |
|---|---|---|
| Temps réel, appel de ligne en direct | Oui | Non |
| Vitesse d'itération sur les modèles | Lente (conversion CoreML/TFLite, quantification, une version par plateforme) | Rapide (Python, PyTorch) |
| Taille des modèles possible | Contrainte forte (mémoire, batterie, chauffe sur 1 h 30) | Libre |
| Précision atteignable | Limitée par le budget de calcul | Modèles lourds, plusieurs passes, lissage hors-ligne sur toute la trajectoire |
| Coût | Nul pour le projet | GPU à payer (ou PC de l'utilisateur) |
| Confidentialité | La vidéo reste sur le téléphone | Vidéo envoyée : RGPD, consentement des personnes filmées |
| Compétences du projet actuel | Swift/Kotlin à acquérir | Python, déjà là |

**Recommandation :** v1 = **traitement hors-ligne** (CLI puis service web), le téléphone sert
seulement à filmer, avec un guide de cadrage. Le hors-ligne est un avantage pour la justesse :
on peut lisser une trajectoire avec le futur comme le passé, faire plusieurs passes, et laisser
l'utilisateur corriger. Le passage sur téléphone (P3) ne se justifie qu'une fois les modèles
stabilisés et validés ; les architectures choisies doivent néanmoins rester **exportables**
(ONNX) pour ne pas fermer cette porte.

Objectif de débit à fixer : traiter **1 h de match en ≤ 1 h** sur un GPU grand public, et en
≤ 4 h sur CPU seul (traitement de nuit acceptable pour un amateur).

### 3.3 Modèles adaptés à la balle

YOLO COCO « sports ball » est **le mauvais outil** : la balle fait 3 à 8 px au fond du court,
elle est floue (à 100 km/h, elle parcourt 0,93 m entre deux images à 30 fps, 0,46 m à 60 fps),
et une seule image ne suffit pas à la distinguer d'une tache claire. La ROADMAP cite TrackNet :
c'est la bonne direction, à rendre ferme.

| Famille | Principe | Pour | Contre |
|---|---|---|---|
| **TrackNet / V2 / V3** | Réseau encodeur-décodeur, entrée = 3 frames consécutives (ou plus), sortie = carte de chaleur de la position de la balle ; V3 ajoute une rectification de trajectoire | Conçu pour la balle de tennis/badminton, exploite le mouvement, poids pré-entraînés publics, ~10 M de paramètres (exportable) | Entraîné sur de la TV, à affiner |
| **WASB** (Tarashima et al., 2023) | Base HRNet en carte de chaleur, plusieurs frames, suivi en ligne | Meilleurs résultats publiés multi-sports au moment de sa sortie, code public | Plus lourd |
| YOLO affiné « tennis ball » à haute résolution, éventuellement en tuiles | Détection image par image | Outils simples | Pas de contexte temporel ; lent en tuiles ; faux positifs sur les courts voisins |

**Recommandation :** famille **carte de chaleur multi-frames** (TrackNetV3 ou WASB : choisir
en mesurant les deux sur le jeu amateur, Lot 4), suivie d'un **lissage hors-ligne de
trajectoire** (segments paraboliques entre frappes et rebonds, plutôt qu'un simple Kalman).
Les joueurs, eux, restent sur YOLO personne + suivi (c'est ce qui marche déjà).

### 3.4 Ne pas tout faire reposer sur la balle

La ROADMAP fait dépendre la détection des frappes (étape 3a) des changements de direction de la
balle. Or la balle sera ratée par moments (occultation par le joueur, filet, sortie du champ).
Deux signaux robustes et bon marché manquent :

- **Le son.** Sur une vidéo amateur, sans public ni commentateur, l'impact raquette-balle est un
  pic sonore très net. La détection d'attaques sonores est une technique classique, peu coûteuse,
  et précise à quelques millisecondes. Les courts voisins créent des faux positifs, que l'image
  permet de filtrer.
- **La pose du joueur.** Le geste de frappe est visible même quand la balle ne l'est pas.

Détection de frappe = **fusion son + pose + balle**, chaque signal pouvant compenser les trous
des autres.

### 3.5 Ce qui manque à la ROADMAP

1. **Identité des joueurs au fil du match** : les joueurs changent de côté aux jeux impairs ;
   « Near/Far » ne suffit pas pour des stats par joueur. Il faut une ré-identification
   (apparence + logique des changements de côté).
2. **Latéralité** (gaucher / droitier) : indispensable pour passer de « côté de frappe » à
   coup droit / revers. À demander à l'utilisateur, avec détection automatique en appoint.
3. **Correction par l'utilisateur** : SwingVision laisse corriger le score et les coups. C'est
   à la fois la garantie de stats justes pour l'utilisateur **et** la source d'annotations la
   moins chère pour améliorer les modèles. Absent de la ROADMAP.
4. **Guide de tournage** et contrôle du cadrage à l'import : la moitié des échecs viendra de
   vidéos mal cadrées. Absent.
5. **Incertitude affichée** : chaque statistique doit porter son niveau de confiance.
6. **Licences** : ultralytics est sous **AGPL-3.0**. Pour un service ouvert à d'autres, cela
   impose de publier le code du service, ou d'acheter une licence, ou d'utiliser d'autres
   implémentations. La licence d'utilisation de TenniSet et des vidéos de diffusion (droits des
   diffuseurs) doit être vérifiée avant toute redistribution d'un modèle entraîné dessus.
7. **Ce qui n'est pas atteignable en monocaméra** doit être écrit noir sur blanc : Hawk-Eye
   triangule la balle en 3D avec une dizaine de caméras rapides calibrées ; un téléphone seul ne
   donne qu'une position 2D, et la projection sur le sol n'est valide **qu'au moment du rebond**.
   Conséquences : appel de ligne seulement indicatif près des lignes (marge d'incertitude
   affichée), pas de hauteur au-dessus du filet, vitesse approximative.
8. **Doubles** : hors périmètre v1, à dire explicitement.

### 3.6 Roadmap révisée (vue d'ensemble)

| Phase | Contenu | Lots |
|---|---|---|
| A. Fondations (P0) | Décisions écrites, socle multi-utilisateur, jeu amateur + protocole, banc d'évaluation | 0, 1, 2, 3 |
| B. Perception (P0/P1) | Balle, court + joueurs | 4, 5 |
| C. Événements (P1) | Frappes, type de coup, rebonds et in/out | 6, 7, 8 |
| D. Logique de jeu (P1) | Points, gagnant, score, correction utilisateur | 9 |
| E. Valeur joueur (P2) | Statistiques, rapport, moments forts, conseils | 10 |
| F. Produit (P2) | Service d'import, confidentialité, licences, performance | 11 |
| G. Bonus (P3) | Traitement sur téléphone / temps réel, vitesse de balle, doubles | 12 |

---

## 4. Hypothèses « usage personnel » à lever

Relevées dans le README, la ROADMAP, le code et le notebook :

| Où | Hypothèse | Pourquoi c'est un problème | Proposition | Lot |
|---|---|---|---|---|
| ROADMAP, étape 6 et intro (« tes propres vidéos », « tes matchs », « un de tes matchs ») | Un seul utilisateur, un seul style de tournage | 10 points d'un seul joueur ne mesurent pas la généralisation | Jeu amateur multi-joueurs, multi-courts (≥ 10 matchs, ≥ 6 joueurs) | 2 |
| `config.py:10` `ROOT_DIR = parents[2]` | Le code tourne depuis le dépôt cloné | Casse dès que le paquet est installé normalement (`pip install tennisvision`) : `ROOT_DIR` pointe dans `site-packages` | Chemins par variables d'environnement / options CLI, dossiers utilisateur via `platformdirs` | 1 |
| `config.py:14`, `cli.py --point` | `data/annotations/points.txt` de TenniSet est *la* source des intervalles | La commande générique `analyze` dépend d'un jeu de recherche | Sortir TenniSet dans `datasets/` + sous-commande `eval`, `analyze` ne connaît que la vidéo | 1 |
| `models/`, téléchargement auto ultralytics | Accès Internet au premier lancement, poids dans le dépôt | Hors-ligne impossible, version des poids non figée | Registre de modèles versionnés (nom, version, somme de contrôle), cache utilisateur | 1 |
| `reports/` dans le dépôt, `summary.json` à chemins absolus | Un seul utilisateur sur une seule machine | Non portable, fuite du nom de compte | Chemins relatifs au dossier de sortie, dossier de sortie paramétrable | 1 |
| Aucune notion de match ni de joueurs | Les 2 joueurs sont anonymes « Near/Far » | Pas de stats par joueur, pas d'historique | Fichier `match.json` : joueurs, latéralité, premier serveur, format de match, surface | 1, 5 |
| README : `.venv\Scripts\activate`, Python 3.14 | Windows uniquement, version la plus récente | Ferme la porte à macOS/Linux et à des versions stables | Instructions multi-OS, CI sur Python 3.12 et 3.13 au minimum, bornes de dépendances testées | 1 |
| Notebook : « si tu veux pousser le palier 1 », écrit dans `data/` | Carnet personnel | Duplique le pipeline, écrit dans les données sources | Le notebook appelle le paquet, écrit dans un dossier de sortie, ou est archivé | 1 |
| README : « pas de mobile » | Usage par le développeur lui-même | Contredit la cible (un amateur avec un téléphone) | Écrire la décision du § 3.2 : capture sur téléphone, traitement serveur en v1 | 0 |

---

## 5. Architecture cible et contrats de données

Pour que les lots soient **indépendants**, ils communiquent uniquement par des fichiers de
format fixé. Le Lot 0 fige ces formats ; chaque lot peut ensuite être développé et testé avec
des fichiers produits à la main ou par un autre lot.

```
vidéo + match.json
   │
   ├─► [Lot 1] ingestion ───────────► frames.parquet (index, horodatage ms)
   ├─► [Lot 4] balle ───────────────► ball.csv
   ├─► [Lot 5] court + joueurs ─────► court.json, players.csv
   ├─► audio ───────────────────────► audio_onsets.csv           (Lot 6)
   │
   ├─► [Lot 6] frappes ─────────────► events.json (type=hit)
   ├─► [Lot 7] type de coup ────────► events.json (stroke=...)
   ├─► [Lot 8] rebonds ─────────────► events.json (type=bounce, in_out, court_xy)
   ├─► [Lot 9] points + score ──────► events.json (type=point, winner, score)
   └─► [Lot 10] stats + rapport ────► stats.json, report.html, highlights/*.mp4
```

Contrats (à préciser en JSON Schema dans le Lot 0, versionnés avec un champ `schema_version`) :

- **`match.json`** : `players[{id, name, handedness}]`, `first_server`, `format`
  (sets, jeu décisif, point décisif), `surface`, `camera` (`fps_nominal`, `position`).
- **`frames.parquet`** : `frame_idx`, `t_ms` (horodatage réel de présentation).
- **`ball.csv`** : `frame_idx, t_ms, x, y, visible, score, source` (`detected` / `interpolated`).
- **`court.json`** : homographie image → court (mètres, origine au centre du filet), 14 points
  clés en pixels, erreur de reprojection, `valid_from_frame` / `valid_to_frame`.
- **`players.csv`** : `frame_idx, player_id, bbox, foot_xy_court, keypoints[17]`.
- **`events.json`** : liste d'événements typés, chacun avec `t_ms`, `frame_idx`, `player_id`,
  attributs propres au type, `confidence`, `source` (`auto` / `user`). Schéma compatible avec
  TenniSet (Point / Serve / Hit / Game / Set), étendu de `bounce`.
- **`stats.json`** : chaque statistique avec `value`, `n` (effectif), `confidence`,
  `validated` (booléen issu du banc d'évaluation), et références aux événements sources.

Les annotations de vérité terrain utilisent **le même format `events.json`** : le banc
d'évaluation compare deux fichiers de même schéma.

---

## 6. Lots de travail

Chaque lot suit le même gabarit : priorité, effort, dépendances, objectif, justification,
tâches, critères de réussite, hors périmètre.

---

### Lot 0 — Décisions d'architecture et contrats de données

- **Priorité** : P0 · **Effort** : S · **Dépend de** : rien
- **Objectif** : figer par écrit les choix structurants et les formats d'échange pour que tous
  les autres lots puissent avancer en parallèle.
- **Justification** : sans contrats, chaque lot invente son format et l'intégration devient le
  goulot. Sans décisions écrites, les mêmes questions (téléphone ou serveur ? TenniSet ou
  amateur ?) reviennent à chaque conversation.
- **Tâches** :
  1. Créer `docs/adr/` et rédiger les décisions (un fichier court par décision : contexte,
     décision, conséquences) :
     ADR-001 périmètre v1 (simple, caméra fixe derrière le fond de court, hors-ligne) ;
     ADR-002 traitement serveur en v1, modèles exportables en ONNX ;
     ADR-003 jeu amateur comme référence, TenniSet secondaire, évaluation par match tenu à
     l'écart ;
     ADR-004 famille de modèle balle (carte de chaleur multi-frames) ;
     ADR-005 politique de licences (AGPL d'ultralytics, TenniSet, TrackNet) ;
     ADR-006 limites assumées de la monocaméra (in/out indicatif, pas de 3D).
  2. Écrire `docs/contracts.md` et les JSON Schema dans `src/tennisvision/schemas/` pour les
     fichiers du § 5.
  3. Ajouter un validateur `tennisvision validate <dossier>` qui vérifie les fichiers d'un
     dossier de sortie contre les schémas.
  4. Mettre à jour README et ROADMAP (retirer « pas de mobile », ajouter le lien vers ce plan).
- **Critères de réussite** :
  - 6 ADR fusionnées ; chaque contrat a un schéma et un exemple valide versionné dans
    `tests/fixtures/`.
  - `tennisvision validate` rejette un fichier non conforme avec un message lisible (test).
- **Hors périmètre** : implémentation des modules qui produisent ces fichiers.

---

### Lot 1 — Socle : portabilité, multi-utilisateur, vidéos de téléphone

- **Priorité** : P0 · **Effort** : M · **Dépend de** : Lot 0 (formats `match.json`,
  `frames.parquet`) ; peut commencer avant en suivant le § 5.
- **Objectif** : un paquet installable et utilisable par n'importe qui, sur n'importe quel OS,
  qui lit correctement les vidéos de téléphone.
- **Justification** : constats C6–C12, C14 et tableau du § 4. Les vidéos de téléphone (cadence
  variable, rotation) feront échouer silencieusement toutes les mesures de durée si l'ingestion
  n'est pas corrigée d'abord.
- **Tâches** :
  1. **Configuration** : remplacer `ROOT_DIR = parents[2]` par une configuration résolue dans
     l'ordre option CLI → variable d'environnement (`TENNISVISION_HOME`, etc.) → dossiers
     utilisateur (`platformdirs`). Aucun chemin vers le dépôt dans le code installé.
  2. **Ingestion avec horodatages** : lire les horodatages de présentation (via PyAV ou
     `CAP_PROP_POS_MSEC`), produire `frames.parquet` ; toutes les durées calculées à partir de
     `t_ms`, jamais de `frame_idx / fps`.
  3. **Orientation** : appliquer la métadonnée de rotation ; refuser explicitement la vidéo
     portrait avec un message (« filmez en paysage »).
  4. **Sortie H.264** (via PyAV ou ffmpeg) pour toute vidéo destinée à être vue ; garder
     `mp4v` seulement en repli.
  5. **Dossier de sortie** : chemins relatifs dans `summary.json` ; créer le dossier après
     validation des bornes (C6) ; une seule définition de chaque indicateur (C7).
  6. **Découplage TenniSet** : `analyze` ne prend qu'une vidéo (et un `match.json` optionnel) ;
     `--point` déplacé dans une sous-commande `tenniset` ou `eval`.
  7. **Registre de modèles** : `models.toml` (nom, version, URL, SHA-256), téléchargement
     explicite (`tennisvision models pull`), cache utilisateur, fonctionnement hors-ligne
     ensuite.
  8. **Corrections de dessin** : la traînée de balle s'efface après N frames sans balle et ne
     relie pas deux points séparés de plus de K frames (C4).
  9. **Tests** : pipeline avec un faux tracker (C12) ; vidéo synthétique en cadence variable ;
     vidéo avec rotation ; test de saut de frame contre lecture séquentielle.
  10. **CI** (GitHub Actions) : Linux + Windows + macOS, Python 3.12 et 3.13, `pytest` +
      `ruff`. Ajuster les bornes de `pyproject.toml` à ce qui est testé.
  11. Notebook : le faire appeler le paquet, ou le déplacer dans `notebooks/archive/`.
- **Critères de réussite** :
  - `pip install .` dans un environnement vierge puis `tennisvision analyze video.mp4` marche
    **depuis un autre dossier que le dépôt** (test CI).
  - Sur une vidéo synthétique VFR (intervalles connus), la durée calculée d'un intervalle a une
    erreur < 1 ms.
  - Les vidéos produites se lisent dans Chrome, Firefox et Safari (vérification manuelle
    documentée une fois, puis test sur le codec dans la CI).
  - CI verte sur les 3 OS ; aucun chemin absolu dans les sorties (test).
- **Hors périmètre** : interface web, comptes utilisateurs (Lot 11).

---

### Lot 2 — Jeu de données amateur, protocole de tournage et outil d'annotation

- **Priorité** : P0 · **Effort** : L (outil : M ; collecte et annotation : en continu)
- **Dépend de** : Lot 0 (format `events.json`)
- **Objectif** : disposer d'une vérité terrain **représentative des utilisateurs** sur
  laquelle toutes les statistiques sont validées.
- **Justification** : § 3.1. Sans ce jeu, aucun chiffre de justesse n'a de valeur pour le
  produit.
- **Tâches** :
  1. **Protocole de tournage** (`docs/filming_guide.md`, futur guide utilisateur) : téléphone
     en paysage, derrière le fond de court, centré, le plus haut possible (≥ 2 m, fixation sur
     grillage ou trépied), court entier visible, 1080p, 60 fps si possible, stabilisation et
     mode économie d'énergie désactivés, objectif nettoyé. Photos d'exemple de bons et de
     mauvais cadrages.
  2. **Collecte** : au moins 10 matchs, au moins 6 joueurs distincts, 3 surfaces (terre, dur,
     intérieur), 30 et 60 fps, au moins 2 modèles de téléphone, niveaux de débutant à
     classé. Formulaire de **consentement écrit** des personnes filmées (droit à l'image, RGPD),
     registre des licences d'utilisation de chaque vidéo.
  3. **Outil d'annotation** : partir d'un outil existant (CVAT, Label Studio) plutôt que d'en
     écrire un, avec export vers `events.json` et `ball.csv`. Pour la balle : annotation d'une
     frame sur N + interpolation assistée. Pour le court : 4 coins + 14 points clés par vidéo.
  4. **Annotations par niveau de coût** :
     - Niveau 1 (tous les matchs) : points (début, fin, serveur, gagnant), services (1re/2de,
       in/faute/let), score par jeu.
     - Niveau 2 (≥ 5 matchs) : chaque frappe (instant, joueur, type, issue), chaque rebond
       (instant, position sur le court, in/out).
     - Niveau 3 (≥ 3 000 frames réparties sur ≥ 5 matchs) : position de la balle image par
       image.
  5. **Découpage** : par match (jamais à l'intérieur d'un match), avec un jeu de test gelé
     auquel on ne touche pas pendant le développement.
  6. **Double annotation** de 10 % des données par une seconde personne pour mesurer l'accord
     inter-annotateurs (c'est le plafond atteignable par les modèles).
  7. **Étiquettes faibles TenniSet** : extraire *forehand/backhand/out/net/winner* des `desc`
     pour enrichir TenniSet ; vérifier la convention du champ `Side` sur 50 frappes à la main
     avant tout usage (la distribution observée sur V006 — Near : Right 59 / Left 36 ;
     Far : Left 52 / Right 42 — suggère un côté **dans l'image** et non coup droit/revers, à
     confirmer).
  8. Récupérer la vidéo **V009** manquante (49 % des points TenniSet, C15).
- **Critères de réussite** :
  - Jeu amateur v1 : ≥ 10 matchs, ≥ 6 joueurs, ≥ 3 surfaces, consentements archivés.
  - Accord inter-annotateurs publié : instant de frappe à ±2 frames sur ≥ 95 % des frappes,
    type de coup κ ≥ 0,9, in/out ≥ 95 %.
  - Fiche descriptive du jeu (`docs/dataset_card.md`) : contenu, conditions, biais connus,
    licence.
- **Hors périmètre** : entraînement de modèles.

---

### Lot 3 — Banc d'évaluation automatique

- **Priorité** : P0 · **Effort** : M · **Dépend de** : Lot 0 ; utilisable dès maintenant sur
  TenniSet, puis sur le jeu amateur quand il arrive.
- **Objectif** : une commande unique qui calcule toutes les métriques du projet et détecte
  toute régression.
- **Justification** : C13. La promesse « stats mesurées et validées » repose entièrement sur ce
  lot. Les chiffres actuels du README sont manuels et non reproductibles.
- **Tâches** :
  1. Créer `src/tennisvision/eval/` avec un comparateur par type d'événement :
     appariement temporel avec tolérance (algorithme hongrois), précision / rappel / F1,
     erreur temporelle moyenne.
  2. Métriques par brique :
     - balle : précision / rappel / F1 à 4 px et à 10 px (à 1080p), taux de trous > 5 frames ;
     - court : erreur de reprojection en px et en cm sur les points clés annotés ;
     - joueurs : % de frames où les 2 joueurs sont trouvés, échanges d'identité (IDF1) ;
     - frappes : F1 à ±2 et ±5 frames ;
     - coups : matrice de confusion, F1 par classe ;
     - rebonds : F1 temporel, erreur de position en cm, exactitude in/out (et séparément
       pour les balles à < 30 cm d'une ligne) ;
     - points : IoU temporel, exactitude du gagnant, % de jeux au score juste.
  3. **Métriques par statistique** (ce que voit l'utilisateur) : pour chaque stat du § 1,
     erreur absolue et relative entre la valeur calculée sur les événements prédits et sur la
     vérité terrain, par match puis agrégée (moyenne et pire cas).
  4. Chargeur TenniSet complet (`generalised/*.json` → `events.json`), découpage
     leave-one-match-out.
  5. Commande `tennisvision eval --dataset amateur|tenniset --split test` → `reports/eval/`
     (JSON + HTML lisible) ; historisation des résultats par commit.
  6. Test de non-régression en CI sur un mini-jeu (2 points annotés, quelques Mo) pour les
     briques rapides ; évaluation complète manuelle ou sur machine GPU.
- **Critères de réussite** :
  - `tennisvision eval` tourne de bout en bout sur TenniSet avec des prédictions factices
    (prédiction = vérité → F1 = 1 ; prédiction vide → rappel = 0) : tests unitaires.
  - Chaque métrique du tableau a un test sur un cas construit à la main.
  - Les résultats d'une version à l'autre sont comparables (même jeu, même graine, même
    tolérance, versions des modèles enregistrées dans le rapport).
- **Hors périmètre** : amélioration des modèles.

---

### Lot 4 — Suivi de balle fiable

- **Priorité** : P0 · **Effort** : L · **Dépend de** : Lot 0 (`ball.csv`) ; Lot 2 niveau 3
  pour l'affinage et la validation ; Lot 3 pour la mesure (peut démarrer avec son propre
  script d'évaluation).
- **Objectif** : position 2D de la balle sur ≥ 90 % des frames où elle est visible, avec très
  peu de fausses détections, sur vidéo amateur.
- **Justification** : C1, C2, C3, § 3.3. Rebonds, zones, in/out et une partie des frappes en
  dépendent.
- **Tâches** :
  1. Sortir la balle de ByteTrack : le tracker YOLO ne traite plus que la classe personne.
  2. **Référence** : TrackNetV3 et WASB avec poids publics, sans affinage, sur le jeu amateur
     (et TenniSet échantillonné) : mesurer avant de modifier.
  3. **Affinage** sur le jeu de données TrackNet + frames amateurs annotées (Lot 2 niveau 3) ;
     augmentations ciblées : flou de mouvement, compression, changement de luminosité,
     grillage simulé, balles parasites des courts voisins.
  4. **Masque de court** : ignorer les détections hors du court étendu (utilise `court.json`
     si disponible, sinon un masque manuel), pour supprimer les balles des courts voisins.
  5. **Post-traitement hors-ligne** : rejet des points aberrants (vitesse physiquement
     impossible), découpage en segments entre événements, ajustement de courbes, interpolation
     des trous ≤ 10 frames marquée `interpolated`.
  6. Gérer 30 et 60 fps (entrée du modèle à cadence normalisée ou modèle entraîné aux deux).
  7. Mesurer le débit et exporter en ONNX.
- **Critères de réussite** (jeu amateur, test, 1080p) :
  - F1 ≥ 0,85 à 10 px ; précision ≥ 0,90 (les faux positifs coûtent plus cher que les trous).
  - Sur le point P00000001 de V006 : la roue de caméra n'est plus jamais retenue.
  - Débit ≥ 30 fps sur un GPU grand public de référence (à choisir et noter), ≥ 5 fps CPU.
  - Comparaison avant/après publiée par le banc d'évaluation.
- **Hors périmètre** : détection des rebonds (Lot 8), 3D.

---

### Lot 5 — Calibration du court, sélection et identité des joueurs, pose

- **Priorité** : P0 · **Effort** : L · **Dépend de** : Lot 0 (`court.json`, `players.csv`,
  `match.json`)
- **Objectif** : une homographie image → court fiable pour toute la vidéo, les 2 joueurs
  isolés de toutes les autres personnes, identifiés de bout en bout du match, avec leur pose.
- **Justification** : C5 (31 « personnes » pour 2 joueurs) ; toutes les zones, positions et
  stats par joueur en dépendent ; § 3.5 points 1 et 2.
- **Tâches** :
  1. **Court** : régression de 14 points clés (réseau de type TennisCourtDetector, ou
     détection de lignes + appariement au modèle ITF 23,77 m × 10,97 m), sur une **image de
     fond** (médiane de frames, sans joueurs) puisque la caméra est fixe ; raffinement par
     ajustement des lignes ; détection de dérive (téléphone qui bouge) par contrôle périodique.
  2. **Repli manuel** : si la confiance est basse, demander à l'utilisateur de placer ou
     corriger 4 coins (c'est aussi ce que font les applications grand public). Prévoir le
     format d'entrée dans `match.json`.
  3. **Sélection des joueurs** : garder les personnes dont les pieds (bas de la boîte, ou
     chevilles de la pose) se projettent dans le court étendu (marge de 4 m derrière les lignes
     de fond, 2 m sur les côtés), une par demi-court.
  4. **Identité au long du match** : apparence (vecteur de ré-identification ou histogramme de
     couleurs des vêtements) + contrainte des changements de côté (jeux impairs, jeu décisif) ;
     correction possible par l'utilisateur.
  5. **Pose** : 17 points clés par joueur (YOLO-pose ou RTMPose), seulement sur les 2 joueurs
     retenus pour économiser le calcul.
  6. **Latéralité** : lue dans `match.json` ; estimation automatique (main qui tient la
     raquette au service) en appoint, avec confirmation.
  7. Minimap vue de dessus incrustée (livrable démontrable).
- **Critères de réussite** :
  - Erreur de reprojection médiane ≤ 10 cm au sol sur les points clés annotés, calibration
    réussie sans intervention sur ≥ 90 % des vidéos amateurs.
  - Les 2 joueurs trouvés sur ≥ 98 % des frames de jeu ; 0 personne hors court retenue sur
    le jeu de test ; échange d'identité entre joueurs : 0 par match après logique de changement
    de côté.
- **Hors périmètre** : estimation de pose 3D.

---

### Lot 6 — Détection des frappes (fusion son + pose + balle)

- **Priorité** : P1 · **Effort** : M · **Dépend de** : contrats `ball.csv`, `players.csv`
  (peut être développé sur la vérité terrain du Lot 2 à la place des sorties des Lots 4–5)
- **Objectif** : l'instant de chaque frappe et le joueur qui frappe.
- **Justification** : § 3.4. C'est l'événement le plus fréquent et la base de tous les comptes
  de coups ; le son le rend robuste aux trous de la balle.
- **Tâches** :
  1. Extraction de la piste audio (resynchronisée sur `t_ms`) ; détection d'attaques
     (flux spectral, bande des impacts) → `audio_onsets.csv`.
  2. Signal balle : changements brusques de direction/vitesse près d'un joueur.
  3. Signal pose : vitesse du poignet, proximité poignet–balle.
  4. Fusion : petit classifieur (gradient boosting) sur des fenêtres centrées sur les candidats
     des trois sources ; attribution au joueur le plus proche de la balle / en mouvement de
     frappe ; filtrage des sons de courts voisins (pas de joueur en geste au même instant).
  5. Mode dégradé documenté : vidéo sans son, ou balle absente.
  6. Validation sur TenniSet (sans son exploitable : commentateurs et public) et surtout sur le
     jeu amateur.
- **Critères de réussite** (jeu amateur, test) :
  - F1 ≥ 0,95 à ±2 frames (30 fps) sur les frappes ; attribution du joueur ≥ 98 %.
  - Ablation publiée : F1 avec chaque source seule et avec la fusion.
- **Hors périmètre** : type de coup.

---

### Lot 7 — Classification des coups

- **Priorité** : P1 · **Effort** : M · **Dépend de** : contrat `events.json` (frappes),
  `players.csv` (pose), `match.json` (latéralité)
- **Objectif** : pour chaque frappe : service, coup droit, revers, volée (CD/R), smash ; et
  en attribut : lifté / coupé quand c'est possible.
- **Justification** : statistiques 1 et 5 du § 1.
- **Tâches** :
  1. Caractéristiques : séquence de poses ±0,5 s autour de la frappe (normalisée par la taille
     du joueur), position de la balle relative au corps, position du joueur sur le court
     (volée = près du filet), contexte (premier coup du point = service).
  2. Règle de base explicite à battre : côté de la balle par rapport au corps + latéralité →
     coup droit / revers. Elle sert de référence et de repli.
  3. Modèle : GRU / 1D-CNN sur les poses, ou gradient boosting sur caractéristiques ;
     pré-entraîné sur TenniSet (étiquettes faibles du Lot 2), affiné sur le jeu amateur.
  4. Gestion des gauchers et du joueur de fond (vu de face, gauche/droite inversées dans
     l'image) : tests dédiés.
  5. Seuil de confiance : sous le seuil, classe « incertain » plutôt qu'une erreur.
- **Critères de réussite** (jeu amateur, test) :
  - Service/coup droit/revers : exactitude ≥ 95 %, F1 par classe ≥ 0,93, y compris sur les
    joueurs gauchers.
  - Volée / smash : F1 ≥ 0,80 (classes rares, publiées à part).
  - Erreur sur le **compte** de coups droits et revers par joueur et par match ≤ 5 %.
- **Hors périmètre** : effet (lift/slice) en v1 : seulement si F1 ≥ 0,8, sinon masqué.

---

### Lot 8 — Rebonds, in/out et zones

- **Priorité** : P1 · **Effort** : L · **Dépend de** : `ball.csv`, `court.json`, frappes
  (`events.json`)
- **Objectif** : chaque rebond horodaté et placé sur le court, avec décision in/out et marge
  d'incertitude ; filets détectés.
- **Justification** : statistiques 1, 2, 4, 6 du § 1. Limite physique : en monocaméra, la
  projection au sol n'est valide qu'au rebond (§ 3.5 point 7).
- **Tâches** :
  1. Détection des rebonds entre deux frappes : rupture de la trajectoire verticale dans
     l'image (classifieur sur caractéristiques de trajectoire, approche classique publiée pour
     le tennis monocaméra).
  2. Projection par homographie → position au sol en mètres ; propagation de l'incertitude
     (erreur de calibration + erreur de position balle + flou) → rayon d'incertitude en cm.
  3. Décision in/out avec trois issues : `in`, `out`, `trop proche pour décider` (balle à
     moins de la marge d'incertitude d'une ligne).
  4. Fin de point : balle dans le filet (arrêt de trajectoire près du filet), double rebond,
     balle hors champ.
  5. Zones : carrés de service découpés en extérieur / corps / T ; fond de court en
     profondeur (avant / après ligne de service, dernier mètre) et en largeur (tiers).
- **Critères de réussite** (jeu amateur, test) :
  - F1 rebonds ≥ 0,90 à ±2 frames ; erreur de position médiane ≤ 30 cm.
  - In/out : exactitude ≥ 97 % hors bande d'incertitude ; bande d'incertitude couvrant
    ≤ 10 % des rebonds ; **jamais** de décision ferme fausse à plus de 30 cm de la ligne.
  - Zone de service correcte ≥ 90 %.
- **Hors périmètre** : hauteur au-dessus du filet, vitesse (P3).

---

### Lot 9 — Points, gagnant du point et score

- **Priorité** : P1 · **Effort** : L · **Dépend de** : `events.json` (frappes, coups,
  rebonds), `match.json`
- **Objectif** : découpage du match en points, gagnant de chaque point, score courant, avec
  correction facile par l'utilisateur.
- **Justification** : statistiques 2 et 3 du § 1 ; le score est ce que l'utilisateur vérifie
  en premier : une erreur ici décrédibilise tout le reste.
- **Tâches** :
  1. Segmentation des points : début = service (pose de service + position derrière la ligne
     de fond), fin = dernier événement + inactivité (balle morte, joueurs qui marchent).
  2. Gagnant du point : règles sur la fin de l'échange (dernier frappeur + issue : faute,
     filet, double rebond, gagnant), avec confiance.
  3. Machine à états du score (jeux, sets, jeu décisif, point décisif / avantage, format lu
     dans `match.json`) ; cohérence vérifiée par le côté de service (déduite de l'image) ; la
     machine à états **corrige** les décisions improbables (ex. un serveur qui changerait au
     milieu d'un jeu).
  4. **Correction par l'utilisateur** : format d'édition (fichier puis interface du Lot 11) ;
     toute correction est enregistrée avec `source: user` et réinjectée dans le jeu de données
     (avec consentement).
  5. Tests unitaires exhaustifs de la machine à états sur des matchs TenniSet complets
     (indépendant de la vision).
- **Critères de réussite** :
  - Machine à états : 100 % des scores de jeux TenniSet reproduits à partir des gagnants
    annotés.
  - Jeu amateur : points retrouvés (IoU ≥ 0,5) ≥ 97 % ; gagnant du point juste ≥ 90 % ;
    longueur d'échange exacte ≥ 85 % des points, à ±1 frappe ≥ 97 %.
  - Après correction de 5 % des points au plus par l'utilisateur, score final juste dans
    100 % des matchs du test.
- **Hors périmètre** : distinction automatique faute directe / faute provoquée (voir Lot 10,
  expérimental).

---

### Lot 10 — Statistiques, rapport, moments forts et conseils

- **Priorité** : P2 · **Effort** : M (rapport + stats) + M (conseils) · **Dépend de** :
  `events.json` complet ; Lot 3 pour le drapeau `validated`
- **Objectif** : ce que le joueur voit : un rapport clair, des extraits vidéo, des conseils
  actionnables, chacun justifié.
- **Justification** : c'est la valeur du produit ; § 1.
- **Tâches** :
  1. Calcul de `stats.json` à partir des seuls événements (fonction pure, testable avec des
     événements construits à la main) : toutes les stats du § 1, par joueur, par set, en
     distinguant 1re et 2de balle.
  2. Chaque stat porte `n`, `confidence`, et `validated` (issu du dernier rapport du banc
     d'évaluation : erreur sous le seuil). Les stats non validées sont masquées ou marquées
     « expérimental ».
  3. **Rapport HTML** (statique, lisible sur téléphone) : tableau de bord, cartes de chaleur
     des rebonds et des positions de frappe sur un court vu de dessus, distribution des
     longueurs d'échange, points gagnés par longueur.
  4. **Moments forts** : plus longs échanges, points gagnants, balles de break ; **match
     condensé** (points seulement, temps morts coupés) ; extraits en H.264.
  5. **Conseils** : moteur de règles explicites, chaque règle = condition chiffrée + message +
     extraits vidéo en preuve. Exemples :
     - « 62 % de vos fautes en revers finissent dans le filet (18/29) » → travailler la
       hauteur au-dessus du filet ; 3 extraits.
     - « 1res balles : 48 % ; points gagnés sur 2de : 31 % » → sécuriser la 1re balle.
     - « 70 % de vos coups de fond rebondissent avant la ligne de service » → profondeur.
     - « Vous perdez 72 % des échanges de plus de 8 coups » → endurance / patience.
     Aucune règle ne se déclenche sous un effectif minimal (ex. n ≥ 15). Un modèle de langage
     peut reformuler, jamais inventer un chiffre : il ne reçoit que `stats.json`.
  6. Faute directe / coup gagnant : **expérimental**, publié avec sa matrice de confusion.
- **Critères de réussite** :
  - Pour chaque stat affichée comme validée : erreur relative ≤ 10 % (≤ 5 % pour les comptes
    de coups et le % de 1res balles) sur le jeu de test amateur, chiffre visible dans le
    rapport.
  - Chaque conseil renvoie à ≥ 3 extraits vidéo et à la stat qui le déclenche.
  - Test utilisateur : 5 joueurs amateurs lisent leur rapport ; ≥ 4 comprennent chaque
    graphique sans explication et jugent au moins un conseil pertinent (grille d'entretien
    écrite avant).
- **Hors périmètre** : comparaison avec des moyennes par niveau (nécessite beaucoup de
  matchs, P3).

---

### Lot 11 — Produit : import, traitement, confidentialité, licences, performance

- **Priorité** : P2 · **Effort** : L · **Dépend de** : Lot 1 ; pipeline de bout en bout
  (même partiel) pour être utile
- **Objectif** : un amateur importe sa vidéo et reçoit son rapport sans rien installer.
- **Justification** : la cible « simple à utiliser » ; § 3.2 et § 3.5 points 3, 4, 6.
- **Tâches** :
  1. **Contrôle à l'import** (en quelques secondes, sur un échantillon de frames) : paysage ?
     court entier visible ? calibration possible ? cadence ? Message clair et actionnable si
     non, avant de lancer un long calcul.
  2. **Formulaire de match** : joueurs, latéralité, premier serveur, format, surface ;
     confirmation/correction des 4 coins du court si la calibration est incertaine.
  3. **Traitement** : file de tâches, reprise après interruption, progression visible ;
     découpage temporel de la vidéo pour paralléliser ; estimation de coût par heure de vidéo.
  4. **Interface de correction** : liste des points avec vignette, édition du gagnant et des
     coups, recalcul instantané des stats.
  5. **Confidentialité / RGPD** : consentement des personnes filmées, durée de conservation,
     suppression à la demande, chiffrement au repos, usage pour l'entraînement **en option
     explicite** ; flouter le public si les vidéos sont partagées.
  6. **Licences** : remplacer ou isoler les composants AGPL (ultralytics) ou mettre le service
     en conformité (publication du code) ; vérifier les licences des poids pré-entraînés et des
     jeux d'entraînement ; documenter dans `docs/licenses.md`.
  7. **Performance** : ne traiter en détail que les fenêtres de jeu (détecteur de « jeu en
     cours » léger en première passe), pose seulement autour des frappes, inférence en lot,
     demi-précision.
- **Critères de réussite** :
  - Un utilisateur qui découvre le produit obtient son rapport sans aide (test avec
    5 personnes, ≥ 4 réussissent).
  - 1 h de vidéo traitée en ≤ 1 h sur la machine de référence ; coût par heure de vidéo
    chiffré.
  - Revue de conformité licences et RGPD écrite et validée avant toute ouverture publique.
- **Hors périmètre** : application mobile native (Lot 12).

---

### Lot 12 — Bonus : téléphone, temps réel et extensions

- **Priorité** : P3 · **Effort** : XL
- **Dépend de** : modèles validés des Lots 4–9
- **Contenu possible** (chacun peut devenir un lot séparé le moment venu) :
  - Inférence sur téléphone (CoreML / TFLite / ONNX Runtime Mobile), d'abord pour le contrôle
    de cadrage, puis pour le comptage de coups en direct.
  - Saisie du score depuis une montre ou par la voix, comme source de vérité alternative.
  - Vitesse de balle approchée (distance au sol entre frappe et rebond / temps), publiée avec
    sa marge d'erreur ; uniquement si l'erreur mesurée est ≤ 10 %.
  - Doubles ; caméra latérale ; multi-caméras (deux téléphones synchronisés → triangulation).
  - Repères par niveau (comparaison aux joueurs de même classement) une fois assez de matchs.
- **Critère d'entrée** : ne démarrer qu'après validation des critères des Lots 4 à 9 sur le
  jeu amateur.

---

## 7. Ordonnancement

### 7.1 Tableau récapitulatif

| Lot | Titre | Priorité | Effort | Peut démarrer |
|---|---|---|---|---|
| 0 | Décisions et contrats | P0 | S | Immédiatement |
| 1 | Socle portable, vidéos de téléphone | P0 | M | Immédiatement |
| 2 | Jeu amateur + annotation | P0 | L | Immédiatement (format : Lot 0) |
| 3 | Banc d'évaluation | P0 | M | Immédiatement (TenniSet d'abord) |
| 4 | Balle | P0 | L | Immédiatement (référence sans affinage) |
| 5 | Court + joueurs + pose | P0 | L | Immédiatement |
| 6 | Frappes | P1 | M | Sur vérité terrain dès que le Lot 2 a 2 matchs |
| 7 | Type de coup | P1 | M | Sur TenniSet tout de suite, amateur ensuite |
| 8 | Rebonds, in/out, zones | P1 | L | Après des sorties de 4 et 5 (ou vérité terrain) |
| 9 | Points et score | P1 | L | Machine à états tout de suite sur TenniSet |
| 10 | Stats, rapport, conseils | P2 | M+M | Sur `events.json` de vérité terrain, tout de suite |
| 11 | Produit | P2 | L | Après Lot 1 |
| 12 | Bonus | P3 | XL | Après validation 4–9 |

### 7.2 Chemin critique

`Lot 0 → Lot 2 (collecte) → Lot 4 → Lot 8 → Lot 9 → Lot 10`

La collecte et l'annotation du jeu amateur (Lot 2) est le plus long délai **incompressible** :
la lancer en premier, en parallèle de tout le reste.

### 7.3 Premiers pas recommandés (2 semaines)

1. Lot 0 (2 jours) — fige le vocabulaire commun.
2. En parallèle : Lot 1 et Lot 3 (TenniSet) ; démarrer la collecte du Lot 2 (protocole +
   2 premiers matchs filmés par d'autres personnes que le développeur).
3. Lot 4, étape « référence » : mesurer TrackNetV3 et WASB pré-entraînés sur les 2 premiers
   matchs amateurs. Ce seul chiffre dira si l'objectif de F1 ≥ 0,85 est à portée.

### 7.4 Comment utiliser ce document dans une autre conversation

Copier : le § 1 (cible), le § 5 (contrats), et le lot visé en entier. Ajouter : « Respecte
les contrats du § 5 ; ne modifie pas les fichiers des autres lots ; livre avec les tests et le
rapport du banc d'évaluation quand il existe. »

---

## 8. Ce qui n'a pas pu être vérifié

- **Aucune vidéo de téléphone** dans le dépôt : tous les constats sur la cadence variable, la
  rotation et le changement de domaine sont des risques connus du domaine, **non mesurés ici**.
- La **vidéo annotée** `reports/V006_25740-25984/annotated.mp4` n'a pas été visionnée ; le
  constat sur la balle s'appuie sur `summary.json` (0 détection) et sur le README.
- Le chiffre « < 2 % de vraie balle » du README n'est pas reproductible (pas de vérité terrain
  balle dans le dépôt).
- Performances GPU : non mesurées (seul le chiffre CPU de 12,5 fps est observé).
- Exactitude du saut de frame : vérifiée sur V006 seulement (2 positions), pas sur V007–V010.
- Licences de TenniSet, du jeu TrackNet et des poids pré-entraînés : à vérifier à la source.
- Signification exacte du champ `Side` de TenniSet : hypothèse « côté dans l'image » tirée de
  la distribution observée, non confirmée.
- Les seuils des critères de réussite sont des **objectifs proposés** (fondés sur ce qu'un
  joueur tolère et sur les ordres de grandeur publiés), à recaler après la première mesure de
  référence du Lot 4 et l'accord inter-annotateurs du Lot 2.
- Les caractéristiques des produits SwingVision et Hawk-Eye citées viennent de connaissances
  générales, non vérifiées dans le cadre de cette revue.

---

## 9. Références

- Faulkner H., Dick A. — *TenniSet: A Dataset for Dense Fine-Grained Event Recognition,
  Localisation and Description*, DICTA 2017.
- Huang Y.-C. et al. — *TrackNet: A Deep Learning Network for Tracking High-speed and Tiny
  Objects in Sports Applications*, AVSS 2019.
- Sun N.-E. et al. — *TrackNetV2: Efficient Shuttlecock Tracking Network*, 2020.
- Chen Y.-J., Wang Y.-S. — *TrackNetV3: Enhancing ShuttleCock Tracking with Augmentations and
  Trajectory Rectification*, 2023.
- Tarashima S. et al. — *Widely Applicable Strong Baseline for Sports Ball Detection and
  Tracking* (WASB), BMVC 2023.
- Zhang Y. et al. — *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*,
  ECCV 2022.
- ITF — *Rules of Tennis* (dimensions du court : 23,77 m × 10,97 m, simple 8,23 m).
