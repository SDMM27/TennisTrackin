# TennisTrackin

Analyse **post-traitement** de vidéos de matchs de tennis (caméra fixe, vue de fond de court)
pour produire des statistiques : services, coups droits, revers, longueur des échanges, zones de
frappe, comptage des points.

Objectif : un MVP honnête **vision → événements → stats**, dont chaque étape est mesurée contre
les annotations du dataset [TenniSet](https://github.com/HaydenFaulkner/Tennis). Pas de temps
réel, pas de mobile, pas d'arbitrage au centimètre.

État actuel : **étape 0** (ingestion vidéo + détection/tracking joueurs et balle).
La suite est décrite dans [ROADMAP.md](ROADMAP.md).

## Installation

Python 3.12+ (développé avec 3.14) :

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Les poids YOLO (`models/yolo11n.pt`) sont téléchargés automatiquement par ultralytics s'ils
sont absents.

## Données

Le dossier `data/` n'est pas versionné. Arborescence attendue :

```
data/
├── videos/        V006.mp4, V007.mp4, V008.mp4, V010.mp4 (1280x720 @ 25 fps)
├── annotations/   points.txt, classes.txt, generalised/*.json, labels/*.txt
└── splits/        découpages train/val/test de TenniSet
```

## Utilisation

```bash
# Détection + tracking sur le 1er point de V006 (bornes lues dans points.txt)
python -m tennisvision analyze data/videos/V006.mp4 --point P00000001

# Sur un intervalle de frames quelconque, sans écrire de vidéo
python -m tennisvision analyze data/videos/V006.mp4 --start 25740 --end 26000 --no-video

# Extraire un rallye dans un petit fichier
python -m tennisvision clip data/videos/V006.mp4 --point P00000001 -o reports/rallye.mp4
```

Sorties dans `reports/<video>_<debut>-<fin>/` :
- `annotated.mp4` : boîtes joueurs avec ID, balle et sa traînée, numéro de frame ;
- `tracks.csv` : une ligne par détection (frame, track_id, classe, confiance, boîte) ;
- `summary.json` : résumé chiffré (identités suivies, taux de détection de la balle…).

## Structure

```
src/tennisvision/
├── cli.py            CLI (`python -m tennisvision ...`)
├── config.py         constantes et chemins
├── pipeline.py       enchaînement des étapes
├── io/video.py       étape 1 : lecture, découpe, écriture vidéo
├── tracking/         étape 2 : détection YOLO + ByteTrack, dessin, export
└── datasets/         chargeurs des annotations TenniSet
notebooks/            explorations (le notebook du palier 1)
tests/                tests unitaires (vidéos synthétiques, pas de modèle)
models/               poids (non versionnés)
reports/              sorties d'analyse (non versionnées)
```

## Tests

```bash
pytest -q
ruff check src tests
```

## Limites connues (étape 0)

- La balle est détectée par le YOLO COCO générique (`sports ball`) : petite, rapide et floue,
  elle est presque toujours ratée. Mesure sur le point `P00000001` de V006 : vraie balle trouvée
  sur < 2 % des frames. Le `taux_detection_balle` de `summary.json` est un taux **brut** qui
  compte aussi les faux positifs (ex. à `--imgsz 1280`, une roue de caméra TV fait monter le
  taux à ~70 %). La vraie mesure (précision/rappel contre des frames annotées) arrive à l'étape 1.
- `n_identites` compte toutes les personnes suivies (arbitres, ramasseurs, public), pas seulement
  les 2 joueurs : le filtrage par position sur le court viendra avec la calibration (étape 2).
- Les vidéos TenniSet sont des images de diffusion TV (changements de plan, ralentis) : le
  tracking est réinitialisé de fait à chaque changement de plan.
