# TennisTrackin

Analyse **post-traitement** de vidéos de matchs de tennis (caméra fixe, vue de fond de court)
pour produire des statistiques : services, coups droits, revers, longueur des échanges, zones de
frappe, comptage des points.

Objectif : un MVP honnête **vision → événements → stats**, dont chaque étape est mesurée contre
les annotations du dataset [TenniSet](https://github.com/HaydenFaulkner/Tennis). La capture se
fait sur téléphone (caméra fixe derrière le fond de court) ; le traitement est hors-ligne, sur
PC ou serveur en v1 (cf. [ADR-001](docs/adr/001-perimetre-v1.md) et
[ADR-002](docs/adr/002-traitement-serveur.md)). Pas de temps réel ni d'arbitrage au centimètre
(cf. [ADR-006](docs/adr/006-limites-monocamera.md)).

État actuel : **étape 0** (ingestion vidéo + détection/tracking joueurs et balle), sur le socle
des deux premiers lots de [docs/PLAN_AMELIORATION.md](docs/PLAN_AMELIORATION.md) :
- **Lot 0** : décisions d'architecture ([docs/adr/](docs/adr/)) et contrats de données
  ([docs/contracts.md](docs/contracts.md), commande `tennisvision validate`) ;
- **Lot 1** : paquet installable et utilisable depuis n'importe quel dossier (aucun chemin vers
  le dépôt), horodatages réels des frames, vidéos de téléphone, sortie H.264, registre de
  modèles, `analyze` découplé de TenniSet, tests avec un faux tracker.

La suite est décrite dans [ROADMAP.md](ROADMAP.md).
Plan de référence : [docs/PLAN_AMELIORATION.md](docs/PLAN_AMELIORATION.md) ; décisions : [docs/adr/](docs/adr/).

## Installation

Python 3.12+ (développé avec 3.14). Créer et activer un environnement virtuel :

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (PowerShell : .venv\Scripts\Activate.ps1)
source .venv/bin/activate       # macOS / Linux
```

Puis installer le paquet :

```bash
pip install .                   # utilisation
pip install -e ".[dev]"         # développement (pytest, ruff)
```

Les poids du modèle ne sont **pas** téléchargés automatiquement : avant le premier lancement,
les récupérer une fois (ils sont gardés dans le cache utilisateur, puis tout fonctionne
hors-ligne) :

```bash
tennisvision models pull
```

### Variables d'environnement

Chaque dossier se résout dans l'ordre : option de la commande, variable d'environnement, valeur
par défaut.

| Variable | Rôle | Défaut |
|---|---|---|
| `TENNISVISION_MODELS_DIR` | cache des modèles | cache utilisateur (`platformdirs`) |
| `TENNISVISION_OUTPUT_DIR` | racine des dossiers de sortie | `./tennisvision-sorties` |
| `TENNISVISION_TENNISET_DIR` | dossier du jeu TenniSet | `./data` |

## Données

Le dossier `data/` n'est pas versionné. Seules les commandes `tennisvision tenniset ...` en ont
besoin. Arborescence attendue :

```
data/
├── videos/        V006.mp4, V007.mp4, V008.mp4, V010.mp4 (1280x720 @ 25 fps)
├── annotations/   points.txt, classes.txt, generalised/*.json, labels/*.txt
└── splits/        découpages train/val/test de TenniSet
```

## Utilisation

```bash
# Détection + tracking sur une vidéo (ou un intervalle de frames), sorties dans
# ./tennisvision-sorties/<video>_<debut>-<fin>/
tennisvision analyze match.mp4
tennisvision analyze match.mp4 --start 250 --end 500 --no-video

# Avec la description du match (match.json validé puis copié dans le dossier de sortie)
tennisvision analyze match.mp4 --match match.json --output-dir sortie/

# Extraire un intervalle dans un petit clip
tennisvision clip match.mp4 --start 250 --end 500 -o rallye.mp4

# Points annotés de TenniSet (bornes et vidéo lues dans le dossier --data-dir)
tennisvision tenniset analyze P00000001 --data-dir data
tennisvision tenniset clip P00000001 --data-dir data -o rallye.mp4

# Vérifier un dossier de sortie contre les contrats de données
tennisvision validate tennisvision-sorties/match_0-499

# Modèles : liste (présent/absent), téléchargement, import d'un fichier local
tennisvision models list
tennisvision models pull yolo11n
tennisvision models import yolo11n chemin/vers/yolo11n.pt
```

`python -m tennisvision ...` est équivalent à `tennisvision ...`. Codes retour : 0 succès, 1
fichier invalide (`validate`), 2 erreur d'argument ou de fichier (bornes invalides, modèle absent
du cache — le message indique `tennisvision models pull` —, point inconnu…).

Sorties dans `tennisvision-sorties/<video>_<debut>-<fin>/` (ou `--output-dir`), toutes
référencées par des chemins **relatifs** dans `summary.json` :
- `annotated.mp4` : vidéo annotée en **H.264** (lisible dans les navigateurs) : boîtes joueurs
  avec ID, balle et sa traînée, numéro de frame et temps réel ;
- `tracks.csv` : une ligne par détection (`frame_idx`, `t_ms`, track_id, classe, confiance,
  boîte) ;
- `frames.parquet` : index complet de la vidéo (`frame_idx`, `t_ms`), seule source de temps du
  pipeline ;
- `summary.json` : résumé chiffré (intervalle en frames et en ms, durée, identités suivies,
  taux de détection de la balle, modèle utilisé avec sa somme SHA-256, version, vitesse) ;
- `match.json` : copie du fichier donné par `--match`, s'il y en a un.

**Vidéos de téléphone** : la cadence variable est gérée (les durées viennent des horodatages
réels, jamais de `frame_idx / fps`) et la métadonnée de rotation est appliquée. Une vidéo qui
reste en **portrait** après rotation est refusée : filmez en paysage.

## Structure

```
src/tennisvision/
├── cli.py            CLI (`tennisvision ...`, `python -m tennisvision ...`)
├── config.py         constantes et résolution des chemins (option -> variable -> défaut)
├── models.py         registre de modèles (models.toml), cache, téléchargement vérifié
├── pipeline.py       enchaînement des étapes
├── validate.py       validation des fichiers contre les contrats de données
├── schemas/          schémas JSON des contrats (cf. docs/contracts.md)
├── io/video.py       ingestion : horodatages, lecture, découpe, écriture H.264
├── tracking/         détection YOLO + ByteTrack, dessin, export
└── datasets/         chargeurs des annotations TenniSet
notebooks/            explorations (`archive/` : notebook du palier 1)
tests/                tests unitaires (vidéos synthétiques, faux tracker, pas de modèle)
models/               notes sur les poids (non versionnés)
reports/              réservé aux rapports d'évaluation futurs (non versionnés)
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
