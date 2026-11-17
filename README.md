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

Sans argument, la commande récupère les modèles utilisés par défaut par `analyze` : YOLO
(joueurs) et `wasb_tennis` (balle).

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
# Joueurs (YOLO) + balle (réseau dédié) sur une vidéo (ou un intervalle de frames), sorties dans
# ./tennisvision-sorties/<video>_<debut>-<fin>/
tennisvision analyze match.mp4
tennisvision analyze match.mp4 --start 250 --end 500 --no-video

# Balle : masque de court (court.json ou polygone), seuil, ou désactivation ; balle seule
tennisvision analyze match.mp4 --court court.json --ball-threshold 0.4
tennisvision analyze match.mp4 --ball-model aucun
tennisvision ball track match.mp4 --start 250 --end 500

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
tennisvision models pull yolo11n wasb_tennis
tennisvision models import yolo11n chemin/vers/yolo11n.pt
```

`python -m tennisvision ...` est équivalent à `tennisvision ...`. Codes retour : 0 succès, 1
fichier invalide (`validate`), 2 erreur d'argument ou de fichier (bornes invalides, modèle absent
du cache — le message indique `tennisvision models pull` —, point inconnu…).

Sorties dans `tennisvision-sorties/<video>_<debut>-<fin>/` (ou `--output-dir`), toutes
référencées par des chemins **relatifs** dans `summary.json` :
- `annotated.mp4` : vidéo annotée en **H.264** (lisible dans les navigateurs) : boîtes joueurs
  avec ID, balle (cercle plein si détectée, creux si interpolée) et sa traînée, numéro de frame
  et temps réel ;
- `tracks.csv` : une ligne par détection de **personne** (`frame_idx`, `t_ms`, track_id, classe,
  confiance, boîte) ;
- `ball.csv` : trajectoire de la balle, une ligne par frame traitée (`frame_idx`, `t_ms`, `x`,
  `y`, `visible`, `score`, `source`), voir `docs/balle.md` ;
- `frames.parquet` : index complet de la vidéo (`frame_idx`, `t_ms`), seule source de temps du
  pipeline ;
- `summary.json` : résumé chiffré (intervalle en frames et en ms, durée, identités suivies,
  bloc `balle` (modèle, taux de frames visibles, diagnostic du post-traitement), modèle YOLO
  utilisé avec sa somme SHA-256, version, vitesse) ;
- `match.json` : copie du fichier donné par `--match`, s'il y en a un.

**Vidéos de téléphone** : la cadence variable est gérée (les durées viennent des horodatages
réels, jamais de `frame_idx / fps`) et la métadonnée de rotation est appliquée. Une vidéo qui
reste en **portrait** après rotation est refusée : filmez en paysage.

## Évaluation

```bash
# Vérité TenniSet convertie en events.json (oracle : l'évaluer doit donner F1 = 1)
tennisvision tenniset export-events --data-dir data -o sortie_oracle/
tennisvision eval --dataset tenniset --predictions sortie_oracle/ --split test

# Jeu amateur : vérité terrain <gt>/<match>/ + splits.json, prédictions <pred>/<match>/
tennisvision eval --dataset amateur --gt-dir verite/ --predictions sorties/
```

Le rapport (`rapport.json` et `rapport.html`) est écrit dans `reports/eval/<date>_<commit>/` et
une ligne est ajoutée à `reports/eval/historique.jsonl`. Formats, définitions exactes des métriques
et conventions TenniSet : [docs/evaluation.md](docs/evaluation.md). Les scores TenniSet sont
optimistes (cf. [ADR-003](docs/adr/003-jeu-amateur-reference.md)).

## Jeu amateur

Vérité terrain représentative des utilisateurs (matchs filmés au téléphone, annotés dans
[CVAT](https://www.cvat.ai)), qui fait foi pour les statistiques
([ADR-003](docs/adr/003-jeu-amateur-reference.md)). Le jeu lui-même n'est **jamais versionné**
(données personnelles, consentements requis) ; ne le sont que les modèles et les guides :
[tournage](docs/filming_guide.md), [consentement](docs/dataset/consentement.md) et
[registre](docs/dataset/README.md), [annotation](docs/annotation.md),
[fiche du jeu](docs/dataset_card.md).

```bash
# Annotation CVAT (« CVAT for video 1.1 ») -> ball.csv, court.json, events.json du match
tennisvision annotation import-cvat export.xml --match verite/m01 --annotateur A1

# Dossier de vérité terrain : avancement vs critères du Lot 2, découpage par match (test gelé),
# accord entre deux annotateurs, statistiques de la fiche du jeu
tennisvision dataset check verite/
tennisvision dataset split verite/
tennisvision dataset accord verite/ verite_b/
tennisvision dataset card verite/

# TenniSet : étiquettes faibles tirées des textes, échantillon de 50 frappes pour contrôler `Side`
tennisvision tenniset weak-labels --data-dir data --sortie etiquettes_faibles.csv
tennisvision tenniset side-sample --n 50 --sortie side_sample.csv
```

## Structure

```
src/tennisvision/
├── cli.py            CLI (`tennisvision ...`, `python -m tennisvision ...`)
├── config.py         constantes et résolution des chemins (option -> variable -> défaut)
├── models.py         registre de modèles (models.toml), cache, téléchargement vérifié
├── pipeline.py       enchaînement des étapes
├── validate.py       validation des fichiers contre les contrats de données
├── schemas/          schémas JSON des contrats (cf. docs/contracts.md)
├── eval/             banc d'évaluation : appariement, métriques, indicateurs, rapport
├── io/video.py       ingestion : horodatages, lecture, découpe, écriture H.264
├── tracking/         détection YOLO + ByteTrack, dessin, export
└── datasets/         chargeurs des annotations TenniSet (points, événements)
notebooks/            explorations (`archive/` : notebook du palier 1)
tests/                tests unitaires (vidéos synthétiques, faux tracker, pas de modèle)
models/               notes sur les poids (non versionnés)
reports/              rapports d'évaluation (non versionnés)
```

## Tests

```bash
pytest -q
ruff check src tests
```

## Limites connues (étape 0)

- La balle vient d'un réseau dédié (WASB-SBDT, poids publics entraînés sur du tennis de
  diffusion), pas de YOLO : voir `docs/balle.md` pour la chaîne de traitement, les licences
  (celle des données d'entraînement des poids n'est pas vérifiée) et les mesures.
- `n_identites` compte toutes les personnes suivies (arbitres, ramasseurs, public), pas seulement
  les 2 joueurs : le filtrage par position sur le court viendra avec la calibration (étape 2).
- Les vidéos TenniSet sont des images de diffusion TV (changements de plan, ralentis) : le
  tracking est réinitialisé de fait à chaque changement de plan.
