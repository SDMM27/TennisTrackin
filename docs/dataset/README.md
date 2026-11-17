# Jeu de données amateur : registre et consentements

> Lot 2 de [PLAN_AMELIORATION.md](../PLAN_AMELIORATION.md). Le jeu amateur est la vérité terrain
> de référence du projet ([ADR-003](../adr/003-jeu-amateur-reference.md)). Ce dossier contient les
> **modèles** (registre, formulaire de consentement) ; le jeu lui-même n'est **jamais versionné**.

## Ce qui est versionné, ce qui ne l'est pas

| Élément | Où | Versionné |
|---|---|---|
| Modèle de registre | `docs/dataset/registre_modele.csv` | oui |
| Modèle de consentement | [`consentement.md`](consentement.md) | oui |
| Fiche descriptive du jeu | [`../dataset_card.md`](../dataset_card.md) | oui (statistiques sans noms) |
| Vidéos, annotations, `registre.csv` rempli, `splits.json` | `<gt_dir>/` (dossier de travail, par exemple sous `data/`) | **non** |
| **Formulaires de consentement signés (originaux)** | espace de stockage protégé, **hors du dépôt** | **non, jamais** |

Les originaux signés contiennent des noms réels : ils ne doivent figurer ni dans le dépôt, ni dans
le registre, ni dans la fiche. Le registre ne contient que les **identifiants de consentement**
(`C001`…), qui permettent de retrouver le formulaire papier ou scanné. Indiquez dans un document
interne (hors dépôt) **où** est rangée l'archive et **qui** y a accès.

## Organisation d'un dossier de vérité terrain

```
<gt_dir>/
├── registre.csv           une ligne par match (colonnes ci-dessous)
├── splits.json            {"train": [...], "val": [...], "test": [...]}  (commande dataset split)
└── <match>/               un dossier par match, nom identique à la colonne `match`
    ├── match.json         contrat 1.0 (docs/contracts.md)
    ├── frames.parquet     index des frames (produit par l'ingestion)
    ├── events.json        niveaux 1 et 2
    ├── ball.csv           niveau 3
    └── court.json         calibration annotée
```

Pour démarrer : copier `docs/dataset/registre_modele.csv` vers `<gt_dir>/registre.csv` et ajouter
une ligne par match filmé.

## Colonnes du registre

Fichier CSV, encodage UTF-8, séparateur virgule, **une ligne par match**. Les colonnes
s'écrivent exactement ainsi, dans cet ordre. Les listes se séparent par `;`.

| Colonne | Contenu | Valeurs |
|---|---|---|
| `match` | nom du dossier du match | texte sans espace, ex. `m01_terre` |
| `date` | date du match | `AAAA-MM-JJ` |
| `surface` | surface de jeu | `terre`, `dur`, `interieur`, `gazon`, `autre` |
| `lieu_type` | type de lieu | texte court : `club`, `public`, `prive`… |
| `telephone` | modèle de téléphone | texte, ex. `Pixel 8` |
| `fps_nominal` | cadence nominale | nombre, ex. `30` ou `60` |
| `resolution` | résolution | ex. `1920x1080` |
| `joueurs` | joueurs, **pseudonymes** | `J01;J02` (jamais de nom réel) |
| `niveau_jeu` | niveau des joueurs | `debutant`, `intermediaire`, `confirme`, `classe` |
| `niveau_annotation` | niveau d'annotation atteint | `1`, `2` ou `3` (cf. [annotation.md](../annotation.md)) |
| `consentements` | identifiants des formulaires | `C001;C002` |
| `consentements_archives` | formulaires signés bien archivés ? | `oui` ou `non` |
| `licence` | licence d'utilisation de la vidéo | texte court, ex. `usage interne`, `extraits publiables`, `jeu publiable` (selon les cases cochées) |
| `annotateurs` | annotateurs (identifiants, pas de noms) | `A1;A2` |
| `double_annotation` | match annoté deux fois ? | `oui` ou `non` |
| `notes` | remarques libres | texte (téléphone déplacé à telle heure, etc.) |

### Règles

- **Pseudonymes** : `J01`, `J02`… sont attribués une fois par personne et réutilisés d'un match à
  l'autre (cela permet de compter les joueurs distincts). La table de correspondance
  pseudonyme ↔ personne n'existe **pas** dans le jeu ; si elle est nécessaire pour traiter un
  retrait de consentement, elle est gardée avec les formulaires signés, hors du dépôt.
- **Licence par vidéo** : la colonne `licence` reflète les cases cochées sur le formulaire. Si
  plusieurs joueurs d'un même match ont fait des choix différents, on retient le plus restrictif
  (cf. [ADR-005](../adr/005-licences.md)).
- **Consentements archivés** : `non` tant que tous les formulaires du match ne sont pas signés et
  rangés. `tennisvision dataset check` le signale ; le critère de réussite du Lot 2 demande `oui`
  partout.
- **Retrait de consentement** : supprimer la ligne et le dossier du match, relancer
  `tennisvision dataset check` et `tennisvision dataset card`, puis refaire les évaluations
  (voir [`consentement.md`](consentement.md)). Si le match retiré était dans le jeu de test gelé,
  la décision de le remplacer est consignée dans la fiche (`dataset_card.md`).

## Commandes

```bash
tennisvision dataset check <gt_dir>               # rapport d'avancement vs critères du Lot 2
tennisvision dataset split <gt_dir>               # écrit splits.json (test gelé)
tennisvision dataset accord <dir_a> <dir_b>       # accord inter-annotateurs
tennisvision dataset card <gt_dir>                # régénère les statistiques de la fiche
```
