# Fiche descriptive du jeu de données amateur

> Lot 2 de [PLAN_AMELIORATION.md](PLAN_AMELIORATION.md). Le jeu amateur est la vérité terrain de
> référence du projet ([ADR-003](adr/003-jeu-amateur-reference.md)). Cette fiche décrit son
> contenu, ses conditions de collecte et ses limites. Les parties entre crochets `[à compléter]`
> sont à remplir par l'équipe au fil de la collecte ; le bloc de statistiques est **généré**.
>
> **Le jeu lui-même n'est pas dans le dépôt** (vidéos et annotations : données personnelles
> soumises à consentement). Ne mettre ici que des statistiques, jamais de noms.

**État** : jeu en cours de constitution. [à compléter : version, date]

## Contenu

Des matchs de tennis amateur filmés au téléphone fixe, derrière le fond de court (cf.
[filming_guide.md](filming_guide.md)), avec leur vérité terrain annotée à la main.

Par match (`<gt_dir>/<match>/`) : `match.json`, `frames.parquet`, `events.json` (points, services,
score, et selon le niveau frappes et rebonds), `ball.csv` (niveau 3), `court.json` (14 repères et
homographie). Formats : [contracts.md](contracts.md). Les métadonnées qui ne sont pas dans
`match.json` (téléphone, niveau, consentements, licence) sont dans `registre.csv`
([dataset/README.md](dataset/README.md)).

### Statistiques (générées)

Ce bloc est **réécrit** par `tennisvision dataset card <gt_dir>`. Ne pas le modifier à la main.

<!-- debut-stats-auto -->
_Statistiques non générées. Lancer `tennisvision dataset card <gt_dir>`._
<!-- fin-stats-auto -->

## Objectifs de la version 1

| Critère | Cible |
|---|---|
| Matchs | ≥ 10 |
| Joueurs distincts | ≥ 6 |
| Surfaces | ≥ 3 parmi terre, dur, intérieur |
| Cadences | 30 et 60 images/s |
| Modèles de téléphone | ≥ 2 |
| Niveau d'annotation 2 | ≥ 5 matchs |
| Niveau d'annotation 3 | ≥ 3 000 frames sur ≥ 5 matchs |
| Consentements | tous archivés |
| Double annotation | ≈ 10 % |

L'état réel contre ces cibles s'obtient avec `tennisvision dataset check <gt_dir>`.

## Conditions de collecte

- **Protocole** : [filming_guide.md](filming_guide.md) (paysage, derrière le fond de court, centré,
  ≥ 2 m, court entier, 1080p, 60 images/s si possible, stabilisation et économie d'énergie
  désactivées).
- **Lieux, dates, surfaces** : [à compléter : résumé, sans lieu identifiant les personnes].
- **Niveaux de jeu** : du débutant au classé, noté par match dans `niveau_jeu`.
- **Écarts au protocole** : [à compléter : téléphones déplacés, cadrages dégradés, etc.].

## Processus d'annotation

- Outil : CVAT, labels de `src/tennisvision/annotation/cvat_labels.json` ;
  guide : [annotation.md](annotation.md).
- Trois niveaux : 1 (points, services, score), 2 (+ frappes, rebonds), 3 (+ balle image par
  image, une image clé sur 5 avec interpolation).
- Conversion : `tennisvision annotation import-cvat`, avec validation contre les contrats.
- Annotateurs : [à compléter : nombre, formation], désignés par des identifiants dans le registre.
- **Double annotation** de ≈ 10 % des données par une seconde personne.

### Accord entre annotateurs

Mesuré avec `tennisvision dataset accord`. Cibles : instant de frappe à ±2 frames ≥ 95 %,
κ du type de coup ≥ 0,9, in/out ≥ 95 %.

| Mesure | Cible | Résultat |
|---|---|---|
| Frappes appariées à ±2 frames | ≥ 95 % | [à compléter] |
| Type de coup, κ de Cohen | ≥ 0,9 | [à compléter] |
| In/out des rebonds | ≥ 95 % | [à compléter] |
| Vainqueur du point | — | [à compléter] |
| Résultat du service | — | [à compléter] |

## Découpage et jeu de test gelé

- Découpage **par match**, jamais à l'intérieur d'un match ([ADR-003](adr/003-jeu-amateur-reference.md)) ;
  `tennisvision dataset split <gt_dir>` écrit `splits.json`.
- Le **jeu de test est gelé** : une fois non vide, il n'est plus modifié. Les nouveaux matchs vont
  en entraînement ou validation. On n'y touche pas pendant le développement ; le refaire demande
  l'option explicite `--refaire-test --je-sais-ce-que-je-fais`, et doit être consigné ici.
- Composition du test : [à compléter : nombre de matchs, surfaces représentées].
- Historique des changements du test : [à compléter].

## Biais connus

[À compléter à mesure que le jeu grandit. Pistes à examiner :]

- surfaces et lieux sur-représentés ; peu de matchs sur gazon ;
- niveaux de jeu : le niveau « classé » peut être rare ;
- nombre réduit de joueurs, donc de styles de frappe ; gauchers ;
- peu de modèles de téléphone et de conditions de lumière (soleil, nuit, intérieur) ;
- matchs en simple uniquement ;
- les annotateurs sont peu nombreux : biais d'annotation propres à eux (définition de l'instant
  de frappe, du rebond) ;
- les joueurs consentants ne sont pas représentatifs de tous les utilisateurs.

## Licence et consentements

- Chaque personne filmée signe le [formulaire de consentement](dataset/consentement.md)
  (droit à l'image et RGPD), avec trois choix séparés : usage interne, publication d'extraits,
  publication du jeu. Le choix figure dans la colonne `licence` du registre.
- Originaux signés : archivés **hors du dépôt**, jamais commités.
- **Redistribution** : sans la case « publication du jeu » cochée par toutes les personnes d'un
  match, la vidéo et ses annotations ne sont pas publiées. La licence retenue pour une éventuelle
  publication est [à décider ; voir [ADR-005](adr/005-licences.md)].
- Retrait de consentement : le match est retiré du jeu et des évaluations suivantes.
- Les joueurs sont désignés par des pseudonymes (`J01`…) ; aucun nom réel dans le jeu.

## Étapes manuelles restantes

Ce que le code ne peut pas faire et qui conditionne le Lot 2 :

- [ ] Filmer **≥ 10 matchs** : ≥ 6 joueurs, 3 surfaces (terre, dur, intérieur), 30 et 60 images/s,
  ≥ 2 modèles de téléphone, du débutant au classé.
- [ ] Recueillir et archiver **tous les consentements**.
- [ ] Annoter par niveau (1 partout, 2 sur ≥ 5 matchs, 3 sur ≥ 3 000 frames sur ≥ 5 matchs).
- [ ] **Double-annoter 10 %** et publier l'accord inter-annotateurs ci-dessus.
- [ ] Vérifier la convention **`Side`** de TenniSet sur **50 frappes** (`tennisvision tenniset
  side-sample`, puis contrôle à la main).
- [ ] Récupérer la vidéo **V009** de TenniSet (`data/annotations/V009.json` existe,
  `data/videos/V009.mp4` manque ; 49 % des points TenniSet).
- [ ] Compléter les sections `[à compléter]` de cette fiche.
