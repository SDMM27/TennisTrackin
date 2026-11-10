# ADR-003 — Jeu amateur comme référence de validation

- **Statut** : accepté
- **Date** : 2026-11-10

## Contexte

Le projet se construit sur TenniSet, cinq matchs professionnels féminins en diffusion TV. La
cible est un téléphone posé par un amateur : caméra plus basse, souvent décentrée, grillage au
premier plan, courts voisins, éclairages variés, 30 ou 60 fps en cadence variable.

Deux défauts méthodologiques de TenniSet sont à prendre en compte :

- les découpages `SPLITS` sont des **intervalles d'une même vidéo** (Train et Test partagent les
  joueuses, le court, la caméra et le jour) : les scores sont optimistes ;
- TenniSet ne couvre qu'un petit nombre de joueurs et de conditions.

Par ailleurs, la ROADMAP repoussait les vidéos réelles à l'étape 6, alors que le changement de
domaine est le risque principal du projet (PLAN_AMELIORATION.md § 3.1).

## Décision

- Le **jeu amateur** est la **référence de validation** : c'est lui qui fait foi pour toutes les
  statistiques affichées. Il vise au minimum 10 matchs, 6 joueurs différents, 3 surfaces et
  2 cadences (30 et 60 fps), tournés selon un protocole écrit (Lot 2).
- **TenniSet** devient un jeu **secondaire** : pré-entraînement des modèles d'événements, tests
  de la logique de score (Lot 9), étiquettes faibles pour les coups (Lot 7).
- **Évaluation par match tenu à l'écart** (leave-one-match-out) : un match entier est retiré
  de l'entraînement et sert au test. Le découpage ne se fait **jamais à l'intérieur d'un match**.
- Un jeu de test amateur est **gelé** : il n'est pas utilisé pendant le développement.

## Conséquences

Positives :

- les chiffres publiés reflètent les conditions réelles d'un utilisateur ;
- la fuite entre entraînement et test à l'intérieur d'un match est exclue par construction ;
- le changement de domaine est mesuré dès le départ, et non en fin de projet.

Négatives :

- la collecte et l'annotation sont longues : c'est le chemin critique du projet (PLAN § 7.2),
  et elles dépendent de personnes filmées qui doivent consentir ;
- avec 10 matchs, les intervalles de confiance restent larges, en particulier pour les classes
  rares (volées, smashs) ;
- les résultats sur TenniSet seront plus élevés que sur le jeu amateur : il faut le dire
  dans les rapports.

## Références

- PLAN_AMELIORATION.md § 3.1 (constat, défaut méthodologique, décision proposée) ;
- PLAN_AMELIORATION.md § 4 (hypothèses « usage personnel » à lever) ;
- PLAN_AMELIORATION.md § 6, Lot 2 (collecte, découpage par match, jeu de test gelé) ;
- PLAN_AMELIORATION.md § 6, Lot 3 tâche 4 (découpage leave-one-match-out) ;
- PLAN_AMELIORATION.md § 6, Lot 7 tâche 3 (pré-entraînement TenniSet, affinage amateur).
