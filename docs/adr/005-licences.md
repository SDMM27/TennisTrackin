# ADR-005 — Politique de licences

- **Statut** : accepté
- **Date** : 2026-11-10

## Contexte

Le projet dépend de plusieurs composants dont les licences conditionnent ce qu'on peut publier,
faire tourner en service ouvert ou redistribuer :

- **ultralytics** (YOLO, suivi ByteTrack) est sous **AGPL-3.0** ;
- le jeu de données **TenniSet** et les **vidéos de diffusion** qu'il référence relèvent du droit
  des diffuseurs ;
- les jeux **TrackNet** et les **poids pré-entraînés** (TrackNetV3, WASB, YOLO) ont leurs propres
  conditions d'utilisation, qui n'ont pas encore été vérifiées à la source.

Un service ouvert à d'autres utilisateurs n'est pas dans la même situation qu'un usage personnel
(PLAN_AMELIORATION.md § 3.5, point 6).

## Décision

1. **AGPL d'ultralytics.** Pour tout service ouvert, une seule des trois issues est retenue
   au moment de l'ouverture publique :
   - publier le code du service sous AGPL-3.0 ;
   - acheter une licence commerciale auprès d'Ultralytics ;
   - remplacer ultralytics par une implémentation sous licence permissive.
   D'ici là, l'usage reste personnel ou interne.
2. **TenniSet, vidéos de diffusion, TrackNet, poids pré-entraînés** : leur licence doit être
   **vérifiée à la source avant toute redistribution** d'un modèle entraîné dessus ou d'un
   extrait de données.
3. **Registre unique** : toutes les licences sont consignées dans `docs/licenses.md`, créé au
   Lot 11 (tâche 6). Chaque vidéo amateure a aussi sa ligne de licence d'utilisation (Lot 2).

## Conséquences

Positives :

- le choix de l'issue pour l'AGPL est explicite et daté, pas découvert au moment de l'ouverture ;
- les contraintes de redistribution sont visibles dès le départ.

Négatives :

- la voie « publier le code » impose un code ouvert ;
- la voie « licence commerciale » a un coût ;
- la voie « remplacement » demande du travail de portage sur le détecteur et le suivi ;
- aucune de ces trois issues n'est encore choisie.

Ce qui **n'est pas encore vérifié** (PLAN_AMELIORATION.md § 8) :

- la licence exacte de TenniSet et ses conditions de redistribution ;
- la licence du jeu TrackNet et des poids TrackNetV3 et WASB ;
- la licence des poids YOLO téléchargés par ultralytics ;
- le statut juridique des vidéos de diffusion contenues dans TenniSet.

## Références

- PLAN_AMELIORATION.md § 3.5 point 6 (constat sur les licences) ;
- PLAN_AMELIORATION.md § 6, Lot 2 tâche 2 (registre des licences des vidéos) ;
- PLAN_AMELIORATION.md § 6, Lot 11 tâche 6 (remplacement ou isolement de l'AGPL, `docs/licenses.md`) ;
- PLAN_AMELIORATION.md § 8 (licences non vérifiées).
