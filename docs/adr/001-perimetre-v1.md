# ADR-001 — Périmètre de la v1

- **Statut** : accepté
- **Date** : 2026-11-10

## Contexte

Le projet part d'une vidéo de match et doit produire des statistiques fiables pour un joueur
amateur. Le périmètre initial reste implicite : le README dit « pas de mobile », la ROADMAP
suppose des vidéos de diffusion TV, et rien n'exclut explicitement les doubles, les caméras
latérales ou le temps réel. Sans périmètre écrit, chaque lot risque de viser un cas différent.

La cible retenue (PLAN_AMELIORATION.md § 1) : un joueur pose son téléphone derrière le fond de
court et filme un match en simple.

## Décision

La v1 couvre uniquement :

- **simple** (2 joueurs) ;
- **caméra fixe** placée derrière le fond de court, filmée depuis un téléphone ;
- **traitement hors-ligne** : la vidéo est traitée après le match, sur PC ou serveur (ADR-002).

Hors périmètre v1, dit explicitement :

- **doubles** (plus de 2 joueurs, règles de service et de rebond différentes) ;
- **caméra latérale** et multi-caméras ;
- **temps réel** (analyse pendant le match, appel de ligne en direct) ;
- **vitesse de balle** (reportée en P3, voir ADR-006).

## Conséquences

Positives :

- un périmètre étroit permet de valider chaque statistique contre une vérité terrain ;
- la géométrie est fixe (une homographie par vidéo), ce qui simplifie la calibration ;
- le traitement hors-ligne autorise le lissage avec le futur et plusieurs passes (ADR-002).

Négatives :

- les doubles, très pratiqués en club, sont exclus de la v1 ;
- le guide de tournage doit être strict : une vidéo mal cadrée est refusée ou mal mesurée ;
- pas de retour immédiat pendant le match, ce qui est une attente naturelle des utilisateurs
  d'applications comme SwingVision.

## Références

- PLAN_AMELIORATION.md § 1 (cible produit) ;
- PLAN_AMELIORATION.md § 3.5 point 8 (doubles hors périmètre v1) ;
- PLAN_AMELIORATION.md § 3.2 (traitement hors-ligne) ;
- PLAN_AMELIORATION.md § 6, Lot 2 tâche 1 (protocole de tournage) et Lot 12 (bonus).
