# ADR-004 — Modèle de détection de la balle

- **Statut** : accepté
- **Date** : 2026-11-10

## Contexte

Le détecteur actuel est YOLO11n COCO, classe `sports ball`. Le constat est bloquant : sur le
point P00000001 de V006, la vraie balle n'est trouvée que sur moins de 2 % des frames
(PLAN_AMELIORATION.md § 2.3, C1). Trois causes sont identifiées :

- **C1** : la balle fait 3 à 8 px au fond du court, elle est floue, et une seule image ne suffit
  pas à la distinguer d'une tache claire ;
- **C2** : la balle passe dans ByteTrack avec les personnes, dont le filtre de Kalman à vitesse
  constante n'est pas adapté à un objet qui rebondit et change de direction à chaque frappe ;
- **C3** : la balle est choisie par confiance maximale, sans cohérence spatiale ou temporelle,
  ce qui laisse gagner la roue de caméra d'une diffusion TV.

À 100 km/h, la balle parcourt 0,93 m entre deux images à 30 fps, et 0,46 m à 60 fps.

## Décision

- **Famille retenue : carte de chaleur multi-frames.** Le réseau reçoit 3 frames consécutives ou
  plus et produit une carte de chaleur de la position de la balle. Le choix entre **TrackNetV3**
  et **WASB** est fait **par mesure** sur le jeu amateur au Lot 4, et non par principe.
- **Lissage hors-ligne de la trajectoire** : segments paraboliques entre frappes et rebonds,
  plutôt qu'un simple filtre de Kalman. Les trous de 10 frames ou moins sont interpolés et
  marqués `interpolated`.
- **La balle sort de ByteTrack.** Le tracker YOLO ne traite plus que la classe personne.
- **Les joueurs restent sur YOLO personne + suivi**, ce qui fonctionne déjà.
- Le résultat est écrit dans `ball.csv` (contrat du § 5 du plan).

## Conséquences

Positives :

- la famille est conçue pour la balle de tennis et de badminton, et exploite le mouvement ;
- les poids pré-entraînés publics permettent une référence avant tout affinage ;
- le lissage hors-ligne tire parti du futur comme du passé (ADR-002) ;
- le modèle reste exportable en ONNX.

Négatives :

- TrackNet est entraîné sur de la diffusion TV : un affinage sur frames amateurs (Lot 2,
  niveau 3) reste nécessaire ;
- WASB est plus lourd ; le débit doit être vérifié contre l'objectif de l'ADR-002 ;
- la licence des poids et du jeu d'entraînement n'est pas encore vérifiée (ADR-005) ;
- le critère « F1 ≥ 0,85 à 10 px » reste un objectif proposé, à recaler après la première mesure.

## Références

- PLAN_AMELIORATION.md § 3.3 (familles de modèles, recommandation) ;
- PLAN_AMELIORATION.md § 2.3, constats C1 à C3 ;
- PLAN_AMELIORATION.md § 3.4 (la détection de frappe ne dépend pas que de la balle) ;
- PLAN_AMELIORATION.md § 6, Lot 4 (tâches 1 à 7 et critères de réussite) ;
- PLAN_AMELIORATION.md § 5 (contrat `ball.csv`).
