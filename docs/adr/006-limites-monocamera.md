# ADR-006 — Limites assumées de la monocaméra

- **Statut** : accepté
- **Date** : 2026-11-10

## Contexte

Hawk-Eye triangule la balle en 3D avec une dizaine de caméras rapides calibrées. Un téléphone
seul ne fournit qu'une image 2D de la balle. Les statistiques attendues par un joueur (in/out,
zones de rebond, vitesse) supposent parfois une information que la monocaméra ne donne pas.
Sans écrire ces limites noir sur blanc, le produit promettrait des chiffres qu'il ne peut pas
mesurer.

## Décision

Limites assumées de la v1 :

- **Position 2D seulement** : la balle est localisée dans l'image, pas dans l'espace.
- **Projection au sol valide uniquement au rebond** : c'est le seul instant où la balle est
  connue sur le plan du court. Les positions projetées entre deux rebonds ne sont pas exposées
  comme des positions au sol.
- **In/out indicatif près des lignes**, avec trois issues : `in`, `out`, et
  `trop proche pour décider` lorsque la balle est dans la marge d'incertitude d'une ligne.
  La marge est calculée et affichée (erreur de calibration, erreur de position, flou).
- **Pas de hauteur au-dessus du filet** : la hauteur n'est pas observable en monocaméra. Les
  statistiques qui la supposent sont exclues.
- **Vitesse de balle approchée** : reportée en P3 (Lot 12), publiée avec sa marge d'erreur, et
  seulement si l'erreur mesurée est ≤ 10 %.

## Conséquences

Positives :

- aucune statistique n'est affichée au-delà de ce que la géométrie permet ;
- l'issue « trop proche pour décider » évite les fausses décisions fermes ;
- les limites sont documentées pour l'utilisateur, ce qui protège la crédibilité du produit.

Négatives :

- l'utilisateur ne peut pas obtenir un arbitrage au centimètre près, attendu d'un système
  professionnel ;
- une part des rebonds (ceux près des lignes) donne une réponse « trop proche pour décider » ;
- la vitesse de balle, très motivante, est repoussée en P3 ;
- le message doit être clair dans le rapport pour éviter une fausse impression de précision.

## Références

- PLAN_AMELIORATION.md § 3.5 point 7 (limites de la monocaméra) ;
- PLAN_AMELIORATION.md § 1, tableau des statistiques (vitesse de balle, rang 8) ;
- PLAN_AMELIORATION.md § 6, Lot 8 (rebonds, in/out à trois issues, hors périmètre : hauteur et
  vitesse) ;
- PLAN_AMELIORATION.md § 6, Lot 12 (vitesse approchée en P3).
