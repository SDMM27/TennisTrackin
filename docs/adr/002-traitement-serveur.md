# ADR-002 — Traitement sur serveur ou PC en v1

- **Statut** : accepté
- **Date** : 2026-11-10

## Contexte

Deux architectures sont possibles : traiter la vidéo directement sur le téléphone (façon
SwingVision, temps réel) ou l'envoyer à un serveur ou un PC pour un traitement par lots. Le
choix conditionne la taille des modèles, la vitesse d'itération, la confidentialité et le coût.

Le projet est aujourd'hui en Python, sans compétence Swift ou Kotlin, et les modèles visés
(TrackNetV3 ou WASB, YOLO-pose) sont trop lourds pour un téléphone sans compromis.

## Décision

- **v1 : traitement hors-ligne sur PC ou serveur** (CLI, puis service web au Lot 11). Le téléphone
  ne sert qu'à filmer, avec un guide de cadrage.
- **Modèles exportables en ONNX** dès leur conception, pour ne pas fermer la porte à un passage
  sur téléphone (P3, Lot 12).
- **Objectif de débit** à respecter :
  - 1 h de match traitée en **≤ 1 h** sur un GPU grand public ;
  - 1 h de match traitée en **≤ 4 h** sur CPU seul (traitement de nuit acceptable).

Comparaison résumée :

| Critère | Téléphone | Serveur / PC (batch) |
|---|---|---|
| Temps réel | oui | non |
| Itération sur les modèles | lente (conversion, quantification) | rapide (Python, PyTorch) |
| Taille des modèles | fortement contrainte | libre |
| Précision atteignable | limitée par le budget de calcul | modèles lourds, lissage hors-ligne |
| Coût pour le projet | nul | GPU à payer ou PC de l'utilisateur |
| Confidentialité | vidéo sur le téléphone | vidéo envoyée : RGPD, consentement |

## Conséquences

Positives :

- itération rapide sur les modèles, en Python, avec le code déjà en place ;
- lissage de trajectoire avec le futur comme le passé, plusieurs passes, correction par
  l'utilisateur (ADR-004) ;
- export ONNX : la voie vers le téléphone reste ouverte.

Négatives :

- la vidéo doit être envoyée ou transférée à un PC : impact sur la confidentialité et le RGPD
  (Lot 11, tâche 5) ;
- un coût d'infrastructure (GPU) ou une dépendance au PC de l'utilisateur ;
- pas de résultat immédiat : le délai dépend du débit visé ci-dessus ;
- le débit cible n'est pas encore mesuré (aucune mesure GPU dans le dépôt, voir PLAN § 8).

## Références

- PLAN_AMELIORATION.md § 3.2 (tableau comparatif complet et recommandation) ;
- PLAN_AMELIORATION.md § 5 (exportabilité ONNX des modèles, Lot 4 tâche 7) ;
- PLAN_AMELIORATION.md § 6, Lot 11 tâche 3 (traitement) et critère de débit ;
- PLAN_AMELIORATION.md § 6, Lot 12 (passage sur téléphone en P3) ;
- PLAN_AMELIORATION.md § 8 (performances GPU non mesurées).
