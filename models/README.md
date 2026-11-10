# Modèles

Les poids (`*.pt`, `*.onnx`, etc.) ne sont pas versionnés dans le dépôt.

## Cache utilisateur

Les modèles sont décrits dans le registre `src/tennisvision/models.toml` (nom, version, URL,
somme SHA-256, licence) et stockés dans le cache utilisateur (via `platformdirs`). Pour changer
d'emplacement, définir `TENNISVISION_MODELS_DIR`.

Aucun téléchargement n'a lieu implicitement : un modèle absent du cache provoque une erreur
qui indique la commande à lancer.

## Récupérer un modèle

```
tennisvision models pull yolo11n
```

Télécharge le fichier et vérifie sa somme SHA-256 avant de le placer dans le cache.

Pour un fichier déjà téléchargé (par exemple `models/yolo11n.pt`) ou pour travailler hors-ligne :

```
tennisvision models import yolo11n models/yolo11n.pt
```

La somme est vérifiée avant la copie dans le cache.

## Hors-ligne

Une fois le modèle dans le cache, plus aucun accès réseau n'est nécessaire.

## Licence

Les poids YOLO d'Ultralytics sont sous licence AGPL-3.0 : voir `docs/adr/005-licences.md`.
