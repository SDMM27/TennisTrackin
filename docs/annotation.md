# Guide de l'annotateur

> Lot 2 de [PLAN_AMELIORATION.md](PLAN_AMELIORATION.md). Comment annoter un match du jeu amateur
> avec **CVAT**, l'importer dans le format du projet ([contracts.md](contracts.md)) et mesurer
> l'accord entre annotateurs. Les annotations deviennent la **vérité terrain** de toutes les
> statistiques : la rigueur compte plus que la vitesse.

## Vue d'ensemble

```
vidéo ──► tennisvision analyze (frames.parquet) ──► CVAT : annotation ──► export « CVAT for video 1.1 »
                                                                              │
<gt_dir>/<match>/  ◄── tennisvision annotation import-cvat ◄──────────────────┘
        │
        └─► tennisvision dataset check / split / accord / card
```

Pourquoi CVAT : pistes de **points** avec interpolation entre images clés (la balle), **étiquettes
de frame** avec attributs (les événements), points pour les 14 repères du court. Le projet ne lit
que le fichier d'export XML : aucune dépendance à CVAT dans le code. Label Studio n'est pas pris
en charge.

## 1. Installer et préparer CVAT

1. Utilisez CVAT hébergé ([app.cvat.ai](https://app.cvat.ai)) ou une instance installée par
   vos soins (voir la documentation de CVAT). Pour des vidéos sous consentement, préférez une
   instance dont vous maîtrisez l'hébergement (cf. [dataset/README.md](dataset/README.md)).
2. Créez un **projet** puis, dans l'éditeur de labels, ouvrez le mode **Raw** et collez le contenu
   de `src/tennisvision/annotation/cvat_labels.json`. Ne modifiez ni les noms de labels, ni les
   attributs, ni leurs valeurs : l'import s'appuie dessus.
3. Créez une **tâche** par match avec la vidéo (ou sa version H.264 produite par
   `tennisvision analyze` ou `tennisvision clip` si le fichier d'origine passe mal). Ne
   rééchantillonnez pas la vidéo : les numéros de frame de CVAT doivent être ceux de
   `frames.parquet`. Choisissez **toutes les images** (pas de saut d'images).
4. Avant d'annoter, produisez l'index des frames du match, que l'import utilise pour les temps :

```bash
tennisvision analyze match.mp4 --no-video --output-dir <gt_dir>/<match>/
```

   Vérifiez que `<gt_dir>/<match>/frames.parquet` existe. L'import refuse de s'exécuter sans lui.
5. Rattachez les annotations à un **pseudonyme** de joueur (`P1` = premier serveur, `P2` = l'autre,
   cf. [contracts.md](contracts.md)), jamais à un nom.

## 2. Les labels

| Label | Type CVAT | Attributs | Quand |
|---|---|---|---|
| `balle` | points, **piste** (une par vidéo) | `visible` (case, défaut vrai) | niveau 3 |
| `court` | points, **un par repère** | `kp` (« 0 » à « 13 ») | tous niveaux, 1 jeu par plan de caméra |
| `frappe` | **étiquette** (tag) de frame | `joueur` (P1/P2), `coup`, `issue` | niveau 2 |
| `rebond` | points (position dans l'image) | `in_out` (in / out / too_close) | niveau 2 |
| `filet` | étiquette | aucun | niveau 2 |
| `service` | étiquette, à l'impact | `numero` (1/2), `resultat` (in / fault / let), `faute` (net / long / wide / foot / unknown), `serveur` (P1/P2) | niveau 1 |
| `debut_point` | étiquette | aucun | niveau 1 |
| `fin_point` | étiquette | `gagnant` (P1/P2), `serveur` (P1/P2), `raison` (texte, facultatif) | niveau 1 |
| `score` | étiquette, en fin de jeu | `jeux` (texte « P1-P2 »), `sets` (texte, ex. « 6-4 », séparés par des espaces) | niveau 1 |

Valeurs de `coup` : `serve`, `forehand`, `backhand`, `forehand_volley`, `backhand_volley`,
`smash`, `uncertain`. Valeurs de `issue` : `in_play`, `winner`, `unforced_error`, `forced_error`,
`unknown`.

## 3. Définitions précises

- **Instant de frappe** : la **première frame où la raquette touche la balle**. En cas de doute
  entre deux frames, prendre celle où la balle commence à changer de direction. Un service est une
  frappe (`service` à l'impact, plus un `frappe` avec `coup = serve` au même instant si vous êtes
  au niveau 2).
- **Rebond** : la **première frame où la balle touche le sol**. Le point `rebond` se place au
  centre de la balle (ou de son ombre) à ce moment ; `in_out` vaut `in` si la balle touche la
  ligne ou l'intérieur (une balle sur la ligne est bonne), `out` si elle tombe entièrement dehors,
  `too_close` si l'on ne peut pas trancher.
- **Joueur** : celui qui frappe, `P1` ou `P2` (cf. [contracts.md](contracts.md)).
- **Coup** : d'après la main et le côté du corps, **pas** d'après le côté de l'image. En cas de
  doute (balle cachée, joueur tourné), `uncertain` plutôt qu'un choix au hasard.
- **Issue** (`issue`) : `winner` (point gagné par cette frappe sans toucher adverse),
  `unforced_error`, `forced_error`, `in_play` (l'échange continue), `unknown`.
- **Début de point** (`debut_point`) : la frame de l'impact du service. **Fin de point**
  (`fin_point`) : la frame où le point est décidé (deuxième rebond, faute, balle hors du court…).
- **Service** : `numero` 1 ou 2 ; `resultat` `in` (bon), `fault` (faute, avec `faute` : filet,
  long, large, pied) ou `let` (le service est rejoué, même numéro).
- **Balle visible** : visible si on distingue la balle. Hors de l'image, cachée ou impossible à
  localiser : `visible` décoché (ou « outside » ou « occluded » dans CVAT : l'import les traite
  tous comme non visible).

## 4. Les trois niveaux d'annotation

Chaque match est annoté au niveau le plus riche prévu pour lui. Le registre
([dataset/README.md](dataset/README.md)) note le niveau atteint dans `niveau_annotation`.

### Niveau 1 : tous les matchs

Points, services, score. Étiquettes : `debut_point`, `fin_point`, `service`, `score`.

1. Au premier service, créez un `debut_point`, puis le `service` au même instant.
2. À chaque faute, un `service` avec `resultat = fault` (même sans `debut_point` supplémentaire),
   puis le second service.
3. À la fin du point, un `fin_point` avec `gagnant` et `serveur`.
4. À la fin de chaque jeu, un `score` : `jeux` = jeux du jeu en cours (« 3-2 »), `sets` = sets
   terminés (« 6-4 »).

### Niveau 2 : au moins 5 matchs

Ajoute, à chaque frappe, un tag `frappe` (instant, `joueur`, `coup`, `issue`) et, à chaque rebond,
un `rebond` (point + `in_out`), plus `filet` quand la balle touche le filet. Ne laissez pas
de trou : un échange compte toutes ses frappes.

### Niveau 3 : au moins 3 000 frames sur au moins 5 matchs

Ajoute la position de la balle. Une seule **piste** `balle` par vidéo, en points :

- placez une **image clé toutes les 5 images** (N = 5) **et à chaque changement de direction**
  (frappe, rebond, filet) ;
- vérifiez que l'**interpolation** de CVAT suit bien la balle entre deux images clés (parcourez
  la piste à la vitesse normale puis image par image) ; ajoutez une image clé là où elle dérive ;
- quand la balle est cachée ou sort du champ, marquez-la non visible (`visible` décoché ou
  « outside ») ; ne la laissez pas interpolée à travers un trou ;
- placez le point au **centre** de la balle (ou de la traînée si elle est floue).

À l'import, les images clés deviennent `source = annotation`, les images interpolées
`source = interpolated`.

### Le court : à tous les niveaux

Placez les repères `court` : un point par repère, attribut `kp` de « 0 » à « 13 ». **Le premier
jeu vaut dès le début de la vidéo** (la caméra est supposée fixe jusqu'au premier déplacement) :
il suffit de l'annoter sur n'importe quelle frame stable. N'en placez un nouveau que **lorsque le
téléphone a bougé**, à la première frame stable suivante (notez l'heure d'après le journal de
tournage). Les rebonds ont besoin d'un court : sans repères `court` (4 au minimum), l'import des
rebonds échoue.

| `kp` | Repère |
|---|---|
| 0 / 1 | fond loin × couloir gauche / droit |
| 2 / 3 | fond proche × couloir gauche / droit |
| 4 / 6 | fond loin × ligne de simple gauche / droite |
| 5 / 7 | fond proche × ligne de simple gauche / droite |
| 8 / 9 | service loin × simple gauche / droite |
| 10 / 11 | service proche × simple gauche / droite |
| 12 | service loin × ligne médiane |
| 13 | service proche × ligne médiane |

La table complète (avec les coordonnées en mètres) est dans [contracts.md](contracts.md#les-14-points-clés-courtjson--keypoints_px).
Un repère hors de l'image ou masqué : ne le placez pas. Il faut au moins 4 repères par jeu pour que
l'homographie puisse être calculée.

> **À insérer** : un schéma numéroté du court avec les 14 repères (image à créer, non fournie
> pour l'instant). La table ci-dessus et celle de `contracts.md` font foi en attendant.

## 5. Exporter et importer

1. Dans CVAT : *Tasks* → menu de la tâche → **Export annotations** → format
   **« CVAT for video 1.1 »**, sans les images. Enregistrez le XML.
2. Importez dans le dossier du match :

```bash
tennisvision annotation import-cvat export.xml --match <gt_dir>/<match> --annotateur A1
```

   Options : `--frames <frames.parquet>` si l'index n'est pas dans `<match_dir>/frames.parquet`.
   La commande affiche le **niveau détecté** (1, 2 ou 3), écrit `events.json`, `ball.csv` (niveau 3),
   `court.json` et un `import_report.json`, après validation contre les contrats. Une incohérence
   (fin de point sans début, service sans frappe…) donne un **avertissement** et figure dans
   `import_report.json` ; relisez-le et corrigez dans CVAT, puis réimportez.
3. Contrôlez le dossier :

```bash
tennisvision validate <gt_dir>/<match>
tennisvision dataset check <gt_dir>
```

## 6. Double annotation (10 %)

Environ **10 % des données** sont annotées par une **seconde personne**, sans qu'elle voie la
première annotation. On en tire l'accord entre annotateurs, qui est le **plafond atteignable** par
un modèle.

1. Choisissez environ un match sur dix (au moins un), idéalement de niveau 2 ou 3.
2. La seconde personne annote le même match dans une tâche CVAT à part, puis exporte son XML.
3. Importez-le dans un second dossier, avec les mêmes noms de matchs :

```bash
tennisvision annotation import-cvat export_b.xml --match <gt_dir_b>/<match> --annotateur A2
tennisvision dataset accord <gt_dir> <gt_dir_b> --json accord.json
```

4. Marquez `double_annotation = oui` dans le registre.

Cibles du Lot 2 : instant de frappe à **±2 frames** sur **≥ 95 %** des frappes, type de coup
**κ de Cohen ≥ 0,9**, in/out **≥ 95 %**. Si une cible n'est pas atteinte, les deux annotateurs
comparent leurs désaccords, précisent les définitions ci-dessus et recommencent ; ne forcez pas
l'accord.

## 7. Étapes manuelles que le code ne fait pas

Le code prépare, calcule et vérifie ; il ne remplace pas ces tâches humaines :

- [ ] **Filmer ≥ 10 matchs** selon le [guide de tournage](filming_guide.md) : ≥ 6 joueurs
  distincts, 3 surfaces (terre, dur, intérieur), 30 et 60 images/s, ≥ 2 modèles de téléphone, du
  débutant au classé.
- [ ] **Recueillir les consentements** écrits de chaque personne filmée
  ([consentement.md](dataset/consentement.md)) et les archiver hors du dépôt.
- [ ] **Annoter par niveau** : niveau 1 sur tous les matchs, niveau 2 sur ≥ 5, niveau 3 sur
  ≥ 3 000 frames réparties sur ≥ 5 matchs.
- [ ] **Double-annoter 10 %** des données par une seconde personne.
- [ ] **Vérifier la convention `Side` de TenniSet** sur 50 frappes (section suivante).
- [ ] **Récupérer la vidéo V009 de TenniSet** : son annotation `data/annotations/V009.json`
  existe, mais `data/videos/V009.mp4` manque ; elle porte 49 % des points TenniSet.

## 8. TenniSet : étiquettes faibles et convention `Side`

Deux commandes préparent le travail sur TenniSet (cf. [evaluation.md](evaluation.md#conventions-tenniset)) :

```bash
# Étiquettes faibles extraites des textes `desc` des points (issue, coup, direction, lob…)
tennisvision tenniset weak-labels --data-dir data --sortie etiquettes_faibles.csv

# Échantillon de 50 frappes à contrôler à l'œil
tennisvision tenniset side-sample --n 50 --seed 0 --sortie side_sample.csv --images side_images/
```

Pour `side-sample` : ouvrez `side_sample.csv` et regardez la frame du milieu de chaque frappe
(dossier `--images`, les vidéos absentes comme V009 sont ignorées avec un avertissement). Remplissez
à la main `verif_coup` (coup droit ou revers réellement joué) et `verif_cote_image` (côté de l'image
où se trouve la raquette). Si `Side` suit le côté de l'image plutôt que le coup, la conversion
TenniSet → `events.json` donne de mauvais `forehand`/`backhand` : **ne pas utiliser ces coups**
avant d'avoir tranché et documenté le résultat (d'après les effectifs observés sur V006, c'est
l'hypothèse la plus probable ; elle reste à confirmer).
