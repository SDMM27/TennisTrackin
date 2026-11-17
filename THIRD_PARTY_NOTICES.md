# Notices des composants tiers

## WASB-SBDT (code des réseaux de balle)

Les fichiers `src/tennisvision/balle/reseaux/hrnet.py`, `src/tennisvision/balle/reseaux/unet.py`
et les parties de `src/tennisvision/balle/detecteur.py` (transformation affine, extraction des
taches) sont portés depuis https://github.com/nttcom/WASB-SBDT, distribué sous la licence
suivante :

```
MIT License

Copyright (c) 2023 NTT Communications Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

À vérifier (ADR-005) :

- `hrnet.py` dérive aussi de HRNet (Microsoft, licence MIT).
- `unet.py` dérive, via WASB-SBDT, de milesial/Pytorch-UNet, dont le dépôt est aujourd'hui sous
  **GPL-3.0**. La licence applicable à cette partie (version d'origine reprise par WASB-SBDT)
  n'est pas vérifiée. Tant que le projet reste sous l'issue AGPL ou en usage interne (ADR-005),
  la contrainte est compatible ; elle est à trancher avant toute autre redistribution.
- Les poids `wasb_tennis` et `tracknetv2_tennis` sont entraînés sur le jeu TrackNet tennis, dont
  la licence n'est pas vérifiée.
