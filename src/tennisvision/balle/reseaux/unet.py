"""TrackNetV2 (U-Net) : 3 images RVB empilées (9 canaux) -> 3 cartes de chaleur.

Portage de `src/models/unet2d.py` et `unet2d_parts.py` du dépôt WASB-SBDT
(https://github.com/nttcom/WASB-SBDT), lui-même dérivé de milesial/Pytorch-UNet.
Copyright (c) 2023 NTT Communications Corporation, licence MIT.
Le framework de configuration est remplacé par des constantes ; seules les variantes utilisées
(`bilinear=True`, `halve_channel=False`, `mode="nearest"`, `bn_first=False`) sont conservées.
Les noms de couches sont identiques afin de charger les poids du dépôt.
"""

import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv -> ReLU -> BN) x 2."""

    def __init__(self, entree: int, sortie: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(entree, sortie, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(sortie),
            nn.Conv2d(sortie, sortie, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(sortie),
        )

    def forward(self, x):
        return self.double_conv(x)


class TripleConv(nn.Module):
    """(Conv -> ReLU -> BN) x 3."""

    def __init__(self, entree: int, sortie: int):
        super().__init__()
        self.triple_conv = nn.Sequential(
            nn.Conv2d(entree, sortie, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(sortie),
            nn.Conv2d(sortie, sortie, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(sortie),
            nn.Conv2d(sortie, sortie, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(sortie),
        )

    def forward(self, x):
        return self.triple_conv(x)


class Down(nn.Module):
    """Sous-échantillonnage (max-pooling) puis double ou triple convolution."""

    def __init__(self, n: int, entree: int, sortie: int):
        super().__init__()
        conv = DoubleConv(entree, sortie) if n == 2 else TripleConv(entree, sortie)
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), conv)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Sur-échantillonnage (plus proche voisin), concaténation avec la skip, puis convolutions."""

    def __init__(self, n: int, entree1: int, entree2: int, sortie: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        conv = DoubleConv if n == 2 else TripleConv
        self.conv = conv(entree1 + entree2, sortie)

    def forward(self, x1, x2):
        return self.conv(torch.cat([x2, self.up(x1)], dim=1))


class OutConv(nn.Module):
    def __init__(self, entree: int, sortie: int):
        super().__init__()
        self.conv = nn.Conv2d(entree, sortie, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TrackNetV2(nn.Module):
    """`forward((B, 9, H, W)) -> logits (B, 3, H, W)`."""

    def __init__(self, n_canaux: int = 9, n_classes: int = 3):
        super().__init__()
        self.inc = DoubleConv(n_canaux, 64)
        self.down1 = Down(2, 64, 128)
        self.down2 = Down(3, 128, 256)
        self.down3 = Down(3, 256, 512)
        self.up1 = Up(3, 512, 256, 256)
        self.up2 = Up(2, 256, 128, 128)
        self.up3 = Up(2, 128, 64, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)
