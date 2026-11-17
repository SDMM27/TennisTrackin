"""HRNet « WASB » : 3 images RVB empilées (9 canaux) -> 3 cartes de chaleur (HRNet à résolution
conservée sur la branche principale).

Portage de `src/models/hrnet.py` du dépôt WASB-SBDT (https://github.com/nttcom/WASB-SBDT,
BMVC 2023), lui-même dérivé de HRNet (Copyright (c) Microsoft, licence MIT, Bin Xiao, Bowen Cheng).
Copyright (c) 2023 NTT Communications Corporation, licence MIT.
Le framework de configuration (hydra/omegaconf) est remplacé par la constante `CONFIG_WASB`
(configs/model/wasb.yaml) ; les déconvolutions (NUM_DECONVS = 0) et l'initialisation des poids
sont retirées. Les noms de couches sont identiques afin de charger les poids du dépôt.
"""

from torch import nn

BN_MOMENTUM = 0.1

# Équivalent de configs/model/wasb.yaml : (frames_in = frames_out = 3, échelle de sortie 0).
CONFIG_WASB = {
    "frames_in": 3,
    "frames_out": 3,
    "out_scales": [0],
    "stem_inplanes": 64,
    "stem_strides": [1, 1],
    "final_conv_kernel": 1,
    "stage1": {"block": "BOTTLENECK", "blocks": [1], "channels": [32]},
    "stage2": {"block": "BASIC", "modules": 1, "blocks": [2, 2], "channels": [16, 32]},
    "stage3": {"block": "BASIC", "modules": 1, "blocks": [2, 2, 2], "channels": [16, 32, 64]},
    "stage4": {
        "block": "BASIC",
        "modules": 1,
        "blocks": [2, 2, 2, 2],
        "channels": [16, 32, 64, 128],
    },
}


def conv3x3(entree: int, sortie: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(entree, sortie, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residu = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residu = self.downsample(x)
        return self.relu(out + residu)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residu = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residu = self.downsample(x)
        return self.relu(out + residu)


BLOCS = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class HighResolutionModule(nn.Module):
    """Branches parallèles à résolutions différentes, fusionnées par somme."""

    def __init__(self, num_branches, block, num_blocks, num_inchannels, num_channels):
        super().__init__()
        if not (num_branches == len(num_blocks) == len(num_channels) == len(num_inchannels)):
            raise ValueError("NUM_BRANCHES incohérent avec NUM_BLOCKS / NUM_CHANNELS / entrées")
        self.num_inchannels = num_inchannels
        self.num_branches = num_branches
        self.branches = nn.ModuleList(
            self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)
        )
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _make_one_branch(self, indice, block, num_blocks, num_channels):
        sortie = num_channels[indice] * block.expansion
        downsample = None
        if self.num_inchannels[indice] != sortie:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[indice], sortie, kernel_size=1, bias=False),
                nn.BatchNorm2d(sortie, momentum=BN_MOMENTUM),
            )
        couches = [block(self.num_inchannels[indice], num_channels[indice], 1, downsample)]
        self.num_inchannels[indice] = sortie
        for _ in range(1, num_blocks[indice]):
            couches.append(block(sortie, num_channels[indice]))
        return nn.Sequential(*couches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None
        ch = self.num_inchannels
        fuse_layers = []
        for i in range(self.num_branches):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(ch[j], ch[i], 1, 1, 0, bias=False),
                            nn.BatchNorm2d(ch[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    convs = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            convs.append(
                                nn.Sequential(
                                    nn.Conv2d(ch[j], ch[i], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(ch[i]),
                                )
                            )
                        else:
                            convs.append(
                                nn.Sequential(
                                    nn.Conv2d(ch[j], ch[j], 3, 2, 1, bias=False),
                                    nn.BatchNorm2d(ch[j]),
                                    nn.ReLU(True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*convs))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        x = [branche(xi) for branche, xi in zip(self.branches, x, strict=True)]
        sortie = []
        for i, fuse in enumerate(self.fuse_layers):
            y = x[0] if i == 0 else fuse[0](x[0])
            for j in range(1, self.num_branches):
                y = y + (x[j] if i == j else fuse[j](x[j]))
            sortie.append(self.relu(y))
        return sortie


class HRNet(nn.Module):
    """`forward((B, 9, H, W)) -> logits (B, 3, H, W)` (échelle de sortie 0 uniquement)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self._frames_out = cfg["frames_out"]
        self._out_scales = cfg["out_scales"]
        inplanes = cfg["stem_inplanes"]
        strides = cfg["stem_strides"]

        self.conv1 = nn.Conv2d(
            3 * cfg["frames_in"], inplanes, kernel_size=3, stride=strides[0], padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            inplanes, inplanes, kernel_size=3, stride=strides[1], padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        stage1 = cfg["stage1"]
        bloc = BLOCS[stage1["block"]]
        planes = stage1["channels"][0]
        self.layer1 = self._make_layer(bloc, inplanes, planes, stage1["blocks"][0])
        pre_channels = [bloc.expansion * planes]

        self._nb_branches = {}
        for numero in (2, 3, 4):
            stage_cfg = cfg[f"stage{numero}"]
            bloc = BLOCS[stage_cfg["block"]]
            channels = [c * bloc.expansion for c in stage_cfg["channels"]]
            transition = self._make_transition_layer(pre_channels, channels)
            stage, pre_channels = self._make_stage(stage_cfg, channels)
            self._nb_branches[numero] = len(stage_cfg["blocks"])
            setattr(self, f"transition{numero - 1}", transition)
            setattr(self, f"stage{numero}", stage)

        self.final_layers = nn.ModuleList(
            nn.Conv2d(pre_channels[scale], self._frames_out, kernel_size=cfg["final_conv_kernel"])
            for scale in self._out_scales
        )

    @staticmethod
    def _make_transition_layer(pre, cur):
        transitions = []
        for i in range(len(cur)):
            if i < len(pre):
                if cur[i] != pre[i]:
                    transitions.append(
                        nn.Sequential(
                            nn.Conv2d(pre[i], cur[i], 3, 1, 1, bias=False),
                            nn.BatchNorm2d(cur[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transitions.append(None)
            else:
                convs = []
                for j in range(i + 1 - len(pre)):
                    entree = pre[-1]
                    sortie = cur[i] if j == i - len(pre) else entree
                    convs.append(
                        nn.Sequential(
                            nn.Conv2d(entree, sortie, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(sortie, momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True),
                        )
                    )
                transitions.append(nn.Sequential(*convs))
        return nn.ModuleList(transitions)

    @staticmethod
    def _make_layer(bloc, inplanes, planes, blocks):
        downsample = None
        if inplanes != planes * bloc.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * bloc.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes * bloc.expansion, momentum=BN_MOMENTUM),
            )
        couches = [bloc(inplanes, planes, 1, downsample)]
        couches += [bloc(planes * bloc.expansion, planes) for _ in range(1, blocks)]
        return nn.Sequential(*couches)

    @staticmethod
    def _make_stage(stage_cfg, num_inchannels):
        bloc = BLOCS[stage_cfg["block"]]
        modules = []
        for _ in range(stage_cfg["modules"]):
            module = HighResolutionModule(
                len(stage_cfg["blocks"]),
                bloc,
                stage_cfg["blocks"],
                num_inchannels,
                stage_cfg["channels"],
            )
            modules.append(module)
            num_inchannels = module.num_inchannels
        return nn.Sequential(*modules), num_inchannels

    def _etage(self, numero, x, y_list=None):
        """Applique l'étage `numero` : transition depuis le précédent, puis modules."""
        transition = getattr(self, f"transition{numero - 1}")
        entrees = []
        for i in range(self._nb_branches[numero]):
            if y_list is None:  # sortie du tronc (étage 1) : une seule branche source
                source = x
            else:  # nouvelles branches (plus basse résolution) issues de la dernière
                source = y_list[-1] if transition[i] is not None else y_list[i]
            entrees.append(transition[i](source) if transition[i] is not None else source)
        return getattr(self, f"stage{numero}")(entrees)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)
        y_list = self._etage(2, x)
        y_list = self._etage(3, x, y_list)
        y_list = self._etage(4, x, y_list)
        # Échelle 0 : résolution pleine, sans déconvolution (NUM_DECONVS = 0).
        return self.final_layers[0](y_list[self._out_scales[0]])


def construire_hrnet_wasb() -> HRNet:
    """HRNet à la configuration du modèle WASB (poids aléatoires)."""
    return HRNet(CONFIG_WASB)
