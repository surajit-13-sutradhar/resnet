import torch
import torch.nn as nn


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class PlainCNN(nn.Module):
    """
    [PAPER, Sec 4.2] CIFAR plain network: 6n+2 layers, no shortcuts.
    Stage 1: 2n layers at 16 filters, 32x32
    Stage 2: 2n layers at 32 filters, 16x16 (first layer stride 2)
    Stage 3: 2n layers at 64 filters, 8x8  (first layer stride 2)
    """

    def __init__(self, n, num_classes=10):
        super().__init__()
        self.n = n

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_stage(16, 16, 2*n, stride=1)
        self.stage2 = self._make_stage(16, 32, 2*n, stride=2)
        self.stage3 = self._make_stage(32, 64, 2*n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n, stride):
        layers = []
        # first layer of the stage may downsample (stride) and change channels
        layers.append(self._layer(in_ch, out_ch, stride))
        for _ in range(1, n):
            layers.append(self._layer(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _layer(self, in_ch, out_ch, stride):
        return nn.Sequential(
            conv3x3(in_ch, out_ch, stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self):
        # [PAPER cites He et al. [13]] -> Kaiming/He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)