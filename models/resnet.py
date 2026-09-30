import torch
import torch.nn as nn
from models.blocks import BasicBlock, conv3x3


class CIFARResNet(nn.Module):
    """
    [PAPER, Sec 4.2] 6n+2 layer CIFAR ResNet, built from BasicBlocks.
    n=3 -> ResNet-20, n=5 -> ResNet-32, n=7 -> ResNet-44, n=9 -> ResNet-56, n=18 -> ResNet-110
    """

    def __init__(self, n, num_classes=10):
        super().__init__()
        self.n = n

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_stage(16, 16, n, stride=1)
        self.stage2 = self._make_stage(16, 32, n, stride=2)
        self.stage3 = self._make_stage(32, 64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]      # first block may downsample
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))   # rest keep shape
        return nn.Sequential(*layers)

    def _init_weights(self):
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


def resnet20(): return CIFARResNet(3)
def resnet32(): return CIFARResNet(5)
def resnet44(): return CIFARResNet(7)
def resnet56(): return CIFARResNet(9)
def resnet110(): return CIFARResNet(18)