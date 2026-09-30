import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class IdentityPadShortcut(nn.Module):
    """
    [PAPER, Sec 3.3, Option A] Used only when stride=2 or channels change.
    Downsamples x spatially by stride, then zero-pads extra channels.
    Zero parameters, as the paper requires for CIFAR-10 (Sec 4.2).
    """

    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.stride = stride
        self.pad_channels = out_ch - in_ch

    def forward(self, x):
        # subsample spatially: keep every `stride`-th pixel
        x = x[:, :, ::self.stride, ::self.stride]
        # pad zero channels: (left, right, top, bottom, front, back) for last 3 dims
        # we only pad the channel dim, so pad only "front/back" of channel axis
        x = F.pad(x, (0, 0, 0, 0, 0, self.pad_channels))
        return x


class BasicBlock(nn.Module):
    """
    [PAPER, Fig 2] y = ReLU( F(x) + shortcut(x) )
    F(x) = BN(Conv3x3( ReLU(BN(Conv3x3(x))) ))
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = IdentityPadShortcut(in_ch, out_ch, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = out + identity      # the residual addition
        out = self.relu(out)      # [PAPER] second ReLU AFTER addition
        return out                # <-- must be indented to match the lines above     # the residual addition