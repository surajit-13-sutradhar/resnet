import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.resnet import CIFARResNet
from models.blocks import IdentityPadShortcut
from models.plain_cnn import PlainCNN


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


table6 = {20: 0.27, 32: 0.46, 44: 0.66, 56: 0.85}

print("=== Param count vs Table 6, and vs the plain-net twin ===")
for n, depth in [(3, 20), (5, 32), (7, 44), (9, 56)]:
    resnet = CIFARResNet(n).cuda()
    plain = PlainCNN(n).cuda()
    rp, pp = count_params(resnet), count_params(plain)
    print(f"n={n:2d} depth={depth:3d}  resnet={rp:,}  plain={pp:,}  "
          f"equal={rp == pp}  paper~{table6[depth]}M")

print("\n=== Shortcut count check (paper says 3n shortcuts total) ===")
for n, depth in [(3, 20), (5, 32)]:
    resnet = CIFARResNet(n)
    n_identity_pad = sum(1 for m in resnet.modules() if isinstance(m, IdentityPadShortcut))
    n_identity = sum(1 for m in resnet.modules() if isinstance(m, nn.Identity))
    print(f"n={n} depth={depth}: IdentityPadShortcut={n_identity_pad}, "
          f"plain Identity={n_identity}, total blocks(shortcuts)={n_identity_pad + n_identity} "
          f"(expected 3n={3*n})")

print("\n=== Forward pass + shape trace for ResNet-20 ===")
model = CIFARResNet(3).cuda()
x = torch.randn(4, 3, 32, 32, device="cuda")
out = model(x)
print("output shape:", tuple(out.shape))

h = model.relu(model.bn1(model.conv1(x))); print("after conv1: ", h.shape)
h = model.stage1(h); print("after stage1:", h.shape)
h = model.stage2(h); print("after stage2:", h.shape)
h = model.stage3(h); print("after stage3:", h.shape)

print("\n=== Backward pass sanity: gradients flow through the whole network ===")
loss = out.sum()
loss.backward()
n_none = sum(1 for p in model.parameters() if p.requires_grad and p.grad is None)
print("params with no gradient:", n_none, "(expected 0)")