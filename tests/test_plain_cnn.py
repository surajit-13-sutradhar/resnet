import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.plain_cnn import PlainCNN


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


for n, depth in [(3, 20), (5, 32), (7, 44), (9, 56)]:
    model = PlainCNN(n).cuda()
    n_params = count_params(model)
    x = torch.randn(4, 3, 32, 32, device="cuda")
    out = model(x)
    print(f"n={n:2d}  depth={depth:3d}  params={n_params:,}  output shape={tuple(out.shape)}")

# Detailed shape trace for n=3 (ResNet-20 scale)
print("\n--- shape trace, n=3 ---")
model = PlainCNN(3).cuda()
x = torch.randn(1, 3, 32, 32, device="cuda")
h = model.relu(model.bn1(model.conv1(x))); print("after conv1:", h.shape)
h = model.stage1(h); print("after stage1:", h.shape)
h = model.stage2(h); print("after stage2:", h.shape)
h = model.stage3(h); print("after stage3:", h.shape)
h = model.avgpool(h).flatten(1); print("after avgpool:", h.shape)
out = model.fc(h); print("after fc:", out.shape)