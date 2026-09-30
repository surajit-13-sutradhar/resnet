import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.blocks import BasicBlock

print("=== Case 1: same channels, stride 1 (identity shortcut) ===")
block = BasicBlock(16, 16, stride=1).cuda()
print("shortcut type:", type(block.shortcut).__name__)
x = torch.randn(2, 16, 32, 32, device="cuda")
y = block(x)
print("input:", x.shape, " output:", y.shape)
params = sum(p.numel() for p in block.shortcut.parameters())
print("shortcut params:", params)

print("\n=== Case 2: downsample + channel change (16->32, stride 2) ===")
block2 = BasicBlock(16, 32, stride=2).cuda()
print("shortcut type:", type(block2.shortcut).__name__)
x2 = torch.randn(2, 16, 32, 32, device="cuda")
y2 = block2(x2)
print("input:", x2.shape, " output:", y2.shape)
params2 = sum(p.numel() for p in block2.shortcut.parameters())
print("shortcut params:", params2)

print("\n=== Verify residual addition actually happens ===")
# Force F(x) to output exactly zero by zeroing conv2's weight and bn2's weight/bias.
# Then y should equal ReLU(identity) = ReLU(x) exactly (since stride=1, in=out here).
block3 = BasicBlock(16, 16, stride=1).cuda()
with torch.no_grad():
    block3.conv2.weight.zero_()
    block3.bn2.weight.zero_()
    block3.bn2.bias.zero_()
x3 = torch.randn(2, 16, 32, 32, device="cuda")
y3 = block3(x3)
expected = torch.relu(x3)
print("max abs diff from ReLU(x):", (y3 - expected).abs().max().item())

print("\n=== Check the identity-pad shortcut's zero-padding directly ===")
from models.blocks import IdentityPadShortcut
sc = IdentityPadShortcut(16, 32, stride=2).cuda()
xin = torch.randn(1, 16, 32, 32, device="cuda")
out = sc(xin)
print("shortcut output shape:", out.shape)
print("first 16 channels equal strided input:",
      torch.equal(out[:, :16], xin[:, :, ::2, ::2]))
print("last 16 channels are all zero:", (out[:, 16:] == 0).all().item())