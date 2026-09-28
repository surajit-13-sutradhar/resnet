import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.data import load_cifar10_raw, compute_per_pixel_mean

train_x, train_y, test_x, test_y = load_cifar10_raw()

print("train_x:", train_x.shape, train_x.dtype)
print("train_y:", train_y.shape, train_y.dtype)
print("test_x: ", test_x.shape)
print("label counts (train):", torch.bincount(train_y).tolist())
print("pixel range:", train_x.min().item(), "to", train_x.max().item())

mean = compute_per_pixel_mean(train_x)
print("per-pixel mean shape:", mean.shape)
print("mean of the mean image, per channel (R,G,B):", mean.mean(dim=(1, 2)).tolist())