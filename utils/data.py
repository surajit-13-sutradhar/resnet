import numpy as np
import torch
from torchvision.datasets import CIFAR10


def load_cifar10_raw(root="data"):
    """Returns uint8 tensors: train_x (50000,3,32,32), train_y, test_x, test_y."""
    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)

    # .data is a numpy array of shape (N, 32, 32, 3), dtype uint8, layout HWC.
    # PyTorch convs expect NCHW, so we permute to (N, 3, 32, 32).
    train_x = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()
    test_x = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous()
    train_y = torch.tensor(train.targets, dtype=torch.long)
    test_y = torch.tensor(test.targets, dtype=torch.long)
    return train_x, train_y, test_x, test_y


def compute_per_pixel_mean(train_x):
    """[PAPER] Per-pixel mean: shape (3, 32, 32), averaged over all training images."""
    return train_x.float().mean(dim=0)

def normalize(x_uint8, mean):
    """[PAPER] Convert to float and subtract the per-pixel mean. No std division."""
    return x_uint8.float() - mean


def augment_batch(x):
    """
    [PAPER] Pad 4 px each side, random 32x32 crop, random horizontal flip.
    x: float tensor (B, 3, 32, 32), already mean-subtracted, on any device.
    Each image gets its OWN random crop offset and its OWN flip decision.
    """
    B, C, H, W = x.shape
    device = x.device

    # 1. Random horizontal flip (per image): flip where mask is True
    flip = torch.rand(B, device=device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(dims=[3]), x)

    # 2. Zero-pad 4 pixels on each side -> (B, 3, 40, 40)
    x = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="constant", value=0.0)

    # 3. Random crop offsets per image, in [0, 8]
    top = torch.randint(0, 9, (B,), device=device)
    left = torch.randint(0, 9, (B,), device=device)

    # Build index grids so every image is cropped at its own offset
    rows = (top[:, None] + torch.arange(H, device=device)[None, :])  # (B, 32)
    cols = (left[:, None] + torch.arange(W, device=device)[None, :])  # (B, 32)
    b_idx = torch.arange(B, device=device)[:, None, None]             # (B, 1, 1)
    x = x.permute(0, 2, 3, 1)                     # (B, 40, 40, 3)
    x = x[b_idx, rows[:, :, None], cols[:, None, :]]  # (B, 32, 32, 3)
    return x.permute(0, 3, 1, 2).contiguous()     # (B, 3, 32, 32)

class CIFARBatcher:
    """
    Keeps the whole dataset on the GPU as float32 (mean-subtracted).
    Yields (images, labels) batches forever; reshuffles at each epoch boundary.
    50000*3*32*32*4 bytes = ~614 MB of VRAM, which is fine on a 6 GB card.
    """

    def __init__(self, x_uint8, y, mean, batch_size=128, augment=True, device="cuda"):
        self.x = normalize(x_uint8, mean).to(device)
        self.y = y.to(device)
        self.batch_size = batch_size
        self.augment = augment
        self.device = device
        self.n = self.x.shape[0]

    def iters_per_epoch(self):
        # ceil division: keeps the last partial batch
        return (self.n + self.batch_size - 1) // self.batch_size

    def epoch(self):
        """One pass over the data (shuffled if augment/training)."""
        idx = torch.randperm(self.n, device=self.device) if self.augment \
            else torch.arange(self.n, device=self.device)
        for start in range(0, self.n, self.batch_size):
            b = idx[start:start + self.batch_size]
            xb, yb = self.x[b], self.y[b]
            if self.augment:
                xb = augment_batch(xb)
            yield xb, yb