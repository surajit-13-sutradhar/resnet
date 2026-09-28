import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
from utils.data import load_cifar10_raw, compute_per_pixel_mean, normalize, augment_batch

train_x, train_y, _, _ = load_cifar10_raw()
mean = compute_per_pixel_mean(train_x)

# 1. Mean subtraction check: the normalized training set should have ~0 mean per pixel
xn = normalize(train_x, mean)
print("max |per-pixel mean| after subtraction:", xn.mean(dim=0).abs().max().item())

# 2. Shape check
batch = xn[:8]
aug = augment_batch(batch)
print("augment output:", aug.shape, aug.dtype)

# 3. Sanity: augmenting must not change the value range much (only zeros added)
print("range before:", batch.min().item(), batch.max().item())
print("range after: ", aug.min().item(), aug.max().item())

# 4. Visual check: original vs 3 augmented versions of the same image
def show(t):  # add the mean back and scale to 0-1 for display
    return ((t + mean).clamp(0, 255) / 255).permute(1, 2, 0).numpy()

img = xn[0:1]
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
axes[0].imshow(show(img[0])); axes[0].set_title("original")
for i in range(1, 5):
    a = augment_batch(img)
    axes[i].imshow(show(a[0])); axes[i].set_title(f"aug {i}")
for ax in axes: ax.axis("off")
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/augmentation_check.png", dpi=120)
print("saved results/augmentation_check.png")