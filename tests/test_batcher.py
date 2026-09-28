import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, torch
from utils.data import load_cifar10_raw, compute_per_pixel_mean, CIFARBatcher

train_x, train_y, test_x, test_y = load_cifar10_raw()
mean = compute_per_pixel_mean(train_x)

train = CIFARBatcher(train_x, train_y, mean, batch_size=128, augment=True)
test = CIFARBatcher(test_x, test_y, mean, batch_size=128, augment=False)

print("train iters/epoch:", train.iters_per_epoch())
print("test  iters/epoch:", test.iters_per_epoch())

# Count everything in one epoch
n_seen, n_batches = 0, 0
label_counts = torch.zeros(10, device="cuda")
torch.cuda.synchronize(); t0 = time.time()
for xb, yb in train.epoch():
    n_seen += xb.shape[0]; n_batches += 1
    label_counts += torch.bincount(yb, minlength=10)
torch.cuda.synchronize()
print(f"one train epoch: {n_batches} batches, {n_seen} images, {time.time()-t0:.2f}s")
print("label counts seen:", label_counts.long().tolist())

# Two epochs should be shuffled differently
first = next(iter(train.epoch()))[1][:10].tolist()
second = next(iter(train.epoch()))[1][:10].tolist()
print("first labels epoch A:", first)
print("first labels epoch B:", second)

# Test set must be deterministic and un-augmented
a = next(iter(test.epoch()))[0]
b = next(iter(test.epoch()))[0]
print("test batches identical across epochs:", torch.equal(a, b))
print("GPU memory allocated (MB):", torch.cuda.memory_allocated() // 2**20)