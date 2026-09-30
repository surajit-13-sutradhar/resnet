import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from utils.data import load_cifar10_raw, compute_per_pixel_mean, CIFARBatcher
from models.resnet import resnet20
from training.train import train

train_x, train_y, test_x, test_y = load_cifar10_raw()
mean = compute_per_pixel_mean(train_x)

train_batcher = CIFARBatcher(train_x, train_y, mean, batch_size=128, augment=True)
test_batcher = CIFARBatcher(test_x, test_y, mean, batch_size=128, augment=False)

model = resnet20()

t0 = time.time()
train(model, train_batcher, test_batcher, run_name="smoke_resnet20",
      max_iters=200, milestones=(120, 160), eval_every=50)
print(f"\ntotal time for 200 iters: {time.time()-t0:.1f}s")
print(f"-> estimated time for full 64000 iters: {(time.time()-t0)/200*64000/60:.1f} minutes")