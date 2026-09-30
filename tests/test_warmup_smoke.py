import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from utils.data import load_cifar10_raw, compute_per_pixel_mean, CIFARBatcher
from models.resnet import CIFARResNet
from training.train import train

train_x, train_y, test_x, test_y = load_cifar10_raw()
mean = compute_per_pixel_mean(train_x)

train_batcher = CIFARBatcher(train_x, train_y, mean, batch_size=128, augment=True)
test_batcher = CIFARBatcher(test_x, test_y, mean, batch_size=128, augment=False)

model = CIFARResNet(18)  # n=18 -> ResNet-110
print("params:", sum(p.numel() for p in model.parameters()))

t0 = time.time()
train(model, train_batcher, test_batcher, run_name="smoke_resnet110",
      max_iters=1000, milestones=(600, 800), eval_every=100,
      warmup=True, warmup_lr=0.01, warmup_error_threshold=0.80)
print(f"\n1000 iters took {time.time()-t0:.1f}s -> "
      f"est. full 64000 iters: {(time.time()-t0)/1000*64000/60:.1f} min")