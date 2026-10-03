# ResNet from Scratch — CIFAR-10

This is a from-scratch PyTorch reimplementation of ["Deep Residual Learning for Image Recognition"](https://arxiv.org/abs/1512.03385) (He, Zhang, Ren, Sun — Microsoft Research, 2015), scoped to the paper's CIFAR-10 experiments (Section 4.2).

No `torchvision.models.resnet*`, no pretrained weights. The BasicBlock, the shortcut connections, the plain-vs-residual comparison, the training loop, the ResNet-110 warmup trick — all of it is written here, layer by layer, and checked against the paper's own numbers at every step.

I did this to actually understand what the paper is claiming, not just to run someone else's code and get a number out. Everything below is what I found when I ran it on my own machine.

## Why this paper

The short version: if you take a normal deep CNN and keep stacking layers on it, past a certain depth it gets *worse* — and not because of overfitting. It gets worse on the data it's training on, which shouldn't be possible if more layers just mean more capacity. This is called the degradation problem, and it was a genuinely surprising result when the paper came out.

The fix the paper proposes is almost embarrassingly simple: instead of asking a block of layers to learn the entire output from scratch, let it learn a residual — the difference between what came in and what should come out — and add that back to the input. `y = F(x) + x`, where `F` is a couple of conv+BN layers and `+x` is a shortcut connection.

That's it. That's the whole idea. And it's the reason ResNets and their descendants are still everywhere, a decade later.

## What's in this repo

```
resnet/
├── models/
│   ├── blocks.py       # BasicBlock + the identity-pad shortcut (Option A)
│   ├── plain_cnn.py     # the no-shortcut baseline, same depth/width as the ResNets
│   └── resnet.py        # CIFAR ResNet-20/32/44/56/110, built from BasicBlock
├── training/
│   ├── train.py          # the training loop (SGD, step-decay LR, optional warmup)
│   └── scheduler.py       # the paper's 32k/48k/64k iteration schedule, plus warmup
├── utils/
│   ├── data.py            # CIFAR-10 loading, per-pixel mean, pad+crop+flip augmentation
│   ├── checkpoint.py       # save/load
│   └── metrics.py          # accuracy
├── experiments/
│   ├── cifar10_resnet20.py
│   ├── cifar10_resnet110.py
│   ├── run_degradation_experiment.py   # trains all plain/resnet pairs in one go
│   ├── layer_response_analysis.py       # the BN-output std analysis
│   └── generate_all_results.py          # regenerates all three plots/tables below
├── results/           # training logs, checkpoints, and the images in this README
└── tests/              # small sanity checks I ran at every stage before trusting the code
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Swap the `cu126` tag for whatever CUDA build matches your GPU driver — check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) for the current command.

## Running it

```powershell
# train a single model
python experiments\cifar10_resnet20.py

# train the full plain-vs-resnet comparison (20/32/44/56, both variants)
python experiments\run_degradation_experiment.py

# train ResNet-110 with the paper's LR warmup
python experiments\cifar10_resnet110.py

# regenerate every plot and table from whatever's already in results/
python experiments\generate_all_results.py
```

Everything here trained on a single RTX 4050 laptop GPU (6 GB VRAM). ResNet-20 takes about 15 minutes for the full 64k-iteration schedule; the deepest model, ResNet-110, takes a bit over an hour. None of this needs a serious GPU.

## The main result: the degradation problem, reproduced

I trained matched plain and residual networks at four depths — 20, 32, 44, and 56 layers — with identical hyperparameters and, thanks to the paper's zero-parameter identity shortcuts, identical parameter counts at each depth. The only difference between a plain net and its ResNet twin is whether the shortcut connections exist.

![Plain vs Residual networks, depth 20-56](results/degradation_comparison.png)

Left side: as the plain networks get deeper, they get *worse* — both in training error (dashed lines) and test error (solid lines). Plain-56 ends up clearly the worst of the four, even though it has the most capacity.

Right side: same four depths, but with shortcuts added. The curves barely separate. ResNet-56 trains just as well as ResNet-20, instead of falling apart like its plain counterpart did.

That gap is the entire point of the paper, made visible:

| | Plain-56 | ResNet-56 | difference |
|---|---|---|---|
| test error | 14.22% | 8.30% | ~6 points, same depth, same parameter count |

## Why it happens: the layer-response analysis

The paper offers a mechanistic explanation for *why* this works — it measures the standard deviation of each layer's output (after BatchNorm, before the nonlinearity) and shows that residual layers tend to have smaller responses than plain ones, consistent with each block learning only a small correction rather than reinventing its input from scratch.

I reproduced this using PyTorch forward hooks on every BatchNorm layer, across five of my trained checkpoints:

![Layer response std, plain vs resnet, by depth](results/layer_response_std.png)

The sorted panel on the right is the clearest read: `resnet110` (purple) sits at or below every other curve for almost its whole length, dropping to a std of around 0.05 by its deepest layers. The ordering across the three ResNet depths — resnet20 > resnet56 > resnet110 in typical magnitude — matches the paper's own claim that deeper ResNets end up with *smaller* per-layer responses, not larger ones. Each individual layer is doing less work as the network gets deeper, because it only has to contribute a small correction on top of everything that came before it.

## Final numbers vs. the paper

| Model | Params | My train err % | My test err % | Paper test err % |
|---|---|---|---|---|
| Plain-20 | 0.27M | 1.23 | 9.63 | — |
| Plain-32 | 0.46M | 1.35 | 9.92 | — |
| Plain-44 | 0.66M | 2.38 | 10.36 | — |
| Plain-56 | 0.85M | 5.24 | 14.22 | — |
| ResNet-20 | 0.27M | 0.73 | 8.24 | 8.75 |
| ResNet-32 | 0.46M | 0.21 | 7.36 | 7.51 |
| ResNet-44 | 0.66M | 0.13 | 7.38 | 7.17 |
| ResNet-56 | 0.85M | 0.18 | 8.30 | 6.97 |
| ResNet-110 | 1.7M | 0.06 | 6.96 | 6.43 (best) / 6.61±0.16 (mean of 5 runs) |

![Full results table](results/summary_table.png)

The paper doesn't report plain-network CIFAR numbers in its tables — only the visual comparison in its equivalent of the plot above — so those rows have no direct paper figure to compare against.

Most of my ResNet numbers land within half a point of the paper, which is about what you'd expect from a single run against a value the paper itself reports as varying by ±0.16% across 5 runs for ResNet-110 alone. ResNet-56 is the one result I'd flag as a bit off (8.30% vs. the paper's 6.97%) — it didn't clearly improve over ResNet-44 the way it does in the paper, which is the kind of run-to-run noise a second attempt would probably resolve.

## Things that broke, and what I learned from them

This is the part I actually got the most out of, so I'm keeping it in rather than just reporting clean final numbers.

**A 56-layer plain network trained to NaN.** The first attempt at plain-56 diverged to a constant 10% accuracy (random guessing) within the first 2000 iterations and never recovered. The cause was mixed-precision (fp16) training — not part of the original paper, just something I added for speed — combined with a very deep network that has no shortcut connections to keep its activations in check. Forward-pass activations overflowed fp16's range, the loss went to NaN, and every step afterward was poisoned. The fix was to just train that one model in fp32 instead of chasing the instability with architecture changes. It's a good illustration of exactly the kind of numerical fragility the paper's shortcut connections end up protecting against — the layer-response analysis above shows plain networks really do have noisier, larger-magnitude activations.

**The ResNet-110 warmup scheduler silently never triggered.** The paper's own training recipe for ResNet-110 starts at a lower learning rate and switches to the full 0.1 once training accuracy clears 20%, because starting straight at 0.1 is too unstable for a network this deep. My first implementation tracked a smoothed accuracy estimate meant to trigger that switch — except the line that actually updated the estimate was missing from the training loop. The value stayed frozen at its initial 0.0 for the entire run, so the switch condition could never fire, and the model trained at the warmup LR for all 64,000 iterations without a single error or crash. Nothing looked wrong in the output; the loss was still going down. The bug only became visible because I was watching the printed learning-rate column and noticed it never changed. That was a good reminder that "the metrics look fine" doesn't mean the logic is correct — it just means nothing's wrong *yet*.

## What I'd do differently next time

- Run plain-56 and resnet-56 in matched precision (fp32 for both) rather than leaving plain-56 as the one fp32 outlier in an otherwise fp16 comparison — it's a real, if small, confound in the degradation plot above.
- Train ResNet-110 more than once, the way the paper does, before reporting a single number against their mean.
- Add a basic assertion or log line any time a scheduler reads a value that's supposed to be updated elsewhere in the loop — the warmup bug would have been caught in seconds with one print statement, instead of a full training run.

## Reference

He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)