import sys, os, time, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from utils.metrics import accuracy
from training.scheduler import IterationLRScheduler, WarmupThenStepScheduler
from utils.checkpoint import save_checkpoint


def evaluate(model, test_batcher, use_amp=True):
    """[PAPER] Testing uses the single un-augmented 32x32 view."""
    model.eval()
    total_correct, total_n, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for xb, yb in test_batcher.epoch():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += yb.size(0)
    model.train()
    return total_loss / total_n, total_correct / total_n


def train(model, train_batcher, test_batcher, run_name,
          max_iters=64000, milestones=(32000, 48000), base_lr=0.1,
          eval_every=1000, log_dir="results", use_amp=True,
          warmup=False, warmup_lr=0.01, warmup_error_threshold=0.80):

    model.cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)

    if warmup:
        scheduler = WarmupThenStepScheduler(optimizer, warmup_lr, base_lr, milestones, 0.1, warmup_error_threshold)
    else: 
        scheduler = IterationLRScheduler(optimizer, base_lr, milestones)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = os.path.join(log_dir, f"{run_name}_log.csv")
    os.makedirs(log_dir, exist_ok=True)
    log_file = open(log_path, "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["iter", "lr", "train_loss", "train_acc", "test_loss", "test_acc"])

    it = 0
    running_loss, running_acc, running_n = 0.0, 0.0, 0
    train_acc_ema = 0.0
    t0 = time.time()

    while it < max_iters:
        for xb, yb in train_batcher.epoch():
            if it >= max_iters:
                break
            if warmup:
                lr = scheduler.step(it, train_acc_ema)
            else:
                lr = scheduler.step(it)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_acc = (logits.argmax(1) == yb).float().mean().item()
            train_acc_ema = 0.99 * train_acc_ema + 0.01 * batch_acc

            running_loss += loss.item() * yb.size(0)
            running_acc += (logits.argmax(1) == yb).sum().item()
            running_n += yb.size(0)
            it += 1

            if warmup and scheduler.warmup_done and it == scheduler.warmup_end_iter + 1:
                print(f"[warmup ended at iter {scheduler.warmup_end_iter}, "f"train_acc_ema={train_acc_ema:.3f}, switching LR to {base_lr}]")

            if it % eval_every == 0 or it == max_iters:
                test_loss, test_acc = evaluate(model, test_batcher, use_amp=use_amp)
                train_loss = running_loss / running_n
                train_acc = running_acc / running_n
                elapsed = time.time() - t0
                print(f"it {it:6d}/{max_iters}  lr {lr:.4f}  "
                      f"train_loss {train_loss:.4f} train_acc {train_acc:.4f}  "
                      f"test_loss {test_loss:.4f} test_acc {test_acc:.4f}  "
                      f"({elapsed:.1f}s elapsed)")
                logger.writerow([it, lr, train_loss, train_acc, test_loss, test_acc])
                log_file.flush()
                running_loss, running_acc, running_n = 0.0, 0.0, 0

    save_checkpoint(f"results/{run_name}_final.pt", model, optimizer, it)
    log_file.close()
    print(f"done. checkpoint: results/{run_name}_final.pt  log: {log_path}")