class IterationLRScheduler:
    """
    [PAPER, Sec 4.2] step decay by iteration count:
    lr = base_lr           for iter < milestones[0]
    lr = base_lr / 10      for milestones[0] <= iter < milestones[1]
    lr = base_lr / 100     for iter >= milestones[1]
    """

    def __init__(self, optimizer, base_lr=0.1, milestones=(32000, 48000), gamma=0.1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.milestones = milestones
        self.gamma = gamma

    def lr_at(self, it):
        lr = self.base_lr
        for m in self.milestones:
            if it >= m:
                lr *= self.gamma
        return lr

    def step(self, it):
        lr = self.lr_at(it)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

class WarmupThenStepScheduler:
    """
    [PAPER, Sec 4.2, ResNet-110] Start at warmup_lr until train accuracy EMA
    exceeds (1 - error_threshold), then switch to the normal step schedule.
    """

    def __init__(self, optimizer, warmup_lr=0.01, base_lr=0.1,
                 milestones=(32000, 48000), gamma=0.1, error_threshold=0.80):
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.base_lr = base_lr
        self.milestones = milestones
        self.gamma = gamma
        self.acc_threshold = 1.0 - error_threshold  # e.g. 0.20
        self.warmup_done = False
        self.warmup_end_iter = None

    def _stepped_lr(self, it):
        lr = self.base_lr
        for m in self.milestones:
            if it >= m:
                lr *= self.gamma
        return lr

    def step(self, it, train_acc_ema):
        if not self.warmup_done:
            if train_acc_ema is not None and train_acc_ema > self.acc_threshold:
                self.warmup_done = True
                self.warmup_end_iter = it
            lr = self.warmup_lr if not self.warmup_done else self._stepped_lr(it)
        else:
            lr = self._stepped_lr(it)

        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr