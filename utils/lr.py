import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWithWarmupLR(_LRScheduler):
    """Linear warmup then cosine decay to 0 — mirrors Exphormer's cosine_with_warmup."""

    def __init__(self, optimizer, warmup, tot, lr, last_epoch=-1, verbose=False):
        self.warmup = max(1, warmup)
        self.tot = tot
        self.lr = lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup:
            lr = self.lr * self._step_count / float(self.warmup)
        else:
            progress = (self._step_count - self.warmup) / max(1, self.tot - self.warmup)
            lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return [lr for _ in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup, tot, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup = warmup
        self.tot = tot
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)


    def get_lr(self):
        if self._step_count <= self.warmup:
            self.warmup_factor = self._step_count / float(self.warmup)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot:
            lr = self.end_lr
        else:
            warmup = self.warmup
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (self.tot - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False
