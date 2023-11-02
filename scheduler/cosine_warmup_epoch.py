import numpy as np
from torch.optim.lr_scheduler import _LRScheduler

class CosineWarmupSchedulerEpoch(_LRScheduler): # ⚡⚡
    def __init__(self, optimizer, warmup, max_epoch, update = "epoch"):
        self.warmup = warmup # unit: epoch
        self.max_epoch = max_epoch # unit: epoch
        self.update = update
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_epoch))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

def Scheduler(optimizer, warmup, max_epoch, update = "epoch", **kwargs):
    sche_fn = CosineWarmupSchedulerEpoch(optimizer, warmup, max_epoch, update = update)

    print('Initialised CosineWarmupSchedulerEpoch scheduler')

    return sche_fn