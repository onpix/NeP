from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from nerfstudio.engine.schedulers import SchedulerConfig, Scheduler


@dataclass
class NePSchedulerConfig(SchedulerConfig):
    _target: Type = field(default_factory=lambda: NePScheduler)
    end_warm: int = 5000
    end_iter: int = 300000
    lr: float = 5e-4
    learning_rate_alpha: float = 0.05


class NePScheduler(Scheduler):
    config: NePSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        def func(step):
            if step < self.config.end_warm:
                learning_factor = step / self.config.end_warm
            else:
                alpha = self.config.learning_rate_alpha
                progress = (step - self.config.end_warm) / \
                           (self.config.end_iter - self.config.end_warm)
                learning_factor = (np.cos(np.pi * progress) +
                                   1.0) * 0.5 * (1 - alpha) + alpha

            return learning_factor

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
