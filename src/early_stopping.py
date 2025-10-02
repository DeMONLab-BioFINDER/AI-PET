# src/early_stopping.py 
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class EarlyStopper:
    mode: str = "max"          # "max" for AUC / combined-reg metric; "min" for MAE/RMSE
    patience: int = 10
    min_delta: float = 1e-2
    best: float = None
    bad_epochs: int = 0
    best_epoch: int = 0

    def __post_init__(self):
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.best = -math.inf if self.mode == "max" else math.inf

    def step(self, val: float, epoch: int) -> tuple[bool, bool]:
        """
        Update with new metric value. Returns (stop, improved).
        """
        if not np.isfinite(val):
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience, False

        improved = (
            (self.mode == "max" and val > self.best + self.min_delta) or
            (self.mode == "min" and val < self.best - self.min_delta)
        )
        if improved:
            self.best = val
            self.best_epoch = epoch
            self.bad_epochs = 0
            return False, True
        else:
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience, False
