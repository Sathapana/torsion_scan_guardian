from dataclasses import dataclass
import numpy as np


@dataclass
class TriggerEvent:
    step: int
    max_atom_std: float
    atom_index: int


class ForceVarianceMonitor:
    """Flags steps where max per-atom force std exceeds threshold (after warmup)."""

    def __init__(self, threshold: float, warmup_steps: int = 0):
        self.threshold = threshold
        self.warmup_steps = warmup_steps

    def check(self, step: int, force_std_per_atom: np.ndarray) -> TriggerEvent | None:
        if step < self.warmup_steps:
            return None
        idx = int(np.argmax(force_std_per_atom))
        max_std = float(force_std_per_atom[idx])
        if max_std > self.threshold:
            return TriggerEvent(step=step, max_atom_std=max_std, atom_index=idx)
        return None
