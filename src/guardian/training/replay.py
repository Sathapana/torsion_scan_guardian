from dataclasses import dataclass, field
from pathlib import Path
import pickle
import numpy as np


@dataclass
class DataPoint:
    positions: np.ndarray
    symbols: list[str]
    energy: float
    forces: np.ndarray
    source: str   # "seed" | "acquired"
    cycle: int = 0


@dataclass
class ReplayBuffer:
    points: list[DataPoint] = field(default_factory=list)

    def add(self, p: DataPoint) -> None:
        self.points.append(p)

    def sample(self, n: int, rng: np.random.Generator, recency_bias: float = 2.0) -> list[DataPoint]:
        if not self.points:
            return []
        weights = np.array([recency_bias if p.source == "acquired" else 1.0 for p in self.points])
        probs = weights / weights.sum()
        idx = rng.choice(len(self.points), size=min(n, len(self.points)), replace=False, p=probs)
        return [self.points[i] for i in idx]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.points, f)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with open(path, "rb") as f:
            return cls(points=pickle.load(f))
