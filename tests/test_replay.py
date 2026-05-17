import numpy as np
from guardian.training.replay import ReplayBuffer, DataPoint


def _mk(source: str) -> DataPoint:
    return DataPoint(
        positions=np.zeros((2, 3)), symbols=["H", "H"],
        energy=0.0, forces=np.zeros((2, 3)), source=source,
    )


def test_buffer_biases_toward_acquired():
    buf = ReplayBuffer()
    for _ in range(10):
        buf.add(_mk("seed"))
    for _ in range(2):
        buf.add(_mk("acquired"))
    rng = np.random.default_rng(0)
    counts = {"seed": 0, "acquired": 0}
    for _ in range(2000):
        s = buf.sample(1, rng, recency_bias=10.0)[0]
        counts[s.source] += 1
    # acquired (weight 10, n=2 → 20) vs seed (weight 1, n=10 → 10): acquired should dominate
    assert counts["acquired"] > counts["seed"]
