import numpy as np
from guardian.uncertainty.monitor import ForceVarianceMonitor


def test_monitor_ignores_warmup():
    mon = ForceVarianceMonitor(threshold=0.1, warmup_steps=10)
    assert mon.check(5, np.array([0.5, 0.5])) is None


def test_monitor_fires_on_threshold():
    mon = ForceVarianceMonitor(threshold=0.1, warmup_steps=0)
    event = mon.check(100, np.array([0.05, 0.5, 0.2]))
    assert event is not None
    assert event.atom_index == 1
    assert event.max_atom_std == 0.5


def test_monitor_silent_below_threshold():
    mon = ForceVarianceMonitor(threshold=1.0, warmup_steps=0)
    assert mon.check(100, np.array([0.5, 0.5, 0.5])) is None
