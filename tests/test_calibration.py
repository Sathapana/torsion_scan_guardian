import numpy as np
from dataclasses import dataclass

from guardian.calibration import calibrate_threshold
from guardian.models.ensemble import EnsemblePrediction


@dataclass
class StubEnsemble:
    """Returns a deterministic force-std vector seeded by the geometry — no MACE needed."""
    n_atoms: int
    base_std: float = 0.05
    spike_atom: int = 0
    spike_scale: float = 2.0

    def predict(self, atoms) -> EnsemblePrediction:
        # std grows with displacement of atom 0 — mimics a high-curvature region.
        disp = np.linalg.norm(atoms.positions[self.spike_atom])
        stds = np.full(self.n_atoms, self.base_std)
        stds[self.spike_atom] = self.base_std + self.spike_scale * disp
        return EnsemblePrediction(
            energy=0.0,
            forces=np.zeros((self.n_atoms, 3)),
            forces_std_per_atom=stds,
        )


def _atoms(n: int):
    from ase import Atoms
    return Atoms("H" * n, positions=np.zeros((n, 3)))


def test_calibration_returns_threshold_above_noise_floor():
    ens = StubEnsemble(n_atoms=3, base_std=0.05, spike_scale=2.0)
    cal = calibrate_threshold(ens, _atoms(3), n_samples=200, sigma_A=0.04, safety_factor=1.5)
    assert cal.n_samples == 200
    assert cal.max_std_p50 < cal.max_std_p95 < cal.max_std_p99
    assert cal.suggested_threshold > cal.max_std_p99   # safety factor applied
    # Should be above the base noise floor since the spike scales with random displacement.
    assert cal.suggested_threshold > 0.05


def test_calibration_safety_factor_scales_threshold():
    ens = StubEnsemble(n_atoms=3, base_std=0.05, spike_scale=2.0)
    cal_a = calibrate_threshold(ens, _atoms(3), n_samples=100, sigma_A=0.04, safety_factor=1.0, seed=42)
    cal_b = calibrate_threshold(ens, _atoms(3), n_samples=100, sigma_A=0.04, safety_factor=2.0, seed=42)
    assert np.isclose(cal_b.suggested_threshold, 2.0 * cal_a.suggested_threshold)
