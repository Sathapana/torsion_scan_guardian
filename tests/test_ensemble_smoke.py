"""Slow smoke test: loads MACE-OFF small and runs the input-perturbation ensemble on H2O.

Skipped unless `RUN_MACE_SMOKE=1` to keep the default `pytest` fast (no model download / GPU).
Run explicitly with: `RUN_MACE_SMOKE=1 pytest tests/test_ensemble_smoke.py -s`.
"""
import os
import numpy as np
import pytest
from ase.build import molecule


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MACE_SMOKE") != "1",
    reason="Set RUN_MACE_SMOKE=1 to run the MACE-OFF download/inference smoke test.",
)


def test_ensemble_predict_h2o():
    from guardian.models.ensemble import MACEOffEnsemble, load_mace_off
    calc = load_mace_off(size="small", device="cpu", dtype="float32")
    ens = MACEOffEnsemble(calc=calc, n_probes=3, position_noise_A=0.005, seed=0)

    atoms = molecule("H2O")
    pred = ens.predict(atoms)

    assert np.isfinite(pred.energy)
    assert pred.forces.shape == (3, 3)
    assert pred.forces_std_per_atom.shape == (3,)
    assert (pred.forces_std_per_atom >= 0).all()
    # Equilibrium H2O should have small forces and small variance.
    assert np.linalg.norm(pred.forces) < 2.0
    assert pred.forces_std_per_atom.max() < 0.5
