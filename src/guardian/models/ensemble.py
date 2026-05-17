from dataclasses import dataclass
from typing import Literal
import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


@dataclass
class EnsemblePrediction:
    energy: float                    # eV, evaluated at the true (unperturbed) geometry
    forces: np.ndarray               # (N, 3) eV/Å, true-geometry forces driving the integrator
    forces_std_per_atom: np.ndarray  # (N,)   eV/Å, std across probes/members — uncertainty proxy


def _pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def load_mace_off(size: Literal["small", "medium", "large"] = "small",
                  device: str = "cuda",
                  dtype: str = "float32"):
    from mace.calculators import mace_off
    return mace_off(model=size, device=_pick_device(device), default_dtype=dtype)


def load_mace_checkpoint(model_path: str, device: str = "cuda", dtype: str = "float32"):
    """Load a single MACE checkpoint (.model file produced by mace_run_train) as an ASE calculator."""
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=model_path, device=_pick_device(device),
                          default_dtype=dtype)


class MACEOffEnsemble:
    """Single MACE-OFF checkpoint queried multiple times with small position jitter.

    Phase-1 stand-in for a real seed-fine-tuned ensemble. See `SeedFinetuneEnsemble`
    for the Phase-2 multi-checkpoint implementation that shares this interface.
    """

    def __init__(self, calc, n_probes: int = 3, position_noise_A: float = 0.005, seed: int = 0):
        if n_probes < 2:
            raise ValueError("n_probes must be >= 2 to compute a std")
        self.calc = calc
        self.n_probes = n_probes
        self.position_noise_A = position_noise_A
        self.rng = np.random.default_rng(seed)

    def predict(self, atoms: Atoms) -> EnsemblePrediction:
        forces = []
        # Probe 0 is unperturbed — used for both std and the integrator's E/F.
        probe = atoms.copy()
        probe.calc = self.calc
        energy0 = float(probe.get_potential_energy())
        f0 = probe.get_forces()
        forces.append(f0)
        for _ in range(self.n_probes - 1):
            probe = atoms.copy()
            probe.positions = probe.positions + self.rng.normal(
                0.0, self.position_noise_A, probe.positions.shape
            )
            probe.calc = self.calc
            probe.get_potential_energy()
            forces.append(probe.get_forces())
        f = np.stack(forces, axis=0)                          # (M, N, 3)
        f_std_vec = f.std(axis=0)                             # (N, 3)
        f_std_per_atom = np.linalg.norm(f_std_vec, axis=1)    # (N,)
        return EnsemblePrediction(
            energy=energy0,
            forces=f0,
            forces_std_per_atom=f_std_per_atom,
        )


class SeedFinetuneEnsemble:
    """Real epistemic ensemble: N independently-fine-tuned MACE checkpoints.

    `predict` returns the *first* member's energy and forces (used by the MD
    integrator — picking member 0 keeps the dynamics deterministic and
    decoupled from ensemble composition), and the per-atom std of forces
    across all members as the uncertainty signal.

    Same interface as `MACEOffEnsemble` — drop-in replacement in the controller.
    """

    def __init__(self, calcs: list):
        if len(calcs) < 2:
            raise ValueError("Need at least 2 members to compute force std")
        self.calcs = calcs

    @classmethod
    def from_checkpoints(cls, paths: list[str], device: str = "cuda", dtype: str = "float32"):
        return cls([load_mace_checkpoint(p, device=device, dtype=dtype) for p in paths])

    def predict(self, atoms: Atoms) -> EnsemblePrediction:
        energies, forces = [], []
        for calc in self.calcs:
            probe = atoms.copy()
            probe.calc = calc
            energies.append(float(probe.get_potential_energy()))
            forces.append(probe.get_forces())
        f = np.stack(forces, axis=0)                          # (M, N, 3)
        f_std_vec = f.std(axis=0)                             # (N, 3)
        f_std_per_atom = np.linalg.norm(f_std_vec, axis=1)    # (N,)
        return EnsemblePrediction(
            energy=energies[0],
            forces=forces[0],
            forces_std_per_atom=f_std_per_atom,
        )


class EnsembleCalculator(Calculator):
    """ASE calculator backed by an MACEOffEnsemble or SeedFinetuneEnsemble.

    Stores per-atom force std on each call for the monitor to read.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, ensemble, **kwargs):
        super().__init__(**kwargs)
        self.ensemble = ensemble
        self.last_force_std: np.ndarray | None = None

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        pred = self.ensemble.predict(self.atoms)
        self.results["energy"] = pred.energy
        self.results["forces"] = pred.forces
        self.last_force_std = pred.forces_std_per_atom
