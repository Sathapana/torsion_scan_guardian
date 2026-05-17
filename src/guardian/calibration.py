from dataclasses import dataclass
import numpy as np
from ase import Atoms
from ase.optimize import BFGS

from .models.ensemble import MACEOffEnsemble


@dataclass
class RelaxReport:
    converged: bool
    n_steps: int
    fmax_final: float
    energy_final: float


@dataclass
class CalibrationReport:
    n_samples: int
    sigma_A: float
    max_std_p50: float
    max_std_p95: float
    max_std_p99: float
    suggested_threshold: float


def relax_geometry(atoms: Atoms, calc, fmax: float = 0.05, max_steps: int = 200,
                   logfile: str | None = None) -> RelaxReport:
    """In-place geometry relaxation with the provided ASE calculator (e.g., a MACE calc)."""
    atoms.calc = calc
    opt = BFGS(atoms, logfile=logfile)
    opt.run(fmax=fmax, steps=max_steps)
    forces = atoms.get_forces()
    fmax_final = float(np.linalg.norm(forces, axis=1).max())
    return RelaxReport(
        converged=fmax_final <= fmax,
        n_steps=opt.get_number_of_steps(),
        fmax_final=fmax_final,
        energy_final=float(atoms.get_potential_energy()),
    )


def calibrate_threshold(ensemble: MACEOffEnsemble, atoms: Atoms, n_samples: int = 50,
                        sigma_A: float = 0.04, safety_factor: float = 1.5,
                        seed: int = 0) -> CalibrationReport:
    """Estimate the natural noise floor of `max_atom_force_std` under thermal-amplitude jitter.

    σ ≈ 0.04 Å approximates the RMS displacement of a stiff bond at 300 K. The
    suggested threshold is `percentile_99 × safety_factor` so genuine OOD spikes
    (typically several× the noise floor) trigger reliably while thermal noise does not.
    """
    rng = np.random.default_rng(seed)
    max_stds = np.empty(n_samples)
    base_positions = atoms.get_positions()
    for i in range(n_samples):
        probe = atoms.copy()
        probe.set_positions(base_positions + rng.normal(0.0, sigma_A, base_positions.shape))
        pred = ensemble.predict(probe)
        max_stds[i] = float(pred.forces_std_per_atom.max())
    p50, p95, p99 = np.percentile(max_stds, [50, 95, 99])
    return CalibrationReport(
        n_samples=n_samples,
        sigma_A=sigma_A,
        max_std_p50=float(p50),
        max_std_p95=float(p95),
        max_std_p99=float(p99),
        suggested_threshold=float(p99 * safety_factor),
    )
