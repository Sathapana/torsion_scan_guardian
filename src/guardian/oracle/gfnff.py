from dataclasses import dataclass
import numpy as np
from ase import Atoms

from .xtb_subprocess import XTBCalculator


@dataclass
class OracleLabel:
    positions: np.ndarray   # (N, 3)
    symbols: list[str]
    energy: float           # eV
    forces: np.ndarray      # (N, 3)


def gfnff_calculator() -> XTBCalculator:
    """Default GFN-FF calculator -- subprocess-driven xtb CLI (see xtb_subprocess.py)."""
    return XTBCalculator(method="gfnff")


def label_with_gfnff(atoms: Atoms) -> OracleLabel:
    work = atoms.copy()
    work.calc = gfnff_calculator()
    E = float(work.get_potential_energy())
    F = work.get_forces()
    return OracleLabel(
        positions=work.get_positions(),
        symbols=list(work.get_chemical_symbols()),
        energy=E,
        forces=F,
    )


def perturb_along_dihedral(atoms: Atoms, dihedral_indices: tuple[int, int, int, int],
                           n_samples: int, jitter_deg: float, rng: np.random.Generator) -> list[Atoms]:
    samples = []
    for _ in range(n_samples):
        clone = atoms.copy()
        delta = float(rng.normal(0.0, jitter_deg))
        clone.set_dihedral(*dihedral_indices,
                           clone.get_dihedral(*dihedral_indices) + delta,
                           indices=list(range(len(clone))))
        samples.append(clone)
    return samples


def label_cloud_with_gfnff(atoms: Atoms, n_samples: int, jitter_A: float,
                           rng: np.random.Generator,
                           max_workers: int | None = None) -> list[OracleLabel]:
    """Label `n_samples` Gaussian-perturbed copies of `atoms` with GFN-FF, in parallel.

    Standard active-learning acquisition: a single triggered geometry under-samples
    the locally-uncertain region of phase space; a small cloud around it gives the
    fine-tune actual gradient signal in the region the Guardian flagged.

    Parallelism: `label_with_gfnff` blocks on a subprocess call to the `xtb` CLI;
    Python releases the GIL during `subprocess.run`, so a ThreadPoolExecutor lets us
    run several xtb children concurrently. `XTBCalculator` writes to a per-call
    `tempfile.TemporaryDirectory`, so parallel calls cannot collide on output files.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Pre-build the perturbed Atoms list serially (cheap, deterministic from `rng`)
    # so the rng draws don't race across threads.
    base_pos = atoms.get_positions()
    clones: list[Atoms] = []
    for _ in range(n_samples):
        clone = atoms.copy()
        clone.set_positions(base_pos + rng.normal(0.0, jitter_A, base_pos.shape))
        clones.append(clone)

    workers = max_workers if max_workers is not None else min(n_samples, 5)
    workers = max(1, workers)

    labels: list[OracleLabel] = []
    if workers == 1:
        # Stay on the simple path when there's no parallelism to exploit.
        for clone in clones:
            try:
                labels.append(label_with_gfnff(clone))
            except Exception:
                continue
        return labels

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(label_with_gfnff, clone) for clone in clones]
        for fut in futures:
            try:
                labels.append(fut.result())
            except Exception:
                # Skip failed labels (e.g. xtb non-convergence on a bad geometry).
                continue
    return labels
