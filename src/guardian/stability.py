"""Stability metrics for MD trajectories — quantify 'is the molecule still intact'.

Used post-hoc to compare base-model and Guardian-corrected MD trajectories.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np


COVALENT_RADII_A = {
    "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Br": 1.20, "I": 1.39,
}


def initial_bond_list(symbols: list[str], positions: np.ndarray, tol: float = 1.30) -> list[tuple[int, int, float]]:
    """Bond list from the first frame: pairs (i, j) where d_ij < tol * (r_i + r_j)."""
    n = len(symbols)
    bonds = []
    for i in range(n):
        ri = COVALENT_RADII_A.get(symbols[i], 0.8)
        for j in range(i + 1, n):
            rj = COVALENT_RADII_A.get(symbols[j], 0.8)
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d < tol * (ri + rj):
                bonds.append((i, j, d))
    return bonds


@dataclass
class StabilityMetrics:
    n_frames: int
    n_atoms: int
    n_bonds: int
    max_bond_stretch_ratio: float       # max over time of (d_t / d_0) for any bond
    max_pairwise_dist_A: float          # max over time of largest pairwise distance
    max_pairwise_growth_ratio: float    # final-frame max pairwise / first-frame max pairwise
    n_broken_bonds_final: int           # bonds stretched > 1.6x their initial length in the final frame
    rmsd_from_initial_A: np.ndarray     # per-frame RMSD vs frame 0 (after rigid Kabsch alignment)


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD between two coordinate sets after optimal rigid alignment (Kabsch).

    Minimises || P_centered @ R^T - Q_centered ||^2 over rotations R.
    """
    P0 = P - P.mean(axis=0)
    Q0 = Q - Q.mean(axis=0)
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R_opt = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T   # column-vector rotation
    aligned = P0 @ R_opt.T                         # apply to row vectors
    return float(np.sqrt(((aligned - Q0) ** 2).sum(axis=1).mean()))


def compute_stability(frames: Iterable, bond_break_ratio: float = 1.6) -> StabilityMetrics:
    """Walk a trajectory (iterable of ASE Atoms) and compute summary metrics.

    `bond_break_ratio` defines what counts as a "broken" bond at the final frame —
    1.6x the initial bond length is the standard threshold used in MD stability papers.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("Empty trajectory")
    first = frames[0]
    symbols = list(first.get_chemical_symbols())
    bonds = initial_bond_list(symbols, first.get_positions())

    max_stretch = 1.0
    max_pairwise = 0.0
    first_max_pairwise = float(np.linalg.norm(
        first.get_positions()[:, None, :] - first.get_positions()[None, :, :], axis=-1).max())
    rmsds = np.empty(len(frames))
    p0 = first.get_positions()
    final_broken = 0

    for t, atoms in enumerate(frames):
        p = atoms.get_positions()
        for (i, j, d0) in bonds:
            d = float(np.linalg.norm(p[i] - p[j]))
            ratio = d / d0 if d0 > 0 else 1.0
            if ratio > max_stretch:
                max_stretch = ratio
        pairwise_max = float(np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1).max())
        if pairwise_max > max_pairwise:
            max_pairwise = pairwise_max
        rmsds[t] = kabsch_rmsd(p, p0)

    # Recount broken bonds at the final frame.
    pf = frames[-1].get_positions()
    for (i, j, d0) in bonds:
        if np.linalg.norm(pf[i] - pf[j]) > bond_break_ratio * d0:
            final_broken += 1

    return StabilityMetrics(
        n_frames=len(frames),
        n_atoms=len(symbols),
        n_bonds=len(bonds),
        max_bond_stretch_ratio=max_stretch,
        max_pairwise_dist_A=max_pairwise,
        max_pairwise_growth_ratio=max_pairwise / max(first_max_pairwise, 1e-12),
        n_broken_bonds_final=final_broken,
        rmsd_from_initial_A=rmsds,
    )


def analyse_run(run_dir: str | Path) -> StabilityMetrics:
    """Convenience wrapper: load `run_dir/traj.traj` and compute metrics."""
    from ase.io.trajectory import Trajectory
    traj = Trajectory(str(Path(run_dir) / "traj.traj"))
    return compute_stability(list(traj))
