import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms


def torsion_scan_energies(atoms: Atoms, dihedral_indices: tuple[int, int, int, int],
                          calc_fn, n_points: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Rotate one dihedral through 360° and collect energies from `calc_fn(atoms) -> energy`."""
    angles = np.linspace(-180, 180, n_points, endpoint=False)
    energies = np.empty(n_points)
    base_angle = atoms.get_dihedral(*dihedral_indices)
    indices = list(range(len(atoms)))
    work = atoms.copy()
    for i, a in enumerate(angles):
        work.set_dihedral(*dihedral_indices, base_angle + a, indices=indices)
        energies[i] = float(calc_fn(work))
    return angles, energies - energies.min()


def plot_scan_comparison(angles, curves: dict[str, np.ndarray], title: str = "") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, e in curves.items():
        ax.plot(angles, e, label=label)
    ax.set_xlabel("Dihedral angle (deg)")
    ax.set_ylabel("Relative energy (eV)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
