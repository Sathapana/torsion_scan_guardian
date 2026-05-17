import numpy as np
from ase.build import molecule
from guardian.stability import compute_stability, initial_bond_list, kabsch_rmsd


def test_initial_bond_list_h2o():
    atoms = molecule("H2O")
    bonds = initial_bond_list(list(atoms.get_chemical_symbols()), atoms.get_positions())
    # 2 O-H bonds, no H-H bond at typical H2O geometry.
    assert len(bonds) == 2


def test_stable_trajectory_metrics():
    atoms = molecule("H2O")
    frames = [atoms.copy() for _ in range(20)]
    # Add tiny thermal noise — should not break any bonds.
    rng = np.random.default_rng(0)
    for f in frames[1:]:
        f.set_positions(f.get_positions() + rng.normal(0, 0.005, f.get_positions().shape))
    m = compute_stability(frames)
    assert m.n_frames == 20
    assert m.n_bonds == 2
    assert m.n_broken_bonds_final == 0
    assert m.max_bond_stretch_ratio < 1.10  # well under the 1.6x break threshold
    assert m.rmsd_from_initial_A[0] == 0.0


def test_exploding_trajectory_detected():
    atoms = molecule("H2O")
    frames = [atoms.copy()]
    # Scale positions outward over time — simulates an exploding molecule.
    for k in range(1, 10):
        f = atoms.copy()
        f.set_positions(atoms.get_positions() * (1.0 + 0.2 * k))
        frames.append(f)
    m = compute_stability(frames)
    assert m.n_broken_bonds_final == m.n_bonds   # all bonds broken
    assert m.max_bond_stretch_ratio > 2.0
    assert m.max_pairwise_growth_ratio > 2.0


def test_kabsch_rmsd_invariant_to_translation_rotation():
    rng = np.random.default_rng(0)
    P = rng.normal(0, 1, (5, 3))
    # Translate + rotate Q.
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]])
    Q = P @ R.T + np.array([2.0, -1.0, 0.5])
    assert kabsch_rmsd(P, Q) < 1e-8
