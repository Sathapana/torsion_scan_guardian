"""Directly probe the ensemble on deliberately-OOD geometries.

Hypothesis discriminator: if the seed-fine-tuned ensemble's force std rises
sharply on geometries clearly outside the training distribution, the ensemble
is working but MD just doesn't reach OOD regions on this molecule. If std
stays flat, the seeds collapsed onto the foundation and the ensemble can't
discriminate.

Probes:
  - relaxed (in-distribution reference)
  - one C-H bond stretched to 1.8 A (~1.6x normal)
  - two H atoms compressed to 0.7 A apart (steric clash)
  - molecule scaled isotropically by 1.3x (all bonds stretched)
  - aryl-CH-COOH dihedral driven to 0 deg (was in seed; in-distribution)
  - COOH OH dihedral (=O-C-O-H) driven to 90 deg (NOT in seed)
"""
import numpy as np
from pathlib import Path
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import SetDihedralDeg

from guardian.config import load_config
from guardian.calibration import relax_geometry
from guardian.models.ensemble import SeedFinetuneEnsemble


import sys
CKPTS_DEFAULT = [
    "runs/finetune_v2/member_seed0/checkpoints/member_seed0_run-0.model",
    "runs/finetune_v2/member_seed1/checkpoints/member_seed1_run-1.model",
    "runs/finetune_v2/member_seed2/checkpoints/member_seed2_run-2.model",
]
CKPTS = sys.argv[1:] if len(sys.argv) > 1 else CKPTS_DEFAULT


def main():
    cfg = load_config(Path("config/default.yaml"))
    ens = SeedFinetuneEnsemble.from_checkpoints(CKPTS, device="cpu", dtype="float32")
    calc = ens.calcs[0]

    # Build relaxed reference geometry.
    mol = Chem.MolFromSmiles(cfg.molecule.smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    relaxed = Atoms(symbols=symbols, positions=mol.GetConformer().GetPositions())
    relax_geometry(relaxed, calc, fmax=0.05)
    print(f"[ood] relaxed reference: E={relaxed.get_potential_energy():.4f} eV")

    def probe(label, atoms):
        pred = ens.predict(atoms)
        e = pred.energy
        f_max = float((pred.forces ** 2).sum(axis=1).max() ** 0.5)
        s_max = float(pred.forces_std_per_atom.max())
        print(f"  {label:42s}  E={e:11.4f}  |F|max={f_max:7.3f}  std_max={s_max:.4f}")
        return s_max

    print("\n[ood] in-distribution baselines:")
    s_ref = probe("relaxed reference", relaxed.copy())

    # Mild thermal jitter (300 K - ish).
    rng = np.random.default_rng(0)
    jitter = relaxed.copy()
    jitter.positions += rng.normal(0, 0.04, jitter.positions.shape)
    probe("thermal jitter sigma=0.04 A", jitter)

    print("\n[ood] deliberate OOD probes:")
    # C-H stretch to 1.8 A
    bond_stretch = relaxed.copy()
    p = bond_stretch.get_positions()
    # find an H attached to atom 1 (any non-H heavy atom). Use first H found.
    heavy_idx = 1
    h_neighbors = [i for i in range(len(bond_stretch)) if bond_stretch.symbols[i] == "H"]
    target_h = min(h_neighbors,
                   key=lambda i: np.linalg.norm(p[i] - p[heavy_idx]))
    vec = p[target_h] - p[heavy_idx]
    new_len = 1.8
    p[target_h] = p[heavy_idx] + vec / np.linalg.norm(vec) * new_len
    bond_stretch.set_positions(p)
    probe(f"C-H stretched to 1.8 A (atom {target_h})", bond_stretch)

    # Steric clash: pull two distant H's to 0.7 A apart.
    clash = relaxed.copy()
    p = clash.get_positions()
    h0, h1 = h_neighbors[0], h_neighbors[-1]
    midpoint = (p[h0] + p[h1]) / 2
    direction = p[h1] - p[h0]
    direction /= np.linalg.norm(direction)
    p[h0] = midpoint - direction * 0.35
    p[h1] = midpoint + direction * 0.35
    clash.set_positions(p)
    probe(f"H-H clash at 0.7 A (atoms {h0},{h1})", clash)

    # Isotropic 1.3x scale (all bonds stretched).
    scaled = relaxed.copy()
    com = scaled.get_center_of_mass()
    scaled.set_positions((scaled.get_positions() - com) * 1.3 + com)
    probe("isotropic 1.3x stretch", scaled)

    # COOH OH torsion (=O - C - O - H), NOT in seed set.
    patt = Chem.MolFromSmarts("C(=O)O[H]")
    match = mol.GetSubstructMatch(patt)
    if match:
        # atoms: C, =O, O(H), H
        i1, i2, i3, i4 = match[1], match[0], match[2], match[3]
        for angle in [0.0, 90.0]:
            twist = relaxed.copy()
            mol_copy = Chem.Mol(mol)
            mol_copy_conf = mol_copy.GetConformer()
            # use the relaxed positions as the starting conformer
            for k, xyz in enumerate(relaxed.get_positions()):
                mol_copy_conf.SetAtomPosition(k, xyz.tolist())
            SetDihedralDeg(mol_copy_conf, i1, i2, i3, i4, angle)
            twist.set_positions(mol_copy_conf.GetPositions())
            probe(f"COOH OH dihedral = {angle:+.0f} deg (OOD)", twist)
    else:
        print("  [ood] SMARTS for COOH-OH dihedral did not match")

    # alpha-carbonyl dihedral at 180 deg (IN seed -- control)
    patt = Chem.MolFromSmarts("[c][C;H1]([CH3])C(=O)O")
    match = mol.GetSubstructMatch(patt)
    if match:
        i1, i2, i3, i4 = match[0], match[1], match[3], match[4]
        twist = relaxed.copy()
        mol_copy = Chem.Mol(mol)
        mol_copy_conf = mol_copy.GetConformer()
        for k, xyz in enumerate(relaxed.get_positions()):
            mol_copy_conf.SetAtomPosition(k, xyz.tolist())
        SetDihedralDeg(mol_copy_conf, i1, i2, i3, i4, 180.0)
        twist.set_positions(mol_copy_conf.GetPositions())
        probe("alpha-carbonyl dihedral = 180 (IN seed)", twist)


if __name__ == "__main__":
    main()
