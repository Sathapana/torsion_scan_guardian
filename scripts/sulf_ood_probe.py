"""Quick OOD probe for sulfanilamide: relaxed + perturbed geometries through the
3-member ensemble. Prints std_max for each and a calibration on the relaxed minimum.
"""
import numpy as np
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import SetDihedralDeg, GetDihedralDeg

from guardian.calibration import relax_geometry, calibrate_threshold
from guardian.models.ensemble import SeedFinetuneEnsemble

SMILES = "Nc1ccc(S(=O)(=O)N)cc1"
CKPTS = [
    "runs/finetune_sulf/member_seed0/checkpoints/member_seed0_run-0.model",
    "runs/finetune_sulf/member_seed1/checkpoints/member_seed1_run-1.model",
    "runs/finetune_sulf/member_seed2/checkpoints/member_seed2_run-2.model",
]


def main():
    ens = SeedFinetuneEnsemble.from_checkpoints(CKPTS, device="cpu", dtype="float32")
    calc = ens.calcs[0]

    mol = Chem.MolFromSmiles(SMILES)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    relaxed = Atoms(symbols=symbols, positions=mol.GetConformer().GetPositions())
    rep = relax_geometry(relaxed, calc, fmax=0.05)
    print(f"[sulf] relaxed: converged={rep.converged}  steps={rep.n_steps}  E={rep.energy_final:.4f}")

    def probe(label, atoms):
        pred = ens.predict(atoms)
        e = pred.energy
        f_max = float((pred.forces ** 2).sum(axis=1).max() ** 0.5)
        s_max = float(pred.forces_std_per_atom.max())
        s_mean = float(pred.forces_std_per_atom.mean())
        print(f"  {label:42s}  E={e:9.4f}  |F|max={f_max:7.3f}  std_max={s_max:.4f}  std_mean={s_mean:.4f}")
        return s_max

    print("\n[sulf] in-distribution baselines:")
    s_ref = probe("relaxed reference", relaxed.copy())
    rng = np.random.default_rng(0)
    jitter = relaxed.copy()
    jitter.positions += rng.normal(0, 0.04, jitter.positions.shape)
    probe("thermal jitter sigma=0.04 A", jitter)

    print("\n[sulf] hard-OOD probes:")
    # N-H bond stretch
    bond_stretch = relaxed.copy()
    p = bond_stretch.get_positions()
    # find an H bonded to the first N (atom 0 in SMILES, the para-NH2)
    n_idx = 0
    h_idx = min((i for i in range(len(bond_stretch)) if bond_stretch.symbols[i] == "H"),
                key=lambda i: np.linalg.norm(p[i] - p[n_idx]))
    vec = p[h_idx] - p[n_idx]
    p[h_idx] = p[n_idx] + vec / np.linalg.norm(vec) * 1.8
    bond_stretch.set_positions(p)
    probe(f"N-H stretched to 1.8 A (atom {h_idx})", bond_stretch)

    # H-H clash
    h_list = [i for i in range(len(relaxed)) if relaxed.symbols[i] == "H"]
    clash = relaxed.copy()
    p = clash.get_positions()
    h0, h1 = h_list[0], h_list[-1]
    mid = (p[h0] + p[h1]) / 2
    d = p[h1] - p[h0]; d /= np.linalg.norm(d)
    p[h0] = mid - d * 0.35; p[h1] = mid + d * 0.35
    clash.set_positions(p)
    probe(f"H-H clash at 0.7 A (atoms {h0},{h1})", clash)

    # Isotropic 1.3x
    scaled = relaxed.copy()
    com = scaled.get_center_of_mass()
    scaled.set_positions((scaled.get_positions() - com) * 1.3 + com)
    probe("isotropic 1.3x stretch", scaled)

    # Aryl-S-N rotation (in seed set)
    patt = Chem.MolFromSmarts("[c][S](=O)(=O)[N]")
    match = mol.GetSubstructMatch(patt)
    if match:
        for angle in (0.0, 90.0):
            twist = relaxed.copy()
            mol2 = Chem.Mol(mol)
            conf2 = mol2.GetConformer()
            for k, xyz in enumerate(relaxed.get_positions()):
                conf2.SetAtomPosition(k, xyz.tolist())
            i1, i2, i3, i4 = match[0], match[1], match[4], match[2]   # c-S-N-O
            SetDihedralDeg(conf2, i1, i2, i3, i4, angle)
            twist.set_positions(conf2.GetPositions())
            probe(f"aryl-S-N dihedral = {angle:+.0f} deg (in seed)", twist)

    # Out-of-set dihedral: the para-NH2 amine rotation about C-N
    patt2 = Chem.MolFromSmarts("[c][c][N]([H])[H]")
    match2 = mol.GetSubstructMatch(patt2)
    if match2:
        i1, i2, i3, i4 = match2[0], match2[1], match2[2], match2[3]
        for angle in (90.0,):
            twist = relaxed.copy()
            mol2 = Chem.Mol(mol)
            conf2 = mol2.GetConformer()
            for k, xyz in enumerate(relaxed.get_positions()):
                conf2.SetAtomPosition(k, xyz.tolist())
            SetDihedralDeg(conf2, i1, i2, i3, i4, angle)
            twist.set_positions(conf2.GetPositions())
            probe(f"aryl-NH2 dihedral = {angle:+.0f} deg (in seed)", twist)

    print("\n[sulf] calibration on relaxed minimum (n=50, sigma=0.04 A):")
    cal = calibrate_threshold(ens, relaxed, n_samples=50, sigma_A=0.04, safety_factor=1.5)
    print(f"  p50={cal.max_std_p50:.4f}  p95={cal.max_std_p95:.4f}  "
          f"p99={cal.max_std_p99:.4f}  threshold(1.5x)={cal.suggested_threshold:.4f}")


if __name__ == "__main__":
    main()
