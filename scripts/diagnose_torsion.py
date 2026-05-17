"""Torsion-scan diagnostic: rotate ibuprofen's aryl-CH-COOH dihedral through 360 deg,
print energy and ensemble force-std at each angle.

Goal: does the input-perturbation ensemble's `max_force_std` respond to torsion?
If std is roughly flat across angles, input-perturbation cannot serve as the
Guardian's epistemic signal and we need Phase-2 seed-fine-tuned ensembles.
"""
import argparse
import numpy as np
from pathlib import Path
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import SetDihedralDeg, GetDihedralDeg

from guardian.config import load_config
from guardian.models.ensemble import (
    MACEOffEnsemble, SeedFinetuneEnsemble, load_mace_off, load_mace_checkpoint,
)
from guardian.calibration import relax_geometry


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--n-angles", type=int, default=24)
    p.add_argument("--out", type=Path, default=Path("runs/torsion_diag.csv"))
    p.add_argument("--checkpoints", type=Path, nargs="+", default=None,
                   help="If given, use SeedFinetuneEnsemble over these checkpoints instead of input-perturbation.")
    args = p.parse_args()

    cfg = load_config(args.config)
    print(f"[diag] molecule={cfg.molecule.smiles}")

    # Build a fresh RDKit mol so we can drive the dihedral exactly.
    mol = Chem.MolFromSmiles(cfg.molecule.smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    # Pick the aryl-CH(CH3)-C(=O)-=O dihedral (the alpha-carbonyl torsion of profens).
    patt = Chem.MolFromSmarts("[c][C;H1]([CH3])C(=O)O")
    match = mol.GetSubstructMatch(patt)
    if not match:
        raise RuntimeError("Couldn't locate aryl-CH-COOH motif in molecule.")
    i1, i2, i3, i4 = match[0], match[1], match[3], match[4]
    start = GetDihedralDeg(conf, i1, i2, i3, i4)
    print(f"[diag] dihedral atoms (1-based for ASE): {i1+1}-{i2+1}-{i3+1}-{i4+1}  "
          f"starting angle = {start:.1f} deg")

    if args.checkpoints:
        print(f"[diag] using SeedFinetuneEnsemble over {len(args.checkpoints)} checkpoints")
        ens = SeedFinetuneEnsemble.from_checkpoints(
            [str(p) for p in args.checkpoints], device=cfg.model.device, dtype=cfg.model.dtype,
        )
        # For BFGS we still need a single calculator; use member 0.
        calc = ens.calcs[0]
    else:
        print("[diag] using input-perturbation MACEOffEnsemble (Phase 1 stand-in)")
        size = cfg.model.backbone.removeprefix("mace-off-")
        calc = load_mace_off(size=size, device=cfg.model.device, dtype=cfg.model.dtype)
        ens = MACEOffEnsemble(calc=calc, n_probes=cfg.model.n_probes,
                              position_noise_A=cfg.model.position_noise_A)

    # Relax once at the starting geometry (using ASE) to give MACE a fair baseline.
    atoms0 = Atoms(symbols=symbols, positions=conf.GetPositions())
    rep = relax_geometry(atoms0, calc, fmax=0.05, max_steps=200)
    print(f"[diag] relax: converged={rep.converged}  steps={rep.n_steps}  "
          f"E0={rep.energy_final:.4f} eV")
    # Copy relaxed positions back into the RDKit conformer so torsion scans start from there.
    for k, xyz in enumerate(atoms0.get_positions()):
        conf.SetAtomPosition(k, xyz.tolist())

    angles = np.linspace(-180.0, 180.0, args.n_angles, endpoint=False)
    rows = []
    e_min = None
    for a in angles:
        SetDihedralDeg(conf, i1, i2, i3, i4, float(a))
        positions = conf.GetPositions()
        atoms = Atoms(symbols=symbols, positions=positions)
        pred = ens.predict(atoms)
        rows.append((float(a), float(pred.energy), float(pred.forces_std_per_atom.max())))
        e_min = pred.energy if e_min is None else min(e_min, pred.energy)
        print(f"  angle={a:6.1f}  E={pred.energy:.4f}  std_max={rows[-1][2]:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("angle_deg,energy_eV,energy_rel_eV,max_force_std_eVperA\n")
        for a, e, s in rows:
            f.write(f"{a},{e},{e - e_min},{s}\n")
    stds = np.array([r[2] for r in rows])
    energies_rel = np.array([r[1] - e_min for r in rows])
    print(f"[diag] std summary: min={stds.min():.3f}  max={stds.max():.3f}  "
          f"range={stds.max() - stds.min():.3f}  ratio={stds.max() / stds.min():.2f}")
    print(f"[diag] energy summary: range={energies_rel.max():.3f} eV "
          f"(~{energies_rel.max() * 23.06:.1f} kcal/mol)")
    print(f"[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
