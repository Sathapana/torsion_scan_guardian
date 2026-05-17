"""Build a seed dataset of ibuprofen conformers labelled by GFN-FF.

Sources of diversity (all labelled by GFN-FF via tblite):
  - ETKDG random embeddings (different RDKit seeds + MMFF pre-opt)
  - GFN-FF Langevin MD at multiple temperatures, sampled at fixed cadence
  - Frozen torsion scans driving named rotatable bonds through 360 deg

Output: extended-xyz file at `data/seed/ibuprofen_seed.xyz`, one frame per
conformer. Each frame carries `energy` and per-atom `forces` (ASE's
SinglePointCalculator format), directly consumable by `mace-torch` training.
"""
import argparse
from pathlib import Path
import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import SetDihedralDeg

from guardian.config import load_config


def gfnff_calc():
    from guardian.oracle.xtb_subprocess import XTBCalculator
    return XTBCalculator(method="gfnff")


def append_frame(out_path: Path, atoms: Atoms, source: str) -> None:
    """Force E/F evaluation, tag the frame, append to extxyz."""
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.info["source"] = source
    write(out_path, atoms, format="extxyz", append=True)


def collect_etkdg(out_path: Path, smiles: str, n: int) -> int:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    ok = 0
    for seed in range(n):
        m = Chem.Mol(mol)
        if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
            print(f"  [etkdg] embed failed seed={seed}")
            continue
        AllChem.MMFFOptimizeMolecule(m)
        atoms = Atoms(symbols=symbols, positions=m.GetConformer().GetPositions())
        atoms.calc = gfnff_calc()
        try:
            append_frame(out_path, atoms, f"etkdg-seed{seed}")
            ok += 1
        except Exception as e:
            print(f"  [etkdg] GFN-FF failed seed={seed}: {e!r}")
    return ok


def collect_md(out_path: Path, smiles: str, temperature_K: float,
               n_steps: int, sample_every: int) -> int:
    from guardian.io.structures import smiles_to_atoms
    atoms = smiles_to_atoms(smiles, seed=0)
    atoms.calc = gfnff_calc()
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    dyn = Langevin(atoms, timestep=0.5 * units.fs,
                   temperature_K=temperature_K, friction=0.01)
    ok = 0
    for step in range(1, n_steps + 1):
        try:
            dyn.run(1)
        except Exception as e:
            print(f"  [md {temperature_K:.0f}K] step {step} failed: {e!r}")
            break
        if step % sample_every == 0:
            try:
                append_frame(out_path, atoms, f"md-{int(temperature_K)}K-step{step}")
                ok += 1
            except Exception as e:
                print(f"  [md {temperature_K:.0f}K] write failed step {step}: {e!r}")
    return ok


# (name, SMARTS pattern, indices_into_match for the 4 dihedral atoms)
# Targets are tried in order -- those without a SMARTS match in the molecule are skipped.
TORSION_TARGETS = [
    # ibuprofen-class
    ("alpha-carbonyl", "[c][C;H1]([CH3])C(=O)O", (0, 1, 3, 4)),
    ("isobutyl-c",     "[c][CH2][CH]([CH3])[CH3]", (0, 1, 2, 3)),
    # sulfonamide-class (e.g. sulfanilamide)
    ("aryl-S-N-O",     "[c][S](=O)(=O)[N]", (0, 1, 4, 2)),   # c-S-N rotation (using one =O as anchor)
    ("aryl-N-arom",    "[c][c][N]([H])[H]", (0, 1, 2, 3)),   # aryl-NH2 rotation
]


def collect_torsion(out_path: Path, smiles: str, n_angles: int) -> int:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    angles = np.linspace(-180.0, 180.0, n_angles, endpoint=False)
    ok = 0
    for name, smarts, idx_tuple in TORSION_TARGETS:
        patt = Chem.MolFromSmarts(smarts)
        match = mol.GetSubstructMatch(patt)
        if not match:
            print(f"  [torsion] no SMARTS match for {name}, skipping")
            continue
        i1, i2, i3, i4 = (match[k] for k in idx_tuple)
        for a in angles:
            SetDihedralDeg(conf, i1, i2, i3, i4, float(a))
            atoms = Atoms(symbols=symbols, positions=conf.GetPositions())
            atoms.calc = gfnff_calc()
            try:
                append_frame(out_path, atoms, f"torsion-{name}-{a:+.0f}deg")
                ok += 1
            except Exception as e:
                print(f"  [torsion] {name} {a:.0f} failed: {e!r}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--smiles", type=str, default=None,
                   help="Override molecule.smiles from the config")
    p.add_argument("--out", type=Path, default=Path("data/seed/ibuprofen_seed.xyz"))
    p.add_argument("--n-etkdg", type=int, default=10)
    p.add_argument("--md-steps", type=int, default=1000)
    p.add_argument("--md-sample-every", type=int, default=50)
    p.add_argument("--md-temperatures", type=float, nargs="+", default=[300.0, 600.0])
    p.add_argument("--torsion-angles", type=int, default=12)
    args = p.parse_args()

    cfg = load_config(args.config)
    smiles = args.smiles if args.smiles is not None else cfg.molecule.smiles
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    print(f"[seed] molecule: {smiles}")
    print(f"[seed] writing to: {args.out}")

    n_etkdg = collect_etkdg(args.out, smiles, args.n_etkdg)
    print(f"[seed] etkdg: {n_etkdg}/{args.n_etkdg} frames")

    n_md_total = 0
    for T in args.md_temperatures:
        n = collect_md(args.out, smiles, T, args.md_steps, args.md_sample_every)
        print(f"[seed] md {int(T)}K ({args.md_steps} steps, every {args.md_sample_every}): {n} frames")
        n_md_total += n

    n_tors = collect_torsion(args.out, smiles, args.torsion_angles)
    print(f"[seed] torsion: {n_tors} frames")

    frames = read(args.out, ":")
    energies = np.array([f.get_potential_energy() for f in frames])
    fmaxes = np.array([np.linalg.norm(f.get_forces(), axis=1).max() for f in frames])
    print(f"[seed] total: {len(frames)} frames")
    print(f"[seed] energy: min={energies.min():.4f}  max={energies.max():.4f}  "
          f"span={energies.max()-energies.min():.3f} eV "
          f"(~{(energies.max()-energies.min())*23.06:.1f} kcal/mol)")
    print(f"[seed] |F|max distribution: median={np.median(fmaxes):.3f}  "
          f"p99={np.percentile(fmaxes, 99):.3f}  max={fmaxes.max():.3f} eV/A")


if __name__ == "__main__":
    main()
