"""Batch active-learning sweep across `data/molecule_library/candidates.csv`.

For each requested molecule:
  1. Ensure a per-molecule config exists (auto-create from the CSV row if not).
  2. Build the GFN-FF seed dataset.
  3. Fine-tune 3 MACE-OFF members on the seed.
  4. Calibrate the uncertainty threshold from the relaxed minimum.
  5. Run a *baseline* MD (Guardian disabled — threshold = 999, max_triggers = 1).
  6. Run an *AL* MD (Guardian active — cloud-5, safeguarded fine-tunes, calibrated threshold).
  7. Compute post-hoc stability metrics on both trajectories.

A running summary is written to `runs/sweep/sweep_summary.csv` after every molecule
so a Colab session interruption doesn't lose work. Per-molecule artifacts land
under `runs/sweep/<name>/{baseline,al}/`.

Usage examples:

    # Process all `todo` molecules (default)
    python scripts/sweep_molecules.py

    # Specific molecules by name (from the `name` column of candidates.csv)
    python scripts/sweep_molecules.py --molecules glycine_zwitterion sulfonyl_chloride

    # All candidates (todo + candidate phases)
    python scripts/sweep_molecules.py --phase-filter todo candidate

    # Skip AL phase, baseline only
    python scripts/sweep_molecules.py --baseline-only

    # Override per-molecule MD step budget (default 4000)
    python scripts/sweep_molecules.py --steps 2000

Outputs:
    runs/sweep/sweep_summary.csv      aggregate table (one row per molecule)
    runs/sweep/<name>/baseline/...    baseline traj, md.csv, summary.json
    runs/sweep/<name>/al/...          AL traj, md.csv, summary.json, cycle_NNN/...
    config/molecules/<name>.yaml      auto-created if missing
    data/seed/<name>_seed.xyz         auto-built if missing
    runs/finetune_<name>/...          auto-built if missing
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CANDIDATES_CSV = REPO_ROOT / "data" / "molecule_library" / "candidates.csv"
CONFIG_DIR = REPO_ROOT / "config" / "molecules"
SEED_DIR = REPO_ROOT / "data" / "seed"
SWEEP_DIR = REPO_ROOT / "runs" / "sweep"


# ---------- helpers ----------

def load_candidates() -> list[dict]:
    with open(CANDIDATES_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_rows(rows: list[dict], names: list[str] | None,
                phases: list[str] | None) -> list[dict]:
    if names:
        present = {r["name"] for r in rows}
        missing = set(names) - present
        if missing:
            raise ValueError(f"Unknown molecule names in candidates.csv: {sorted(missing)}")
        return [r for r in rows if r["name"] in names]
    if phases:
        return [r for r in rows if r["phase"] in phases]
    return rows


def ensure_config(row: dict, default_steps: int, default_temperature: float) -> Path:
    """Return path to a per-molecule YAML, creating a minimal one from the CSV row if missing."""
    path = CONFIG_DIR / f"{row['name']}.yaml"
    if path.exists():
        return path
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {
        "molecule": {
            "smiles": row["smiles"],
            "charge": int(row["charge"]),
            "multiplicity": int(row["multiplicity"]),
        },
        "md": {
            "temperature_K": default_temperature,
            "total_steps": default_steps,
            "log_every": 50,
            "checkpoint_every": 100,
            "timestep_fs": 0.5,
            "friction": 0.01,
        },
        "model": {
            "backbone": "mace-off-small",
            "ensemble_mode": "seed-fine-tune",
            "n_probes": 3,
            "position_noise_A": 0.005,
            "device": "cuda",
            "dtype": "float32",
        },
        "uncertainty": {
            "threshold": 0.05,           # placeholder; --calibrate updates this in run-time
            "warmup_steps": 100,
            "metric": "max_atom_force_std",
        },
        "io": {
            "run_dir": f"runs/{row['name']}/",
            "oracle_cache": "data/oracle_cache/",
        },
        "oracle": {
            "method": "gfn-ff",
            "perturb_cloud": {"enabled": True, "n_samples": 5, "dihedral_jitter_deg": 15.0},
        },
        "training": {
            "lr": 1.0e-4, "epochs_per_cycle": 5, "batch_size": 8, "grad_clip": 10.0,
            "ema_decay": 0.999, "val_force_mae_regression_tol": 0.10,
        },
        "wandb": {"enabled": False, "project": "torsion-scan-guardian"},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"[sweep] created config {path.relative_to(REPO_ROOT)} from CSV row")
    return path


def run_subprocess(cmd: list[str], label: str, env: dict | None = None,
                   echo: bool = True) -> tuple[int, float, str, str]:
    """Run, return (returncode, elapsed_seconds, stdout, stderr). Streams stdout on echo=True."""
    t0 = time.time()
    full_env = dict(os.environ)
    full_env.setdefault("PYTHONIOENCODING", "utf-8")
    full_env.setdefault("MPLBACKEND", "Agg")
    if env:
        full_env.update(env)
    if echo:
        print(f"[sweep] [{label}] $ {' '.join(cmd[:6])} ...")
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", env=full_env, cwd=REPO_ROOT)
    elapsed = time.time() - t0
    if res.returncode != 0 and echo:
        print(f"[sweep] [{label}] FAILED (exit {res.returncode}) in {elapsed:.0f}s")
        print(res.stderr[-1500:])
    elif echo:
        print(f"[sweep] [{label}] ok in {elapsed:.0f}s")
    return res.returncode, elapsed, res.stdout, res.stderr


# ---------- per-stage wrappers ----------

def build_seed(row: dict, seed_path: Path) -> int:
    if seed_path.exists():
        print(f"[sweep] seed exists at {seed_path.relative_to(REPO_ROOT)} — skip build")
        return 0
    return run_subprocess([
        sys.executable, "scripts/build_seed_dataset.py",
        "--smiles", row["smiles"], "--out", str(seed_path),
    ], f"seed-{row['name']}")[0]


def finetune_ensemble(row: dict, seed_path: Path, ft_root: Path, device: str) -> int:
    expected = [ft_root / f"member_seed{s}" / "checkpoints" / f"member_seed{s}_run-{s}.model"
                for s in (0, 1, 2)]
    if all(p.exists() for p in expected):
        print(f"[sweep] all 3 members exist at {ft_root.relative_to(REPO_ROOT)} — skip fine-tune")
        return 0
    for s in (0, 1, 2):
        rc, _, _, _ = run_subprocess([
            sys.executable, "scripts/finetune_member.py",
            "--seed", str(s), "--epochs", "5", "--lr", "5e-4",
            "--train-file", str(seed_path), "--out-root", str(ft_root),
            "--device", device,
        ], f"finetune-{row['name']}-seed{s}")
        if rc != 0:
            return rc
    return 0


def member_paths(ft_root: Path) -> list[Path]:
    return sorted(ft_root.glob("member_seed*/checkpoints/*.model"))


def calibrate(row: dict, cfg_path: Path, ckpts: list[Path]) -> float | None:
    rc, _, stdout, _ = run_subprocess([
        sys.executable, "-m", "guardian.cli",
        "--config", str(cfg_path), "--smiles", row["smiles"],
        "--calibrate", "--calibrate-samples", "30",
        "--checkpoints", *map(str, ckpts),
    ], f"calibrate-{row['name']}")
    if rc != 0:
        return None
    for ln in stdout.splitlines():
        if "suggested uncertainty.threshold" in ln:
            try:
                return float(ln.split("=")[1].split("eV")[0].strip())
            except Exception:
                pass
    return None


def run_md(row: dict, cfg_path: Path, ckpts: list[Path], threshold: float,
           run_dir: Path, steps: int, temperature: float, online_ft: bool,
           cloud_size: int, max_triggers: int) -> tuple[int, float]:
    cmd = [
        sys.executable, "-m", "guardian.cli",
        "--config", str(cfg_path), "--smiles", row["smiles"],
        "--steps", str(steps), "--temperature", str(temperature),
        "--threshold", str(threshold),
        "--max-triggers", str(max_triggers), "--cooldown-steps", "200",
        "--checkpoints", *map(str, ckpts),
        "--run-dir", str(run_dir),
    ]
    if online_ft:
        seed_path = SEED_DIR / f"{row['name']}_seed.xyz"
        cmd += [
            "--online-finetune", "--seed-data-file", str(seed_path),
            "--cloud-size", str(cloud_size), "--cloud-jitter", "0.05",
            "--finetune-epochs", "2", "--finetune-lr", "1e-4",
            "--ft-regression-tol", "0.10",
        ]
    label = f"md-{row['name']}-{'al' if online_ft else 'baseline'}"
    rc, elapsed, _, _ = run_subprocess(cmd, label)
    return rc, elapsed


def stability_summary(run_dir: Path) -> dict:
    from ase.io.trajectory import Trajectory
    from guardian.stability import compute_stability
    traj_path = run_dir / "traj.traj"
    if not traj_path.exists():
        return {}
    metrics = compute_stability(list(Trajectory(str(traj_path))))
    return {
        "n_frames": metrics.n_frames,
        "n_bonds": metrics.n_bonds,
        "max_bond_stretch": round(metrics.max_bond_stretch_ratio, 3),
        "max_pairwise_growth": round(metrics.max_pairwise_growth_ratio, 3),
        "broken_bonds_final": metrics.n_broken_bonds_final,
        "final_rmsd_A": round(float(metrics.rmsd_from_initial_A[-1]), 3),
    }


# ---------- main ----------

def process_one(row: dict, *, steps: int, temperature: float, device: str,
                cloud_size: int, max_triggers: int, baseline_only: bool,
                threshold_override: float | None) -> dict:
    name = row["name"]
    print(f"\n========================= {name} =========================")
    result: dict = {
        "name": name, "smiles": row["smiles"],
        "heavy_atoms": int(row["heavy_atoms"]),
        "phase_in_csv": row["phase"],
    }
    t_total = time.time()

    cfg_path = ensure_config(row, default_steps=steps, default_temperature=temperature)
    seed_path = SEED_DIR / f"{name}_seed.xyz"
    if build_seed(row, seed_path) != 0:
        result["status"] = "seed-failed"
        return result

    ft_root = REPO_ROOT / "runs" / f"finetune_{name}"
    if finetune_ensemble(row, seed_path, ft_root, device) != 0:
        result["status"] = "finetune-failed"
        return result
    ckpts = member_paths(ft_root)
    if len(ckpts) < 3:
        result["status"] = f"only-{len(ckpts)}-checkpoints-found"
        return result

    if threshold_override is not None:
        thr = threshold_override
        print(f"[sweep] {name}: using threshold override {thr}")
    else:
        thr = calibrate(row, cfg_path, ckpts) or 0.05
    result["calibrated_threshold"] = round(thr, 4)

    sweep_mol_dir = SWEEP_DIR / name
    bl_dir = sweep_mol_dir / "baseline"
    rc, bl_time = run_md(row, cfg_path, ckpts, threshold=999.0,
                         run_dir=bl_dir, steps=steps, temperature=temperature,
                         online_ft=False, cloud_size=cloud_size, max_triggers=1)
    result["baseline_wall_s"] = round(bl_time, 1)
    if rc == 0:
        result.update({f"baseline_{k}": v for k, v in stability_summary(bl_dir).items()})
    else:
        result["baseline_status"] = "md-failed"

    if not baseline_only:
        al_dir = sweep_mol_dir / "al"
        rc, al_time = run_md(row, cfg_path, ckpts, threshold=thr,
                             run_dir=al_dir, steps=steps, temperature=temperature,
                             online_ft=True, cloud_size=cloud_size, max_triggers=max_triggers)
        result["al_wall_s"] = round(al_time, 1)
        if rc == 0:
            summary_path = al_dir / "summary.json"
            if summary_path.exists():
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                result["al_triggers"] = s["n_triggers"]
                result["al_labels"] = sum(c["labels_acquired"] for c in s["cycles"])
            result.update({f"al_{k}": v for k, v in stability_summary(al_dir).items()})
        else:
            result["al_status"] = "md-failed"

    result["total_wall_s"] = round(time.time() - t_total, 1)
    result["status"] = "ok"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--molecules", nargs="*", default=None,
                   help="Specific molecule names from candidates.csv")
    p.add_argument("--phase-filter", nargs="*", default=["todo"],
                   help="If --molecules not given, filter by these phase tags (default: todo)")
    p.add_argument("--steps", type=int, default=4000, help="MD step budget per run")
    p.add_argument("--temperature", type=float, default=300.0, help="MD temperature in K")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--cloud-size", type=int, default=5)
    p.add_argument("--max-triggers", type=int, default=5)
    p.add_argument("--baseline-only", action="store_true",
                   help="Skip the AL phase; useful for first-pass screening")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override calibration; same threshold for all molecules")
    args = p.parse_args()

    rows = filter_rows(load_candidates(), args.molecules, args.phase_filter)
    print(f"[sweep] processing {len(rows)} molecule(s): {[r['name'] for r in rows]}")
    if not rows:
        print("[sweep] nothing to do; check --molecules / --phase-filter")
        return

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = SWEEP_DIR / "sweep_summary.csv"
    results: list[dict] = []
    sweep_t0 = time.time()

    for row in rows:
        try:
            result = process_one(
                row, steps=args.steps, temperature=args.temperature,
                device=args.device, cloud_size=args.cloud_size,
                max_triggers=args.max_triggers, baseline_only=args.baseline_only,
                threshold_override=args.threshold,
            )
        except Exception as e:
            print(f"[sweep] EXCEPTION on {row['name']}: {e!r}")
            result = {"name": row["name"], "status": f"exception: {e}"}
        results.append(result)
        # Persist after every molecule so a Colab disconnect doesn't lose progress.
        pd.DataFrame(results).to_csv(summary_csv, index=False)
        print(f"[sweep] progress saved: {summary_csv.relative_to(REPO_ROOT)}")

    print(f"\n========================= sweep complete =========================")
    print(f"total wall time: {(time.time() - sweep_t0) / 60:.1f} min")
    print(f"summary CSV:     {summary_csv.relative_to(REPO_ROOT)}")
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
