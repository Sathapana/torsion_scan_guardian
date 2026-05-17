"""Online fine-tuning for Phase 4: a single short re-train step per AL cycle.

Each call shells out to `mace_run_train` so we don't have to maintain the
training-loop state in-process. Input: a single member's current checkpoint
path + a combined-data extxyz file. Output: a path to the updated checkpoint.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import subprocess
import sys
import time

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from .replay import DataPoint, ReplayBuffer


@dataclass
class FineTuneReport:
    cycle: int
    member: int
    seconds: float
    initial_force_rmse_meV_per_A: float | None
    final_force_rmse_meV_per_A: float | None
    new_checkpoint_path: str
    accepted: bool
    reason: str = ""


def datapoint_to_atoms(dp: DataPoint) -> Atoms:
    """Turn a buffered (positions, forces, energy) record into an ASE Atoms with E/F."""
    atoms = Atoms(symbols=dp.symbols, positions=dp.positions)
    atoms.calc = SinglePointCalculator(atoms, energy=dp.energy, forces=dp.forces)
    # Touch them so the SinglePointCalculator populates info/arrays for extxyz.
    atoms.get_potential_energy()
    atoms.get_forces()
    return atoms


def write_combined_train(seed_file: Path | None, buffer: ReplayBuffer, out_path: Path) -> int:
    """Write seed_file + all DataPoints from the buffer to out_path. Returns frame count."""
    if out_path.exists():
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    if seed_file is not None and Path(seed_file).exists():
        from ase.io import read
        for f in read(str(seed_file), ":"):
            write(out_path, f, format="extxyz", append=True)
            n += 1
    for dp in buffer.points:
        write(out_path, datapoint_to_atoms(dp), format="extxyz", append=True)
        n += 1
    return n


def online_finetune_member(
    member_checkpoint: Path,
    train_file: Path,
    out_dir: Path,
    seed: int,
    epochs: int = 2,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: str = "cpu",
    valid_fraction: float = 0.10,
    regression_tol: float = 0.10,
) -> FineTuneReport:
    """Run a short fine-tune on `train_file` starting from `member_checkpoint`.

    Uses subprocess invocation of `mace_run_train` so we don't share argparse state
    with the parent process. The new model is at `out_dir / "checkpoints" / <name>_run-<seed>.model`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"member_seed{seed}_cycle"
    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", name,
        "--train_file", str(train_file),
        "--valid_fraction", str(valid_fraction),
        "--foundation_model", str(member_checkpoint),
        "--multiheads_finetuning", "False",
        "--energy_key", "energy",
        "--forces_key", "forces",
        "--E0s", "average",
        "--max_num_epochs", str(epochs),
        "--lr", str(lr),
        "--batch_size", str(batch_size),
        "--seed", str(seed),
        "--device", device,
        "--default_dtype", "float32",
        "--loss", "weighted",
        "--forces_weight", "10.0",
        "--energy_weight", "1.0",
        "--model_dir", str(out_dir),
        "--checkpoints_dir", str(out_dir / "checkpoints"),
        "--results_dir", str(out_dir / "results"),
        "--log_dir", str(out_dir / "logs"),
        "--save_cpu",
    ]
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8")
    elapsed = time.time() - t0
    if res.returncode != 0:
        log_path = out_dir / "stderr.log"
        log_path.write_text(res.stderr or "")
        raise RuntimeError(f"online finetune failed (exit {res.returncode}); stderr in {log_path}")

    # Locate the new checkpoint file. mace writes "<name>_run-<seed>.model" under checkpoints/.
    ckpt_dir = out_dir / "checkpoints"
    new_path = ckpt_dir / f"{name}_run-{seed}.model"
    if not new_path.exists():
        candidates = sorted(ckpt_dir.glob("*.model"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No .model written under {ckpt_dir}")
        new_path = candidates[0]

    # Parse initial (= pre-FT val MAE of the starting checkpoint) and final epoch RMSE_F from stdout.
    initial_rmse, final_rmse = None, None
    lines = (res.stdout or "").splitlines()
    for ln in lines:
        if "Initial:" in ln and "RMSE_F=" in ln:
            try:
                initial_rmse = float(ln.split("RMSE_F=")[1].split("meV")[0].strip())
            except Exception:
                pass
            break
    for ln in lines[::-1]:
        if "RMSE_F=" in ln and "Epoch" in ln:
            try:
                final_rmse = float(ln.split("RMSE_F=")[1].split("meV")[0].strip())
            except Exception:
                pass
            break

    # Safeguard: if val force RMSE regressed by more than `regression_tol` relative to the
    # input checkpoint, refuse the update -- keep the old weights for this member this cycle.
    accepted, reason = True, "improved or within tolerance"
    if initial_rmse is not None and final_rmse is not None:
        if final_rmse > initial_rmse * (1.0 + regression_tol):
            accepted = False
            reason = (f"final RMSE_F={final_rmse:.1f} > "
                      f"initial {initial_rmse:.1f} x (1 + {regression_tol:.2f})"
                      f" = {initial_rmse * (1 + regression_tol):.1f}; reverting")
            new_path = member_checkpoint   # keep the prior checkpoint

    return FineTuneReport(
        cycle=-1, member=seed, seconds=elapsed,
        initial_force_rmse_meV_per_A=initial_rmse,
        final_force_rmse_meV_per_A=final_rmse,
        new_checkpoint_path=str(new_path),
        accepted=accepted, reason=reason,
    )
