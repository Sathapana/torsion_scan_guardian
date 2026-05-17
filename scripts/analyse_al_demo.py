"""Post-run analysis of a Phase-4/5 active-learning demo.

Reads a run directory containing md.csv, traj.traj, summary.json, and per-cycle
sub-directories, then produces:
  - figures/al_timeline.png       : u(t) over MD with triggers marked + cycle bands
  - figures/al_stability.png      : per-cycle stability metrics (max bond stretch, RMSD)
  - figures/al_buffer_growth.png  : training-set size vs cycle

Usage:
  python scripts/analyse_al_demo.py runs/sulf_phase5
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory

from guardian.stability import compute_stability, initial_bond_list


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/sulf_phase5")
    figs = Path("figures"); figs.mkdir(exist_ok=True)
    out_prefix = run_dir.name

    md = pd.read_csv(run_dir / "md.csv")
    summary = json.loads((run_dir / "summary.json").read_text())
    cycles = summary["cycles"]
    threshold = summary["threshold"]
    print(f"[ana] run_dir={run_dir}  steps={summary['global_steps']}  "
          f"triggers={summary['n_triggers']}  wall={summary['elapsed_s']:.0f}s")

    # ---- Figure 1: timeline of u with triggers ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(md.step, md.max_force_std_eVperA, lw=0.8, color="#1f77b4", label="MD u")
    ax.axhline(threshold, color="black", linestyle="--", lw=1, alpha=0.6,
               label=f"threshold = {threshold:.3f}")
    for c in cycles:
        ax.axvline(c["trigger_step"], color="#d62728", alpha=0.5, lw=1)
        ax.annotate(f"c{c['cycle']}", xy=(c["trigger_step"], c["trigger_std"]),
                    xytext=(0, 8), textcoords="offset points", fontsize=8,
                    ha="center", color="#d62728")
    ax.set_xlabel("MD step")
    ax.set_ylabel("max per-atom force std u (eV/A)")
    ax.set_title(f"Active-learning timeline — {summary['molecule']}  "
                 f"({summary['n_triggers']} cycles in {summary['elapsed_s']:.0f}s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / f"{out_prefix}_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: stability metrics ----
    traj = list(Trajectory(str(run_dir / "traj.traj")))
    metrics = compute_stability(traj)
    print(f"[ana] traj frames={metrics.n_frames}  bonds={metrics.n_bonds}  "
          f"max_bond_stretch={metrics.max_bond_stretch_ratio:.3f}  "
          f"broken_bonds_final={metrics.n_broken_bonds_final}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(metrics.rmsd_from_initial_A, color="#2ca02c", lw=1)
    for c in cycles:
        # md.csv has rows every log_every steps; map step -> traj index roughly
        idx = int(c["trigger_step"] / max(summary["global_steps"], 1) * (len(traj) - 1))
        axes[0].axvline(idx, color="#d62728", alpha=0.4, lw=1)
    axes[0].set_xlabel("Trajectory frame")
    axes[0].set_ylabel("RMSD vs frame 0 (A)")
    axes[0].set_title("Molecular drift")
    axes[0].grid(alpha=0.3)

    axes[1].axhline(1.0, color="gray", lw=0.5)
    axes[1].axhline(1.6, color="#d62728", lw=0.5, linestyle="--",
                    label="bond-break threshold")
    axes[1].text(0.02, 0.92, f"max bond stretch ratio = {metrics.max_bond_stretch_ratio:.3f}\n"
                 f"broken bonds at final frame = {metrics.n_broken_bonds_final}/{metrics.n_bonds}",
                 transform=axes[1].transAxes, fontsize=10, va="top",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    axes[1].plot([], [])  # placeholder
    axes[1].set_xticks([]); axes[1].set_yticks([])
    axes[1].set_title("Stability summary")

    fig.suptitle(f"Stability — {summary['molecule']}", y=1.02)
    fig.tight_layout()
    fig.savefig(figs / f"{out_prefix}_stability.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: training-set size + per-cycle FT info from cycle dirs ----
    train_sizes, cycle_ids = [], []
    for c in cycles:
        cdir = run_dir / f"cycle_{c['cycle']:03d}"
        tfile = cdir / "train.xyz"
        if tfile.exists():
            # quick count: number of lines starting with a digit followed by space (extxyz frame headers)
            n = 0
            with open(tfile) as fh:
                for ln in fh:
                    s = ln.strip()
                    if s.isdigit():
                        n += 1
            train_sizes.append(n)
            cycle_ids.append(c["cycle"])
    if train_sizes:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(cycle_ids, train_sizes, "o-", color="#9467bd")
        ax.set_xlabel("AL cycle")
        ax.set_ylabel("training-set size (frames)")
        ax.set_title("Replay buffer growth across cycles")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / f"{out_prefix}_buffer.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    print(f"[ana] wrote figures: {out_prefix}_timeline.png  {out_prefix}_stability.png  "
          f"{out_prefix}_buffer.png")


if __name__ == "__main__":
    main()
