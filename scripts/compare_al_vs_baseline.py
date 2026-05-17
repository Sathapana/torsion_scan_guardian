"""Compare AL run vs baseline run on the same molecule and step budget.

Side-by-side u(t), RMSD(t), and a metrics table to figures/al_vs_baseline.png.

Usage:
  python scripts/compare_al_vs_baseline.py runs/sulf_phase5 runs/sulf_baseline
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

from guardian.stability import compute_stability


def load_run(run_dir: Path):
    md = pd.read_csv(run_dir / "md.csv")
    summary = json.loads((run_dir / "summary.json").read_text())
    traj = list(Trajectory(str(run_dir / "traj.traj")))
    metrics = compute_stability(traj)
    return dict(md=md, summary=summary, metrics=metrics, name=run_dir.name)


def main():
    al = load_run(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/sulf_phase5"))
    bl = load_run(Path(sys.argv[2] if len(sys.argv) > 2 else "runs/sulf_baseline"))
    figs = Path("figures"); figs.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # u(t)
    ax = axes[0, 0]
    ax.plot(bl["md"].step, bl["md"].max_force_std_eVperA, lw=0.8, color="#7f7f7f", label=f"baseline ({bl['name']})")
    ax.plot(al["md"].step, al["md"].max_force_std_eVperA, lw=0.8, color="#1f77b4", label=f"AL ({al['name']})")
    for c in al["summary"]["cycles"]:
        ax.axvline(c["trigger_step"], color="#d62728", alpha=0.4, lw=1)
    ax.axhline(al["summary"]["threshold"], color="black", linestyle="--", lw=1, alpha=0.5,
               label=f"AL threshold = {al['summary']['threshold']:.3f}")
    ax.set_xlabel("MD step")
    ax.set_ylabel("max u (eV/A)")
    ax.set_title("Ensemble uncertainty over MD")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # RMSD(t)
    ax = axes[0, 1]
    ax.plot(bl["metrics"].rmsd_from_initial_A, color="#7f7f7f", lw=1.0, label="baseline")
    ax.plot(al["metrics"].rmsd_from_initial_A, color="#1f77b4", lw=1.0, label="AL")
    ax.set_xlabel("Trajectory frame")
    ax.set_ylabel("RMSD vs initial (A)")
    ax.set_title("Molecular drift (Kabsch RMSD)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    # Bond stretch & metrics table
    ax = axes[1, 0]
    labels = ["baseline", "AL"]
    stretches = [bl["metrics"].max_bond_stretch_ratio, al["metrics"].max_bond_stretch_ratio]
    growths = [bl["metrics"].max_pairwise_growth_ratio, al["metrics"].max_pairwise_growth_ratio]
    rmsd_finals = [bl["metrics"].rmsd_from_initial_A[-1], al["metrics"].rmsd_from_initial_A[-1]]
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, stretches, w, label="max bond stretch ratio", color="#1f77b4")
    ax.bar(x,     growths, w, label="max pairwise growth ratio", color="#2ca02c")
    ax.bar(x + w, rmsd_finals, w, label="final RMSD (A)", color="#d62728")
    ax.axhline(1.6, color="#d62728", linestyle="--", lw=0.6, alpha=0.4, label="bond-break = 1.6")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Stability metrics")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Text panel: cycle and timing summary
    ax = axes[1, 1]
    ax.axis("off")
    text = (
        f"Molecule: {al['summary']['molecule']}\n"
        f"MD steps: {al['summary']['global_steps']} (AL) vs {bl['summary']['global_steps']} (baseline)\n\n"
        f"AL:\n"
        f"  triggers: {al['summary']['n_triggers']}\n"
        f"  labels acquired: {sum(c['labels_acquired'] for c in al['summary']['cycles'])}\n"
        f"  wall time: {al['summary']['elapsed_s']:.0f} s\n"
        f"  threshold: {al['summary']['threshold']:.3f}\n"
        f"  bonds intact at end: {al['metrics'].n_bonds - al['metrics'].n_broken_bonds_final}"
        f"/{al['metrics'].n_bonds}\n\n"
        f"Baseline:\n"
        f"  triggers: {bl['summary']['n_triggers']} (Guardian disabled)\n"
        f"  wall time: {bl['summary']['elapsed_s']:.0f} s\n"
        f"  bonds intact at end: {bl['metrics'].n_bonds - bl['metrics'].n_broken_bonds_final}"
        f"/{bl['metrics'].n_bonds}\n"
    )
    ax.text(0.0, 0.98, text, transform=ax.transAxes, fontsize=10, va="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="#fafafa", edgecolor="#cccccc"))

    fig.suptitle("AL vs baseline — same molecule, same MD step budget", y=1.00)
    fig.tight_layout()
    fig.savefig(figs / "al_vs_baseline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/al_vs_baseline.png")
    print(f"AL  final RMSD: {al['metrics'].rmsd_from_initial_A[-1]:.3f} A   "
          f"max bond stretch: {al['metrics'].max_bond_stretch_ratio:.3f}")
    print(f"BL  final RMSD: {bl['metrics'].rmsd_from_initial_A[-1]:.3f} A   "
          f"max bond stretch: {bl['metrics'].max_bond_stretch_ratio:.3f}")


if __name__ == "__main__":
    main()
