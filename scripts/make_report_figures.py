"""Generate figures for REPORT.md from the run artifacts produced so far.

Inputs (paths hard-coded for now):
  - runs/torsion_diag.csv               (torsion-scan diagnostic)
  - runs/20260516-230003/md.csv          (300 K live run)
  - runs/20260516-232519/md.csv          (600 K live run)
Outputs:
  - figures/torsion_scan.png
  - figures/md_timeseries.png
  - figures/std_distribution.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
THRESHOLD = 1.67

# --- Torsion scan: energy + std vs dihedral angle ---------------------------
ts = pd.read_csv("runs/torsion_diag.csv").sort_values("angle_deg")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ts.angle_deg, ts.energy_rel_eV * 23.06, "o-", color="#1f77b4")
axes[0].set_xlabel("Dihedral angle (deg)")
axes[0].set_ylabel("Relative energy (kcal/mol)")
axes[0].set_title("Aryl-CH-COOH torsion profile (MACE-OFF small)")
axes[0].grid(alpha=0.3)

axes[1].plot(ts.angle_deg, ts.max_force_std_eVperA, "o-", color="#d62728")
axes[1].axhline(THRESHOLD, color="black", linestyle="--", linewidth=1,
                label=f"trigger threshold = {THRESHOLD}")
axes[1].set_xlabel("Dihedral angle (deg)")
axes[1].set_ylabel("max per-atom force std (eV/A)")
axes[1].set_title("Input-perturbation uncertainty vs torsion angle")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, max(THRESHOLD * 1.1, ts.max_force_std_eVperA.max() * 1.1))
fig.tight_layout()
fig.savefig(FIG_DIR / "torsion_scan.png", dpi=140)
plt.close(fig)

# --- MD time series at 300 K and 600 K --------------------------------------
md300 = pd.read_csv("runs/20260516-230003/md.csv")
md600 = pd.read_csv("runs/20260516-232519/md.csv")
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
axes[0].plot(md300.step, md300.max_force_std_eVperA, color="#1f77b4", label="300 K")
axes[0].axhline(THRESHOLD, color="black", linestyle="--", linewidth=1,
                label=f"threshold = {THRESHOLD}")
axes[0].set_ylabel("max force std (eV/A)")
axes[0].set_title("Guardian uncertainty trace, 300 K (5000 steps, 0 triggers)")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, THRESHOLD * 1.1)

axes[1].plot(md600.step, md600.max_force_std_eVperA, color="#d62728", label="600 K")
axes[1].axhline(THRESHOLD, color="black", linestyle="--", linewidth=1,
                label=f"threshold = {THRESHOLD}")
axes[1].set_xlabel("MD step")
axes[1].set_ylabel("max force std (eV/A)")
axes[1].set_title("Guardian uncertainty trace, 600 K (2000 steps, 0 triggers)")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, THRESHOLD * 1.1)
fig.tight_layout()
fig.savefig(FIG_DIR / "md_timeseries.png", dpi=140)
plt.close(fig)

# --- Std distributions: calibration (taken from 300K trace as proxy) vs both runs ---
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, max(md300.max_force_std_eVperA.max(),
                          md600.max_force_std_eVperA.max(),
                          ts.max_force_std_eVperA.max()) * 1.05, 30)
ax.hist(md300.max_force_std_eVperA, bins=bins, alpha=0.55, label="MD 300 K (n=101)",
        color="#1f77b4", density=True)
ax.hist(md600.max_force_std_eVperA, bins=bins, alpha=0.55, label="MD 600 K (n=41)",
        color="#d62728", density=True)
ax.hist(ts.max_force_std_eVperA, bins=bins, alpha=0.55, label="Torsion scan (n=24)",
        color="#2ca02c", density=True)
ax.axvline(THRESHOLD, color="black", linestyle="--", linewidth=1,
           label=f"threshold = {THRESHOLD}")
ax.set_xlabel("max per-atom force std (eV/A)")
ax.set_ylabel("density")
ax.set_title("Uncertainty distributions overlap: signal does not separate from noise floor")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "std_distribution.png", dpi=140)
plt.close(fig)

print("Wrote:", *list(FIG_DIR.glob("*.png")), sep="\n  ")
