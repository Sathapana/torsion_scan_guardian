"""Phase 2 figures for REPORT.md: ensemble comparison, OOD bar chart, std distributions."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path("figures"); FIG.mkdir(exist_ok=True)

# ---------- 1. Torsion-scan comparison: Phase 1 vs Phase 2 ----------
p1 = pd.read_csv("runs/torsion_diag.csv").sort_values("angle_deg")
p2 = pd.read_csv("runs/torsion_diag_phase2.csv").sort_values("angle_deg")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(p1.angle_deg, p1.max_force_std_eVperA, "o-", color="#d62728", label="Phase 1 input-perturbation")
axes[0].axhline(1.67, color="#d62728", linestyle=":", linewidth=1, alpha=0.6, label="P1 calibrated threshold (1.67)")
axes[0].set_xlabel("Dihedral angle (deg)")
axes[0].set_ylabel("max per-atom force std (eV/A)")
axes[0].set_title("Phase 1: input-perturbation (M=3, sigma=0.005 A)")
axes[0].legend(loc="upper right", fontsize=9)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 1.9)

axes[1].plot(p2.angle_deg, p2.max_force_std_eVperA, "o-", color="#2ca02c",
             label="Phase 2 seed-fine-tune (3 members)")
axes[1].axhline(0.033, color="#2ca02c", linestyle=":", linewidth=1, alpha=0.6,
                label="P2 calibrated threshold (0.033)")
axes[1].set_xlabel("Dihedral angle (deg)")
axes[1].set_ylabel("max per-atom force std (eV/A)")
axes[1].set_title("Phase 2: 3 seed-fine-tuned MACE-OFF members")
axes[1].legend(loc="upper right", fontsize=9)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 0.04)
fig.suptitle("Noise floor drops 50x with real ensemble (Phase 1 vs Phase 2)", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "phase2_torsion_compare.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---------- 2. OOD diagnostic bar chart ----------
ood = [
    ("Relaxed reference",            0.0105, "in-dist"),
    ("Thermal jitter (sigma=0.04A)", 0.0128, "in-dist"),
    ("Alpha-CH torsion 180 (in seed)", 0.0123, "in-dist"),
    ("COOH-OH dihedral 0 (OOD)",     0.0105, "near-in-dist"),
    ("COOH-OH dihedral 90 (OOD)",    0.0126, "near-in-dist"),
    ("C-H stretched to 1.8 A",       0.0212, "OOD"),
    ("Iso 1.3x stretch",             0.0194, "OOD"),
    ("H-H clash at 0.7 A",           0.0278, "OOD"),
]
labels = [x[0] for x in ood]
vals = [x[1] for x in ood]
groups = [x[2] for x in ood]
colors = {"in-dist": "#2ca02c", "near-in-dist": "#ff7f0e", "OOD": "#d62728"}
bar_colors = [colors[g] for g in groups]

fig, ax = plt.subplots(figsize=(10, 4.5))
y = np.arange(len(labels))
ax.barh(y, vals, color=bar_colors, alpha=0.85)
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Ensemble max force std (eV/A)")
ax.set_title("Phase 2 ensemble response by geometry class — modest 2.6x dynamic range")
ax.axvline(0.022, color="black", linestyle="--", linewidth=1, alpha=0.6, label="calibration p99 = 0.022")
ax.axvline(0.033, color="black", linestyle=":",  linewidth=1, alpha=0.6, label="calibrated threshold = 0.033")
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3)
# legend for color groups
from matplotlib.patches import Patch
group_handles = [Patch(color=colors[g], label=g) for g in ["in-dist", "near-in-dist", "OOD"]]
ax.legend(handles=group_handles + [
    plt.Line2D([0], [0], color="black", linestyle="--", label="calibration p99 = 0.022"),
    plt.Line2D([0], [0], color="black", linestyle=":",  label="threshold = 0.033"),
], loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "phase2_ood_bar.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# ---------- 3. MD-std distribution comparison: Phase 1 vs Phase 2 ----------
md1_300 = pd.read_csv("runs/20260516-230003/md.csv")
md1_600 = pd.read_csv("runs/20260516-232519/md.csv")
md2_300 = pd.read_csv("runs/phase2_300K/md.csv")
md2_600 = pd.read_csv("runs/phase2_final/md.csv")  # the 5000-step 600 K final run

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# Phase 1
bins1 = np.linspace(0, 2.0, 30)
axes[0].hist(md1_300.max_force_std_eVperA, bins=bins1, alpha=0.55, label="300 K",
             color="#1f77b4", density=True)
axes[0].hist(md1_600.max_force_std_eVperA, bins=bins1, alpha=0.55, label="600 K",
             color="#d62728", density=True)
axes[0].axvline(1.67, color="black", linestyle="--", linewidth=1, label="threshold = 1.67")
axes[0].set_xlabel("max force std (eV/A)")
axes[0].set_ylabel("density")
axes[0].set_title("Phase 1 (input-perturbation)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Phase 2
bins2 = np.linspace(0, 0.04, 30)
axes[1].hist(md2_300.max_force_std_eVperA, bins=bins2, alpha=0.55, label="300 K",
             color="#1f77b4", density=True)
axes[1].hist(md2_600.max_force_std_eVperA, bins=bins2, alpha=0.55, label="600 K (5000 steps)",
             color="#d62728", density=True)
axes[1].axvline(0.022, color="black", linestyle="--", linewidth=1, alpha=0.6,
                label="calibration p99 = 0.022")
axes[1].axvline(0.033, color="black", linestyle=":",  linewidth=1, alpha=0.6,
                label="threshold = 0.033")
axes[1].set_xlabel("max force std (eV/A)")
axes[1].set_ylabel("density")
axes[1].set_title("Phase 2 (seed-fine-tune ensemble)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
fig.suptitle("MD std distributions: noise floor dropped 50x, but MD stays in-distribution at both temperatures",
             y=1.02)
fig.tight_layout()
fig.savefig(FIG / "phase2_md_dist.png", dpi=140, bbox_inches="tight")
plt.close(fig)

print("Wrote:", *list(FIG.glob("phase2_*.png")), sep="\n  ")
