# 🛡️ Project: The Torsion Scan Guardian
### Active Learning for Conformational Stability in ML Force Fields

**Objective:** To develop an automated, end-to-end Active Learning (AL) pipeline that ensures Machine Learning Force Fields (MLFFs) remain stable and accurate during long-range Molecular Dynamics (MD) simulations of flexible drug-like molecules.

**Status:** Phase 0–5 complete (verified active-learning instrument running end-to-end on CPU, with acquisition clouds, safeguarded fine-tunes, stability metrics, and a 5-cycle long-run demo). Full technical writeup in [REPORT.md](REPORT.md). Next milestone is Phase 6: demonstrate AL *stabilising a divergent baseline* on a molecule where MACE-OFF collapses without help.

---

## 🌟 The "Why": The Problem of Structural Collapse
Traditional ML Force Fields often suffer from **extrapolation failure**. When a molecule explores a high-energy conformation (like a rare dihedral torsion) not present in the training data, the model's predicted forces can become unphysically large. This leads to "structural collapse," where the molecule unrealistically flies apart or "explodes" during simulation.

**The Torsion Scan Guardian** solves this by acting as a real-time autopilot, detecting uncertainty and teaching the model new physics on the fly.

---

## 🛠️ System Architecture & Workflow

The project implements a closed-loop "Active Learning" cycle:

1.  **Exploration (MD Engine):** Use a pre-trained Equivariant Graph Neural Network (GNN) to run MD simulations of a target molecule (e.g., a flexible sulfonamide or biphenyl derivative) via the `ASE` interface.
2.  **Uncertainty Monitoring (The Guardian):** An ensemble of GNNs monitors the predicted forces. If the variance between models exceeds a calibrated threshold (indicating a novel torsion angle), the simulation is paused.
3.  **Data Acquisition (The Oracle):** The pipeline automatically triggers a high-fidelity reference calculation. It computes the "ground truth" energy and forces for the problematic conformation, plus a small cloud of perturbed copies around it for richer gradient signal.
4.  **On-the-Fly Retraining:** The new data is added to the training set. Each ensemble member undergoes a short, *safeguarded* fine-tune (reverted automatically if validation force MAE regresses) to patch the hole in its chemical knowledge.
5.  **Resumption:** The simulation resumes from the last stable point with a smarter, more robust force field, after a cooldown window so the integrator escapes the flagged geometry under the new force surface.

---

## 🧰 Tech Stack — options considered, what we chose, and why

Each design decision had multiple defensible candidates. The table records what was on the table at each layer, what we picked, and the reasoning — so the cost of swapping any one component later is clear.

| Layer | Options surveyed | **Chosen** | Why this, not the others |
| --- | --- | --- | --- |
| MLFF backbone | MACE-OFF (small / medium / large), SchNet, ANI-2x, NequIP | **MACE-OFF23 small** (~5 M params) | Pretrained on SPICE drug-like data; strict E(3)-equivariance; small enough for fast CPU iteration; cleanest active-learning fine-tune story via `mace_run_train --foundation_model`. |
| Uncertainty signal | Input-perturbation (cheap Phase-1 stand-in), 3-member seed-fine-tuned ensemble, MC-dropout, latent-feature std | **3-member seed-fine-tuned ensemble** | Phase 1 empirically showed input-perturbation measures local PES curvature, *not* novelty (REPORT §6). Real model-weight ensembles are the published standard and dropped our noise floor 50× (§9.3). |
| MD engine | ASE Langevin, OpenMM, LAMMPS | **ASE Langevin** | Direct in-Python integration with MACE's ASE calculator; sufficient for small-molecule MD; trivial to swap to OpenMM later if simulation length becomes the bottleneck. |
| Oracle (reference labels) | DFT via Psi4 (ωB97X-D/def2-TZVP), GFN-FF, GFN2-xTB, OpenFF | **GFN-FF via the `xtb` CLI** | ~100 ms per single point vs DFT's ~30 min — lets us iterate AL cycles in seconds. Honest about fidelity: GFN-FF is a fast empirical reference, not a quantum ground truth. DFT is reserved as the validation "judge" for Phase 6 (§11.5, §12.7). |
| Oracle integration | `tblite-python`, `xtb-python`, `xtb` CLI via subprocess | **`xtb` CLI via subprocess** | The two Python bindings both crashed on Windows with a delay-load DLL bug inside `singlepoint`. The CLI binary works perfectly; we parse its Turbomole-format `energy` and `gradient` files. ~150 ms total overhead per call, dominated by subprocess startup. |
| Fine-tuning loop | In-process `mace-torch`, subprocess `mace_run_train`, ML-framework workers (Modal / Ray / SLURM) | **Subprocess `mace_run_train`** | Cleanest isolation from the parent process's argparse / torch state; no shared-cache bugs; ~15 s startup overhead acceptable for short runs. Worker-process upgrade documented as the throughput target (§12.4) for runs > 10 cycles. |
| Safeguard | None, gradient clipping only, val-MAE rollback, EMA on weights | **Validation-MAE rollback** (+ revert checkpoint if final RMSE_F > initial × (1 + tol)) | Empirically caught the Phase-2 v2 pathology where aggressive fine-tunes silently degraded forces and made BFGS diverge to E ≈ 10⁹ eV (§9.2). |
| Acquisition strategy | Single triggered geometry, dihedral-driven cloud, Gaussian position cloud, active-querying | **Gaussian position cloud** (default 5 perturbations at σ = 0.05 Å) | Gives the fine-tune real *local* gradient signal in the flagged region; ~0.5 s extra cost per trigger vs ~120 s for the fine-tune itself — essentially free (§12.1). |
| Experiment tracking | Weights & Biases, MLflow, plain CSV + JSON summaries | **CSV + JSON summaries + matplotlib** | Zero external dependencies; trivially diffable in git; W&B is pre-wired via `cfg.wandb` and can be turned on for hyperparameter sweeps without code changes. |
| Compute environment | Local CPU (dev), Google Colab (free T4), RunPod spot (~$0.30/h), Lambda Labs on-demand, Modal serverless | **Local CPU for Phase 0–5; Colab + Docker for Phase 6 onwards** | CPU was sufficient to verify every phase. Cloud is needed for the long experiments (100+ cycles, DFT oracle calls). Colab is free; Docker image works on RunPod / Lambda / any Linux GPU host. Detailed cost analysis in REPORT §13. |

---

## 📊 What's done vs what's next

**Done (Phase 0–5; full technical detail in [REPORT.md](REPORT.md)):**

| Phase | Deliverable | Outcome |
| --- | --- | --- |
| 0 | Env + skeleton + tests | 11 tests passing |
| 1 | MACE-OFF integration + input-perturbation ensemble | Empirically shown to fail as an uncertainty signal — measures curvature, not novelty (§6) |
| 2 | Seed-fine-tuned 3-member ensemble | Noise floor dropped 50×; OOD response real but modest (2.6×) on ibuprofen (§9) |
| 3 | GFN-FF oracle | Subprocess wrapper, ~100 ms/call (§9.1, §11.1) |
| 4 | Closed AL loop end-to-end | 2 cycles in 5 min on CPU (§11) |
| 5 | Acquisition clouds + safeguarded FT + stability metrics + long-run demo + matched baseline | 5 cycles, 30 acquired labels, 16 min wall time, 0 broken bonds, matched baseline shows AL doesn't destabilise (§12) |
| Deployment | Dockerfile + Colab-ready notebook | One-command launch on any GPU host (§13) |

**Phase 6 (next experimental milestone):**

| To do | Why |
| --- | --- |
| Find a molecule where the unguarded baseline actually collapses | Candidates: glycine zwitterion, sulfonyl chloride, charged dipeptide. Required to demonstrate AL *stabilisation*, not just *non-degradation*. |
| Run baseline-vs-AL stability comparison | Reuse `scripts/compare_al_vs_baseline.py` — already produces side-by-side u(t), RMSD, bond-stretch figures. |
| Swap oracle to DFT (Psi4) for the validation judge | Final torsion-profile comparison vs gold standard; the `oracle/` module is interface-clean, swap is ~50 LOC. |

---

## 📊 Key Highlights & Portfolio Features
* **Automated Labeling:** Human-out-of-the-loop data engineering — every trigger acquires a GFN-FF label and feeds the next fine-tune without supervision.
* **Physical Fidelity:** Torsion-scan diagnostics compare base, Phase-1, and Phase-2 models against GFN-FF; DFT validation is the Phase-6 deliverable.
* **Visualization:** Side-by-side stability metrics (max bond stretch, Kabsch RMSD, broken-bond count) for AL vs baseline runs (REPORT §12.6, `figures/al_vs_baseline.png`).
* **Scalability:** Framework is molecule-agnostic — swap the SMILES, re-run the same CLI; bootstrap-style seed datasets are auto-generated.
* **Honest empirical reporting:** Every negative result (Phase 1 failed, Phase 2 had narrow dynamic range, sulfanilamide didn't trigger naturally) is documented in the report with mechanistic explanation, not hidden.
* **Reproducible deployment:** Dockerfile + Colab notebook; cloud cost analysis (§13) shows the project runs end-to-end for **$0 on Colab Free** or **~$2 on RunPod spot**.

---

## 🚀 Future Scope
* **Phase 6: demonstrate stabilisation** on a molecule that collapses without AL — the missing ingredient between "verified instrument" and "publishable result".
* **Multi-Molecule Scaling:** Expand the pipeline to handle a library of drug fragments simultaneously (currently molecule-at-a-time).
* **Delta-Learning:** Implement a "Delta" approach where the ML model learns the difference between GFN-FF and DFT — preserves MACE-OFF's prior while adding DFT-quality corrections.
* **Worker-process fine-tuner** (REPORT §12.4): drop per-cycle wall-time from ~120 s to ~30 s, making 100-cycle runs tractable on CPU.
* **W&B integration:** the config hook is already in place; enable for hyperparameter sweeps over `cloud_size`, `finetune_epochs`, `ft_regression_tol`.

---
*Created by Sathapana Chawananon as part of a Data Science Portfolio focused on Computational Drug Discovery.*
