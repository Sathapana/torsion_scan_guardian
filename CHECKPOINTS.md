# Torsion Scan Guardian — CHECKPOINTS

Status snapshot of the project, derived from a scan of [`src/`](src/), [`config/`](config/), [`scripts/`](scripts/), [`notebooks/colab/`](notebooks/colab/), and [`REPORT.md`](REPORT.md).

- **Branch:** `chore/seeds-medium5-update`
- **Date of snapshot:** 2026-05-22
- **Project goal:** Active-learning pipeline that stabilises MACE-OFF molecular dynamics on flexible drug-like molecules by detecting OOD conformations on-the-fly and fine-tuning during MD.

---

## 1. Where we are at a glance

| Phase | Scope | Status | Evidence |
| --- | --- | --- | --- |
| 0 — Scaffolding | Env, package, tests, MACE-OFF download, geom prep | ✅ Done | REPORT §4 |
| 1 — Input-perturbation ensemble | Cheap stand-in uncertainty signal | ✅ Done, **negative result** (measures curvature, not novelty) | REPORT §§5–8 |
| 2 — Seed-fine-tune ensemble (small-3) | 3 MACE-OFF small members on 74-frame GFN-FF seed | ✅ Done. Noise floor ↓50× but OOD range only ~2.6× | REPORT §9 |
| 2.5 — Cross-molecule (sulfanilamide) | Validate ensemble signal on a 2nd molecule | ✅ Done, **negative**: disagreement structural, not geometric | REPORT §10 |
| 4 — Online fine-tune during MD | Closed AL loop (trigger → label → retrain → reload → cooldown) | ✅ Done (verified, 2 cycles in 5 min on CPU) | REPORT §11 |
| 5 — AL instrument hardening | Acquisition clouds, safeguarded FT, stability module, AL-vs-baseline | ✅ Done. 5-cycle demo, 30 labels, 0 broken bonds | REPORT §12 |
| 6a — Compute / deployment | Dockerfile, Colab notebook, T4 GPU validation | ✅ Done. Cloud path verified, 2.1× speedup on T4 | REPORT §13.1–§13.7 |
| 6b — Throughput optimisation | Parallel cloud labels, parallel FT, batch 8→32 | ✅ Done. 3–4× sweep speedup | REPORT §13.8 |
| 6c — Scientific upgrade | MACE-OFF *medium* foundation + 5-member ensemble | ✅ Done. Defaults updated, configs migrated | REPORT §13.9 |
| 6d — First multi-molecule sweep (medium-5) | glycine_zwitterion, sulfonyl_chloride, diglycine on Colab T4 | ⚠ Half-done. Instrument runs cleanly but **0 triggers, no baseline collapse** | REPORT §14 |
| 6e — Demonstrated AL stabilisation | A molecule where baseline collapses *and* AL recovers it | ❌ Not yet | REPORT §14.5–§14.6 |

**One-line summary:** the AL instrument is feature-complete and verified at scale on the new medium-5 foundation; the publishable-paper milestone (showing AL prevents a collapsing trajectory) is the only remaining science gap.

---

## 2. Code surface

### [`src/guardian/`](src/guardian/) — package layout
| Module | Purpose | Phase added |
| --- | --- | --- |
| `io/structures.py` | SMILES → RDKit → ASE `Atoms` | 0 |
| `md/driver.py` | Langevin MD wrapper | 0 |
| `models/ensemble.py` | MACE-OFF loader, `MACEOffEnsemble`, `SeedFinetuneEnsemble`, device fallback | 1→2 |
| `oracle/gfnff.py` | GFN-FF single point + `label_cloud_with_gfnff` (parallel) | 0, §12.1 |
| `oracle/xtb_subprocess.py` | `xtb.exe` CLI wrapper (works around Windows DLL bug) | 2 |
| `pipeline/controller.py` | AL state machine `RUN→PAUSE→LABEL→RETRAIN→RESUME`, `_do_finetune_cycle`, `max_parallel_finetunes` | 4–§13.9 |
| `uncertainty/monitor.py` | Threshold check with warmup + cooldown suppression | 0, 4 |
| `training/finetune.py` | `online_finetune_member` (subprocess to `mace_run_train`, safeguarded) | 4, §12.2 |
| `training/replay.py` | Replay buffer with recency-biased sampling | 0 |
| `viz/torsion_scan.py` | Frozen dihedral scans for diagnostics | 1 |
| `calibration.py` | BFGS relaxation + thermal-cloud threshold derivation | 0 |
| `stability.py` | Bond-stretch ratio, RMSD, broken-bond count metrics | §12.3 |
| `cli.py`, `config.py` | CLI plumbing, YAML config, device override | 0+ |

### [`config/`](config/)
- `default.yaml` — **already on the §13.9 upgrade** (`backbone: mace-off-medium`, `batch_size: 32`). Threshold `1.67` is the legacy ibuprofen small-3 calibration; per-molecule files override.
- `molecules/glycine_zwitterion.yaml`, `molecules/ibuprofen.yaml`, `molecules/sulfanilamide.yaml` — three checked-in molecule configs.
- Other candidates (`sulfonyl_chloride`, `diglycine`, `n_methylacetamide`, `biphenyl`, `norbornadiene`, `caffeine`) are auto-created from `data/molecule_library/candidates.csv` and not yet committed.

### [`scripts/`](scripts/)
| Script | Role |
| --- | --- |
| `run_guardian.py` | Main AL-run entry point |
| `build_seed_dataset.py` | GFN-FF labelled seed XYZ generator |
| `finetune_member.py` | Per-member fine-tune driver (`--foundation-size medium` default per §13.9) |
| `sweep_molecules.py` | Outer loop over `candidates.csv`; defaults `--n-members 5 --foundation-size medium` |
| `diagnose_ood.py` | Hand-crafted OOD probe diagnostic (§9.4 / §10.2) |
| `diagnose_torsion.py` | Frozen-torsion scan diagnostic |
| `analyse_al_demo.py`, `compare_al_vs_baseline.py` | Post-run analysis |
| `make_phase2_figures.py`, `make_report_figures.py` | Figure regeneration |
| `sulf_ood_probe.py` | Sulfanilamide-specific OOD test from §10 |
| `build_bootstrap_splits.py` | (Legacy) bootstrap subset generator from §9.2 v2/v3 attempts |

### [`notebooks/colab/`](notebooks/colab/)
- `guardian_colab.ipynb` — single-molecule Phase-5 demo on Colab T4 (validated 2026-05-17, §13.7).
- `guardian_sweep_colab.ipynb` — multi-molecule sweep notebook used for the Phase-6 first run (§14).
- Both switched to `uv` install (recent commit `e8466a1`, ~10× faster `pip`).

### Seed datasets in [`data/seed/`](data/seed/)
- Already labelled: `ibuprofen`, `sulfanilamide`, plus the 7 Phase-6 candidates: `biphenyl`, `caffeine`, `diglycine`, `glycine_zwitterion`, `n_methylacetamide`, `norbornadiene`, `sulfonyl_chloride`.

---

## 3. Key empirical findings to keep in mind

1. **Input-perturbation uncertainty (Phase 1) ≠ epistemic uncertainty.** It measures local PES curvature (bond-stretch stiffness dominates). Discarded. — §6
2. **Foundation-anchored fine-tunes on ~74 examples produce SGD-noise ensembles, not epistemic ones.** On ibuprofen OOD/in-dist ratio = 2.6×; on sulfanilamide ≈ 1.0× (flat regardless of geometry). — §9.6, §10.4
3. **Medium-5 ensemble dropped the noise floor 2–80×** vs small-3 (calibrated `τ` now 0.019–0.052 eV/Å across glycine_zwitterion / sulfonyl_chloride / diglycine vs 0.10–1.67 before). The OOD-discrimination half of the §13.9 prediction is **untested** — `scripts/diagnose_ood.py` has not been re-run on medium-5 checkpoints. — §14.2, §14.4
4. **Baseline (unguarded) MD does not collapse on any molecule tested so far** at 300 K / 4000 steps. Max bond stretch tops out ~1.08–1.15 (break threshold 1.6). Guardian's value is conditional on baseline failing → not yet demonstrable. — §14.2
5. **Float64 vs float32 changes ensemble noise floor ~15 %.** Threshold must be re-calibrated per device/dtype. Colab loads MACE-OFF in float64 by default. — §13.7
6. **Per-cycle wall time is dominated by `mace_run_train` CLI startup (~10–15 s per call).** Worker-process fine-tuner (§12.4) is the deferred but obvious next engineering win.

---

## 4. Outstanding work, in priority order

| # | Item | Cost | Source |
| --- | --- | --- | --- |
| 1 | **Find a molecule whose baseline collapses.** Hotter MD (600–1000 K) on existing candidates; or biased sampling (metadynamics on a known floppy dihedral); or chemistry outside MACE-OFF coverage (transition-metal complex, partial bond cleavage). | ~30 min/molecule on T4 | §12.7, §14.5 |
| 2 | **Re-run `diagnose_ood.py` on medium-5 checkpoints.** Decides whether §13.9's noise-floor drop translated into a real OOD/in-dist dynamic-range win. | ~30 min/molecule on T4 | §14.5 |
| 3 | **Worker-process fine-tuner.** Per-cycle wall-time 120 s → 30–40 s; unblocks 100-cycle runs on CPU. | ~200 LOC | §12.4 |
| 4 | **Cloud-size × FT-epochs scan** on whichever molecule reaches milestone #1. | small | §12.7 |
| 5 | **DFT-quality oracle** (`guardian.oracle.dft`, e.g. Psi4 ωB97X-D / def2-TZVP) for final-cycle judge step. Interfaces already swappable. | medium | §12.7 |
| 6 | **Longer MD runs** (40 000 steps) — needs tmux/Vast.ai to escape Colab's 90-min idle limit. | infra | §14.5 |
| 7 | **Commit reproducible seed datasets + auto-generated `config/molecules/*.yaml`** for the 7 Phase-6 candidates currently only in the download bundle. | trivial | §14.7 |

---

## 5. Reproduction one-liners

```bash
# Local CPU single-molecule AL run (Phase 5 config)
python scripts/run_guardian.py --config config/molecules/sulfanilamide.yaml \
    --online-finetune --cloud-size 5 --max-triggers 5 --threshold 0.05

# Multi-molecule sweep (medium-5 defaults; rebuilds against new foundation)
python scripts/sweep_molecules.py --phase-filter todo

# Direct OOD diagnostic (re-run after §13.9 to fill the §14.5 gap)
python scripts/diagnose_ood.py --molecule sulfanilamide --ensemble runs/finetune_sulfanilamide_medium/
```

For Colab, open [`notebooks/colab/guardian_sweep_colab.ipynb`](notebooks/colab/guardian_sweep_colab.ipynb) on a T4 runtime and run all cells.

---

## 6. The bottom line

> **Pipeline: feature-complete. Science: half-way through Phase 6.**
>
> Eight molecules swept, eight zero-trigger AL runs, zero broken bonds across all baselines. The instrument works; we just haven't yet found a substrate where the foundation model fails badly enough to give the Guardian something to catch. The next experiment that matters is **item #1** above — push MD conditions or molecule choice until the baseline breaks, then show AL recovers it.
