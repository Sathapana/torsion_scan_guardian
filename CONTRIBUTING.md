# Contributing to Torsion Scan Guardian

Thanks for your interest in the project. This guide covers the local setup, how the code is organised, and the conventions a PR should follow to be quick to review.

## Quick local setup

```bash
git clone git@github.com:Sathapana/torsion_scan_guardian.git
cd torsion_scan_guardian
conda env create -f environment.yml
conda activate guardian
pip install -e .
pytest                                 # 11 tests should pass
python -m guardian.cli --help          # sanity check the CLI loads
```

For GPU runs, set `device: cuda` in [config/default.yaml](config/default.yaml) (CUDA-enabled `torch` is already in `environment.yml`). Falls back to CPU if no GPU.

For everything else (Docker, Colab, cloud), see [README.md](README.md) §*Quickstart* and [REPORT.md](REPORT.md) §13.

## Running tests

```bash
pytest                                 # fast suite (no model downloads); ~5 s
RUN_MACE_SMOKE=1 pytest -s             # also runs the MACE-OFF download/inference smoke test (~30 s)
```

If you change anything in `src/guardian/`, also run the relevant integration script(s):

| Changed area | Quick integration check |
| --- | --- |
| `models/ensemble.py` | `python -m guardian.cli --config config/default.yaml --calibrate --dry-run` |
| `oracle/*.py` | `python -c "from guardian.oracle.gfnff import label_with_gfnff; from ase.build import molecule; print(label_with_gfnff(molecule('H2O')).energy)"` |
| `pipeline/controller.py` | `python -m guardian.cli ... --steps 200 --max-triggers 1` on the smallest molecule you can manage |
| `training/finetune.py` | `python scripts/finetune_member.py --seed 0 --epochs 1 --train-file data/seed/sulfanilamide_seed.xyz --out-root /tmp/ft_test` |
| `stability.py` | `pytest tests/test_stability.py -v` |

## How the code is organised

```
src/guardian/
├── cli.py               # argparse entry point — every flag is documented in --help
├── config.py            # pydantic schema for config/default.yaml
├── calibration.py       # geometry relaxation + threshold calibration
├── stability.py         # post-hoc trajectory stability metrics
├── md/
│   └── driver.py        # ASE Langevin loop with per-step callback
├── models/
│   └── ensemble.py      # MACEOffEnsemble (Phase 1) and SeedFinetuneEnsemble (Phase 2)
├── oracle/
│   ├── gfnff.py         # high-level GFN-FF labelling + acquisition cloud
│   └── xtb_subprocess.py  # ASE calculator wrapping the xtb CLI (Windows DLL workaround)
├── pipeline/
│   └── controller.py    # state machine: RUN → PAUSE → LABEL → RETRAIN → RESUME
├── training/
│   ├── finetune.py      # subprocess wrapper around mace_run_train + safeguard
│   └── replay.py        # in-memory + pickle-persisted DataPoint buffer
├── uncertainty/
│   └── monitor.py       # force-std threshold trigger
├── viz/
│   └── torsion_scan.py  # plotting helpers for torsion-scan diagnostics
└── io/
    └── structures.py    # SMILES → ASE Atoms via RDKit ETKDG + MMFF
```

`scripts/` contains one-shot CLIs for dataset building, fine-tuning, diagnostics, and analysis. `tests/` contains pytest unit tests; the heavy MACE-OFF inference test is gated by `RUN_MACE_SMOKE=1`.

## Conventions

### Code style
- **Type hints on public functions** (the existing code does this consistently). Internal helpers can skip them.
- **Dataclasses for return values** with more than two fields (`EnsemblePrediction`, `OracleLabel`, `FineTuneReport`, `StabilityMetrics`).
- **No emojis in code or output** — they break the Thai cp874 codepage Windows console used during development.
- **ASCII in print statements** — use `eV/A` not `eV/Å` (same reason as above). Set `PYTHONIOENCODING=utf-8` if you need Unicode at runtime.

### Tests
- Every new module gets at least one smoke test in `tests/test_<module>.py`.
- Stub calculators are preferred over loading MACE for fast tests; see `tests/test_calibration.py::StubEnsemble` for the pattern.
- Heavyweight tests (model downloads, multi-second computations) should be gated behind an env var like `RUN_MACE_SMOKE=1`.

### Configuration
- New runtime knobs go through the pydantic schema in `src/guardian/config.py`, the default YAML in `config/default.yaml`, and the CLI in `src/guardian/cli.py` (in that order).
- CLI flags follow the existing kebab-case convention (`--finetune-epochs`, not `--finetune_epochs`).

### Commit messages
- Short imperative subject line, ~50 chars. Body wrapped at ~72.
- Reference the report section that motivates the change when relevant (e.g. "implements REPORT §12.7 item 2").
- One logical change per commit.

## Pull request flow

1. Branch from `main`: `git checkout -b feature/short-description`.
2. Make the change. Run `pytest` locally — must be green.
3. Update [REPORT.md](REPORT.md) if the change affects experimental results or interpretation; update [README.md](README.md) if the change affects the public interface.
4. Push and open a PR. The CI workflow ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the test suite on Linux Python 3.11; the PR is unmergeable while CI is red.
5. Keep PRs scoped — easier to review one focused change than five mixed ones.

## Adding a new molecule

The project uses a three-tier molecule convention so molecule-specific work doesn't pollute the controller code:

| Tier | Where it lives | When to use it |
| --- | --- | --- |
| **1. One-off** | `--smiles "..."` on the CLI | Quick test, single experiment, no plan to re-run |
| **2. Per-molecule config** | `config/molecules/<name>.yaml` | You'll run experiments on this molecule more than once and need to remember the threshold / temperature / run-dir settings |
| **3. Catalog entry** | Append a row to `data/molecule_library/candidates.csv` | The molecule is a Phase-6 candidate or a benchmark you want documented even if not yet run |

Most contributions will touch tiers 2 + 3 together: a new YAML config plus a catalog row pointing at it. See [`config/molecules/glycine_zwitterion.yaml`](config/molecules/glycine_zwitterion.yaml) for the recommended template — it includes a suggested-run-order comment block at the bottom that walks through `--calibrate` → seed-build → fine-tune → baseline-vs-AL.

Things to remember when adding a charged or radical species:
- Set `molecule.charge` and `molecule.multiplicity` correctly. Most molecules are `0 / 1`; zwitterions are also net `0 / 1` but visibly carry formal charges in the SMILES.
- The current `guardian.oracle.xtb_subprocess.XTBCalculator` does **not** pass `--chrg` / `--uhf` to xtb. GFN-FF still works because it autodetects from the input geometry; GFN2-xTB and DFT would need the kwarg added (~10 LOC change). Document this in the YAML comment header if you swap oracles.
- Calibrate the threshold *after* the seed-fine-tune ensemble exists for the molecule — calibrating against the input-perturbation ensemble (Phase 1) gives the wrong answer (REPORT §6).

## What's worth working on right now

Pick from the list in [REPORT.md §12.7](REPORT.md):

1. **A molecule where the baseline collapses.** Strongest practical PR — pick a candidate (glycine zwitterion, sulfonyl chloride, charged dipeptide), get a baseline-vs-AL comparison showing AL stabilises, write the result up.
2. **Hyperparameter sweep over `--cloud-size` / `--finetune-epochs`.** Modal/W&B-friendly; produces a sensitivity plot for the report.
3. **DFT oracle (Psi4).** Add `src/guardian/oracle/dft.py` implementing the same interface as `gfnff.py`. The controller is interface-clean and will pick it up via a config flag.
4. **Worker-process fine-tuner.** Drop per-cycle wall time from ~120 s to ~30 s. ~200 LOC, documented in [REPORT.md §12.4](REPORT.md).

## Reporting issues

Open a GitHub issue with:
- What you ran (full CLI line)
- What you expected vs what happened
- Log output (paste relevant lines, attach `runs/<dir>/summary.json` if the run completed)
- Environment: OS, conda env (`conda list --explicit` if convenient), GPU model

## Questions

Open an issue tagged `question`. For longer discussions, use the repo's Discussions tab (if enabled).
