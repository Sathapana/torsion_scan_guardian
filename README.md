# Torsion Scan Guardian

[![CI](https://github.com/Sathapana/torsion_scan_guardian/actions/workflows/ci.yml/badge.svg)](https://github.com/Sathapana/torsion_scan_guardian/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

Active-learning pipeline for stabilising MACE-OFF molecular dynamics on flexible drug-like molecules. Detects ML force-field uncertainty in real time, acquires GFN-FF reference labels for the flagged geometries, and re-fine-tunes the ensemble on the fly.

**Status:** Phase 0–5 complete (verified active-learning instrument running end-to-end on CPU and GPU). Phase 6 (demonstrating stabilisation of a baseline-collapsing trajectory) is the next experimental milestone.

## What's here

| Path | What it is |
| --- | --- |
| [`Torsion_Scan_Guardian_Project.md`](Torsion_Scan_Guardian_Project.md) | Project overview, tech stack, design-decisions table, status |
| [`REPORT.md`](REPORT.md) | Full technical report (Phase 0 → 5, ~700 lines, with figures and math) |
| [`src/guardian/`](src/guardian/) | The package — ensemble, oracle, controller, training, stability, calibration, MD driver |
| [`scripts/`](scripts/) | One-shot CLIs: build seed dataset, fine-tune member, OOD probes, AL-vs-baseline comparison |
| [`config/default.yaml`](config/default.yaml) | Default experiment config |
| [`notebooks/`](notebooks/) | Per-platform notebooks: Colab Free, Thunder Compute, Vast.ai (single-molecule + sweep each) |
| [`Dockerfile`](Dockerfile) | Reproducible Linux image (CPU default, CUDA via build-arg) |
| [`tests/`](tests/) | 11 pytest tests (calibration, monitor, replay, stability, ensemble smoke) |
| [`data/seed/`](data/seed/) | Two GFN-FF labelled seed datasets (ibuprofen, sulfanilamide; 74 frames each) |
| [`figures/`](figures/) | Phase 1, 2, and 5 figures referenced from the report |

## Quickstart — three ways

### Local (CPU, ~10 min env build once)
```bash
conda env create -f environment.yml
conda activate guardian
pip install -e .
pytest                                            # 11 tests pass
python -m guardian.cli --config config/default.yaml --calibrate
```

### Docker (CPU)
```bash
docker build -t guardian:cpu .
docker run --rm -it -v "$PWD/runs:/app/runs" guardian:cpu \
    python -m guardian.cli --config config/default.yaml --calibrate
```

### Google Colab (GPU, free)
Open [`notebooks/colab/guardian_colab.ipynb`](notebooks/colab/guardian_colab.ipynb) (single-molecule demo) or [`notebooks/colab/guardian_sweep_colab.ipynb`](notebooks/colab/guardian_sweep_colab.ipynb) (multi-molecule sweep over `candidates.csv`), set runtime to T4 GPU, edit the `REPO_URL` cell, run all. Total time: ~10 min for the Phase-5 demo, ~2 h for the 7-molecule sweep.

**First time on Colab? Read [`Colab_WAY.md`](Colab_WAY.md)** — full tutorial, tips, and the 12 problems we hit (with fixes) so you don't re-hit them.

### Thunder Compute (paid GPU, VSCode-integrated)

If Colab keeps disconnecting on your sweep, or you want a proper IDE: see [`Thunder_WAY.md`](Thunder_WAY.md) for the Thunder Compute + VSCode Remote-SSH setup. ~$0.50 for a 7-molecule sweep on a T4 spot instance, with no idle timeout and persistent storage.

### Vast.ai (cheapest paid GPU marketplace)

Peer-to-peer GPU marketplace, typically the lowest hourly rate of any cloud option (~$0.20–0.30/h for RTX 3090). More variance in reliability (you filter for it), but ~$0.40–0.60 for a 7-molecule Phase-6 sweep makes it the cheapest non-free option. See [`VastAI_WAY.md`](VastAI_WAY.md) for the full VSCode Remote-SSH workflow.

## What the pipeline actually does

```
SMILES → ASE atoms → relax with MACE-OFF
                        ↓
                   Langevin MD (ASE)  ←─────────────────────────┐
                        ↓                                       │
              3-member MACE-OFF ensemble                        │
                        ↓                                       │
              per-step max-atom force std                       │ cooldown
                        ↓                                       │ steps
                if std > calibrated threshold ────► PAUSE       │
                                                     ↓          │
                                          GFN-FF (xtb) labels   │
                                          trigger + cloud       │
                                                     ↓          │
                                          subprocess fine-tune  │
                                          3 members (safeguarded)
                                                     ↓          │
                                          reload ensemble ──────┘
```

Implementation lives in [`src/guardian/pipeline/controller.py`](src/guardian/pipeline/controller.py). Every step is configurable via CLI flags; see `python -m guardian.cli --help`.

## Headline results

See [REPORT.md §12.5–12.6](REPORT.md) for the full long-run demo and AL-vs-baseline comparison. Summary on sulfanilamide:

| Metric | Baseline (no AL) | AL (Phase 5) |
| --- | ---: | ---: |
| MD steps | 1550 | 1550 |
| Wall time (CPU) | 285 s | 974 s |
| Triggers fired | 0 (Guardian off) | 5 |
| Labels acquired | 0 | 30 |
| Broken bonds at end | 0 / 19 | 0 / 19 |
| Max bond stretch | 1.06× | 1.09× |

Both runs stable — AL does not destabilise. To demonstrate AL *stabilising a divergent baseline*, a molecule outside MACE-OFF's coverage is needed (Phase 6, see [REPORT.md §12.7](REPORT.md)).

## License
[MIT](LICENSE) for the code in this repository. Third-party models (notably MACE-OFF23 under the Academic Software License) retain their own terms — see LICENSE for the full list.

## Author
Sathapana Chawananon — Data Science portfolio, Computational Drug Discovery.
