# Notebooks

Per-platform notebooks for running the Torsion Scan Guardian pipeline on a GPU. Each platform has **two** notebooks: a single-molecule Phase-5 demo and a multi-molecule Phase-6 sweep over `data/molecule_library/candidates.csv`.

```
notebooks/
├── colab/                      ← Google Colab Free (T4)
│   ├── guardian_colab.ipynb           single-molecule demo (sulfanilamide)
│   └── guardian_sweep_colab.ipynb     multi-molecule sweep
├── thunder/                    ← Thunder Compute (owned-fleet GPU rental)
│   ├── guardian_thunder.ipynb         single-molecule demo
│   └── guardian_sweep_thunder.ipynb   multi-molecule sweep
└── vastai/                     ← Vast.ai (peer-to-peer GPU marketplace)
    ├── guardian_vastai.ipynb          single-molecule demo
    └── guardian_sweep_vastai.ipynb    multi-molecule sweep
```

## How the platforms differ

| | Colab | Thunder | Vast.ai |
| --- | --- | --- | --- |
| **Cost (Phase-5 demo)** | $0 | ~$0.10 | ~$0.02 |
| **Cost (Phase-6 sweep, 7 molecules)** | $0 | ~$0.55 | ~$0.40 |
| **Setup time first session** | 10 min | 15 min | 15 min |
| **Idle timeout** | ~90 min on Free tier | none | none |
| **Persistent storage** | Google Drive (manual mount) | per instance | per instance |
| **Push results to GitHub** | Bundle download → push from local | scp / PAT / SSH key | scp / PAT / SSH key |
| **Reliability** | Variable (queue at peak) | Owned fleet (predictable) | Marketplace (filter for verified hosts) |
| **Setup tutorial** | [`../Colab_WAY.md`](../Colab_WAY.md) | [`../Thunder_WAY.md`](../Thunder_WAY.md) | [`../VastAI_WAY.md`](../VastAI_WAY.md) |
| **Status of tutorial** | Battle-tested (12 problems documented) | Predicted | Predicted |

## Which one should I use?

| Situation | Pick |
| --- | --- |
| First time, want free, OK with browser-only Jupyter | **Colab** |
| Want VSCode IDE + persistent disk + predictable reliability | **Thunder** |
| Cheapest paid option, OK filtering marketplace hosts | **Vast.ai** |
| Sweep that must complete reliably overnight | **Thunder** (or Vast with `verified=true reliability>0.99`) |
| Iteration on a single molecule, no budget | **Colab** |

## Common structure across all six notebooks

| Section | Colab | Thunder / Vast.ai |
| --- | --- | --- |
| GPU verify | cell 1 | cell 1 |
| Env setup (mount Drive / clone / pip / condacolab / xtb) | cells 2–7 | done from shell per the `*_WAY.md` doc, before opening the notebook |
| Pre-warm MACE-OFF cache | cell 8 | cell 2 |
| Pytest smoke | cell 9 (sweep notebook only) | cell 3 |
| Build seed dataset | cell 10 (skip if exists) | cell 4 |
| Fine-tune 3 ensemble members | cell 11 | cell 5 |
| AL run (single-mol) OR sweep (sweep nb) | cell 12 | cell 6 |
| Analysis figures | cell 13 | cell 7 |
| Bundle results for download | cell 14 (Colab-only) | n/a — push from local via git or scp |

The Thunder and Vast.ai notebooks are intentionally shorter because their env setup is done from the shell (per their `*_WAY.md` doc) before the notebook is opened, not from notebook cells. They start at the equivalent of Colab cell 8.

## Cross-references

- Project overview: [`../README.md`](../README.md)
- Full technical report: [`../REPORT.md`](../REPORT.md)
- Adding a new molecule: [`../CONTRIBUTING.md`](../CONTRIBUTING.md) §Adding a new molecule
- Molecule catalog: [`../data/molecule_library/candidates.csv`](../data/molecule_library/candidates.csv)
- Sweep orchestrator script (used by all sweep notebooks): [`../scripts/sweep_molecules.py`](../scripts/sweep_molecules.py)
