# Molecule library

A curated catalog of molecules used or considered for Torsion Scan Guardian
experiments. The CSV is the source of truth; per-molecule YAML configs in
[`config/molecules/`](../../config/molecules/) hold the runnable settings for
the subset that has been parameterised.

## `candidates.csv` columns

| Column | Meaning |
| --- | --- |
| `name` | Lowercase, underscore-separated identifier — matches the YAML basename when one exists |
| `smiles` | RDKit-parseable SMILES with explicit charges where applicable |
| `charge` | Net molecular charge (passed to oracles like xtb via `--chrg`) |
| `multiplicity` | Spin multiplicity; 1 for closed-shell, 2 for radicals, etc. |
| `heavy_atoms` | Non-H atom count — quick filter for "small/medium/large" |
| `why_interesting` | Free-text rationale: scientific motivation, links to REPORT sections, expected failure mode |
| `phase` | Lifecycle tag: `done` (used in REPORT), `todo` (queued for Phase 6), `candidate` (worth trying), `rejected` |
| `config_path` | Relative path to the per-molecule YAML if one exists; empty otherwise |

## Adding a molecule

1. Append a row to `candidates.csv`. The `phase` starts as `candidate` or `todo`.
2. (Optional, for Tier-2 reuse) Create `config/molecules/<name>.yaml` — see
   `glycine_zwitterion.yaml` for the recommended template (includes a
   suggested-run-order comment block at the bottom).
3. Update `config_path` in the CSV row.
4. When you've actually run an experiment with the molecule, promote `phase`
   to `done` and add a REPORT.md section / link.

## Why this format

CSV instead of JSON / YAML for the catalog: trivially renderable on GitHub,
diffable in PRs, and Excel-openable for non-programmer collaborators. The
free-text `why_interesting` column is intentional — it's the field that
documents *scientific* judgement that doesn't fit into structured config.
