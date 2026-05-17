# Colab Way — running Torsion Scan Guardian on Google Colab

Practical tutorial + the complete list of problems we hit and how each was fixed. Read top-to-bottom the first time; use the **Problems we hit** table as a lookup after that.

## TL;DR — first-time setup, 10 minutes of clicks

1. Open [`notebooks/colab/guardian_sweep_colab.ipynb`](notebooks/colab/guardian_sweep_colab.ipynb) (multi-molecule sweep) or [`notebooks/colab/guardian_colab.ipynb`](notebooks/colab/guardian_colab.ipynb) (single-molecule demo) on github.com, click *Open in Colab*.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run cells **1 → 2 → 3 → 4 → 5** (verify GPU, mount Drive + clone, sanity-check, pip install, condacolab install).
4. **Condacolab restarts the kernel automatically.** When it does, re-run cells **1 → 2 → 3 → 4** (don't re-run 5).
5. Run cells **6 → 7 → 8** (conda install xtb, pre-warm MACE-OFF, pytest smoke).
6. Pick what to run (single molecule or sweep), edit the config cell, run the sweep cell.
7. When it's done, run the bundle cell → downloads a zip to your laptop.
8. Locally: unzip into the repo, `git add` / `commit` / `push` (push from local — keep GitHub credentials out of Colab).

Total wall time, including ~3 min of installs: **~30 min for a single molecule, ~2 h for the 7-molecule sweep**, all on Colab Free.

---

## Detailed setup (first session on a fresh machine / Drive)

### Step 1 — request the GPU runtime

Colab Free gives you a T4 GPU on request:

`Runtime → Change runtime type → Hardware accelerator: GPU → GPU type: T4`

The notebook checks this in cell 1 (`torch.cuda.is_available()`). If you see `Device: CPU`, you didn't get the GPU yet — usually a queue issue at peak hours; retry in 10 minutes.

#### Which accelerator should I pick on Colab? (decision table for *this project*)

Colab Pro and Pro+ unlock additional GPUs and TPUs. Their value for the Torsion Scan Guardian workload (MACE-OFF small, ~5 M params, ≤40-atom drug-like molecules, 3-member ensemble) is not the same as the raw spec sheet — see the rightmost column.

| Accelerator | Memory | FP32 / FP16 (TFLOPS) | Arch / year | Colab tier | Verdict for **this** project |
| --- | ---: | ---: | --- | --- | --- |
| **T4 GPU** | 16 GB | 8.1 / 65 | Turing 2018 | **Free** + Pro | **✓ Default; what we tested.** Phase-5 demo in ~5 min, full 7-molecule sweep in ~2 h (~30–45 min with the §13.8 throughput optimisations active, default since commit `<this hash>`). Fine for ≤50 heavy atoms. |
| **L4 GPU** | 24 GB | 30 / 121 | Ada 2023 | Pro / Pro+ | **✓ Best price/perf if you have Pro.** Roughly 2–3× T4 speed for our workload; finishes sweeps in ~1 h. Burns Pro compute units a bit faster but the wall-time win is real. |
| **A100 GPU** (40 GB) | 40 GB | 19.5 / 312 | Ampere 2020 | Pro / Pro+ | **✓ for ≥50-atom molecules or 5+ ensemble members.** Overkill for sulfanilamide-scale (19 atoms) — see [REPORT §13.7](REPORT.md): GPU speedup is bounded by graph-construction overhead at small atom counts, so A100 is only ~1.3× faster than T4 here. |
| **H100 GPU** | 80 GB | 67 / 1979 (FP16-sparse) | Hopper 2022 | Pro+ (rare/queued) | **✗ Wrong fit.** Built for FP8 LLM training and very large-batch workloads. MACE-OFF small uses ~5 M params; you'd burn 30–50× T4 compute units for ~0% improvement on this workload. Useful elsewhere, not here. |
| **G4 GPU** | (varies) | (varies) | — | (verify in Colab UI) | **? — not a standard NVIDIA SKU.** If Colab exposes this in the runtime selector, click "Show resources" and check `nvidia-smi`; it's likely an A4 / L4 / T4 re-badge for one of Google's instance families. Treat as "verify before relying on it." |
| **TPU v5e-1** | 16 GB HBM | ~197 BF16 | TPU v5e 2023 | Pro / Pro+ | **✗ Not compatible.** MACE-OFF runs on PyTorch + e3nn; e3nn's spherical-harmonic kernels don't go through `torch_xla` cleanly. Porting would be a project of its own. TPUs shine on TensorFlow / JAX / Flax workloads. |
| **TPU v6e-1** (Trillium) | 32 GB HBM | ~918 BF16 | TPU v6e 2024 | Pro+ | **✗ Same reason as v5e-1.** Brand-new + more memory, but no torch_xla path that runs e3nn out of the box. |

**Bottom line for this project:**
- **Free tier:** stick with **T4**. The shipped notebooks are tested on it; the §13.7 GPU validation numbers all come from T4.
- **Have Colab Pro / Pro+:** **L4** if it's available (sweet-spot price/perf), otherwise T4. Step up to A100 only when your molecule is ≥40 heavy atoms or you've increased `--n_probes` (ensemble members) past 5.
- **Skip H100, TPUs, and any "G4 GPU" you can't verify.** Wrong tool for a 5 M-param equivariant GNN inferring on small molecules.

#### Why "bigger GPU" isn't a linear speedup here

This bites people coming from LLM workflows. For LLM inference you scale roughly linearly with FP16 TFLOPS, so going T4 → A100 → H100 is a real ~30× total speedup. For *this* project:

- MACE-OFF small's forward pass on a 20-atom molecule is dominated by **graph construction** (neighbour-list build, edge-feature gathers) and **small-kernel launches**, not by big matmuls.
- GPU underutilisation: the actual FLOPs per step are tiny. A T4 finishes a forward pass in ~30 ms; an A100 in ~23 ms; an H100 in ~21 ms. The MD step takes the same wall time on all of them ± noise.
- This is **why we only saw ~2× speedup vs CPU on Colab T4** (REPORT §13.7) — the bottleneck is overhead, not compute.

So pick the cheapest accelerator that's available to you (T4 on free, L4 on Pro). The difference between T4 and H100 for this workload is negligible in wall time and large in compute-unit burn rate.

### Step 2 — mount Google Drive and clone the repo into it

Why Drive: results (`runs/`, `figures/`, fine-tune checkpoints, seed datasets) **persist across Colab sessions**. If you don't mount Drive, every Colab disconnect wipes all of your work.

Cell 2 in both notebooks does this:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
REPO_URL = 'https://github.com/Sathapana/torsion_scan_guardian.git'
REPO_DIR = '/content/drive/MyDrive/torsion-scan-guardian'
!test -d torsion-scan-guardian || git clone $REPO_URL torsion-scan-guardian
%cd torsion-scan-guardian
```

First run: clones the repo (~5 sec). Subsequent runs: no-op except `cd`.

### Step 3 — install Python dependencies via `uv`

Colab already has CUDA-enabled PyTorch. All other runtime dependencies are declared in [`pyproject.toml`](pyproject.toml)'s `[project] dependencies`, so a single editable install pulls everything in. We use [`uv`](https://github.com/astral-sh/uv) — a drop-in pip replacement that's ~10× faster.

```bash
pip install -q uv
cd /content/drive/MyDrive/torsion-scan-guardian
uv pip install --system -q -e ".[dev]"
```

Takes ~20–40 s (vs ~3 min on plain pip). The `[dev]` extras include pytest for the smoke-test cell; drop the suffix if you don't want it.

The `cd` is **essential** in `%%bash` cells — see problem #2 below. The `--system` flag tells `uv` to install into the kernel's current Python environment (not a `.venv/`), which keeps things simple on Colab — the kernel finds the freshly-installed packages without any kernel switching.

**Why not literal `uv sync`?** `uv sync` is uv's project workflow: it creates a `.venv/` and installs there. On Colab the Jupyter kernel uses the system Python, not `.venv/python`, so a literal `uv sync` would require registering and switching kernels mid-session — defeats the "one command" benefit. `uv pip install --system` gives you the same speed without the venv dance.

### Step 4 — install xtb (for the GFN-FF oracle)

xtb isn't on Colab by default and isn't on PyPI for the version we want. The clean path is conda-forge via `condacolab`:

```python
!pip install -q condacolab
import condacolab
condacolab.install()    # !! RESTARTS THE KERNEL !!
```

When this finishes, the kernel **restarts automatically**. You'll lose all imports, mounted Drive state, and IPython's `%cd`. **Re-run cells 1–4** (verify GPU, mount Drive, sanity check, pip install) before continuing.

Then install xtb itself:
```bash
conda install -y -c conda-forge xtb
```

### Step 5 — pre-warm the MACE-OFF cache

This step prevents the most common failure mode (problem #4). One cell:

```python
import torch
from mace.calculators import mace_off
mace_off(model='small', device='cuda' if torch.cuda.is_available() else 'cpu')
import os
p = os.path.expanduser('~/.cache/mace/MACE-OFF23_small.model')
assert os.path.exists(p), f'MACE-OFF cache missing at {p}'
print('cache:', os.path.getsize(p) // 1024, 'KB')
```

Downloads ~7 MB on first run. Subsequent runs are no-ops.

### Step 6 — pick your experiment

Two notebooks, two different scopes:

- **[`guardian_colab.ipynb`](notebooks/colab/guardian_colab.ipynb)** — single-molecule. Runs Phase-5 demo on sulfanilamide by default. ~5 min on GPU. Use this for quick iteration on one molecule.
- **[`guardian_sweep_colab.ipynb`](notebooks/colab/guardian_sweep_colab.ipynb)** — multi-molecule sweep over [`data/molecule_library/candidates.csv`](data/molecule_library/candidates.csv). Configurable via a single editable cell. Use this for Phase-6 work (try several candidates, see which baseline collapses).

The sweep is **idempotent**: re-running after a disconnect skips already-built seed datasets and fine-tune checkpoints. So if Colab disconnects mid-sweep, just re-run the sweep cell and it picks up where it stopped.

### Step 7 — bundle and download results

The last cell in both notebooks zips the small, version-worthy artifacts (figures, summary JSONs, seed datasets, auto-created configs) and triggers a browser download. **Never push from Colab** — instead:

1. Unzip the downloaded `guardian_results.zip` (or `guardian_sweep_results.zip`) into the repo on your local machine
2. `git status` to review what changed
3. `git add` the figures you want versioned + any new seed datasets + any new per-molecule configs
4. `git commit -m "Phase 6 results: ..."` then `git push` (works because your local SSH key is already trusted)

This keeps GitHub credentials out of Colab entirely.

---

## Tips

| | |
| --- | --- |
| **Use Option A (Drive-mounted clone), not Option B (ephemeral)** | Sweep checkpoints survive disconnects. Per-molecule fine-tune runs take 1–2 min each, you don't want to redo them. |
| **Pre-warm MACE-OFF before any batch operation** | Failing 3 minutes into a 2-hour sweep is much worse than failing at setup. |
| **Check `os.getcwd()` after every kernel restart** | If you're not in `/content/drive/MyDrive/torsion-scan-guardian`, relative paths in subsequent cells will silently fail. |
| **Keep the Colab tab visible during long sweeps** | Free tier has a ~90-min idle timeout. The sweep's incremental CSV write means you don't lose finished molecules, but avoiding the timeout means you can leave it unattended. |
| **`MPLBACKEND=Agg` for any matplotlib-touching script** | Colab's default backend needs a display; headless invocations crash without it. |
| **For multi-account GitHub SSH**, configure `~/.ssh/config` with separate `Host` aliases per account | See the project repo's `README.md` setup notes. Colab uses HTTPS clone so this is local-only. |
| **`git pull` after each fix push from a local session** | Colab's autosaved notebook conflicts can be resolved with `git restore notebooks/*.ipynb` before pulling. |
| **Wall time on T4 vs CPU is ~2×, not 10×, for tiny molecules** | Graph construction dominates at small atom counts. Don't expect linear-in-N speedup for ≤20-atom molecules. |
| **Calibrate the threshold per device** | float64 (Colab default) vs float32 (local default) gives ~15% different noise floors. Threshold from one doesn't transfer. See REPORT §13.7. |
| **The cell numbering shifts when you insert/delete cells** | "Re-run cells 1–4 after kernel restart" assumes the notebook hasn't been edited. If you've inserted cells, re-count. |

---

## Problems we hit (and how each was fixed)

Listed in roughly the order encountered. Most are now baked into the shipped notebooks or scripts so you won't hit them again — but worth knowing in case you fork a new variant.

### 1. `tblite-python` / `xtb-python` Windows DLL crash

**Symptom (local Windows, not Colab):** `Windows fatal exception: code 0xc06d007f` inside `singlepoint` → `handle_context_error`. Affects both Python bindings of tblite and xtb.

**Cause:** Windows delay-load DLL failure in the Fortran error-handling path. The standalone `xtb.exe` CLI works fine; only the Python bindings crash.

**Fix:** [`src/guardian/oracle/xtb_subprocess.py`](src/guardian/oracle/xtb_subprocess.py) — thin ASE Calculator that shells out to `xtb.exe` and parses the Turbomole `energy`/`gradient` outputs. Used by default everywhere. Adds ~150 ms per call (subprocess startup), which is negligible compared to typical MD timings.

### 2. `%%bash` cells don't see `%cd`

**Symptom:** A script invoked from `%%bash` says "file not found" for a path that exists.

**Cause:** `%%bash` spawns a fresh subprocess. The cwd it inherits is the kernel's process cwd at start, **not** the cwd set by `%cd` magic in an earlier IPython cell.

**Fix:** Every `%%bash` cell in the shipped notebooks starts with an explicit:
```bash
cd /content/drive/MyDrive/torsion-scan-guardian
```

### 3. `condacolab.install()` restarts the kernel

**Symptom:** Cells run before `condacolab.install()` "stop working" — imports gone, Drive unmounted, `%cd` lost.

**Cause:** condacolab swaps the Python interpreter. The kernel restarts is by design.

**Fix:** After running the condacolab cell, **re-run cells 1–4** (verify GPU, mount Drive, sanity check, pip install). The shipped notebooks document this in the markdown header.

### 4. MACE-OFF cache missing → all fine-tunes fail

**Symptom:** Sweep cell shows every molecule with status `finetune-failed`:
```
FileNotFoundError: MACE-OFF small not cached at /root/.cache/mace/MACE-OFF23_small.model
```

**Cause:** [`scripts/finetune_member.py`](scripts/finetune_member.py) needs the MACE-OFF foundation checkpoint as input to `mace_run_train --foundation_model`. The cache is empty on fresh Colab. The single-molecule notebook had a pre-download cell; the sweep notebook didn't.

**Fix** (commit `ce83fda`):
- The script now **auto-downloads** MACE-OFF if the cache file is missing, so it's self-contained.
- The sweep notebook gained a pre-warm cell that downloads + asserts the file exists, failing fast at setup instead of 30 min into a sweep.

### 5. Em-dash (U+2014) in Python source rejected by Colab tokenizer

**Symptom:**
```
File "scripts/sweep_molecules.py", line 8
    5. Run a *baseline* MD (Guardian disabled — threshold = 999, ...).
                                              ^
SyntaxError: invalid character '—' (U+2014)
```
Worked locally on Windows, failed on Colab.

**Cause:** Python 3.12+ tokenizer treats em-dash as a "confusable Unicode operator" and refuses it at the module top level — even inside a triple-quoted docstring. Older Python is more lenient. Colab is on a stricter version.

**Fix** (commit `db35c40`): replaced **17 em-dashes across 9 `.py` files** with ASCII `--`. Markdown files (this one, REPORT.md, etc.) keep their em-dashes because they're not Python-parsed.

### 6. Colab autosaves notebooks → `git pull` conflicts

**Symptom:**
```
error: Your local changes to the following files would be overwritten by merge:
    notebooks/colab/guardian_sweep_colab.ipynb
Please commit your changes or stash them before you merge.
```

**Cause:** Colab autosaves the open `.ipynb` back to Drive whenever you run cells (cell output state changes). Git sees this as a local modification even though you didn't intentionally edit.

**Fix** — two acceptable workflows:

**Option A — discard before pulling** (simplest, what we typically do):
```bash
!git restore notebooks/*.ipynb && git pull --ff-only origin main
```
Safe because the cell-output state has no scientific value — all real results go to `runs/` and `figures/`, not into the notebook.

**Option B — strip outputs automatically** (one-time persistent setup):
```bash
!pip install -q nbstripout && nbstripout --install
!git config filter.nbstripout.required true
```
After this, git ignores notebook cell-output diffs entirely.

### 7. matplotlib backend missing display on Colab

**Symptom:** Scripts that call `matplotlib.pyplot` from a subprocess (not from a notebook cell) crash with `_tkinter.TclError: no display name and no $DISPLAY environment variable`.

**Cause:** Colab's default matplotlib backend assumes a display.

**Fix:** Prefix script invocations with `MPLBACKEND=Agg`:
```bash
MPLBACKEND=Agg python scripts/analyse_al_demo.py runs/sulf_phase5_colab
```
The shipped notebooks already do this. Internally, `analyse_al_demo.py` and `make_phase2_figures.py` also do `matplotlib.use("Agg")` as belt-and-suspenders.

### 8. `find -exec` escape mangled through JSON cell source

**Symptom:** A `find ... -exec cp --parents {} dest/ \;` line in the bundle cell silently does nothing on Colab.

**Cause:** JSON-encoding a `\;` in the notebook source requires `\\;`, which decodes back to `\;`. But IPython's `!` magic passes that to bash, which interprets `\\` as a literal backslash + `;` as command separator. find sees `\` as its terminator (which doesn't match `\;`) and errors silently.

**Fix:** Replaced the `find -exec` with a Python `glob.glob` loop in the bundle cell. No shell escaping involved.

### 9. Drive idle timeout disconnects during long sweeps

**Symptom:** Colab tab shows "Runtime disconnected" after ~90 minutes of inactivity. Some sweep results lost.

**Cause:** Colab Free has an idle-timeout policy. The "idle" is measured by Colab's frontend; if you have the tab visible, you're fine.

**Fix** — three independent mitigations:
1. **The sweep writes its CSV after every molecule**, not at the end. So if you disconnect after molecules 1–4 of 7, the CSV has rows for 4. Re-running the sweep is idempotent (skips already-built artifacts), so you resume from molecule 5.
2. **Mount Drive (Option A)** so the seed datasets and fine-tune checkpoints persist. Re-running picks them up.
3. **Keep the tab visible.** Or use a browser autoclicker / a tab-keep-alive extension. (Some people consider this against TOS; check first.)

### 10. float64 vs float32 changes the calibrated threshold

**Symptom:** A threshold calibrated on local CPU (float32) gives "0 triggers" on Colab GPU (float64), and vice versa.

**Cause:** `mace_off(..., default_dtype=...)` defaults to float64 unless you force float32. The ensemble's relaxed-minimum disagreement drops ~15% from float32 to float64 because the three members produce more similar predictions at higher precision.

**Fix / workaround:**
- **Always `--calibrate` on the device you'll run on.** The shipped notebooks call calibration before each per-molecule MD.
- If you want exact reproducibility across devices, hard-code `default_dtype="float32"` in `mace_off(...)` calls. The local pipeline does this in `config/default.yaml`.

Documented in detail in [REPORT.md §13.7](REPORT.md).

### 11. GPU speedup is ~2× for tiny molecules, not 10×

**Symptom:** Expected "Colab T4 is 10× faster than CPU"; got "Colab T4 is 2.1× faster" on sulfanilamide.

**Cause:** For ≤20-atom molecules, the MACE-OFF forward pass is dominated by graph-construction overhead and small CUDA kernel launches. The actual FLOPs are tiny; GPU underutilised. The 10× quoted in benchmark papers is for larger systems (≥50 atoms) where kernels are big enough to amortise the overhead.

**Mitigation:** For tiny molecules, CPU is competitive. For Phase-6 work on small targets (glycine, sulfonyl chloride, NMA), CPU local + sweep at home overnight may beat Colab GPU + babysitting the tab. For larger molecules (peptides, drug-likes ≥30 atoms), GPU is meaningfully faster.

### 12. Mounting Drive when already mounted

**Symptom:**
```
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
```

**Cause:** Re-running the mount cell.

**Fix:** It's a warning, not an error. Ignore. Or pass `force_remount=True` if you genuinely need a fresh mount (rare).

---

## Cost summary

| Setup | Cost per experiment | Best for |
| --- | --- | --- |
| Colab Free (T4 GPU) | $0 | Phase 6 first attempts; multi-molecule sweeps under ~3 h |
| Local CPU | $0 (electricity) | Iteration on tiny molecules; overnight sweeps |
| RunPod spot RTX 3090 | ~$0.20/h, ~$1 per molecule | When Colab keeps disconnecting and you need 5+ h uninterrupted |
| Lambda Labs on-demand T4 | ~$0.50/h, ~$2 per molecule | Predictable, no preemption, slightly more expensive than spot |
| Modal serverless | per-second billing | Once you have a worker-process fine-tuner (REPORT §12.4) — best $/throughput for production sweeps |

Most of the project's actual experiments through Phase 5 ran on **local CPU** because each was ~10–30 min. For Phase 6 (longer sweeps, several molecules), Colab Free is the natural choice.

---

## Pushing results back to GitHub from Colab

Don't try to push from Colab itself — it requires putting credentials (a Personal Access Token) inside the notebook, which is shared with Google and might log to telemetry. Instead, use the **bundle + local push** flow:

### From Colab

The last cell in both notebooks (`Bundle results for download`) creates a zip:
```python
import shutil, os, glob
from google.colab import files
bundle_dir = '/content/guardian_sweep_results'
os.makedirs(bundle_dir, exist_ok=True)
# ... copies figures, summary JSONs, seed datasets, configs ...
shutil.make_archive('/content/guardian_sweep_results', 'zip', bundle_dir)
files.download('/content/guardian_sweep_results.zip')
```

You get a `.zip` in your Downloads folder.

### Locally

```powershell
cd "C:\Users\satha\Documents\DataProject_Sathapana\TorsionalScanGrandian_mol_project"
Expand-Archive -Force -Path "$env:USERPROFILE\Downloads\guardian_sweep_results.zip" -DestinationPath .

# Review what changed
git status

# Stage selectively — figures + new seed datasets + new configs, not raw runs/
git add figures/sweep_summary.png figures/<molecule>_*.png
git add data/seed/<new_molecule>_seed.xyz
git add config/molecules/<new_molecule>.yaml

# Commit + push (your local SSH is already set up)
git commit -m "Phase 6 sweep: <description>"
git push
```

CI runs automatically on push. If green, the sweep results are live on the GitHub repo home.

---

## When you get really stuck

In order of "what to try first":

1. **`git pull --ff-only origin main`** in your Colab session. Most production issues are fixed and pushed by the time you hit them.
2. **Restart the runtime entirely** (`Runtime → Disconnect and delete runtime`), open a fresh one, re-run from cell 1. Solves about 60% of "weird state" problems.
3. **Check `os.getcwd()`** — wrong cwd accounts for half the rest.
4. **Read the stderr file** for any failed subprocess: `!cat runs/finetune_<molecule>/member_seed0/stderr.log | tail -30`. The sweep prints the last 1500 chars to stdout but the full log is in the stderr file.
5. **File a GitHub issue** at https://github.com/Sathapana/torsion_scan_guardian/issues with: command run, last 30 lines of output, output of `os.getcwd()` and `torch.cuda.is_available()`.
