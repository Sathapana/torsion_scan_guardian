# Thunder Way — running Torsion Scan Guardian on Thunder Compute via VSCode

[Thunder Compute](https://thundercompute.com) rents GPU instances (T4, A100, H100, …) typically cheaper than RunPod / Lambda / AWS, with first-class CLI + VSCode integration. The big win over Colab: **a real Linux box you SSH into**, no 90-min idle timeout, no condacolab kernel-restart dance, no Drive-mount fiddling. The trade-off: it costs money (Colab Free is $0), and you must remember to stop the instance.

When this is worth doing:
- A multi-hour sweep keeps disconnecting on Colab and you need it to *finish*.
- You want VSCode's interactive debugger / notebook editor instead of Jupyter-in-browser.
- The molecule is large (≥40 atoms) and Colab T4's ~2× speedup is no longer enough.
- You want the same workflow as local-dev (open a folder in VSCode, run cells) but on someone else's GPU.

Pricing as of writing (verify on the [Thunder pricing page](https://www.thundercompute.com/pricing) — these change): T4 spot ≈ $0.27/h, A100 spot ≈ $0.78/h. For this project, a T4 is more than enough.

> **Note:** Thunder commands (`tnr ...`) and exact pricing change between releases. Where this doc shows `tnr <verb>`, treat the verb as a placeholder — run `tnr --help` on your machine after installing to see the current syntax, or consult [docs.thundercompute.com](https://docs.thundercompute.com). The *workflow* (install CLI → create instance → SSH from VSCode → run notebook) is stable; only the exact command names drift.

---

## TL;DR — five steps, ~10 min the first time

1. **Sign up** at https://console.thundercompute.com, add a payment method.
2. **Install the Thunder CLI** (`pip install tnr` or per [docs](https://docs.thundercompute.com)) and `tnr login`.
3. **Create a T4 instance** via the CLI or web console; note the instance ID.
4. **In VSCode**, install the **Thunder Compute extension** *or* the standard **Remote-SSH extension**. Either way, connect to the instance.
5. **In the connected VSCode**, clone the repo, install deps, open `notebooks/guardian_colab.ipynb` or `notebooks/guardian_sweep_colab.ipynb`, run cells. Same notebooks as Colab — they're written generically, the only Colab-specific cells (Drive mount, `condacolab`, `google.colab.files.download`) are skipped or replaced.

**When done:** `tnr stop <id>` (or "Stop" in the console). **Forgetting this is the #1 way to be surprised by a Thunder bill.**

---

## Detailed setup

### Step 1 — sign up and add billing

- Go to https://console.thundercompute.com
- Sign up (Google or email)
- Add a payment method. Thunder usually credits new accounts a few dollars for testing; the whole Phase-5 demo fits inside that credit.
- Set a **spending limit** in the account settings so a forgotten instance doesn't ruin your week. Recommended: $10/mo for prototyping.

### Step 2 — install the Thunder CLI

The CLI is named `tnr` (Thunder Runner). Installation varies by platform; the most common path is pip:

```bash
pip install tnr
tnr --version
tnr login                      # opens browser for OAuth
```

After login, your auth token is cached so subsequent commands don't re-prompt.

> Verify the exact install command against [docs.thundercompute.com/installation](https://docs.thundercompute.com) — they sometimes ship a standalone installer script (`curl ... | bash`) which is what I'd use on a fresh machine to avoid Python-version skew.

### Step 3 — install VSCode + extensions

If you don't already have VSCode:
- Download from https://code.visualstudio.com (free, all platforms)
- Install

Required extensions (search by name in VSCode's Extensions panel):
- **Python** (Microsoft) — the standard Python language support
- **Jupyter** (Microsoft) — runs `.ipynb` notebooks natively inside VSCode
- One of these, depending on your preference:
  - **Thunder Compute** (Thunder's official extension) — one-click connect, instance management UI inside VSCode. Preferred if it works on your platform.
  - **Remote - SSH** (Microsoft) — the generic SSH-into-anywhere extension. Works on every cloud, not just Thunder. Use if the Thunder extension isn't available or doesn't fit your workflow.

### Step 4 — create a GPU instance

Either from the web console (Console → Instances → New) or from the CLI:

```bash
# Example shape — check current Thunder CLI for exact subcommands and flags:
tnr create --gpu t4
tnr list
```

A T4 instance typically boots in 1–3 minutes. You'll get an instance ID (e.g., `tnr-abc123`) and an SSH address.

**Choose persistent storage** if Thunder offers it for your instance type — for this project, you want the seed datasets, fine-tune checkpoints, and run directories to persist between sessions. Otherwise you'll re-download MACE-OFF and re-build seed datasets every time you spin up a new instance.

### Step 5a — connect via the Thunder Compute VSCode extension

If you installed Thunder's extension:
1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Search for **"Thunder: Connect to instance"** (exact name may differ)
3. Pick your running instance
4. A new VSCode window opens, connected to the instance. The bottom-left status bar shows the instance ID.

### Step 5b — connect via Remote-SSH (universal fallback)

If you're using the generic Remote-SSH extension:
1. `tnr ssh <id>` from your local terminal — Thunder injects the SSH key into your `~/.ssh/config` as a host alias (e.g., `thunder-abc123`)
2. In VSCode: Command Palette → **"Remote-SSH: Connect to Host…"** → pick `thunder-abc123`
3. VSCode reopens connected to the instance, status bar shows `SSH: thunder-abc123`

Either path lands you in the same place: a VSCode window where every terminal, file browse, and notebook runs on the Thunder GPU instance.

### Step 6 — set up the Python environment on the instance

In the VSCode terminal (which is the Thunder instance's shell):

```bash
# Verify the GPU is visible
nvidia-smi

# Clone the repo. SSH or HTTPS — HTTPS is simpler the first time.
git clone https://github.com/Sathapana/torsion_scan_guardian.git
cd torsion_scan_guardian

# Option A: conda env (matches local dev environment exactly)
conda env create -f environment.yml
conda activate guardian

# Option B: pip-only (faster, smaller)
pip install --upgrade pip
pip install mace-torch ase rdkit matplotlib pandas pydantic pyyaml tqdm pytest numpy
pip install -e .

# In both cases: install xtb for the GFN-FF oracle
conda install -y -c conda-forge xtb
# (if not using conda: download xtb binary from https://github.com/grimme-lab/xtb/releases
# and add it to your PATH)

# Smoke test
pytest -q tests/ -k 'not test_ensemble_predict_h2o'
```

Then pre-warm the MACE-OFF cache (~7 MB download):

```bash
python -c "
import torch
from mace.calculators import mace_off
mace_off(model='small', device='cuda' if torch.cuda.is_available() else 'cpu')
import os; p = os.path.expanduser('~/.cache/mace/MACE-OFF23_small.model')
assert os.path.exists(p); print('cache ready:', os.path.getsize(p)//1024, 'KB')
"
```

### Step 7 — open a notebook in VSCode and run it

1. **File → Open Folder…** → pick the `torsion_scan_guardian` directory you cloned
2. Open `notebooks/guardian_sweep_colab.ipynb` (or `guardian_colab.ipynb`)
3. **Top-right of the notebook** → click the kernel selector → pick the `guardian` conda env (or your pip env)
4. Run cells

**Skip the Colab-specific cells**:
- Cell that mounts Google Drive (`drive.mount(...)`) — you're not on Colab, skip it. The repo is already on the instance.
- The `condacolab.install()` cell — already done via the conda install above.
- The "verify clone present" cell — only relevant after a Drive clone; skip.
- The bundle-and-download cell at the end (`files.download`) — this is Colab-only. Use git instead (see Step 9).

You can either delete these cells in your local copy, or just *not click* them when running cells one-by-one. The notebook is structured so it works either way.

Alternative: skip the notebook entirely and run the script directly from a VSCode terminal — for the sweep, this is actually cleaner since you get streaming output without notebook overhead:

```bash
MPLBACKEND=Agg python scripts/sweep_molecules.py \
    --phase-filter todo candidate \
    --steps 4000 --temperature 300 \
    --device cuda
```

### Step 8 — monitor

In VSCode:
- **The notebook output** streams live as cells run.
- **A second terminal** (Terminal → New Terminal) can run `nvidia-smi -l 5` to watch GPU utilisation.
- **The Files panel** shows new files appearing under `runs/sweep/` as molecules complete.

You can disconnect VSCode and reconnect later — the running processes keep going on the Thunder instance as long as the instance itself is up. Use `nohup ... &` or `tmux`/`screen` if you want absolute certainty that VSCode disconnects don't kill long-running scripts (most don't, but the safest pattern is `tmux new -s sweep; python scripts/sweep_molecules.py …; <Ctrl-B d to detach>`).

### Step 9 — push results back to GitHub

Two ways. Pick A if your local SSH key is set up for the repo (the simplest); B if you want to push from the Thunder instance directly.

**Option A — pull-then-push-from-local (no credentials on Thunder)**

On the Thunder instance:
```bash
git add figures/sweep_summary.png figures/sweep_*.png
git add data/seed/*.xyz config/molecules/*.yaml
git status
git diff --stat
git push      # only works if you set up SSH on Thunder (see Option B)
```

Actually wait — Option A really means: push from the Thunder instance using HTTPS + a token, OR push the results from the Thunder instance into Drive / S3 / SCP and reassemble locally. The cleanest is to give Thunder write access *to this one repo* via a fine-grained PAT:

**Option B — Personal Access Token on Thunder**

1. Create a fine-grained PAT at https://github.com/settings/personal-access-tokens/new
   - **Repo access:** Only `torsion_scan_guardian`
   - **Permissions:** `Contents: Read and write`
   - **Expiration:** 30 days (rotate)
2. On the Thunder instance:
   ```bash
   git remote set-url origin https://Sathapana:<TOKEN>@github.com/Sathapana/torsion_scan_guardian.git
   git config user.email "sathapana.chawananon@gmail.com"
   git config user.name  "Sathapana"
   git add ... && git commit -m "..." && git push
   ```
3. When you're done with the instance, **revoke the token** at https://github.com/settings/personal-access-tokens (don't leave it lying around — instances can be snapshotted/imaged).

**Option C — SSH key on Thunder**

Generate a new ed25519 key on the Thunder instance, add the `.pub` to your GitHub account's SSH keys (same flow as the local-setup README), update `git remote set-url origin git@github.com:Sathapana/torsion_scan_guardian.git`. Slightly more secure than a PAT because keys can't be globally exfiltrated, but more setup.

### Step 10 — STOP THE INSTANCE

The single most important step in this whole document:

```bash
# From your local terminal (NOT the Thunder instance's shell):
tnr stop <instance-id>

# Or in the web console: Instances → your instance → Stop
```

**An idle T4 left running for a week costs ~$45.** Make this a habit: every time you're done for the session, stop the instance. Restarting later is fast (1–3 min) and persistent storage (if you enabled it) preserves your work.

For longer-term peace of mind, set an alarm in the web console to auto-stop after N hours of idle CPU.

---

## Tips

| | |
| --- | --- |
| **Use VSCode's `Remote: Show Resource Usage`** | Bottom status bar shows CPU/RAM/GPU of the remote machine. Notice when you've finished a job and the instance is idle. |
| **Set a `auto-stop on idle` rule** in the Thunder console | Belt to the suspenders of "remembering to stop". Saves money when you forget. |
| **Persistent storage > ephemeral storage** | The first MACE-OFF download is ~7 MB but the seed datasets you build are ~300 KB each, and the fine-tune checkpoints are ~5 MB × 3 = 15 MB per molecule. Lose those, you re-build them. |
| **`tmux` for long sweeps** | `tmux new -s sweep; python scripts/sweep_molecules.py ...; <Ctrl-B d>` detaches without killing. Reconnect later with `tmux attach -t sweep`. Protects against VSCode disconnects. |
| **Use the Thunder CLI's `tnr scp` if available** | Bulk-copying figures back to local is faster via scp than via git for binary files. |
| **Two terminals in VSCode** | One running the sweep, one with `watch -n 5 nvidia-smi` to monitor GPU. |
| **Push during the sweep, not after** | The script writes incrementally to `runs/sweep/sweep_summary.csv`. Periodic `git add runs/sweep/sweep_summary.csv && git commit && git push` from a second terminal lets you check progress from your laptop without VSCode being connected. |
| **Save the instance image** if Thunder supports it | After you've installed conda + xtb + pre-downloaded MACE-OFF, snapshot the instance. Next time you spin up, you skip 5 min of setup. |
| **Use `conda env export > environment.yml.lock`** | If you change any deps on the instance, snapshot the env so reproducing on a fresh instance is one command. |
| **Don't forget GPU vs CPU dtype** | Thunder GPU runs MACE-OFF as float64 by default (same as Colab), so threshold calibrated on local CPU (float32) won't transfer. Always `--calibrate` on the device you'll run on. See [REPORT §13.7](REPORT.md). |

---

## Problems you might hit (predicted; cross-check with experience)

This list is shorter than [Colab_WAY.md](Colab_WAY.md)'s because Thunder is closer to "regular Linux + a GPU" — none of the Colab-specific issues (`%%bash` cd, condacolab kernel restart, Drive mount conflicts) apply. The ones below are educated guesses, not battle-tested. Open an issue if you hit something not here.

### 1. `nvidia-smi` shows "no GPU detected" inside VSCode terminal but works from raw SSH

**Cause:** VSCode's terminal can sometimes inherit a different `$PATH` or `$LD_LIBRARY_PATH` than a fresh login shell, missing CUDA libs.

**Fix:** Add to your `~/.bashrc` on the instance:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Then reload the terminal. Or close-and-reopen the VSCode SSH connection.

### 2. `conda install -y -c conda-forge xtb` is slow or fails to resolve

**Cause:** conda solver is slow on large environments. The Thunder base image may have a lot pre-installed.

**Fix:** Use `mamba` instead — drop-in conda replacement with a much faster solver:
```bash
conda install -y -c conda-forge mamba
mamba install -y -c conda-forge xtb
```

### 3. Notebook cell hangs at "Connecting to kernel…"

**Cause:** VSCode's Jupyter extension is starting a Python process on the remote. Sometimes the kernel selector picks the wrong interpreter.

**Fix:** Top-right of the notebook → kernel picker → explicitly choose `Python 3.11 (guardian)` or the path that matches `which python` in the terminal. If conda env doesn't appear, click "Select Another Kernel" → "Python Environments" → browse to `~/miniconda3/envs/guardian/bin/python`.

### 4. `pip install -e .` errors because gcc is missing

**Cause:** Some MACE/torch wheels need a C compiler available at install time even for binary wheel install (rare, but happens with old setuptools).

**Fix:**
```bash
sudo apt-get update && sudo apt-get install -y build-essential
```

### 5. SSH connection drops every few minutes

**Cause:** Local-to-Thunder SSH connection is going through NAT / your laptop's wifi sleep behavior.

**Fix:** Add to your **local** `~/.ssh/config`:
```
Host thunder-*
    ServerAliveInterval 60
    ServerAliveCountMax 10
```
And run long jobs inside `tmux` so VSCode disconnects don't kill them.

### 6. Bill is bigger than expected after a weekend

**Cause:** You forgot to stop the instance.

**Fix:** Use the Thunder console's spending alerts. Set up auto-stop rules. Make a habit of `tnr list` at the start of each session — if you see any instance you don't recognise, stop it.

---

## Cost summary

For this project specifically, comparing Thunder vs Colab vs local:

| Setup | Phase-5 demo (1 molecule) | Phase-6 sweep (3–7 molecules) | Notes |
| --- | --- | --- | --- |
| Local CPU | $0 | $0, ~3 h | Free. Slow but private and zero hassle. |
| Colab Free (T4) | $0 | $0, ~2 h | Great but 90-min idle timeout + condacolab dance. |
| Thunder T4 spot | ~$0.10 | ~$0.50 | Comfortable VSCode, no idle timeout, persistent storage. |
| Thunder A100 spot | ~$0.30 | ~$1.50 | Worth it only for large molecules (≥40 atoms) or many ensemble members. |
| Local + Thunder mix | per Thunder run | per Thunder run | Develop locally, batch-run on Thunder when needed. **My recommended workflow.** |

The break-even between "use Colab Free" and "pay a dollar for Thunder" is: how much your *time* is worth vs the cost of restarting after a disconnect. For 2-hour sweeps, paying $0.50 to skip the babysitting is usually worth it.

---

## When to use Thunder vs Colab vs local

| Situation | Use this |
| --- | --- |
| One-off experiment on a tiny molecule | Local CPU |
| Quick prototype on a medium molecule, $0 budget | Colab Free |
| Multi-hour sweep, $0 budget, willing to babysit | Colab Free |
| Multi-hour sweep, want to leave unattended | **Thunder** (or RunPod spot) |
| You want VSCode debugging / proper IDE | **Thunder** (or any SSH-able cloud) |
| Production runs ≥10 cycles per molecule | Thunder + worker-process fine-tuner (REPORT §12.4 future work) |
| You forget to stop instances and have a fixed budget | Colab Free (you can't accidentally overspend) |

---

## Reference: official Thunder docs

- [docs.thundercompute.com](https://docs.thundercompute.com) — Thunder's official documentation (verify exact CLI commands here)
- [console.thundercompute.com](https://console.thundercompute.com) — Web console for instance management + billing
- [Thunder VSCode extension marketplace page](https://marketplace.visualstudio.com) — search "Thunder Compute" inside VSCode

For the Torsion Scan Guardian project specifically:
- [`README.md`](README.md) — project overview
- [`REPORT.md`](REPORT.md) §13 — compute environment analysis comparing CPU / Colab / Thunder / Modal
- [`Colab_WAY.md`](Colab_WAY.md) — the Colab-specific equivalent of this doc, including 12 problems we've battle-tested
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — local-dev conventions that mostly apply on Thunder too
