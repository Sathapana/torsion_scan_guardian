"""Fine-tune one MACE-OFF small member on the GFN-FF seed dataset.

Usage:
  python scripts/finetune_member.py --seed 0 --epochs 5 --lr 5e-4

Output: a checkpoint at runs/finetune/member_seed<N>/ that the multi-checkpoint
ensemble loader can pick up in Step 3 of Phase 2.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# Force UTF-8 stdout so mace's help/log strings don't crash the Thai cp874 codec on Windows.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", type=Path, default=Path("data/seed/ibuprofen_seed.xyz"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--valid-fraction", type=float, default=0.18)
    p.add_argument("--out-root", type=Path, default=Path("runs/finetune"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    out_dir = args.out_root / f"member_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"member_seed{args.seed}"

    foundation = str(Path.home() / ".cache" / "mace" / "MACE-OFF23_small.model")
    if not Path(foundation).exists():
        raise FileNotFoundError(
            f"MACE-OFF small not cached at {foundation}; run a Phase-1 inference first to download it."
        )

    argv = [
        "mace_run_train",
        "--name", name,
        "--train_file", str(args.train_file),
        "--valid_fraction", str(args.valid_fraction),
        "--foundation_model", foundation,
        "--multiheads_finetuning", "False",
        "--energy_key", "energy",
        "--forces_key", "forces",
        "--E0s", "average",
        "--max_num_epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--device", args.device,
        "--default_dtype", "float32",
        "--loss", "weighted",
        "--forces_weight", "10.0",
        "--energy_weight", "1.0",
        "--model_dir", str(out_dir),
        "--checkpoints_dir", str(out_dir / "checkpoints"),
        "--results_dir", str(out_dir / "results"),
        "--log_dir", str(out_dir / "logs"),
        "--save_cpu",
    ]
    print("[finetune] argv:", " ".join(argv), flush=True)
    sys.argv = argv

    from mace.cli.run_train import main as mace_main
    t0 = time.time()
    mace_main()
    elapsed = time.time() - t0
    print(f"[finetune] member_seed{args.seed} done in {elapsed:.1f}s "
          f"({elapsed/max(args.epochs,1):.1f}s/epoch)", flush=True)


if __name__ == "__main__":
    main()
