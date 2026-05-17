"""Build 3 bootstrap subsets of the seed dataset, one per ensemble member.

Each output file gets a different 80% random subset of the input frames,
seeded so the splits are deterministic. This forces meaningful divergence
between ensemble members during fine-tuning: classic deep-ensemble recipe.
"""
import argparse
from pathlib import Path
import numpy as np
from ase.io import read, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("data/seed/ibuprofen_seed.xyz"))
    p.add_argument("--out-dir", type=Path, default=Path("data/seed"))
    p.add_argument("--n-members", type=int, default=3)
    p.add_argument("--fraction", type=float, default=0.80)
    args = p.parse_args()

    frames = read(args.input, ":")
    n = len(frames)
    k = int(round(n * args.fraction))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[boot] input: {n} frames in {args.input}")
    print(f"[boot] writing {args.n_members} subsets of {k} frames each (fraction={args.fraction})")

    for m in range(args.n_members):
        rng = np.random.default_rng(seed=m * 17 + 3)
        idx = rng.choice(n, size=k, replace=False)
        idx.sort()
        out = args.out_dir / f"ibuprofen_boot{m}.xyz"
        if out.exists():
            out.unlink()
        for i in idx:
            write(out, frames[i], format="extxyz", append=True)
        from collections import Counter
        sources = Counter(frames[i].info.get("source", "?").split("-")[0] for i in idx)
        print(f"[boot] member {m}: {out.name}  ({k} frames) sources: {dict(sources)}")


if __name__ == "__main__":
    main()
