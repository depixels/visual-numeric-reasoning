"""Build Qwen3-VL training datasets for 3-stage pipeline.

Example:
python tools/generate/make_trainsets_qwen3vl.py \
  --out_dir /data/rege_trainsets_qwen3vl \
  --seed 1 --resolution 512 \
  --n_stage1_pairs 200000 \
  --n_stage2_sft_single 50000 \
  --n_stage2_sft_pair 50000 \
  --n_stage3_prefs 1000
"""

import argparse
import os
import shutil
import subprocess
import sys


def _parse_args():
    parser = argparse.ArgumentParser(description="Make Qwen3-VL trainsets")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--n_stage1_triplets", type=int, default=None)
    parser.add_argument("--n_stage1_pairs", type=int, default=200000)
    parser.add_argument("--n_stage2_sft_single", type=int, default=50000)
    parser.add_argument("--n_stage2_sft_pair", type=int, default=50000)
    parser.add_argument("--n_stage3_prefs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _run(cmd):
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    subprocess.run(cmd, check=True)


def main():
    args = _parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    base_dir = os.path.dirname(__file__)
    
    n_stage1_pairs = args.n_stage1_pairs if args.n_stage1_pairs is not None else args.n_stage1_triplets

    # # Stage 1: Contrastive pairs
    # print("\n" + "="*80)
    # print("STAGE 1: Generating contrastive pairs")
    # print("="*80)
    # stage1_dir = os.path.join(out_dir, "stage1_contrastive")
    # cmd = [
    #     sys.executable,
    #     os.path.join(base_dir, "train_stage1_contrastive.py"),
    #     "--out_dir", stage1_dir,
    #     "--seed", str(args.seed),
    #     "--resolution", str(args.resolution),
    #     "--n_pairs", str(n_stage1_pairs),
    # ]
    # if args.resume:
    #     cmd.append("--resume")
    # _run(cmd)
    
    # Stage 2: Single-image SFT
    print("\n" + "="*80)
    print("STAGE 2a: Generating single-image SFT data")
    print("="*80)
    stage2_single_dir = os.path.join(out_dir, "stage2_sft_single")
    cmd = [
        sys.executable,
        os.path.join(base_dir, "train_stage2_sft_single.py"),
        "--out_dir", stage2_single_dir,
        "--seed", str(args.seed),
        "--resolution", str(args.resolution),
        "--n_samples", str(args.n_stage2_sft_single),
    ]
    if args.resume:
        cmd.append("--resume")
    _run(cmd)
    
    # Stage 2: Pair SFT
    print("\n" + "="*80)
    print("STAGE 2b: Generating pair SFT data")
    print("="*80)
    stage2_pair_dir = os.path.join(out_dir, "stage2_sft_pair")
    cmd = [
        sys.executable,
        os.path.join(base_dir, "train_stage2_sft_pair.py"),
        "--out_dir", stage2_pair_dir,
        "--seed", str(args.seed),
        "--resolution", str(args.resolution),
        "--n_pairs", str(args.n_stage2_sft_pair),
    ]
    if args.resume:
        cmd.append("--resume")
    _run(cmd)
    
    # Stage 3: Preference data (optional)
    if args.n_stage3_prefs > 0:
        print("\n" + "="*80)
        print("STAGE 3: Generating preference data")
        print("="*80)
        stage3_dir = os.path.join(out_dir, "stage3_prefs")
        cmd = [
            sys.executable,
            os.path.join(base_dir, "train_stage3_prefs.py"),
            "--out_dir", stage3_dir,
            "--seed", str(args.seed),
            "--n_prefs", str(args.n_stage3_prefs),
            "--stage2_single_dir", stage2_single_dir,
            "--stage2_pair_dir", stage2_pair_dir,
        ]
        if args.resume:
            cmd.append("--resume")
        _run(cmd)
    
    print("\n" + "="*80)
    print("✅ All stages completed!")
    print(f"Output directory: {out_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
