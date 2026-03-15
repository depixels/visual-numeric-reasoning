#!/usr/bin/env python3
"""Generate OOD Blender benchmark splits with custom style/view distributions.

Example:
python tools/generate/make_ood_blender_benchmark.py \
  --out_dir /data/rege_ood_benchmark \
  --clean_n 100 --noisy_n 100 --resolution 512 --seed 2026
"""

import argparse
import os
import subprocess
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make OOD Blender benchmark")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clean_n", type=int, default=100)
    p.add_argument("--noisy_n", type=int, default=100)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--style_bank_dir",
        default=os.path.join(
            os.path.dirname(__file__), "blender", "assets", "styles_ood_benchmark"
        ),
    )
    p.add_argument("--blender_bin", default="blender")
    p.add_argument("--validate", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 90)
    print("Running:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def _render_split(
    blender_bin: str,
    render_script: str,
    out_dir: str,
    split: str,
    n: int,
    resolution: int,
    seed: int,
    style_bank_dir: str,
    id_prefix: str,
    view_yaw_min: float,
    view_yaw_max: float,
    view_pitch_min: float,
    view_pitch_max: float,
    view_roll_min: float,
    view_roll_max: float,
    pose_yaw_max: float,
    pose_pitch_max: float,
    pose_roll_max: float,
    pose_x_max: float,
    pose_y_max: float,
) -> None:
    cmd = [
        blender_bin,
        "-b",
        "-P",
        render_script,
        "--",
        "--out_dir",
        out_dir,
        "--n",
        str(n),
        "--split",
        split,
        "--seed",
        str(seed),
        "--resolution",
        str(resolution),
        "--style_bank_dir",
        style_bank_dir,
        "--id_prefix",
        id_prefix,
        "--clean_view_mode",
        "mild",
        "--view_yaw_min",
        str(view_yaw_min),
        "--view_yaw_max",
        str(view_yaw_max),
        "--view_pitch_min",
        str(view_pitch_min),
        "--view_pitch_max",
        str(view_pitch_max),
        "--view_roll_min",
        str(view_roll_min),
        "--view_roll_max",
        str(view_roll_max),
        "--pose_yaw_max",
        str(pose_yaw_max),
        "--pose_pitch_max",
        str(pose_pitch_max),
        "--pose_roll_max",
        str(pose_roll_max),
        "--pose_x_max",
        str(pose_x_max),
        "--pose_y_max",
        str(pose_y_max),
    ]
    _run(cmd)


def _validate_split(validate_script: str, out_dir: str, split: str) -> None:
    jsonl = os.path.join(out_dir, split, "samples.jsonl")
    root = os.path.join(out_dir, split)
    cmd = [sys.executable, validate_script, "--jsonl", jsonl, "--images_root", root, "--type", "sample"]
    _run(cmd)


def main() -> None:
    args = _parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    base_dir = os.path.dirname(__file__)
    render_script = os.path.join(base_dir, "blender", "render_batch.py")
    validate_script = os.path.join(base_dir, "..", "validate", "validate_annotations.py")

    if not os.path.isdir(args.style_bank_dir):
        raise FileNotFoundError(f"style_bank_dir not found: {args.style_bank_dir}")

    # Clean OOD: still readable, but with noticeable tilt and viewpoint drift.
    _render_split(
        blender_bin=args.blender_bin,
        render_script=render_script,
        out_dir=out_dir,
        split="clean",
        n=args.clean_n,
        resolution=args.resolution,
        seed=args.seed,
        style_bank_dir=args.style_bank_dir,
        id_prefix="rege_ood_clean_blender",
        view_yaw_min=-35.0,
        view_yaw_max=35.0,
        view_pitch_min=42.0,
        view_pitch_max=78.0,
        view_roll_min=-7.0,
        view_roll_max=7.0,
        pose_yaw_max=7.0,
        pose_pitch_max=7.0,
        pose_roll_max=8.0,
        pose_x_max=0.035,
        pose_y_max=0.035,
    )

    # Noisy OOD: stronger perspective and object pose perturbation.
    _render_split(
        blender_bin=args.blender_bin,
        render_script=render_script,
        out_dir=out_dir,
        split="noisy",
        n=args.noisy_n,
        resolution=args.resolution,
        seed=args.seed + 100000,
        style_bank_dir=args.style_bank_dir,
        id_prefix="rege_ood_noisy_blender",
        view_yaw_min=-70.0,
        view_yaw_max=70.0,
        view_pitch_min=22.0,
        view_pitch_max=80.0,
        view_roll_min=-14.0,
        view_roll_max=14.0,
        pose_yaw_max=18.0,
        pose_pitch_max=18.0,
        pose_roll_max=18.0,
        pose_x_max=0.07,
        pose_y_max=0.07,
    )

    if args.validate:
        _validate_split(validate_script, out_dir, "clean")
        _validate_split(validate_script, out_dir, "noisy")

    print("\nDone.")
    print(f"- clean: {os.path.join(out_dir, 'clean', 'samples.jsonl')}")
    print(f"- noisy: {os.path.join(out_dir, 'noisy', 'samples.jsonl')}")


if __name__ == "__main__":
    main()
