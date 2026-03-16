#!/usr/bin/env python3
"""Generate controlled Blender OOD benchmark splits for paper experiments."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make controlled Blender OOD benchmark")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clean_n", type=int, default=100)
    p.add_argument("--noisy_n", type=int, default=100)
    p.add_argument("--viewpoint_only_n", type=int, default=0)
    p.add_argument("--illumination_only_n", type=int, default=0)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--style_bank_dir",
        default=os.path.join(os.path.dirname(__file__), "blender", "assets", "styles_ood_benchmark"),
    )
    p.add_argument("--blender_bin", default="blender")
    p.add_argument("--validate", action="store_true")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print("\n" + "=" * 90)
    print("Running:", " ".join(cmd))
    print("=" * 90)
    subprocess.run(cmd, check=True)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _tilt_bucket(yaw: Optional[float]) -> str:
    if yaw is None:
        return "unknown"
    yaw = abs(float(yaw))
    if yaw < 10:
        return "Front"
    if yaw < 20:
        return "10-20"
    if yaw < 30:
        return "20-30"
    if yaw < 40:
        return "30-40"
    if yaw < 50:
        return "40-50"
    if yaw < 60:
        return "50-60"
    if yaw < 70:
        return "60-70"
    return "70+"


def _specular_bucket(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    if value <= 0.0:
        return "0.0"
    if value < 0.1:
        return "0.0-0.1"
    if value < 0.3:
        return "0.1-0.3"
    if value < 0.6:
        return "0.3-0.6"
    return "0.6+"


def _blur_bucket(motion_blur: Optional[float], defocus: Optional[float]) -> str:
    level = max(float(motion_blur or 0.0), float(defocus or 0.0))
    if level <= 0.0:
        return "0.0"
    if level < 0.05:
        return "0.0-0.05"
    if level < 0.15:
        return "0.05-0.15"
    if level < 0.30:
        return "0.15-0.30"
    return "0.30+"


def _postprocess_split(out_dir: str, split: str, severity: str, benchmark_source: str) -> None:
    path = os.path.join(out_dir, split, "samples.jsonl")
    rows = _load_jsonl(path)
    updated: List[Dict[str, Any]] = []
    for row in rows:
        meta = dict(row.get("meta", {}))
        view = meta.get("view", {}) or {}
        degradation = meta.get("degradation", {}) or {}
        lighting = meta.get("lighting", {}) or {}
        meta["benchmark_split"] = split
        meta["ood_severity"] = severity
        meta["source"] = benchmark_source
        meta["view_bucket"] = meta.get("view_bucket") or _tilt_bucket(view.get("yaw"))
        meta["tilt_bucket"] = _tilt_bucket(view.get("yaw"))
        meta["specular_bucket"] = _specular_bucket(degradation.get("specular"))
        meta["blur_bucket"] = _blur_bucket(degradation.get("motion_blur"), degradation.get("defocus"))
        meta["lighting_env_id"] = lighting.get("env_id")
        row["split"] = split
        row["meta"] = meta
        updated.append(row)
    _write_jsonl(path, updated)


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
    clean_view_mode: str,
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
    specular_min: float,
    specular_max: float,
    motion_blur_min: float,
    motion_blur_max: float,
    defocus_min: float,
    defocus_max: float,
    env_choices: str,
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
        clean_view_mode,
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
        "--specular_min",
        str(specular_min),
        "--specular_max",
        str(specular_max),
        "--motion_blur_min",
        str(motion_blur_min),
        "--motion_blur_max",
        str(motion_blur_max),
        "--defocus_min",
        str(defocus_min),
        "--defocus_max",
        str(defocus_max),
    ]
    if env_choices:
        cmd.extend(["--env_id_choices", env_choices])
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

    # clean = moderate physical OOD, mainly viewpoint shift with mild photometric change
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
        clean_view_mode="mild",
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
        specular_min=0.0,
        specular_max=0.15,
        motion_blur_min=0.0,
        motion_blur_max=0.04,
        defocus_min=0.0,
        defocus_max=0.04,
        env_choices="studio_softbox,studio_softbox_round,top_light",
    )
    _postprocess_split(out_dir, "clean", severity="moderate", benchmark_source="blender_ood_clean")

    # noisy = severe physical OOD, joint viewpoint + illumination + degradation
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
        clean_view_mode="mild",
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
        specular_min=0.3,
        specular_max=1.0,
        motion_blur_min=0.05,
        motion_blur_max=0.6,
        defocus_min=0.05,
        defocus_max=0.6,
        env_choices="studio_softbox,studio_softbox_round,studio_softbox_ellipse,top_light,top_light_round,top_light_ellipse,window_side",
    )
    _postprocess_split(out_dir, "noisy", severity="severe", benchmark_source="blender_ood_noisy")

    if args.viewpoint_only_n > 0:
        tmp_root = os.path.join(out_dir, "_tmp_viewpoint_only")
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root)
        _render_split(
            blender_bin=args.blender_bin,
            render_script=render_script,
            out_dir=tmp_root,
            split="clean",
            n=args.viewpoint_only_n,
            resolution=args.resolution,
            seed=args.seed + 200000,
            style_bank_dir=args.style_bank_dir,
            id_prefix="rege_ood_viewpoint_only",
            clean_view_mode="mild",
            view_yaw_min=-70.0,
            view_yaw_max=70.0,
            view_pitch_min=25.0,
            view_pitch_max=80.0,
            view_roll_min=-12.0,
            view_roll_max=12.0,
            pose_yaw_max=2.0,
            pose_pitch_max=2.0,
            pose_roll_max=2.0,
            pose_x_max=0.01,
            pose_y_max=0.01,
            specular_min=0.0,
            specular_max=0.05,
            motion_blur_min=0.0,
            motion_blur_max=0.0,
            defocus_min=0.0,
            defocus_max=0.0,
            env_choices="studio_softbox",
        )
        shutil.move(os.path.join(tmp_root, "clean"), os.path.join(out_dir, "viewpoint_only"))
        shutil.rmtree(tmp_root)
        _postprocess_split(out_dir, "viewpoint_only", severity="factorized_viewpoint", benchmark_source="blender_viewpoint_only")

    if args.illumination_only_n > 0:
        tmp_root = os.path.join(out_dir, "_tmp_illumination_only")
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root)
        _render_split(
            blender_bin=args.blender_bin,
            render_script=render_script,
            out_dir=tmp_root,
            split="clean",
            n=args.illumination_only_n,
            resolution=args.resolution,
            seed=args.seed + 300000,
            style_bank_dir=args.style_bank_dir,
            id_prefix="rege_ood_illumination_only",
            clean_view_mode="front",
            view_yaw_min=-5.0,
            view_yaw_max=5.0,
            view_pitch_min=82.0,
            view_pitch_max=90.0,
            view_roll_min=-2.0,
            view_roll_max=2.0,
            pose_yaw_max=0.0,
            pose_pitch_max=0.0,
            pose_roll_max=0.0,
            pose_x_max=0.0,
            pose_y_max=0.0,
            specular_min=0.2,
            specular_max=0.8,
            motion_blur_min=0.0,
            motion_blur_max=0.12,
            defocus_min=0.0,
            defocus_max=0.12,
            env_choices="studio_softbox_round,studio_softbox_ellipse,top_light,window_side",
        )
        shutil.move(os.path.join(tmp_root, "clean"), os.path.join(out_dir, "illumination_only"))
        shutil.rmtree(tmp_root)
        _postprocess_split(out_dir, "illumination_only", severity="factorized_illumination", benchmark_source="blender_illumination_only")

    if args.validate:
        for split in ("clean", "noisy", "viewpoint_only", "illumination_only"):
            if os.path.exists(os.path.join(out_dir, split, "samples.jsonl")):
                _validate_split(validate_script, out_dir, split)

    print("\nDone.")
    for split in ("clean", "noisy", "viewpoint_only", "illumination_only"):
        path = os.path.join(out_dir, split, "samples.jsonl")
        if os.path.exists(path):
            print(f"- {split}: {path}")


if __name__ == "__main__":
    main()
