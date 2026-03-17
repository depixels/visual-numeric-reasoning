#!/usr/bin/env python3
"""Generate a formal Blender gauge benchmark aligned with the clock benchmark layout."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make controlled Blender gauge benchmark")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--clean_n", type=int, default=100)
    p.add_argument("--noisy_n", type=int, default=100)
    p.add_argument("--viewpoint_only_n", type=int, default=0)
    p.add_argument("--illumination_only_n", type=int, default=0)
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--blender_bin", default="blender")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _render_benchmark(out_dir: str, clean_n: int, noisy_n: int, resolution: int, seed: int, blender_bin: str) -> None:
    script = os.path.join(os.path.dirname(__file__), "make_ood_blender_gauge_smoke.py")
    _run(
        [
            sys.executable,
            script,
            "--out_dir",
            out_dir,
            "--clean_n",
            str(clean_n),
            "--noisy_n",
            str(noisy_n),
            "--resolution",
            str(resolution),
            "--seed",
            str(seed),
            "--blender_bin",
            blender_bin,
        ]
    )


def _tag_split(root: str, split: str, severity: str) -> None:
    path = os.path.join(root, split, "samples.jsonl")
    rows = _load_jsonl(path)
    for row in rows:
        meta = row.setdefault("meta", {})
        meta["benchmark_split"] = split
        meta["ood_severity"] = severity
        meta["view_bucket"] = meta.get("tilt_bucket")
        row["split"] = split
    _write_jsonl(path, rows)


def _move_split(src_root: str, src_split: str, dst_root: str, dst_split: str) -> None:
    if os.path.exists(os.path.join(dst_root, dst_split)):
        shutil.rmtree(os.path.join(dst_root, dst_split))
    shutil.move(os.path.join(src_root, src_split), os.path.join(dst_root, dst_split))


def main() -> None:
    args = _parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _render_benchmark(out_dir, args.clean_n, args.noisy_n, args.resolution, args.seed, args.blender_bin)
    _tag_split(out_dir, "clean", "moderate")
    _tag_split(out_dir, "noisy", "severe")

    if args.viewpoint_only_n > 0:
        tmp_root = os.path.join(out_dir, "_tmp_viewpoint_only")
        _render_benchmark(tmp_root, args.viewpoint_only_n, 0, args.resolution, args.seed + 10000, args.blender_bin)
        _move_split(tmp_root, "clean", out_dir, "viewpoint_only")
        _tag_split(out_dir, "viewpoint_only", "factorized_viewpoint")
        shutil.rmtree(tmp_root, ignore_errors=True)

    if args.illumination_only_n > 0:
        tmp_root = os.path.join(out_dir, "_tmp_illumination_only")
        _render_benchmark(tmp_root, 0, args.illumination_only_n, args.resolution, args.seed + 20000, args.blender_bin)
        _move_split(tmp_root, "noisy", out_dir, "illumination_only")
        _tag_split(out_dir, "illumination_only", "factorized_illumination")
        shutil.rmtree(tmp_root, ignore_errors=True)

    print("\nDone.")
    for split in ("clean", "noisy", "viewpoint_only", "illumination_only"):
        path = os.path.join(out_dir, split, "samples.jsonl")
        if os.path.exists(path):
            print(f"- {split}: {path}")


if __name__ == "__main__":
    main()
