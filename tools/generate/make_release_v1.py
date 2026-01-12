"""Orchestrate release v1 dataset generation."""

import argparse
import json
import os
import shutil
import subprocess
from collections import Counter
from typing import Any, Dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make release v1 dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    return parser.parse_args()


def _run(cmd: list) -> None:
    subprocess.run(cmd, check=True)


def _ensure_empty(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def _move_split(src: str, dst: str) -> None:
    _ensure_empty(dst)
    shutil.move(src, dst)


def _validate(out_dir: str, split_name: str, jsonl_name: str, split_type: str, reports_dir: str) -> None:
    jsonl_path = os.path.join(out_dir, split_name, jsonl_name)
    images_root = os.path.join(out_dir, split_name)
    report_path = os.path.join(reports_dir, f"validate_{split_name}.txt")
    cmd = [
        "python",
        os.path.join(os.path.dirname(__file__), "..", "validate", "validate_annotations.py"),
        "--jsonl",
        jsonl_path,
        "--images_root",
        images_root,
        "--type",
        split_type,
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _write_stats_single(split_dir: str, reports_dir: str, split_name: str, jsonl_name: str) -> None:
    style_counts = Counter()
    env_counts = Counter()
    view_counts = Counter()

    for row in _load_jsonl(os.path.join(split_dir, jsonl_name)):
        meta = row.get("meta", {})
        style_counts[meta.get("style_id", "unknown")] += 1
        lighting = meta.get("lighting", {})
        env_counts[lighting.get("env_id", "unknown")] += 1
        view_counts[meta.get("view_bucket", "unknown")] += 1

    stats = {
        "style_id_counts": dict(style_counts),
        "env_id_counts": dict(env_counts),
        "view_bucket_counts": dict(view_counts),
    }
    out_path = os.path.join(reports_dir, f"stats_{split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def _write_stats_pair(split_dir: str, reports_dir: str, split_name: str) -> None:
    style_counts = Counter()
    env_counts = Counter()
    view_counts = Counter()
    pair_type_counts = Counter()
    delta_hist = Counter()

    for row in _load_jsonl(os.path.join(split_dir, "pairs.jsonl")):
        meta = row.get("meta", {})
        style_counts[meta.get("style_id_a", "unknown")] += 1
        style_counts[meta.get("style_id_b", "unknown")] += 1
        pair_type_counts[meta.get("pair_type", "unknown")] += 1
        label = row.get("label", {})
        delta_hist[str(label.get("delta_minutes"))] += 1
        lighting = meta.get("lighting", {})
        env_counts[lighting.get("env_id", "unknown")] += 1
        view_counts[meta.get("view_bucket", "unknown")] += 1

    stats = {
        "style_id_counts": dict(style_counts),
        "env_id_counts": dict(env_counts),
        "view_bucket_counts": dict(view_counts),
        "pair_type_counts": dict(pair_type_counts),
        "delta_minutes_hist": dict(delta_hist),
    }
    out_path = os.path.join(reports_dir, f"stats_{split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def main() -> None:
    args = _parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Matplot single
    matplot_out = os.path.join(out_dir, "rege_clean_matplot")
    _ensure_empty(matplot_out)
    _run(
        [
            "python",
            os.path.join(os.path.dirname(__file__), "matplot", "render_matplot_batch.py"),
            "--out_dir",
            out_dir,
            "--n",
            "100",
            "--seed",
            str(args.seed),
            "--resolution",
            str(args.resolution),
            "--id_prefix",
            "rege_clean_matplot",
        ]
    )
    _validate(out_dir, "rege_clean_matplot", "annotations.jsonl", "sample", reports_dir)
    _write_stats_single(os.path.join(out_dir, "rege_clean_matplot"), reports_dir, "rege_clean_matplot", "annotations.jsonl")

    # Blender clean single
    _run(
        [
            "blender",
            "-b",
            "-P",
            os.path.join(os.path.dirname(__file__), "blender", "render_batch.py"),
            "--",
            "--out_dir",
            out_dir,
            "--n",
            "100",
            "--split",
            "clean",
            "--seed",
            str(args.seed),
            "--resolution",
            str(args.resolution),
            "--id_prefix",
            "rege_clean_blender",
        ]
    )
    _move_split(os.path.join(out_dir, "clean"), os.path.join(out_dir, "rege_clean_blender"))
    _validate(out_dir, "rege_clean_blender", "samples.jsonl", "sample", reports_dir)
    _write_stats_single(os.path.join(out_dir, "rege_clean_blender"), reports_dir, "rege_clean_blender", "samples.jsonl")

    # Blender noisy single
    _run(
        [
            "blender",
            "-b",
            "-P",
            os.path.join(os.path.dirname(__file__), "blender", "render_batch.py"),
            "--",
            "--out_dir",
            out_dir,
            "--n",
            "100",
            "--split",
            "noisy",
            "--seed",
            str(args.seed),
            "--resolution",
            str(args.resolution),
            "--id_prefix",
            "rege_noisy_blender",
        ]
    )
    _move_split(os.path.join(out_dir, "noisy"), os.path.join(out_dir, "rege_noisy_blender"))
    _validate(out_dir, "rege_noisy_blender", "samples.jsonl", "sample", reports_dir)
    _write_stats_single(os.path.join(out_dir, "rege_noisy_blender"), reports_dir, "rege_noisy_blender", "samples.jsonl")

    # Blender pairs
    _run(
        [
            "blender",
            "-b",
            "-P",
            os.path.join(os.path.dirname(__file__), "blender", "render_batch.py"),
            "--",
            "--out_dir",
            out_dir,
            "--n",
            "200",
            "--split",
            "pair",
            "--seed",
            str(args.seed),
            "--resolution",
            str(args.resolution),
            "--id_prefix",
            "rege_pair_blender",
            "--pair_quota_json",
            "{\"same_style_same_tz\":50,\"cross_style_same_tz\":50,\"same_style_cross_tz\":50,\"cross_style_cross_tz\":50}",
            "--delta_quota_json",
            "{\"hard_n\":100,\"hard_min\":1,\"hard_max\":5,\"easy_n\":100,\"easy_min\":30,\"easy_max\":180}",
        ]
    )
    _move_split(os.path.join(out_dir, "pair"), os.path.join(out_dir, "rege_pair_blender"))
    _validate(out_dir, "rege_pair_blender", "pairs.jsonl", "pair", reports_dir)
    _write_stats_pair(os.path.join(out_dir, "rege_pair_blender"), reports_dir, "rege_pair_blender")

    # Spotcheck
    spot_src = os.path.join(out_dir, "spotcheck")
    spot_dst = os.path.join(reports_dir, "spotcheck_blender")
    if os.path.exists(spot_src):
        shutil.rmtree(spot_src)
    _run(
        [
            "blender",
            "-b",
            "-P",
            os.path.join(os.path.dirname(__file__), "blender", "render_batch.py"),
            "--",
            "--out_dir",
            out_dir,
            "--n",
            "1",
            "--split",
            "clean",
            "--seed",
            str(args.seed),
            "--resolution",
            str(args.resolution),
            "--spotcheck",
        ]
    )
    if os.path.exists(spot_dst):
        shutil.rmtree(spot_dst)
    shutil.move(spot_src, spot_dst)


if __name__ == "__main__":
    main()
