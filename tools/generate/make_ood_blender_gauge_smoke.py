#!/usr/bin/env python3
"""Generate a minimal automotive gauge smoke benchmark and preview sheets."""

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List

from PIL import Image, ImageDraw


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate gauge smoke benchmark")
    p.add_argument("--out_dir", default="data/bench/gauge_smoke")
    p.add_argument("--clean_n", type=int, default=16)
    p.add_argument("--noisy_n", type=int, default=16)
    p.add_argument("--resolution", type=int, default=384)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--blender_bin", default="blender")
    p.add_argument("--fast_render", action="store_true")
    return p.parse_args()


def _run(cmd: List[str]) -> None:
    print("\nRunning:", " ".join(cmd))
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


def _crop_render_to_bbox(image_path: str, bbox_view: List[float], output_size: int, pad: float) -> None:
    min_x, max_x, min_y, max_y = bbox_view
    with Image.open(image_path) as img:
        width, height = img.size
        left = min_x * width
        right = max_x * width
        top = (1.0 - max_y) * height
        bottom = (1.0 - min_y) * height

        bbox_w = max(1.0, right - left)
        bbox_h = max(1.0, bottom - top)
        side = max(bbox_w, bbox_h) * (1.0 + pad * 2.0)
        cx = (left + right) * 0.5
        cy = (top + bottom) * 0.5

        crop_left = max(0.0, cx - side * 0.5)
        crop_top = max(0.0, cy - side * 0.5)
        crop_right = min(float(width), cx + side * 0.5)
        crop_bottom = min(float(height), cy + side * 0.5)

        crop_w = crop_right - crop_left
        crop_h = crop_bottom - crop_top
        side = min(crop_w, crop_h)
        crop_left = max(0.0, min(crop_left, width - side))
        crop_top = max(0.0, min(crop_top, height - side))
        crop_box = (
            int(round(crop_left)),
            int(round(crop_top)),
            int(round(crop_left + side)),
            int(round(crop_top + side)),
        )
        cropped = img.crop(crop_box).resize((output_size, output_size), Image.Resampling.LANCZOS)
        cropped.save(image_path)


def _postprocess_split(out_dir: str, split: str, resolution: int) -> None:
    samples_path = os.path.join(out_dir, split, "samples.jsonl")
    if not os.path.exists(samples_path):
        return
    rows = _load_jsonl(samples_path)
    for row in rows:
        bbox_view = row.get("meta", {}).get("crop_bbox_view")
        if bbox_view:
            pad = 0.08 if split == "clean" else 0.10
            _crop_render_to_bbox(os.path.join(out_dir, split, row["image"]), bbox_view, resolution, pad)
    _write_jsonl(samples_path, rows)


def _render_split(args: argparse.Namespace, split: str, n: int, seed: int) -> None:
    if n <= 0:
        return
    script_name = "render_gauge_smoke.py"
    render_script = os.path.join(os.path.dirname(__file__), "blender", script_name)
    cmd = [
        args.blender_bin,
        "-b",
        "-P",
        render_script,
        "--",
        "--out_dir",
        os.path.abspath(args.out_dir),
        "--split",
        split,
        "--n",
        str(n),
        "--resolution",
        str(args.resolution),
        "--seed",
        str(seed),
        "--id_prefix",
        f"gauge_{split}",
    ]
    _run(cmd)


def _build_preview(out_dir: str, split: str) -> str:
    rows = _load_jsonl(os.path.join(out_dir, split, "samples.jsonl"))
    rows = rows[:16]
    if not rows:
        raise RuntimeError(f"No rows found for split={split}")

    cell_w = 220
    cell_h = 250
    cols = 4
    rows_n = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cell_w * cols, cell_h * rows_n), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    for idx, row in enumerate(rows):
        x0 = (idx % cols) * cell_w
        y0 = (idx // cols) * cell_h
        img = Image.open(os.path.join(out_dir, split, row["image"])).convert("RGB")
        img.thumbnail((cell_w - 12, 170))
        img_x = x0 + (cell_w - img.width) // 2
        img_y = y0 + 6
        canvas.paste(img, (img_x, img_y))

        value = row["label"]["gauge_value"]
        yaw = row["meta"]["view"]["yaw"]
        specular = row["meta"]["degradation"]["specular"]
        style_id = row["meta"].get("style_id", "unknown")
        lines = [
            f"value={value}",
            f"style={style_id}",
            f"yaw={yaw:.1f}",
            f"spec={specular:.2f}",
            f"split={split}",
        ]
        text_y = y0 + 180
        for line in lines:
            draw.text((x0 + 8, text_y), line, fill=(20, 20, 20))
            text_y += 15

    preview_path = os.path.join(out_dir, f"preview_{split}.png")
    canvas.save(preview_path)
    return preview_path


def main() -> None:
    args = _parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _render_split(args, "clean", args.clean_n, args.seed)
    _render_split(args, "noisy", args.noisy_n, args.seed + 10000)
    _postprocess_split(out_dir, "clean", args.resolution)
    _postprocess_split(out_dir, "noisy", args.resolution)

    clean_preview = _build_preview(out_dir, "clean") if args.clean_n > 0 else None
    noisy_preview = _build_preview(out_dir, "noisy") if args.noisy_n > 0 else None

    print("\nDone.")
    if clean_preview:
        print(f"- clean preview: {clean_preview}")
    if noisy_preview:
        print(f"- noisy preview: {noisy_preview}")


if __name__ == "__main__":
    main()
