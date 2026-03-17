"""Generate stage2 single-image SFT data for analog gauge reading."""

import argparse
import json
import os
import random
import shutil
import subprocess
from collections import Counter
from typing import Any, Dict, List


USER_PROMPT = "Read the exact value shown on this analog gauge. Answer with an integer from 0 to 100."


def _parse_args():
    parser = argparse.ArgumentParser(description="Stage2 gauge SFT single generator")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--reuse_pools_dir", default=None)
    parser.add_argument("--pool_blender_splits", choices=["clean", "noisy", "both"], default="both")
    parser.add_argument("--pool_use_matplot", action="store_true")
    parser.add_argument("--target_format", choices=["answer_only", "grounded_rationale"], default="grounded_rationale")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _run(cmd):
    subprocess.run(cmd, check=True)


def _load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _safe_link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _load_stage1_pools(base_dir: str, blender_splits: str, use_matplot: bool) -> List[Dict]:
    samples: List[Dict] = []
    direct_benchmark = all(os.path.exists(os.path.join(base_dir, split, "samples.jsonl")) for split in ("clean", "noisy"))
    blender_root = base_dir if direct_benchmark else os.path.join(base_dir, "_pool_blender")
    splits = ["clean", "noisy"] if blender_splits == "both" else [blender_splits]
    for split in splits:
        path = os.path.join(blender_root, split, "samples.jsonl")
        if not os.path.exists(path):
            continue
        rows = _load_jsonl(path)
        for row in rows:
            row["_source"] = f"blender_{split}"
            row["_image_path"] = os.path.join(blender_root, split, row["image"])
        samples.extend(rows)
    if use_matplot:
        matplot_root = os.path.join(base_dir, "_pool_matplot", "gauge_clean_matplot")
        path = os.path.join(matplot_root, "annotations.jsonl")
        if os.path.exists(path):
            rows = _load_jsonl(path)
            for row in rows:
                row["_source"] = "matplot"
                row["_image_path"] = os.path.join(matplot_root, row["image"])
            samples.extend(rows)
    return samples


def _answer(label: Dict[str, Any]) -> str:
    return str(int(label["gauge_value"]))


def _view_desc(meta: Dict[str, Any]) -> str:
    view = meta.get("view") or {}
    yaw = float(view.get("yaw", 0) or 0)
    pitch = float(view.get("pitch", 0) or 0)
    parts = []
    if abs(yaw) > 25:
        parts.append("seen from a strong side angle")
    elif abs(yaw) > 10:
        parts.append("seen from a mild side angle")
    if pitch > 65:
        parts.append("with a fairly top-down view")
    elif pitch < 45:
        parts.append("with a lower viewing angle")
    if not parts:
        return "The gauge is viewed close to the front."
    return "The gauge is " + " and ".join(parts) + "."


def _quality_desc(meta: Dict[str, Any], source: str) -> str:
    degradation = meta.get("degradation") or {}
    spec = float(degradation.get("specular", 0) or 0)
    blur = max(float(degradation.get("motion_blur", 0) or 0), float(degradation.get("defocus", 0) or 0))
    parts = []
    if spec > 0.5:
        parts.append("strong glass glare")
    elif spec > 0.15:
        parts.append("some reflection")
    if blur > 0.12:
        parts.append("visible blur")
    if "noisy" in source:
        parts.append("a harder automotive-style rendering condition")
    if not parts:
        return "The scale markings look clear."
    return "I notice " + ", ".join(parts) + "."


def _pointer_desc(value: int) -> str:
    if value <= 10:
        return "near the low end of the scale"
    if value <= 30:
        return "between the low end and the lower-middle range"
    if value <= 45:
        return "around the lower-middle range"
    if value <= 55:
        return "near the middle of the semicircle"
    if value <= 70:
        return "around the upper-middle range"
    if value <= 90:
        return "close to the high end"
    return "very close to the maximum end of the scale"


def _generate_grounded_rationale(label: Dict[str, Any], meta: Dict[str, Any], source: str) -> str:
    value = int(label["gauge_value"])
    angle = float(label["pointer_angle_deg"])
    thoughts = [
        _view_desc(meta),
        _quality_desc(meta, source),
        "There is only one pointer on the semicircular dial.",
        f"The pointer appears { _pointer_desc(value) }.",
        f"On a 0 to 100 semicircular scale, that pointer position corresponds to about {value}.",
        f"The simulator angle is {angle:.1f} degrees, which is consistent with value {value}.",
    ]
    return f"<think>{' '.join(thoughts)}</think>\n<answer>{value}</answer>"


def _format_target(label: Dict[str, Any], meta: Dict[str, Any], source: str, target_format: str) -> str:
    value = _answer(label)
    if target_format == "answer_only":
        return f"<answer>{value}</answer>"
    return _generate_grounded_rationale(label, meta, source)


def _build_pools(out_dir: str, seed: int, resolution: int, n_samples: int) -> List[Dict]:
    base_dir = os.path.dirname(__file__)
    per_split = max(64, n_samples // 3)
    bench_root = os.path.join(out_dir, "_pool_blender")
    matplot_root = os.path.join(out_dir, "_pool_matplot")
    _run(
        [
            "python",
            os.path.join(base_dir, "make_ood_blender_gauge_benchmark.py"),
            "--out_dir",
            bench_root,
            "--clean_n",
            str(per_split),
            "--noisy_n",
            str(per_split),
            "--resolution",
            str(resolution),
            "--seed",
            str(seed),
        ]
    )
    _run(
        [
            "python",
            os.path.join(base_dir, "matplot", "render_matplot_gauge_batch.py"),
            "--out_dir",
            matplot_root,
            "--n",
            str(per_split),
            "--seed",
            str(seed + 17),
            "--resolution",
            str(resolution),
            "--id_prefix",
            "stage2_matplot_gauge",
        ]
    )
    return _load_stage1_pools(out_dir, "both", True)


def main():
    args = _parse_args()
    rng = random.Random(args.seed)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "annotations.jsonl")

    if args.reuse_pools_dir:
        samples = _load_stage1_pools(args.reuse_pools_dir, args.pool_blender_splits, args.pool_use_matplot)
    else:
        samples = _build_pools(out_dir, args.seed, args.resolution, args.n_samples)
    if not samples:
        raise RuntimeError("No gauge samples available")

    rng.shuffle(samples)
    while len(samples) < args.n_samples:
        samples.extend(samples[: max(1, args.n_samples - len(samples))])
    samples = samples[: args.n_samples]

    existing_rows = _load_jsonl(out_jsonl) if args.resume and os.path.exists(out_jsonl) else []
    start_index = len(existing_rows)
    if start_index >= args.n_samples:
        return

    with open(out_jsonl, "a", encoding="utf-8") as f:
        for idx, row in enumerate(samples[start_index:], start=start_index):
            filename = f"sample_{idx:06d}.png"
            dst = os.path.join(images_dir, filename)
            _safe_link(row["_image_path"], dst)

            label = row["label"]
            meta = row.get("meta", {})
            source = row.get("_source", "unknown")
            target = _format_target(label, meta, source, args.target_format)

            record = {
                "id": f"gauge_stage2_single_{idx:06d}",
                "images": [os.path.join("images", filename)],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": os.path.join("images", filename)},
                            {"type": "text", "text": USER_PROMPT},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": target}],
                    },
                ],
                "label": {
                    "gauge_value": int(label["gauge_value"]),
                    "pointer_angle_deg": float(label["pointer_angle_deg"]),
                    "value_norm": float(label.get("value_norm", int(label["gauge_value"]) / 100.0)),
                },
                "meta": {
                    "source": source,
                    "style_id": meta.get("style_id", "unknown"),
                    "target_format": args.target_format,
                },
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    rows = _load_jsonl(out_jsonl)
    stats = Counter(row.get("meta", {}).get("source", "unknown") for row in rows)
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"source_counts": dict(stats)}, f, indent=2)


if __name__ == "__main__":
    main()
