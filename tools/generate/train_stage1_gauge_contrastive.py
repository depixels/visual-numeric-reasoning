"""Generate stage1 contrastive triplets for analog gauge reading."""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
from collections import Counter
from typing import Dict, List


def _parse_args():
    parser = argparse.ArgumentParser(description="Stage1 gauge contrastive generator")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--n_triplets", type=int, required=True)
    parser.add_argument("--margin_base", type=float, default=0.2)
    parser.add_argument("--margin_alpha", type=float, default=0.1)
    parser.add_argument("--ensure_pools", action="store_true")
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


def _pool_size(n_triplets):
    return min(20000, max(64, int(n_triplets * 0.06)))


def _build_pools(out_dir, seed, resolution, n_triplets, resume):
    pool_n = _pool_size(n_triplets)
    base_dir = os.path.dirname(__file__)

    blender_root = os.path.join(out_dir, "_pool_blender")
    blender_samples = []
    bench_script = os.path.join(base_dir, "make_ood_blender_gauge_benchmark.py")
    if not (resume and os.path.exists(os.path.join(blender_root, "clean", "samples.jsonl"))):
        if os.path.exists(blender_root) and not resume:
            shutil.rmtree(blender_root)
        _run(
            [
                "python",
                bench_script,
                "--out_dir",
                blender_root,
                "--clean_n",
                str(pool_n),
                "--noisy_n",
                str(pool_n),
                "--resolution",
                str(resolution),
                "--seed",
                str(seed),
            ]
        )
    for split in ("clean", "noisy"):
        samples_path = os.path.join(blender_root, split, "samples.jsonl")
        rows = _load_jsonl(samples_path)
        for row in rows:
            row["_source"] = f"blender_{split}"
            row["_image_path"] = os.path.join(blender_root, split, row["image"])
        blender_samples.extend(rows)

    matplot_root = os.path.join(out_dir, "_pool_matplot")
    matplot_samples_path = os.path.join(matplot_root, "gauge_clean_matplot", "annotations.jsonl")
    if not (resume and os.path.exists(matplot_samples_path)):
        if os.path.exists(matplot_root) and not resume:
            shutil.rmtree(matplot_root)
        _run(
            [
                "python",
                os.path.join(base_dir, "matplot", "render_matplot_gauge_batch.py"),
                "--out_dir",
                matplot_root,
                "--n",
                str(pool_n),
                "--seed",
                str(seed + 7),
                "--resolution",
                str(resolution),
                "--id_prefix",
                "stage1_matplot_gauge",
            ]
        )
    matplot_samples = _load_jsonl(matplot_samples_path)
    for row in matplot_samples:
        row["_source"] = "matplot"
        row["_image_path"] = os.path.join(matplot_root, "gauge_clean_matplot", row["image"])

    return blender_samples, matplot_samples


def _extract_meta(sample: Dict) -> Dict:
    label = sample["label"]
    return {
        "gauge_value": int(label["gauge_value"]),
        "pointer_angle_deg": float(label["pointer_angle_deg"]),
        "value_norm": float(label.get("value_norm", int(label["gauge_value"]) / 100.0)),
        "style_id": sample.get("meta", {}).get("style_id"),
        "source": sample.get("_source", sample.get("meta", {}).get("source", "unknown")),
    }


def _index_by_value(samples: List[Dict]) -> Dict[int, List[Dict]]:
    buckets: Dict[int, List[Dict]] = {}
    for s in samples:
        value = int(s["label"]["gauge_value"])
        buckets.setdefault(value, []).append(s)
    return buckets


def _index_by_value_source(samples: List[Dict]) -> Dict[int, Dict[str, List[Dict]]]:
    buckets: Dict[int, Dict[str, List[Dict]]] = {}
    for s in samples:
        value = int(s["label"]["gauge_value"])
        src = s.get("_source", "unknown")
        buckets.setdefault(value, {}).setdefault(src, []).append(s)
    return buckets


def _sample_positive_cross_domain(rng, value_sources):
    for _ in range(1000):
        value = rng.choice(list(value_sources.keys()))
        sources = value_sources[value]
        if "matplot" in sources and any(src.startswith("blender") for src in sources):
            a = rng.choice(sources["matplot"])
            blender_src = rng.choice([src for src in sources if src.startswith("blender")])
            b = rng.choice(sources[blender_src])
            return a, b, "blender_matplot"
    raise RuntimeError("Failed to sample cross-domain positive")


def _sample_positive_clean_noisy(rng, value_sources):
    for _ in range(1000):
        value = rng.choice(list(value_sources.keys()))
        sources = value_sources[value]
        if "blender_clean" in sources and "blender_noisy" in sources:
            return rng.choice(sources["blender_clean"]), rng.choice(sources["blender_noisy"]), "blender_clean_noisy"
    raise RuntimeError("Failed to sample clean/noisy positive")


def _sample_positive_same_domain_diff_style(rng, value_sources):
    for _ in range(2000):
        value = rng.choice(list(value_sources.keys()))
        src = rng.choice(list(value_sources[value].keys()))
        candidates = value_sources[value][src]
        if len(candidates) < 2:
            continue
        anchor = rng.choice(candidates)
        anchor_style = anchor.get("meta", {}).get("style_id")
        positives = [c for c in candidates if c.get("meta", {}).get("style_id") != anchor_style]
        if positives:
            return anchor, rng.choice(positives), f"{src}_{src}"
    raise RuntimeError("Failed to sample same-domain diff-style positive")


def _sample_negative_for_anchor(rng, all_buckets, anchor_value, delta_range):
    for _ in range(2000):
        delta = rng.randint(delta_range[0], delta_range[1])
        target = anchor_value + (delta if rng.random() < 0.5 else -delta)
        if not (0 <= target <= 100):
            continue
        if target not in all_buckets:
            continue
        return rng.choice(all_buckets[target]), target - anchor_value
    raise RuntimeError("Failed to sample negative")


def main():
    args = _parse_args()
    rng = random.Random(args.seed)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, "annotations.jsonl")

    existing_rows = _load_jsonl(out_jsonl) if args.resume and os.path.exists(out_jsonl) else []
    start_index = len(existing_rows)
    if start_index >= args.n_triplets:
        print(f"Already have {start_index} triplets, skipping")
        return

    blender_samples, matplot_samples = _build_pools(out_dir, args.seed, args.resolution, args.n_triplets, args.resume)
    all_samples = blender_samples + matplot_samples
    all_buckets = _index_by_value(all_samples)
    value_sources = _index_by_value_source(all_samples)

    n_pos_cross = int(args.n_triplets * 0.5)
    n_pos_clean_noisy = int(args.n_triplets * 0.3)
    n_pos_same = args.n_triplets - n_pos_cross - n_pos_clean_noisy
    n_neg_hard = int(args.n_triplets * 0.4)
    n_neg_medium = int(args.n_triplets * 0.2)
    n_neg_easy = args.n_triplets - n_neg_hard - n_neg_medium

    stats = Counter()
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for idx in range(start_index, args.n_triplets):
            if idx % 1000 == 0:
                print(f"Progress: {idx}/{args.n_triplets}")

            if idx < n_pos_cross:
                anchor, positive, pos_source_pair = _sample_positive_cross_domain(rng, value_sources)
            elif idx < n_pos_cross + n_pos_clean_noisy:
                anchor, positive, pos_source_pair = _sample_positive_clean_noisy(rng, value_sources)
            else:
                anchor, positive, pos_source_pair = _sample_positive_same_domain_diff_style(rng, value_sources)

            neg_roll = idx % (n_neg_hard + n_neg_medium + n_neg_easy)
            anchor_value = int(anchor["label"]["gauge_value"])
            if neg_roll < n_neg_hard:
                negative, delta = _sample_negative_for_anchor(rng, all_buckets, anchor_value, (1, 5))
                delta_bucket = "hard"
            elif neg_roll < n_neg_hard + n_neg_medium:
                negative, delta = _sample_negative_for_anchor(rng, all_buckets, anchor_value, (6, 20))
                delta_bucket = "medium"
            else:
                negative, delta = _sample_negative_for_anchor(rng, all_buckets, anchor_value, (21, 60))
                delta_bucket = "easy"

            filename_anchor = f"triplet_{idx:06d}_anchor.png"
            filename_pos = f"triplet_{idx:06d}_positive.png"
            filename_neg = f"triplet_{idx:06d}_negative.png"
            _safe_link(anchor["_image_path"], os.path.join(images_dir, filename_anchor))
            _safe_link(positive["_image_path"], os.path.join(images_dir, filename_pos))
            _safe_link(negative["_image_path"], os.path.join(images_dir, filename_neg))

            anchor_meta = _extract_meta(anchor)
            positive_meta = _extract_meta(positive)
            negative_meta = _extract_meta(negative)
            delta_value = negative_meta["gauge_value"] - anchor_meta["gauge_value"]
            margin = args.margin_base + args.margin_alpha * math.log(1.0 + abs(delta_value))

            row = {
                "id": f"gauge_stage1_{idx:06d}",
                "anchor": os.path.join("images", filename_anchor),
                "positive": os.path.join("images", filename_pos),
                "negative": os.path.join("images", filename_neg),
                "label": {
                    "anchor_gauge_value": anchor_meta["gauge_value"],
                    "positive_gauge_value": positive_meta["gauge_value"],
                    "negative_gauge_value": negative_meta["gauge_value"],
                    "negative_delta": delta_value,
                },
                "meta": {
                    "anchor_source": anchor_meta["source"],
                    "positive_source": positive_meta["source"],
                    "negative_source": negative_meta["source"],
                    "pos_source_pair": pos_source_pair,
                    "delta_bucket": delta_bucket,
                    "triplet_margin": margin,
                },
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            if idx % 200 == 0:
                f.flush()
            stats[pos_source_pair] += 1

    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"triplet_counts": dict(stats)}, f, indent=2)


if __name__ == "__main__":
    main()
