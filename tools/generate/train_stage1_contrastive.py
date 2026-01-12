"""Generate stage1 contrastive triplets for Qwen3-VL.

Example:
python tools/generate/train_stage1_contrastive.py \
  --out_dir /data/stage1_contrastive \
  --seed 1 --resolution 512 --n_triplets 200000
"""

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
    parser = argparse.ArgumentParser(description="Stage1 contrastive generator")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--n_triplets", type=int, default=None)
    parser.add_argument("--n_pairs", type=int, default=None)
    parser.add_argument("--blender_split", choices=["clean", "noisy", "both"], default="both")
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
    return min(30000, max(3000, int(n_triplets * 0.08)))


def _build_pools(out_dir, seed, resolution, n_triplets, resume, blender_split):
    """生成 Blender + Matplot 图像池"""
    pool_n = _pool_size(n_triplets)
    base_dir = os.path.dirname(__file__)
    
    # Blender pool
    blender_root = os.path.join(out_dir, "_pool_blender")
    blender_splits = ["clean", "noisy"] if blender_split == "both" else [blender_split]
    blender_samples = []
    for split in blender_splits:
        blender_samples_path = os.path.join(blender_root, split, "samples.jsonl")
        if not (resume and os.path.exists(blender_samples_path)):
            print(f"\n[Stage1] Generating Blender {split} pool ({pool_n} images)...")
            if os.path.exists(blender_root) and not resume:
                shutil.rmtree(blender_root)
            cmd = [
                "blender", "-b", "-P",
                os.path.join(base_dir, "blender", "render_batch.py"),
                "--",
                "--out_dir", blender_root,
                "--n", str(pool_n),
                "--split", split,
                "--seed", str(seed),
                "--resolution", str(resolution),
                "--id_prefix", f"stage1_blender_{split}",
                "--second_hand_prob", "0.7",
                "--alarm_hand_prob", "0.3",
                "--time_mode", "random",
            ]
            if split == "clean":
                cmd.extend(["--clean_view_mode", "front"])
            if resume:
                cmd.append("--resume")
            _run(cmd)
        split_samples = _load_jsonl(blender_samples_path)
        for row in split_samples:
            row["_source"] = f"blender_{split}"
            row["_image_path"] = os.path.join(blender_root, split, row["image"])
        blender_samples.extend(split_samples)
    
    # Matplot pool
    matplot_root = os.path.join(out_dir, "_pool_matplot")
    matplot_samples_path = os.path.join(matplot_root, "rege_clean_matplot", "annotations.jsonl")
    if not (resume and os.path.exists(matplot_samples_path)):
        print(f"\n[Stage1] Generating Matplot pool ({pool_n} images)...")
        if os.path.exists(matplot_root):
            shutil.rmtree(matplot_root)
    cmd = [
        "python",
        os.path.join(base_dir, "matplot", "render_matplot_batch.py"),
        "--out_dir", matplot_root,
        "--n", str(pool_n),
        "--seed", str(seed + 7),
        "--resolution", str(resolution),
        "--id_prefix", "stage1_matplot",
        "--second_hand_prob", "0.7",
        "--alarm_hand_prob", "0.3",
        "--time_mode", "random",
    ]
    if resume:
        cmd.append("--resume")
    _run(cmd)
    matplot_samples = _load_jsonl(matplot_samples_path)
    for row in matplot_samples:
        row["_source"] = "matplot"
        row["_image_path"] = os.path.join(matplot_root, "rege_clean_matplot", row["image"])
    
    return blender_samples, matplot_samples


def _extract_meta(sample: Dict) -> Dict:
    label = sample.get("label", {})
    seconds = label.get("seconds")
    seconds_total = label.get("time_seconds_total")
    if seconds is None or seconds_total is None:
        seconds = None
        seconds_total = None
    meta = sample.get("meta", {})
    has_second = bool(meta.get("has_second")) if seconds is not None else False
    has_alarm = bool(meta.get("has_alarm")) if seconds is not None else False
    hand_config = meta.get("hand_config", 2 if seconds is None else 3)
    return {
        "time_minutes": int(label["time_minutes"]),
        "seconds": seconds,
        "seconds_total": seconds_total,
        "hand_config": hand_config,
        "has_second": has_second,
        "has_alarm": has_alarm,
        "style_id": sample.get("meta", {}).get("style_id"),
        "source": sample.get("_source", "unknown"),
    }


def _index_by_time(samples: List[Dict]) -> Dict[int, List[Dict]]:
    buckets: Dict[int, List[Dict]] = {}
    for s in samples:
        t = int(s["label"]["time_minutes"])
        buckets.setdefault(t, []).append(s)
    return buckets


def _index_by_time_source(samples: List[Dict]) -> Dict[int, Dict[str, List[Dict]]]:
    buckets: Dict[int, Dict[str, List[Dict]]] = {}
    for s in samples:
        t = int(s["label"]["time_minutes"])
        src = s.get("_source", "unknown")
        buckets.setdefault(t, {}).setdefault(src, []).append(s)
    return buckets


def _signed_delta_minutes(a: int, b: int) -> int:
    delta = b - a
    if delta > 360:
        delta -= 720
    if delta < -360:
        delta += 720
    return delta


def _sample_positive_cross_domain(rng, time_sources):
    for _ in range(1000):
        t = rng.choice(list(time_sources.keys()))
        sources = time_sources[t]
        if "matplot" in sources and any(src.startswith("blender") for src in sources):
            a = rng.choice(sources["matplot"])
            blender_src = rng.choice([s for s in sources if s.startswith("blender")])
            b = rng.choice(sources[blender_src])
            return a, b, "blender_matplot"
    raise RuntimeError("Failed to sample cross-domain positive")


def _sample_positive_clean_noisy(rng, time_sources):
    for _ in range(1000):
        t = rng.choice(list(time_sources.keys()))
        sources = time_sources[t]
        if "blender_clean" in sources and "blender_noisy" in sources:
            a = rng.choice(sources["blender_clean"])
            b = rng.choice(sources["blender_noisy"])
            return a, b, "blender_clean_noisy"
    raise RuntimeError("Failed to sample clean/noisy positive")


def _sample_positive_same_domain_diff_style(rng, time_sources):
    for _ in range(1000):
        t = rng.choice(list(time_sources.keys()))
        sources = time_sources[t]
        src = rng.choice(list(sources.keys()))
        candidates = sources[src]
        if len(candidates) < 2:
            continue
        a = rng.choice(candidates)
        a_style = a.get("meta", {}).get("style_id")
        b_candidates = [c for c in candidates if c.get("meta", {}).get("style_id") != a_style]
        if not b_candidates:
            continue
        b = rng.choice(b_candidates)
        return a, b, f"{src}_{src}"
    raise RuntimeError("Failed to sample same-domain diff-style positive")


def _sample_negative_for_anchor(rng, all_buckets, anchor_time, delta_range):
    for _ in range(1000):
        delta = rng.randint(delta_range[0], delta_range[1])
        if rng.random() < 0.5:
            delta = -delta
        t2 = (anchor_time + delta) % 720
        if t2 not in all_buckets:
            continue
        b = rng.choice(all_buckets[t2])
        return b, delta
    raise RuntimeError("Failed to sample negative")


def main():
    args = _parse_args()
    rng = random.Random(args.seed)

    n_triplets = args.n_triplets if args.n_triplets is not None else args.n_pairs
    if n_triplets is None:
        raise ValueError("Must provide --n_triplets (or legacy --n_pairs).")
    
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    out_jsonl = os.path.join(out_dir, "annotations.jsonl")
    existing_rows = []
    resume_rebuild = False
    if args.resume and os.path.exists(out_jsonl):
        existing_rows = _load_jsonl(out_jsonl)
    elif args.resume and os.path.exists(images_dir):
        # If images exist but JSONL is missing (e.g. interrupted before flush), rebuild labels.
        resume_rebuild = True
    start_index = 0 if resume_rebuild else len(existing_rows)
    
    if args.ensure_pools:
        _build_pools(out_dir, args.seed, args.resolution, n_triplets, args.resume, args.blender_split)
    if start_index >= n_triplets:
        print(f"✅ Already have {start_index} triplets, skipping generation")
        return
    
    # 生成图像池
    blender_samples, matplot_samples = _build_pools(out_dir, args.seed, args.resolution, n_triplets, args.resume, args.blender_split)
    
    all_samples = blender_samples + matplot_samples
    all_buckets = _index_by_time(all_samples)
    time_sources = _index_by_time_source(all_samples)

    n_pos_cross = int(n_triplets * 0.5)
    n_pos_clean_noisy = int(n_triplets * 0.3)
    n_pos_same = n_triplets - n_pos_cross - n_pos_clean_noisy

    n_neg_hard = int(n_triplets * 0.4)
    n_neg_easy = int(n_triplets * 0.4)
    n_neg_medium = n_triplets - n_neg_hard - n_neg_easy

    stats = Counter()

    print(f"\n[Stage1] Generating {n_triplets} triplets...")
    print(f"  - pos cross-domain: {n_pos_cross}")
    print(f"  - pos clean/noisy: {n_pos_clean_noisy}")
    print(f"  - pos same-domain diff-style: {n_pos_same}")
    print(f"  - neg hard 1-5: {n_neg_hard}")
    print(f"  - neg easy 30-180: {n_neg_easy}")
    print(f"  - neg medium 6-20: {n_neg_medium}\n")

    write_mode = "w" if resume_rebuild else "a"
    with open(out_jsonl, write_mode, encoding="utf-8") as f:
        for idx in range(start_index, n_triplets):
            if idx % 1000 == 0:
                print(f"  Progress: {idx}/{n_triplets}")

            if idx < n_pos_cross:
                anchor, positive, pos_source_pair = _sample_positive_cross_domain(rng, time_sources)
            elif idx < n_pos_cross + n_pos_clean_noisy:
                anchor, positive, pos_source_pair = _sample_positive_clean_noisy(rng, time_sources)
            else:
                anchor, positive, pos_source_pair = _sample_positive_same_domain_diff_style(rng, time_sources)

            neg_roll = idx % (n_neg_hard + n_neg_medium + n_neg_easy)
            if neg_roll < n_neg_hard:
                negative, delta = _sample_negative_for_anchor(rng, all_buckets, anchor["label"]["time_minutes"], (1, 5))
                delta_bucket = "hard"
            elif neg_roll < n_neg_hard + n_neg_medium:
                negative, delta = _sample_negative_for_anchor(rng, all_buckets, anchor["label"]["time_minutes"], (6, 20))
                delta_bucket = "medium"
            else:
                negative, delta = _sample_negative_for_anchor(rng, all_buckets, anchor["label"]["time_minutes"], (30, 180))
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

            delta_minutes = _signed_delta_minutes(anchor_meta["time_minutes"], negative_meta["time_minutes"])
            margin = args.margin_base + args.margin_alpha * math.log(1.0 + abs(delta_minutes))

            row = {
                "id": f"stage1_{idx:06d}",
                "anchor": os.path.join("images", filename_anchor),
                "positive": os.path.join("images", filename_pos),
                "negative": os.path.join("images", filename_neg),
                "label": {
                    "anchor_time_minutes": anchor_meta["time_minutes"],
                    "positive_time_minutes": positive_meta["time_minutes"],
                    "negative_time_minutes": negative_meta["time_minutes"],
                    "negative_delta": delta_minutes,
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
    
    # 保存统计
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"triplet_counts": dict(stats)}, f, indent=2)
    
    print(f"\n✅ Stage1 completed: {n_triplets} triplets")
    print(f"   Output: {out_jsonl}")


if __name__ == "__main__":
    main()
