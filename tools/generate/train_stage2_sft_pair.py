"""Generate stage2 pair SFT data for Qwen3-VL.

Example:
python tools/generate/train_stage2_sft_pair.py \
  --out_dir /data/stage2_sft_pair \
  --seed 1 --resolution 512 --n_pairs 50000
"""

import argparse
import json
import os
import random
import shutil
import subprocess
from collections import Counter


def _parse_args():
    parser = argparse.ArgumentParser(description="Stage2 SFT pair generator")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--n_pairs", type=int, required=True)
    parser.add_argument("--reuse_pools_dir", default=None)
    parser.add_argument("--pool_blender_splits", choices=["clean", "noisy", "both"], default="both")
    parser.add_argument("--pool_use_matplot", action="store_true")
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


def _hhmm_to_minutes(hhmm):
    hh, mm = map(int, hhmm.split(":"))
    if hh == 12:
        hh = 0
    return hh * 60 + mm


def _minutes_to_hhmm(time_minutes: int) -> str:
    hh = (time_minutes // 60) % 12
    mm = time_minutes % 60
    if hh == 0:
        hh = 12
    return f"{hh:02d}:{mm:02d}"


def _load_stage1_pools(base_dir: str, blender_splits: str, use_matplot: bool):
    samples = []
    blender_root = os.path.join(base_dir, "_pool_blender")
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
        matplot_root = os.path.join(base_dir, "_pool_matplot", "rege_clean_matplot")
        matplot_path = os.path.join(matplot_root, "annotations.jsonl")
        if os.path.exists(matplot_path):
            rows = _load_jsonl(matplot_path)
            for row in rows:
                row["_source"] = "matplot"
                row["_image_path"] = os.path.join(matplot_root, row["image"])
            samples.extend(rows)
    return samples


def _index_by_time(samples):
    buckets = {}
    for s in samples:
        t = int(s["label"]["time_minutes"])
        buckets.setdefault(t, []).append(s)
    return buckets


def _describe_hand_geometry(hand_name: str, value: int, total_steps: int, is_hour: bool = False) -> str:
    """
    (复用自 Single Image 脚本) 生成纯几何的视觉描述。
    """
    if is_hour:
        cycle = 12
        position = value % 12
        if position == 0: position = 12
        return f"pointing near the {position} o'clock direction"
    else:
        clock_pos = value // 5
        if clock_pos == 0: clock_pos = 12
        remainder = value % 5
        
        if remainder == 0:
            return f"pointing exactly at the {clock_pos}"
        elif remainder <= 2:
            return f"pointing just past the {clock_pos}"
        else:
            next_pos = (clock_pos % 12) + 1
            return f"pointing just before the {next_pos}"

def _analyze_single_clock(hhmm: str, label_prefix: str) -> dict:
    """
    生成单个时钟的分析步骤。
    """
    hh, mm = map(int, hhmm.split(":"))
    
    # 视觉描述
    hh_desc = _describe_hand_geometry("hour", hh, 12, is_hour=True)
    if mm > 30:
        current_h = hh % 12
        if current_h == 0: current_h = 12
        next_h = (current_h % 12) + 1
        hh_desc = f"positioned between {current_h} and {next_h}"
        
    mm_desc = _describe_hand_geometry("minute", mm, 60)
    
    # 转换为分钟数 (0-719)
    minutes_total = (hh % 12) * 60 + mm
    
    steps = [
        f"Looking at **{label_prefix}**:",
        f"- The short hour hand is {hh_desc}.",
        f"- The long minute hand is {mm_desc}.",
        f"- This visually corresponds to {hh}:{mm:02d}.",
        f"- Converting this to minutes from 12:00: ({hh if hh!=12 else 0} * 60) + {mm} = {minutes_total} minutes."
    ]
    
    return {
        "text": "\n".join(steps),
        "minutes": minutes_total,
        "time_str": f"{hh}:{mm:02d}"
    }

def _generate_cot_pair(rng, label: dict) -> str:
    """
    生成两图对比的 CoT，包含独立的视觉分析和包含周期修正的计算过程。
    """
    time_a_str = label["time_a_hhmm"]
    time_b_str = label["time_b_hhmm"]
    target_delta = label["delta_minutes"]

    # 1. 独立分析两个时钟
    analysis_a = _analyze_single_clock(time_a_str, "Clock A (First Image)")
    analysis_b = _analyze_single_clock(time_b_str, "Clock B (Second Image)")

    # 2. 计算逻辑
    min_a = analysis_a["minutes"]
    min_b = analysis_b["minutes"]
    
    # 初步差值
    raw_diff = min_b - min_a
    
    # 3. 构建推理文本
    thoughts = []
    thoughts.append("I need to calculate the time difference in minutes from Clock A to Clock B (B - A).")
    thoughts.append("Let's analyze each clock individually based on the hand positions.")
    
    # 添加视觉分析部分
    thoughts.append(analysis_a["text"])
    thoughts.append(analysis_b["text"])
    
    thoughts.append("\nNow, let's calculate the difference (B - A):")
    thoughts.append(f"{min_b} (Clock B) - {min_a} (Clock A) = {raw_diff} minutes.")

    # 4. 处理周期性 (Wrap-around Logic)
    # 如果直接相减的结果不等于 label 中的 delta，说明跨越了 12 点
    if raw_diff != target_delta:
        # 计算修正值
        diff_12h = 12 * 60 # 720
        
        if raw_diff < target_delta: 
            # 例如 A=11:00 (660), B=01:00 (60). Raw = -600. Target = +120.
            # 需要 +720
            thoughts.append(f"The result {raw_diff} seems large and negative, but visually the clocks are close in time.")
            thoughts.append(f"Since clocks operate on a 12-hour cycle (720 minutes), we add 720 to account for the wrap-around.")
            thoughts.append(f"{raw_diff} + 720 = {target_delta} minutes.")
        else:
            # 例如 A=01:00 (60), B=11:00 (660). Raw = +600. Target = -120.
            # 需要 -720
            thoughts.append(f"The result {raw_diff} seems large, implying we might be looking backwards across the 12 o'clock boundary.")
            thoughts.append(f"Adjusting for the 12-hour cycle: {raw_diff} - 720 = {target_delta} minutes.")
    else:
        thoughts.append("The difference is straightforward and requires no 12-hour cycle adjustment.")

    thoughts.append(f"So, the final time difference is {target_delta} minutes.")

    think_block = "\n".join(thoughts)
    return f"<think>{think_block}</think>\n<answer>{target_delta}</answer>"

def main():
    args = _parse_args()
    rng = random.Random(args.seed)
    
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    base_dir = os.path.dirname(__file__)
    
    out_jsonl = os.path.join(out_dir, "annotations.jsonl")
    if args.resume and os.path.exists(out_jsonl):
        existing_rows = _load_jsonl(out_jsonl)
        if len(existing_rows) >= args.n_pairs:
            print(f"✅ Already have {len(existing_rows)} pairs, skipping")
            return
    
    hard_n = int(args.n_pairs * 0.3)
    easy_n = args.n_pairs - hard_n

    print(f"\n[Stage2-Pair] Generating {args.n_pairs} pairs...")
    print(f"  Delta distribution: hard={hard_n}, easy={easy_n}\n")

    rows = []
    if args.reuse_pools_dir:
        samples = _load_stage1_pools(args.reuse_pools_dir, args.pool_blender_splits, args.pool_use_matplot)
        if not samples:
            raise RuntimeError("No samples found in reuse_pools_dir")
        buckets = _index_by_time(samples)
        rng.shuffle(samples)

        deltas = []
        for _ in range(hard_n):
            delta = rng.randint(1, 5)
            deltas.append(delta if rng.random() < 0.5 else -delta)
        for _ in range(easy_n):
            delta = rng.randint(10, 180)
            deltas.append(delta if rng.random() < 0.5 else -delta)
        rng.shuffle(deltas)

        for idx in range(args.n_pairs):
            for _ in range(2000):
                time_a = rng.choice(list(buckets.keys()))
                delta = deltas[idx]
                time_b = (time_a + delta) % 720
                if time_b not in buckets:
                    continue
                sample_a = rng.choice(buckets[time_a])
                sample_b = rng.choice(buckets[time_b])
                if sample_a["id"] == sample_b["id"]:
                    continue
                row = {
                    "image_a": sample_a["image"],
                    "image_b": sample_b["image"],
                    "label": {
                        "time_a_hhmm": _minutes_to_hhmm(time_a),
                        "time_b_hhmm": _minutes_to_hhmm(time_b),
                        "delta_minutes": delta,
                    },
                    "meta": {
                        "pair_type": "same_style_same_tz"
                        if sample_a.get("meta", {}).get("style_id") == sample_b.get("meta", {}).get("style_id")
                        else "cross_style_same_tz",
                        "style_id_a": sample_a.get("meta", {}).get("style_id"),
                        "style_id_b": sample_b.get("meta", {}).get("style_id"),
                    },
                    "_image_path_a": sample_a["_image_path"],
                    "_image_path_b": sample_b["_image_path"],
                }
                rows.append(row)
                break
        pool_root = args.reuse_pools_dir
    else:
        # 配置 pair 类型分布
        pair_quota = {
            "same_style_same_tz": args.n_pairs // 4,
            "cross_style_same_tz": args.n_pairs // 4,
            "same_style_cross_tz": args.n_pairs // 4,
        }
        pair_quota["cross_style_cross_tz"] = args.n_pairs - sum(pair_quota.values())

        delta_quota = {
            "hard_n": hard_n,
            "hard_min": 1,
            "hard_max": 5,
            "easy_n": easy_n,
            "easy_min": 10,
            "easy_max": 180,
        }

        print(f"  Pair types: {pair_quota}")

        pool_root = os.path.join(out_dir, "_pool_pairs")
        if os.path.exists(pool_root):
            shutil.rmtree(pool_root)

        _run([
            "blender", "-b", "-P",
            os.path.join(base_dir, "blender", "render_batch.py"),
            "--",
            "--out_dir", pool_root,
            "--n", str(args.n_pairs),
            "--split", "pair",
            "--seed", str(args.seed),
            "--resolution", str(args.resolution),
            "--id_prefix", "stage2_pair",
            "--pair_quota_json", json.dumps(pair_quota),
            "--delta_quota_json", json.dumps(delta_quota),
        ])

        rows = _load_jsonl(os.path.join(pool_root, "pair", "pairs.jsonl"))
    
    stats = Counter()
    existing_rows = []
    if args.resume and os.path.exists(out_jsonl):
        existing_rows = _load_jsonl(out_jsonl)
    start_index = len(existing_rows)
    if start_index >= args.n_pairs:
        print(f"✅ Already have {start_index} pairs, skipping generation")
        return
    
    print(f"\n[Stage2-Pair] Writing {len(rows)} pairs...")
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for idx, row in enumerate(rows[start_index:], start=start_index):
            if idx % 1000 == 0:
                print(f"  Progress: {idx}/{len(rows)}")
            
            image_a = row["image_a"]
            image_b = row["image_b"]
            if args.reuse_pools_dir:
                src_a = row["_image_path_a"]
                src_b = row["_image_path_b"]
            else:
                src_a = os.path.join(pool_root, "pair", image_a)
                src_b = os.path.join(pool_root, "pair", image_b)
            
            filename_a = f"pair_{idx:06d}_a.png"
            filename_b = f"pair_{idx:06d}_b.png"
            dst_a = os.path.join(images_dir, filename_a)
            dst_b = os.path.join(images_dir, filename_b)
            _safe_link(src_a, dst_a)
            _safe_link(src_b, dst_b)
            
            label = row["label"]
            time_a_min = _hhmm_to_minutes(label["time_a_hhmm"])
            time_b_min = _hhmm_to_minutes(label["time_b_hhmm"])
            delta = label["delta_minutes"]
            
            # 生成 CoT
            label_with_minutes = {
                **label,
                "time_a_minutes": time_a_min,
                "time_b_minutes": time_b_min,
            }
            cot = _generate_cot_pair(rng, label_with_minutes)
            
            meta = row.get("meta", {})
            
            record = {
                "id": f"stage2_pair_{idx:06d}",
                "images": [
                    os.path.join("images", filename_a),
                    os.path.join("images", filename_b),
                ],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": os.path.join("images", filename_a)},
                            {"type": "image", "image": os.path.join("images", filename_b)},
                            {
                                "type": "text",
                                "text": "These two clocks may have different styles or timezones. Calculate the time difference in minutes from Clock A to Clock B (B - A). Answer as an integer number of minutes.",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": cot}],
                    },
                ],
                "label": {
                    "delta_minutes": delta,
                    "time_a_minutes": time_a_min,
                    "time_b_minutes": time_b_min,
                    "time_a_hhmm": label["time_a_hhmm"],
                    "time_b_hhmm": label["time_b_hhmm"],
                },
                "meta": {
                    "pair_type": meta.get("pair_type"),
                    "style_id_a": meta.get("style_id_a"),
                    "style_id_b": meta.get("style_id_b"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            stats[meta.get("pair_type", "unknown")] += 1
    
    # 保存统计
    all_rows = _load_jsonl(out_jsonl)
    stats = Counter()
    for row in all_rows:
        stats[row.get("meta", {}).get("pair_type", "unknown")] += 1
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"pair_type_counts": dict(stats)}, f, indent=2)
    
    print(f"\n✅ Stage2-Pair completed: {len(rows)} pairs")
    print(f"   Output: {out_jsonl}")


if __name__ == "__main__":
    main()
