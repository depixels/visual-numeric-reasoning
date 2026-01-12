"""Generate stage2 single-image SFT data for Qwen3-VL.

Example:
python tools/generate/train_stage2_sft_single.py \
  --out_dir /data/stage2_sft_single \
  --seed 1 --resolution 512 --n_samples 50000
"""

import argparse
import json
import os
import random
import shutil
import subprocess
from collections import Counter
from typing import Dict, List, Any


def _parse_args():
    parser = argparse.ArgumentParser(description="Stage2 SFT single generator")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--reuse_pools_dir", default=None)
    parser.add_argument("--pool_blender_splits", choices=["clean", "noisy", "both"], default="both")
    parser.add_argument("--pool_use_matplot", action="store_true")
    parser.add_argument("--dry_run", type=int, default=0)
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


def _load_stage1_pools(base_dir: str, blender_splits: str, use_matplot: bool) -> List[Dict]:
    samples: List[Dict] = []
    blender_root = os.path.join(base_dir, "_pool_blender")
    splits = ["clean", "noisy"] if blender_splits == "both" else [blender_splits]
    for split in splits:
        samples_path = os.path.join(blender_root, split, "samples.jsonl")
        if not os.path.exists(samples_path):
            continue
        rows = _load_jsonl(samples_path)
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


def _safe_link(src, dst):
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _label_answer(label: Dict[str, Any]) -> tuple[str, bool]:
    if label.get("seconds") is not None and label.get("time_hhmmss"):
        return label["time_hhmmss"], True
    return label["time_hhmm"], False


def _summarize_conditions(meta: Dict[str, Any], source: str) -> List[str]:
    conditions: List[str] = []
    view = meta.get("view") or {}
    pose = meta.get("pose") or {}
    view_bucket = meta.get("view_bucket")

    def _near_zero(value: Any) -> bool:
        try:
            return abs(float(value)) < 1.0
        except Exception:
            return False

    yaw = view.get("yaw", pose.get("yaw"))
    pitch = view.get("pitch", pose.get("pitch"))
    roll = view.get("roll", pose.get("roll"))
    if any(not _near_zero(v) for v in (yaw, pitch, roll) if v is not None):
        conditions.append(
            f"This image is tilted (yaw={yaw}, pitch={pitch}, roll={roll})."
        )

    degradation = meta.get("degradation") or {}
    specular = degradation.get("specular")
    motion_blur = degradation.get("motion_blur")
    defocus = degradation.get("defocus")
    if specular is not None and float(specular) > 0.3:
        conditions.append(f"There is glare/reflection on the glass (specular={specular}).")
    if motion_blur is not None and float(motion_blur) > 0.05:
        conditions.append(f"There is motion blur (motion_blur={motion_blur}).")
    if defocus is not None and float(defocus) > 0.05:
        conditions.append(f"The image is slightly out of focus (defocus={defocus}).")

    if view_bucket and not str(view_bucket).startswith("front_"):
        conditions.append("This is a challenging viewing angle.")
    if source == "blender_noisy":
        conditions.append("This is a challenging view with synthetic degradation.")
    return conditions


def _get_perspective_description(meta: Dict[str, Any]) -> str:
    """
    将角度元数据翻译成人类的视觉描述。
    不输出具体数字，只输出视觉感受。
    """
    view = meta.get("view") or {}
    pose = meta.get("pose") or {}
    
    # 获取角度，默认为0
    yaw = float(view.get("yaw", pose.get("yaw", 0)) or 0)
    pitch = float(view.get("pitch", pose.get("pitch", 0)) or 0)
    roll = float(view.get("roll", pose.get("roll", 0)) or 0)

    # 设定阈值，模拟人类感知
    descriptions = []
    
    # 1. 左右侧视 (Yaw)
    if abs(yaw) > 45:
        descriptions.append("viewed from a steep side angle")
    elif abs(yaw) > 20:
        descriptions.append("viewed from a side angle")
    
    # 2. 俯仰视 (Pitch)
    if pitch > 20:
        descriptions.append("viewed from above")
    elif pitch < -20:
        descriptions.append("viewed from below")
        
    # 3. 旋转 (Roll)
    if abs(roll) > 15:
        descriptions.append("the clock face is tilted/rotated")

    if not descriptions:
        return "The clock is viewed directly from the front."
    
    return "The clock is " + ", and ".join(descriptions) + "."


def _get_quality_description(meta: Dict[str, Any], source: str) -> str:
    """
    将画质元数据翻译成视觉干扰描述。
    """
    degradation = meta.get("degradation") or {}
    specular = float(degradation.get("specular", 0))
    blur = float(degradation.get("motion_blur", 0)) + float(degradation.get("defocus", 0))
    noise = float(degradation.get("noise", 0))

    issues = []
    
    # 强反光
    if specular > 0.4:
        issues.append("bright glare or reflections on the glass surface")
    elif specular > 0.1:
        issues.append("slight reflections")
        
    # 模糊
    if blur > 0.1:
        issues.append("blurriness affecting the edges")
    
    # 噪声 (通常 Blender noisy split 会有)
    if "noisy" in source or noise > 0.1:
        issues.append("visual noise or grain")

    if not issues:
        return "The image is clear and the details are sharp."
    
    return f"I notice {', '.join(issues)}, which might make reading the hands slightly difficult."


def _describe_hand_geometry(hand_name: str, value: int, total_steps: int, is_hour: bool = False) -> str:
    """
    根据数值生成纯几何的视觉描述（指向哪里）。
    """
    # 将数值映射到表盘刻度 (0-12)
    # 分针/秒针: 0-60 -> 0-12
    # 时针: 0-12 -> 0-12
    
    if is_hour:
        # 时针的位置是连续的，例如 10:30，时针在 10 和 11 中间
        cycle = 12
        position = value % 12
        if position == 0: position = 12
        return f"pointing near the {position} o'clock direction"
    else:
        # 分针秒针
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


def _generate_grounded_cot(label: Dict[str, Any], meta: Dict[str, Any], source: str) -> str:
    # 1. 获取答案
    answer_str, has_seconds = _label_answer(label)
    
    # 解析时间用于生成几何描述
    try:
        parts = answer_str.split(":")
        hh = int(parts[0])
        mm = int(parts[1])
        ss = int(parts[2]) if has_seconds else 0
    except:
        return f"<think>Time is {answer_str}.</think>\n<answer>{answer_str}</answer>"

    # 2. 构建思维链
    thoughts = []
    
    # --- Step 1: 整体观察 (Observation) ---
    thoughts.append(_get_perspective_description(meta))
    thoughts.append(_get_quality_description(meta, source))
    thoughts.append("Let's identify the hands based on their length and shape.")

    # --- Step 2: 视觉定位 (Visual Evidence) ---
    # 时针 (Short/Thick)
    hh_desc = _describe_hand_geometry("hour", hh, 12, is_hour=True)
    # 如果分钟过半，时针应该描述为 "between X and Y"
    if mm > 30:
        current_h = hh % 12
        if current_h == 0: current_h = 12
        next_h = (current_h % 12) + 1
        hh_desc = f"positioned between {current_h} and {next_h}"
    
    thoughts.append(f"The **short, thick hand** (hour hand) is {hh_desc}.")
    
    # 分针 (Long)
    mm_desc = _describe_hand_geometry("minute", mm, 60)
    thoughts.append(f"The **longer hand** (minute hand) is {mm_desc}.")
    
    # 秒针 (Thin)
    if has_seconds:
        ss_desc = _describe_hand_geometry("second", ss, 60)
        thoughts.append(f"The **thinnest hand** (second hand) is {ss_desc}.")

    # --- Step 3: 逻辑推理 (Reasoning) ---
    thoughts.append("Now, let's translate these positions into time:")
    
    thoughts.append(f"- Hour hand at that position indicates the hour is {hh}.")
    
    # 分针推理：把视觉位置转回数字
    # 比如 pointing at 3 -> 15 minutes
    m_clock = mm // 5
    if m_clock == 0: m_clock = 12
    m_rem = mm % 5
    if m_rem == 0:
        thoughts.append(f"- Minute hand on the {m_clock} mark corresponds to {mm} minutes.")
    else:
        thoughts.append(f"- Minute hand past the {m_clock} mark by {m_rem} ticks corresponds to {mm} minutes.")

    if has_seconds:
        thoughts.append(f"- Second hand position corresponds to {ss} seconds.")

    # --- Step 4: 结论 (Conclusion) ---
    thoughts.append(f"Putting it all together, the time is {answer_str}.")

    # 拼接并格式化
    think_block = " ".join(thoughts)
    return f"<think>{think_block}</think>\n<answer>{answer_str}</answer>"

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
        if len(existing_rows) >= args.n_samples:
            print(f"✅ Already have {len(existing_rows)} samples, skipping")
            return
    
    if args.reuse_pools_dir:
        all_samples = _load_stage1_pools(
            args.reuse_pools_dir,
            args.pool_blender_splits,
            args.pool_use_matplot,
        )
        if not all_samples:
            raise RuntimeError("No samples found in reuse_pools_dir")
        rng.shuffle(all_samples)
        while len(all_samples) < args.n_samples:
            all_samples.extend(all_samples[: max(1, args.n_samples - len(all_samples))])
        all_samples = all_samples[: args.n_samples]
    else:
        # 分配数量
        n_blender_clean = args.n_samples // 3
        n_blender_noisy = args.n_samples // 3
        n_matplot = args.n_samples - n_blender_clean - n_blender_noisy

        # Blender clean
        print(f"\n[Stage2-Single] Generating Blender clean ({n_blender_clean})...")
        blender_clean_root = os.path.join(out_dir, "_pool_blender_clean")
        if os.path.exists(blender_clean_root):
            shutil.rmtree(blender_clean_root)
        _run([
            "blender", "-b", "-P",
            os.path.join(base_dir, "blender", "render_batch.py"),
            "--",
            "--out_dir", blender_clean_root,
            "--n", str(n_blender_clean),
            "--split", "clean",
            "--seed", str(args.seed),
            "--resolution", str(args.resolution),
            "--id_prefix", "stage2_clean",
        ])
        clean_samples = _load_jsonl(os.path.join(blender_clean_root, "clean", "samples.jsonl"))
        for row in clean_samples:
            row["_source"] = "blender_clean"
            row["_image_path"] = os.path.join(blender_clean_root, "clean", row["image"])

        # Blender noisy
        print(f"\n[Stage2-Single] Generating Blender noisy ({n_blender_noisy})...")
        blender_noisy_root = os.path.join(out_dir, "_pool_blender_noisy")
        if os.path.exists(blender_noisy_root):
            shutil.rmtree(blender_noisy_root)
        _run([
            "blender", "-b", "-P",
            os.path.join(base_dir, "blender", "render_batch.py"),
            "--",
            "--out_dir", blender_noisy_root,
            "--n", str(n_blender_noisy),
            "--split", "noisy",
            "--seed", str(args.seed + 7),
            "--resolution", str(args.resolution),
            "--id_prefix", "stage2_noisy",
        ])
        noisy_samples = _load_jsonl(os.path.join(blender_noisy_root, "noisy", "samples.jsonl"))
        for row in noisy_samples:
            row["_source"] = "blender_noisy"
            row["_image_path"] = os.path.join(blender_noisy_root, "noisy", row["image"])

        # Matplot
        print(f"\n[Stage2-Single] Generating Matplot ({n_matplot})...")
        matplot_root = os.path.join(out_dir, "_pool_matplot")
        if os.path.exists(matplot_root):
            shutil.rmtree(matplot_root)
        _run([
            "python",
            os.path.join(base_dir, "matplot", "render_matplot_batch.py"),
            "--out_dir", matplot_root,
            "--n", str(n_matplot),
            "--seed", str(args.seed + 13),
            "--resolution", str(args.resolution),
            "--id_prefix", "stage2_matplot",
        ])
        matplot_samples = _load_jsonl(os.path.join(matplot_root, "rege_clean_matplot", "annotations.jsonl"))
        for row in matplot_samples:
            row["_source"] = "matplot"
            row["_image_path"] = os.path.join(matplot_root, "rege_clean_matplot", row["image"])

        # 合并并打乱
        all_samples = clean_samples + noisy_samples + matplot_samples
        rng.shuffle(all_samples)
        all_samples = all_samples[: args.n_samples]
    
    stats = Counter()
    existing_rows = []
    if args.resume and os.path.exists(out_jsonl):
        existing_rows = _load_jsonl(out_jsonl)
    start_index = len(existing_rows)
    if start_index >= args.n_samples:
        print(f"✅ Already have {start_index} samples, skipping generation")
        return
    
    print(f"\n[Stage2-Single] Writing {len(all_samples)} samples...")
    with open(out_jsonl, "a", encoding="utf-8") as f:
        for idx, row in enumerate(all_samples[start_index:], start=start_index):
            if idx % 1000 == 0:
                print(f"  Progress: {idx}/{len(all_samples)}")
            
            filename = f"sample_{idx:06d}.png"
            dst = os.path.join(images_dir, filename)
            _safe_link(row["_image_path"], dst)
            
            label = row["label"]
            meta = row.get("meta", {})
            source = row.get("_source", "unknown")
            cot = _generate_grounded_cot(label, meta, source)
            answer, has_seconds = _label_answer(label)
            expected_answer = label.get("time_hhmmss") if has_seconds else label.get("time_hhmm")
            if expected_answer not in cot:
                raise AssertionError("CoT answer does not match label.")
            if not cot.strip().endswith(f"<answer>{expected_answer}</answer>"):
                raise AssertionError("Final <answer> tag does not match label.")
            
            record = {
                "id": f"stage2_single_{idx:06d}",
                "images": [os.path.join("images", filename)],
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": os.path.join("images", filename)},
                            {
                                "type": "text",
                                "text": "Read the exact time shown on this analog clock. Answer in HH:MM:SS format."
                                if has_seconds
                                else "Read the exact time shown on this analog clock. Answer in HH:MM format.",
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": cot}],
                    },
                ],
                "label": {
                    "time_minutes": label["time_minutes"],
                    "time_hhmm": label["time_hhmm"],
                    "time_hhmmss": label.get("time_hhmmss"),
                    "time_seconds_total": label.get("time_seconds_total"),
                    "seconds": label.get("seconds"),
                },
                "meta": {
                    "source": source,
                    "style_id": meta.get("style_id", "unknown"),
                },
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            stats[row["_source"]] += 1
            if args.dry_run and (idx - start_index + 1) >= args.dry_run:
                print("\n[Stage2-Single] Dry run sample")
                print("label:", label)
                print("cot:", cot)
                return
    
    # 保存统计
    all_rows = _load_jsonl(out_jsonl)
    stats = Counter()
    for row in all_rows:
        stats[row.get("meta", {}).get("source", "unknown")] += 1
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"source_counts": dict(stats)}, f, indent=2)
    
    print(f"\n✅ Stage2-Single completed: {len(all_samples)} samples")
    print(f"   Output: {out_jsonl}")


if __name__ == "__main__":
    main()
