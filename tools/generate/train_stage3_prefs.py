"""Generate stage3 preference data for Qwen3-VL.

⚠️ 注意：这个脚本需要先训练 Stage2 模型，然后用模型预测真实图来构造 preference pairs
如果还没有训练好的模型，可以先用合成数据的"错误答案"作为 rejected（临时方案）

Example:
python tools/generate/train_stage3_prefs.py \
  --out_dir /data/stage3_prefs \
  --seed 1 --n_prefs 1000 \
  --stage2_single_dir /data/stage2_sft_single \
  --stage2_pair_dir /data/stage2_sft_pair \
  --mode synthetic  # 或 --mode real --model_checkpoint /path/to/stage2_model
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter


def _parse_args():
    parser = argparse.ArgumentParser(description="Stage3 prefs generator")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_prefs", type=int, required=True)
    parser.add_argument("--stage2_single_dir", required=True)
    parser.add_argument("--stage2_pair_dir", required=True)
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                        help="synthetic: 从合成数据生成错误答案; real: 用模型预测真实图")
    parser.add_argument("--model_checkpoint", default=None,
                        help="Stage2 模型路径 (仅 mode=real 时需要)")
    parser.add_argument("--real_images_jsonl", default=None,
                        help="真实图 JSONL 路径 (仅 mode=real 时需要)")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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


def _bad_time(hhmm, offset):
    """生成错误的时间（偏移 offset 分钟）"""
    hh, mm = map(int, hhmm.split(":"))
    if hh == 12:
        hh = 0
    minutes = (hh * 60 + mm + offset) % 720
    hh = (minutes // 60) % 12
    mm = minutes % 60
    if hh == 0:
        hh = 12
    return f"{hh:02d}:{mm:02d}"


def _bad_delta(delta, mode):
    """生成错误的 delta"""
    if mode == "wrong_sign":
        return -delta
    elif mode == "off_by_5":
        return delta + 5 if delta >= 0 else delta - 5
    else:  # off_by_1
        return delta + 1


def _generate_synthetic_prefs(args, rng):
    """从合成数据生成 preference pairs（临时方案）"""
    print("\n[Stage3] Mode: synthetic (generating from Stage2 data)")
    
    single_rows = _load_jsonl(os.path.join(args.stage2_single_dir, "annotations.jsonl"))
    pair_rows = _load_jsonl(os.path.join(args.stage2_pair_dir, "annotations.jsonl"))
    
    rng.shuffle(single_rows)
    rng.shuffle(pair_rows)
    
    n_single = args.n_prefs // 2
    n_pair = args.n_prefs - n_single
    
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    out_jsonl = os.path.join(out_dir, "annotations.jsonl")
    stats = Counter()
    
    print(f"\n[Stage3] Generating {args.n_prefs} preference pairs...")
    print(f"  - Single-image: {n_single}")
    print(f"  - Pair: {n_pair}\n")
    
    with open(out_jsonl, "w", encoding="utf-8") as f:
        idx = 0
        
        # Single-image preferences
        for row in single_rows[:n_single]:
            image = row["images"][0]
            src_image = os.path.join(args.stage2_single_dir, image)
            filename = f"pref_{idx:06d}.png"
            dst = os.path.join(images_dir, filename)
            _safe_link(src_image, dst)
            
            label = row["label"]
            correct = label["time_hhmm"]
            wrong = _bad_time(correct, rng.choice([1, 2, -1, -2]))
            
            prompt = row["messages"][0]["content"][1]["text"]
            chosen = row["messages"][1]["content"][0]["text"]
            rejected = chosen.replace(correct, wrong)
            
            record = {
                "id": f"stage3_single_{idx:06d}",
                "images": [os.path.join("images", filename)],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "label": label,
                "meta": {
                    "source": "synthetic",
                    "error_type": "off_by_1_or_2_min",
                },
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            stats["single_off_by_1_or_2_min"] += 1
            idx += 1
        
        # Pair preferences
        for row in pair_rows[:n_pair]:
            image_a, image_b = row["images"]
            src_a = os.path.join(args.stage2_pair_dir, image_a)
            src_b = os.path.join(args.stage2_pair_dir, image_b)
            filename_a = f"pref_{idx:06d}_a.png"
            filename_b = f"pref_{idx:06d}_b.png"
            dst_a = os.path.join(images_dir, filename_a)
            dst_b = os.path.join(images_dir, filename_b)
            _safe_link(src_a, dst_a)
            _safe_link(src_b, dst_b)
            
            label = row["label"]
            delta = label["delta_minutes"]
            prompt = row["messages"][0]["content"][2]["text"]
            chosen = row["messages"][1]["content"][0]["text"]
            
            error_type = rng.choice(["wrong_sign", "off_by_5", "off_by_1"])
            wrong_delta = _bad_delta(delta, error_type)
            rejected = chosen.replace(str(delta), str(wrong_delta))
            
            record = {
                "id": f"stage3_pair_{idx:06d}",
                "images": [
                    os.path.join("images", filename_a),
                    os.path.join("images", filename_b),
                ],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "label": label,
                "meta": {
                    "source": "synthetic",
                    "error_type": error_type,
                },
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            stats[f"pair_{error_type}"] += 1
            idx += 1
    
    # 保存统计
    with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"error_type_counts": dict(stats)}, f, indent=2)
    
    print(f"\n✅ Stage3 completed: {args.n_prefs} preference pairs")
    print(f"   Output: {out_jsonl}")
    print(f"\n⚠️  注意：这是临时方案（synthetic mode）")
    print(f"   建议训练完 Stage2 后，用 mode=real 重新生成")


def _generate_real_prefs(args):
    """用 Stage2 模型预测真实图，生成 preference pairs（推荐方案）"""
    print("\n[Stage3] Mode: real (using Stage2 model predictions)")
    
    if not args.model_checkpoint:
        raise ValueError("--model_checkpoint is required for mode=real")
    if not args.real_images_jsonl:
        raise ValueError("--real_images_jsonl is required for mode=real")
    
    # TODO: 实现模型加载和预测逻辑
    # 这部分需要根据你的训练框架来实现
    print("\n⚠️  Real mode 需要实现模型预测逻辑")
    print("   请参考以下伪代码：")
    print("""
    from your_training_framework import load_model
    
    model = load_model(args.model_checkpoint)
    real_images = _load_jsonl(args.real_images_jsonl)
    
    preferences = []
    for img in real_images:
        pred = model.predict(img['image'])
        pred_time = parse_time(pred)
        gt_time = img['label']['time_hhmm']
        
        if abs(pred_time - gt_time) > 5:  # 误差 > 5 分钟
            preferences.append({
                'chosen': generate_correct_cot(img),
                'rejected': pred,
                'label': img['label'],
            })
    """)
    
    raise NotImplementedError("Real mode not implemented yet. Please use mode=synthetic first.")


def main():
    args = _parse_args()
    rng = random.Random(args.seed)
    
    if args.mode == "synthetic":
        _generate_synthetic_prefs(args, rng)
    else:
        _generate_real_prefs(args)


if __name__ == "__main__":
    main()
