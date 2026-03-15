#!/usr/bin/env python3
"""
绘制读表准确率随相机倾斜角度变化的曲线图 (Accuracy vs. Tilt Angle)
支持数据分箱 (Binning) 和 均衡采样 (Balanced Sampling)。
"""

import argparse
import json
import random
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Accuracy vs Tilt Angle with Sampling")
    parser.add_argument("--pred_json", required=True, help="Path to predictions.json")
    parser.add_argument("--gt_jsonl", required=True, help="Path to original samples.jsonl")
    parser.add_argument("--output", default="accuracy_vs_tilt.png", help="Output image path")
    parser.add_argument("--tolerance", type=int, default=0, help="Minute tolerance for correctness")
    parser.add_argument("--bin_size", type=int, default=10, help="Bin size for tilt angles (degrees)")
    parser.add_argument("--balance", action="store_true", help="Randomly downsample all bins to the same size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)

    # 1. 读取 Ground Truth
    print(f"Loading ground truth from {args.gt_jsonl}...")
    samples_meta = {}
    with open(args.gt_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            samples_meta[row['id']] = row

    # 2. 读取预测结果
    print(f"Loading predictions from {args.pred_json}...")
    with open(args.pred_json, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    # 3. 将预测结果按倾斜角度区间 (Bins) 归类
    bin_data = defaultdict(list)
    missing_count = 0
    
    for pred in preds:
        sample_id = pred['id']
        if sample_id not in samples_meta:
            missing_count += 1
            continue
            
        meta = samples_meta[sample_id].get('meta', {})
        view = meta.get('view', {})
        
        pitch = float(view.get('pitch', 90.0))
        tilt_angle = 90.0 - pitch
        bin_start = int(tilt_angle // args.bin_size) * args.bin_size
        
        bin_data[bin_start].append(pred)

    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} predictions were missing from GT and skipped.")

    # 过滤掉完全没有数据的异常区间
    bin_data = {k: v for k, v in bin_data.items() if len(v) > 0}

    # 4. 均衡采样 (如果开启了 --balance)
    if args.balance and len(bin_data) > 0:
        min_n = min(len(v) for v in bin_data.values())
        print(f"\n⚖️ [Balance Mode] Downsampling all bins to {min_n} samples...")
        for b in bin_data:
            bin_data[b] = random.sample(bin_data[b], min_n)

    # 5. 计算准确率
    sorted_bins = sorted(bin_data.keys())
    x_labels = []
    accuracies = []
    
    print("\n--- Accuracy Report ---")
    for b in sorted_bins:
        items = bin_data[b]
        total = len(items)
        correct = 0
        
        for pred in items:
            gt = pred.get('gt_time_minutes')
            pr = pred.get('pred_time_minutes')
            if gt is not None and pr is not None:
                error = abs(gt - pr)
                error = min(error, 720 - error) # 12小时周期
                if error <= args.tolerance:
                    correct += 1
                    
        acc = (correct / total) * 100 if total > 0 else 0
        accuracies.append(acc)
        
        # 构造 X 轴标签，包含 N 的数量
        label = f"{b}°-{b + args.bin_size}°\n(N={total})"
        x_labels.append(label)
        print(f"Tilt {b:2d}°-{b+args.bin_size:2d}° | N={total:3d} | Accuracy {acc:5.1f}% ({correct}/{total})")

    if not accuracies:
        print("Error: No valid data points to plot.")
        return

    # 6. 画图
    plt.figure(figsize=(10, 6.5), dpi=300)
    x_positions = range(len(sorted_bins))
    
    plt.plot(x_positions, accuracies, marker='o', linestyle='-', 
             linewidth=3, markersize=10, color='#1f77b4', label='Model Accuracy')
    
    plt.title('Model Robustness under Perspective Distortion', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Camera Tilt Angle (Larger = More Distortion)', fontsize=14, labelpad=10)
    plt.ylabel(f'Accuracy (%) [Tol ≤ {args.tolerance} min]', fontsize=14)
    
    plt.ylim(-5, 105)
    plt.xticks(x_positions, x_labels, fontsize=11)
    plt.yticks(range(0, 101, 20), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    
    # 标注数值
    for x, y in zip(x_positions, accuracies):
        plt.annotate(f'{y:.1f}%', 
                     (x, y), 
                     textcoords="offset points", 
                     xytext=(0, 12), 
                     ha='center',
                     fontsize=11,
                     fontweight='bold',
                     color='#1f77b4')
        
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"\n✅ Plot successfully saved to: {args.output}")

if __name__ == '__main__':
    main()