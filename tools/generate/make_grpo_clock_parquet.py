#!/usr/bin/env python3
"""Generate a GRPO-ready clock dataset with embedded images as parquet."""

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterator

from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as ImageData
from PIL import Image
from sklearn.model_selection import train_test_split


DEFAULT_PROMPT = (
    "Read the exact time shown on this analog clock. "
    "Answer in HH:MM:SS format."
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build GRPO clock parquet from Blender OOD benchmark")
    p.add_argument("--benchmark_out_dir", default="data/bench/clock_ood_grpo_2k")
    p.add_argument("--output_dir", default="data/rl/clock_ood_grpo_2k")
    p.add_argument("--clean_n", type=int, default=1000)
    p.add_argument("--noisy_n", type=int, default=1000)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--blender_bin", default="blender")
    p.add_argument("--fast_render", action="store_true")
    p.add_argument("--style_bank_dir", default=None)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--push_to_hub", default=None, help="Push to HuggingFace Hub (e.g., username/dataset-name)")
    return p.parse_args()


def _load_jsonl(path: str) -> list[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _run(cmd: list[str]) -> None:
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# def generate_data(benchmark_root: str, prompt_text: str, sample_ids: list[str]) -> Iterator[Dict[str, Any]]:
#     """Generator function to yield samples one by one."""
#     # Load all samples
#     all_samples = {}
#     for split in ("clean", "noisy"):
#         jsonl_path = os.path.join(benchmark_root, split, "samples.jsonl")
#         for sample in _load_jsonl(jsonl_path):
#             sample_id = f"{split}_{sample['id']}"
#             all_samples[sample_id] = (split, sample)
    
#     # Yield samples in the order specified by sample_ids
#     for sample_id in sample_ids:
#         if sample_id not in all_samples:
#             continue
        
#         split, sample = all_samples[sample_id]
#         label = sample.get("label", {})
#         gt = label.get("time_hhmmss")
        
#         if not gt:
#             raise ValueError(
#                 f"Sample {sample.get('id')} does not contain time_hhmmss. "
#                 "Generate with --time_mode hms and --force_hand_config 3 or 4."
#             )
        
#         image_rel = sample["image"]
#         image_abs = os.path.abspath(os.path.join(benchmark_root, split, image_rel))
        
#         # Load image
#         image = Image.open(image_abs).convert("RGB")
        
#         yield {
#             "images": [image],
#             "problem": prompt_text,
#             "answer": gt,
#         }

def generate_data(benchmark_root: str, prompt_text: str, sample_ids: list[str]) -> Iterator[Dict[str, Any]]:
    """Generator function to yield samples one by one."""
    # Load all samples
    all_samples = {}
    for split in ("clean", "noisy"):
        jsonl_path = os.path.join(benchmark_root, split, "samples.jsonl")
        for sample in _load_jsonl(jsonl_path):
            sample_id = f"{split}_{sample['id']}"
            all_samples[sample_id] = (split, sample)
    
    # Yield samples in the order specified by sample_ids
    for sample_id in sample_ids:
        if sample_id not in all_samples:
            continue
        
        split, sample = all_samples[sample_id]
        label = sample.get("label", {})
        gt = label.get("time_hhmmss")
        
        if not gt:
            raise ValueError(
                f"Sample {sample.get('id')} does not contain time_hhmmss. "
                "Generate with --time_mode hms and --force_hand_config 3 or 4."
            )
        
        image_rel = sample["image"]
        image_abs = os.path.abspath(os.path.join(benchmark_root, split, image_rel))
        
        # Load image
        image = Image.open(image_abs).convert("RGB")
        
        # 保留 <image> 占位符，使用 images 列表
        yield {
            "images": [image],                    # 复数，列表格式
            "problem": f"<image>\n{prompt_text}", # 添加 <image> 占位符
            "answer": gt,
        }


def _get_all_sample_ids(benchmark_root: str) -> list[str]:
    """Get all sample IDs from the benchmark."""
    sample_ids = []
    for split in ("clean", "noisy"):
        jsonl_path = os.path.join(benchmark_root, split, "samples.jsonl")
        for sample in _load_jsonl(jsonl_path):
            sample_id = f"{split}_{sample['id']}"
            sample_ids.append(sample_id)
    return sample_ids


def main() -> None:
    args = _parse_args()
    benchmark_out_dir = os.path.abspath(args.benchmark_out_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Generate benchmark if needed
    if not args.skip_generation:
        make_benchmark_script = os.path.join(os.path.dirname(__file__), "make_ood_blender_benchmark.py")
        cmd = [
            sys.executable,
            make_benchmark_script,
            "--out_dir",
            benchmark_out_dir,
            "--clean_n",
            str(args.clean_n),
            "--noisy_n",
            str(args.noisy_n),
            "--resolution",
            str(args.resolution),
            "--seed",
            str(args.seed),
            "--blender_bin",
            args.blender_bin,
            "--time_mode",
            "hms",
            "--force_hand_config",
            "3",
            "--max_seconds",
            "59",
        ]
        if args.fast_render:
            cmd.append("--fast_render")
        if args.style_bank_dir:
            cmd.extend(["--style_bank_dir", args.style_bank_dir])
        _run(cmd)

    print("\nLoading samples and splitting into train/validation...")
    
    # Get all sample IDs and split them
    all_sample_ids = _get_all_sample_ids(benchmark_out_dir)
    train_ids, val_ids = train_test_split(
        all_sample_ids,
        train_size=0.9,
        random_state=args.seed,
        shuffle=True
    )
    
    print(f"Total samples: {len(all_sample_ids)}")
    print(f"Train samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    
    # Create datasets using generators
    print("\nCreating train dataset...")
    trainset = Dataset.from_generator(
        generate_data,
        gen_kwargs={
            "benchmark_root": benchmark_out_dir,
            "prompt_text": args.prompt,
            "sample_ids": train_ids,
        }
    )
    
    print("Creating validation dataset...")
    valset = Dataset.from_generator(
        generate_data,
        gen_kwargs={
            "benchmark_root": benchmark_out_dir,
            "prompt_text": args.prompt,
            "sample_ids": val_ids,
        }
    )
    
    # Cast images column to proper type
    print("\nCasting image column...")
    dataset = DatasetDict({
        "train": trainset,
        "validation": valset
    }).cast_column("images", Sequence(ImageData()))
    
    # Save to parquet
    print(f"\nSaving datasets to {output_dir}...")
    dataset["train"].to_parquet(os.path.join(output_dir, "train.parquet"))
    dataset["validation"].to_parquet(os.path.join(output_dir, "validation.parquet"))
    
    # Save metadata
    meta = {
        "num_train": len(dataset["train"]),
        "num_validation": len(dataset["validation"]),
        "total": len(dataset["train"]) + len(dataset["validation"]),
        "features": list(dataset["train"].features.keys()),
        "prompt": args.prompt,
        "benchmark_out_dir": benchmark_out_dir,
        "output_dir": output_dir,
    }
    
    meta_path = os.path.join(output_dir, "dataset_info.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print("Dataset created successfully!")
    print("="*60)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\nDataset info:")
    print(dataset)
    
    # Push to hub if requested
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}")
        dataset.push_to_hub(args.push_to_hub)
        print("✅ Successfully pushed to Hub!")


if __name__ == "__main__":
    main()
