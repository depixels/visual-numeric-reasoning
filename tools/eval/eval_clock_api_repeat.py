#!/usr/bin/env python3
"""Run eval_clock_api.py multiple times and aggregate metrics.

Example:
python tools/eval/eval_clock_api_repeat.py \
  --num_runs 5 \
  --output_dir /data/hyz/workspace/rege_bench/runs_v2/eval_qwen3_noisy_repeat5 \
  --provider vllm_qwen \
  --base_url http://127.0.0.1:8001/v1 \
  --api_key EMPTY \
  --model Qwen3-VL-4B-Instruct \
  --gt_jsonl /data/hyz/workspace/rege_bench/data/benchmark_ood_blender_ood_v1/noisy/samples.jsonl \
  --images_root /data/hyz/workspace/rege_bench/data/benchmark_ood_blender_ood_v1/noisy
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from typing import Any, Dict, List


METRIC_KEYS = [
    "parsed_rate",
    "hour_acc",
    "minute_acc",
    "exact_hhmm_acc",
    "minute_given_hour_acc",
    "tol_1min_acc",
    "tol_5min_acc",
    "mae_minutes",
    "median_abs_error_minutes",
    "avg_latency_sec",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeat eval_clock_api.py and average metrics")
    p.add_argument("--num_runs", type=int, default=5)
    p.add_argument("--output_dir", required=True)

    # Forwarded args for eval_clock_api.py
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--images_root", default=None)
    p.add_argument(
        "--provider",
        choices=["vllm_qwen", "gemini_3_pro", "azure_gpt", "qwen_dashscope"],
        required=True,
    )
    p.add_argument("--model", default="/data/hyz/workspace/hf/Qwen3-VL-4B-Instruct")
    p.add_argument("--base_url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--api_key", required=True)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--developer_prompt", default=None)
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--user_prompt", default=None)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_sleep", type=float, default=0.8)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_every", type=int, default=20)
    return p.parse_args()


def _append_opt(cmd: List[str], key: str, val: Any) -> None:
    if val is None:
        return
    cmd.extend([key, str(val)])


def _run_one(args: argparse.Namespace, run_id: int) -> Dict[str, Any]:
    run_dir = os.path.join(args.output_dir, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    script = os.path.join(os.path.dirname(__file__), "eval_clock_api.py")
    cmd: List[str] = [
        sys.executable,
        script,
        "--gt_jsonl",
        args.gt_jsonl,
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--base_url",
        args.base_url,
        "--api_key",
        args.api_key,
        "--timeout",
        str(args.timeout),
        "--max_tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--max_retries",
        str(args.max_retries),
        "--retry_sleep",
        str(args.retry_sleep),
        "--start_index",
        str(args.start_index),
        "--save_every",
        str(args.save_every),
        "--output_dir",
        run_dir,
    ]
    _append_opt(cmd, "--images_root", args.images_root)
    _append_opt(cmd, "--developer_prompt", args.developer_prompt)
    _append_opt(cmd, "--system_prompt", args.system_prompt)
    _append_opt(cmd, "--user_prompt", args.user_prompt)
    _append_opt(cmd, "--limit", args.limit)

    print(f"\n[run {run_id}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics["run_id"] = run_id
    metrics["run_dir"] = run_dir
    return metrics


def _aggregate(run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "num_runs": len(run_metrics),
        "runs": run_metrics,
    }
    stats: Dict[str, Any] = {}

    for key in METRIC_KEYS:
        vals = [m.get(key) for m in run_metrics if isinstance(m.get(key), (int, float))]
        if not vals:
            continue
        stats[key] = {
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }
    out["aggregate"] = stats
    return out


def main() -> None:
    args = _parse_args()
    if args.num_runs <= 0:
        raise ValueError("--num_runs must be > 0")
    os.makedirs(args.output_dir, exist_ok=True)

    run_metrics: List[Dict[str, Any]] = []
    for i in range(1, args.num_runs + 1):
        run_metrics.append(_run_one(args, i))

    summary = _aggregate(run_metrics)
    summary["provider"] = args.provider
    summary["model"] = args.model
    summary["gt_jsonl"] = args.gt_jsonl
    summary["images_root"] = args.images_root
    summary["temperature"] = args.temperature

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nAll runs done.")
    print(f"Summary saved: {summary_path}")
    for key, stat in summary.get("aggregate", {}).items():
        mean = stat["mean"]
        std = stat["std"]
        print(f"- {key}: mean={mean:.6f}, std={std:.6f}")


if __name__ == "__main__":
    main()
