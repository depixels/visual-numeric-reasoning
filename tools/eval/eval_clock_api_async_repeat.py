#!/usr/bin/env python3
"""Run eval_clock_api_async.py multiple times and aggregate run-level and sample-level stability."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

from eval_common import compute_metrics, load_jsonl, write_csv, write_json, write_jsonl


METRIC_KEYS = [
    "parsed_rate",
    "hour_acc",
    "minute_acc",
    "second_acc",
    "exact_acc",
    "tol1_acc",
    "tol5_acc",
    "mae",
    "median_abs_error_minutes",
    "avg_latency_sec",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeat eval_clock_api_async.py and aggregate metrics")
    p.add_argument("--num_runs", type=int, default=5)
    p.add_argument("--output_dir", required=True)

    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--images_root", default=None)
    p.add_argument(
        "--provider",
        choices=["vllm_qwen", "gemini_3_pro", "azure_gpt", "qwen_dashscope"],
        required=True,
    )
    p.add_argument("--model", default="/data/hyz/workspace/hf/Qwen3-VL-4B-Instruct")
    p.add_argument("--base_url", default=None)
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
    # ── 新增并发参数 ──
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--rpm_limit", type=int, default=0)
    return p.parse_args()


def _append_opt(cmd: List[str], key: str, val: Any) -> None:
    if val is None:
        return
    cmd.extend([key, str(val)])


def _run_one(args: argparse.Namespace, run_id: int) -> Dict[str, Any]:
    run_dir = os.path.join(args.output_dir, f"run_{run_id:02d}")
    os.makedirs(run_dir, exist_ok=True)

    # ← 改为调用 eval_clock_api_async.py
    script = os.path.join(os.path.dirname(__file__), "eval_clock_api_async.py")
    cmd: List[str] = [
        sys.executable,
        script,
        "--gt_jsonl", args.gt_jsonl,
        "--provider", args.provider,
        "--model", args.model,
        "--api_key", args.api_key,
        "--timeout", str(args.timeout),
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--max_retries", str(args.max_retries),
        "--retry_sleep", str(args.retry_sleep),
        "--start_index", str(args.start_index),
        "--save_every", str(args.save_every),
        "--concurrency", str(args.concurrency),   # ← 新增
        "--rpm_limit", str(args.rpm_limit),        # ← 新增
        "--output_dir", run_dir,
    ]
    _append_opt(cmd, "--base_url", args.base_url)
    _append_opt(cmd, "--images_root", args.images_root)
    _append_opt(cmd, "--developer_prompt", args.developer_prompt)
    _append_opt(cmd, "--system_prompt", args.system_prompt)
    _append_opt(cmd, "--user_prompt", args.user_prompt)
    _append_opt(cmd, "--limit", args.limit)

    print(f"\n[run {run_id}] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    metrics = load_jsonl(os.path.join(run_dir, "per_sample_results.jsonl"))
    with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as f:
        metric_payload = json.load(f)
    run_metrics = {
        **metric_payload,
        "run_id": run_id,
        "run_dir": run_dir,
        "per_sample_results_jsonl": os.path.join(run_dir, "per_sample_results.jsonl"),
        "num_rows_loaded": len(metrics),
    }
    return run_metrics


def _aggregate_metrics(run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    return stats


def _majority_value(values: List[Any]) -> Any:
    counts = Counter(values)
    best = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    return best[0][0]


def _aggregate_per_sample(output_dir: str, run_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run_info in run_metrics:
        for row in load_jsonl(run_info["per_sample_results_jsonl"]):
            per_id[row["id"]].append({**row, "run_id": run_info["run_id"]})

    majority_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []

    for sample_id, sample_runs in sorted(per_id.items()):
        base_row = dict(sample_runs[0])
        preds = [row.get("pred_time_minutes") for row in sample_runs if row.get("parsed_ok")]
        pred_strings = [row.get("pred_time_hhmm") for row in sample_runs if row.get("parsed_ok")]
        parsed_count = sum(1 for row in sample_runs if row.get("parsed_ok"))
        vote_minutes = _majority_value(preds) if preds else None
        vote_string = _majority_value(pred_strings) if pred_strings else None
        agreement = safe_agreement(pred_strings)

        vote_row = dict(base_row)
        vote_row["pred_time_minutes"] = vote_minutes
        vote_row["pred_time_hhmm"] = vote_string
        vote_row["parsed_ok"] = vote_minutes is not None
        if vote_minutes is None:
            vote_row["is_exact"] = False
            vote_row["tol_1"] = False
            vote_row["tol_5"] = False
            vote_row["abs_err_minutes"] = None
            vote_row["hour_correct"] = None
            vote_row["minute_correct"] = None
            vote_row["second_correct"] = None
        else:
            matching = next(
                (
                    row
                    for row in sample_runs
                    if row.get("pred_time_minutes") == vote_minutes and row.get("parsed_ok")
                ),
                None,
            )
            if matching is not None:
                for key in (
                    "is_exact",
                    "tol_1",
                    "tol_5",
                    "abs_err_minutes",
                    "hour_correct",
                    "minute_correct",
                    "second_correct",
                    "pred_time_hhmmss",
                    "pred_time_seconds_total",
                ):
                    vote_row[key] = matching.get(key)
        vote_row["vote_support"] = Counter(preds).get(vote_minutes, 0) if vote_minutes is not None else 0
        vote_row["num_runs"] = len(sample_runs)
        majority_rows.append(vote_row)

        any_exact = any(row.get("is_exact") is True for row in sample_runs)
        any_tol1 = any(row.get("tol_1") is True for row in sample_runs)
        any_tol5 = any(row.get("tol_5") is True for row in sample_runs)
        stability_rows.append(
            {
                "id": sample_id,
                "split": base_row.get("split"),
                "image": base_row.get("image"),
                "num_runs": len(sample_runs),
                "parsed_count": parsed_count,
                "parsed_rate": parsed_count / len(sample_runs),
                "num_unique_predictions": len(set(pred_strings)),
                "agreement_rate": agreement,
                "vote_pred_time_hhmm": vote_string,
                "vote_pred_time_minutes": vote_minutes,
                "vote_support": Counter(preds).get(vote_minutes, 0) if vote_minutes is not None else 0,
                "any_exact": any_exact,
                "any_tol1": any_tol1,
                "any_tol5": any_tol5,
            }
        )

    majority_jsonl = os.path.join(output_dir, "majority_vote_results.jsonl")
    majority_csv = os.path.join(output_dir, "majority_vote_results.csv")
    stability_jsonl = os.path.join(output_dir, "per_sample_stability.jsonl")
    stability_csv = os.path.join(output_dir, "per_sample_stability.csv")
    write_jsonl(majority_jsonl, majority_rows)
    write_csv(majority_csv, majority_rows)
    write_jsonl(stability_jsonl, stability_rows)
    write_csv(stability_csv, stability_rows)

    majority_metrics = compute_metrics(majority_rows)
    oracle_metrics = {
        "total": len(stability_rows),
        "exact_acc": statistics.fmean([1.0 if row["any_exact"] else 0.0 for row in stability_rows])
        if stability_rows
        else 0.0,
        "tol1_acc": statistics.fmean([1.0 if row["any_tol1"] else 0.0 for row in stability_rows])
        if stability_rows
        else 0.0,
        "tol5_acc": statistics.fmean([1.0 if row["any_tol5"] else 0.0 for row in stability_rows])
        if stability_rows
        else 0.0,
    }
    stability_summary = {
        "mean_agreement_rate": statistics.fmean([row["agreement_rate"] for row in stability_rows])
        if stability_rows
        else 0.0,
        "mean_unique_predictions": statistics.fmean([row["num_unique_predictions"] for row in stability_rows])
        if stability_rows
        else 0.0,
        "mean_parsed_rate": statistics.fmean([row["parsed_rate"] for row in stability_rows])
        if stability_rows
        else 0.0,
    }
    write_json(os.path.join(output_dir, "majority_vote_metrics.json"), majority_metrics)
    write_json(os.path.join(output_dir, "oracle_best_of_n_metrics.json"), oracle_metrics)

    return {
        "majority_vote_metrics": majority_metrics,
        "oracle_best_of_n_metrics": oracle_metrics,
        "stability_summary": stability_summary,
        "majority_vote_results_jsonl": majority_jsonl,
        "per_sample_stability_jsonl": stability_jsonl,
    }


def safe_agreement(pred_strings: List[Any]) -> float:
    if not pred_strings:
        return 0.0
    best = Counter(pred_strings).most_common(1)[0][1]
    return best / len(pred_strings)


def main() -> None:
    args = _parse_args()
    if args.num_runs <= 0:
        raise ValueError("--num_runs must be > 0")
    os.makedirs(args.output_dir, exist_ok=True)

    run_metrics: List[Dict[str, Any]] = []
    for i in range(1, args.num_runs + 1):
        run_metrics.append(_run_one(args, i))

    summary: Dict[str, Any] = {
        "num_runs": len(run_metrics),
        "runs": run_metrics,
        "aggregate": _aggregate_metrics(run_metrics),
        "provider": args.provider,
        "model": args.model,
        "gt_jsonl": args.gt_jsonl,
        "images_root": args.images_root,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "rpm_limit": args.rpm_limit,
    }
    summary.update(_aggregate_per_sample(args.output_dir, run_metrics))

    summary_path = os.path.join(args.output_dir, "summary.json")
    write_json(summary_path, summary)

    print("\nAll runs done.")
    print(f"Summary saved: {summary_path}")
    for key, stat in summary.get("aggregate", {}).items():
        print(f"- {key}: mean={stat['mean']:.6f}, std={stat['std']:.6f}")
    print(f"- majority exact_acc: {summary['majority_vote_metrics'].get('exact_acc', 0.0):.6f}")
    print(f"- oracle best-of-n exact_acc: {summary['oracle_best_of_n_metrics'].get('exact_acc', 0.0):.6f}")


if __name__ == "__main__":
    main()