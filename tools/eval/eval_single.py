"""Evaluate single-image clock readout predictions with unified per-sample outputs."""

import argparse
import json
import os
from typing import Any, Dict

from eval_common import (
    compute_metrics,
    extract_gt_label,
    extract_image_relpath,
    finalize_prediction_row,
    infer_split,
    load_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)
from parse_time import parse_hhmm, parse_hhmmss


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-image predictions")
    parser.add_argument("--gt_jsonl", required=True)
    parser.add_argument("--pred_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pred_json", default="predictions.json")
    parser.add_argument("--pred_results_jsonl", default="per_sample_results.jsonl")
    parser.add_argument("--pred_results_csv", default="per_sample_results.csv")
    parser.add_argument("--metrics_json", default="metrics.json")
    return parser.parse_args()


def _load_preds(path: str) -> Dict[str, Dict[str, Any]]:
    preds: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        preds[row["id"]] = row
    return preds


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    preds = _load_preds(args.pred_jsonl)
    out_rows = []

    for row in load_jsonl(args.gt_jsonl):
        sample_id = row["id"]
        pred_row = preds.get(sample_id, {})
        raw_output = str(pred_row.get("raw_output") or pred_row.get("output") or "")
        pred_seconds_total = parse_hhmmss(raw_output)
        pred_minutes = pred_seconds_total // 60 if pred_seconds_total is not None else parse_hhmm(raw_output)
        out_rows.append(
            finalize_prediction_row(
                sample_id=sample_id,
                split=infer_split(row, args.gt_jsonl),
                image_rel=extract_image_relpath(row),
                gt_label=extract_gt_label(row),
                raw_output=raw_output,
                pred_minutes=pred_minutes,
                pred_seconds_total=pred_seconds_total,
                latency_sec=None,
                error=None,
            )
        )

    metrics = compute_metrics(out_rows)
    metrics["gt_jsonl"] = args.gt_jsonl
    metrics["input_pred_jsonl"] = args.pred_jsonl

    pred_json = os.path.join(args.output_dir, args.pred_json)
    pred_results_jsonl = os.path.join(args.output_dir, args.pred_results_jsonl)
    pred_results_csv = os.path.join(args.output_dir, args.pred_results_csv)
    metrics_json = os.path.join(args.output_dir, args.metrics_json)

    write_json(pred_json, out_rows)
    write_jsonl(pred_results_jsonl, out_rows)
    write_csv(pred_results_csv, out_rows)
    write_json(metrics_json, metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
