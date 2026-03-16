#!/usr/bin/env python3
"""Build main tables or ablation tables from result directories."""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from common import write_csv, write_json


TABLE_METRICS = ["exact_acc", "tol1_acc", "tol5_acc", "hour_acc", "minute_acc", "mae", "parsed_rate"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ablation/main result table")
    p.add_argument("--setting", action="append", required=True, help="name=/path/to/result_dir")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_json", default=None)
    return p.parse_args()


def _load_metrics(path: str) -> Optional[Dict[str, Any]]:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    candidate = os.path.join(path, "metrics.json")
    if os.path.exists(candidate):
        with open(candidate, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_split_metrics(result_dir: str, split: str) -> Optional[Dict[str, Any]]:
    candidates = [
        os.path.join(result_dir, split, "metrics.json"),
        os.path.join(result_dir, split, "majority_vote_metrics.json"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def main() -> None:
    args = _parse_args()
    rows: List[Dict[str, Any]] = []
    raw: Dict[str, Any] = {}

    for item in args.setting:
        if "=" not in item:
            raise ValueError(f"Invalid --setting: {item}")
        name, result_dir = item.split("=", 1)
        result_dir = os.path.abspath(result_dir)
        row: Dict[str, Any] = {"setting": name}
        raw[name] = {"path": result_dir}

        direct_metrics = _load_metrics(result_dir)
        if direct_metrics:
            for metric in TABLE_METRICS:
                row[metric] = direct_metrics.get(metric)
            raw[name]["default"] = direct_metrics

        for split in ("clean", "noisy"):
            split_metrics = _load_split_metrics(result_dir, split)
            if split_metrics:
                raw[name][split] = split_metrics
                for metric in TABLE_METRICS:
                    row[f"{split}_{metric}"] = split_metrics.get(metric)

        rows.append(row)

    write_csv(args.output_csv, rows)
    if args.output_json:
        write_json(args.output_json, raw)
    print(f"settings={len(rows)}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
