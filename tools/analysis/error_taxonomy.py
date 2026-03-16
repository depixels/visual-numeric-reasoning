#!/usr/bin/env python3
"""Summarize prediction errors into a simple taxonomy for paper analysis."""

import argparse
from collections import Counter
from typing import Any, Dict, List

from common import load_rows, write_csv, write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build error taxonomy from joined jsonl")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_csv", default=None)
    p.add_argument("--output_jsonl", default=None)
    return p.parse_args()


def _classify(row: Dict[str, Any]) -> Dict[str, Any]:
    parsed_ok = row.get("parsed_ok") is True
    hour_ok = row.get("hour_correct") is True
    minute_ok = row.get("minute_correct") is True
    second_flag = isinstance(row.get("second_correct"), bool)
    second_ok = row.get("second_correct") is True
    err = row.get("abs_err_minutes")
    err = float(err) if isinstance(err, (int, float)) else None

    flags = {
        "unparsed": not parsed_ok,
        "hour_only_wrong": parsed_ok and (not hour_ok) and minute_ok,
        "minute_only_wrong": parsed_ok and hour_ok and (not minute_ok),
        "both_wrong": parsed_ok and (not hour_ok) and (not minute_ok),
        "near_miss_1to5": parsed_ok and err is not None and 1 < err <= 5,
        "large_error_5plus": parsed_ok and err is not None and err > 5,
        "second_only_wrong": parsed_ok and second_flag and hour_ok and minute_ok and (not second_ok),
    }

    if flags["unparsed"]:
        category = "unparsed"
    elif flags["second_only_wrong"]:
        category = "second_only_wrong"
    elif flags["hour_only_wrong"]:
        category = "hour_only_wrong"
    elif flags["minute_only_wrong"]:
        category = "minute_only_wrong"
    elif flags["both_wrong"]:
        category = "both_wrong"
    elif flags["near_miss_1to5"]:
        category = "near_miss_1to5"
    elif flags["large_error_5plus"]:
        category = "large_error_5plus"
    else:
        category = "correct"

    return {"error_category": category, **flags}


def main() -> None:
    args = _parse_args()
    rows = load_rows(args.input_jsonl)
    per_sample: List[Dict[str, Any]] = []
    counter = Counter()

    for row in rows:
        info = _classify(row)
        counter[info["error_category"]] += 1
        per_sample.append({**row, **info})

    summary = {
        "total": len(rows),
        "counts": dict(counter),
        "rates": {key: value / len(rows) for key, value in counter.items()} if rows else {},
    }
    write_json(args.output_json, summary)

    if args.output_jsonl:
        write_jsonl(args.output_jsonl, per_sample)
    if args.output_csv:
        table = [{"error_category": key, "count": value, "rate": value / len(rows)} for key, value in counter.items()]
        write_csv(args.output_csv, table)

    print(f"total={len(rows)}")
    print(f"output_json={args.output_json}")


if __name__ == "__main__":
    main()
