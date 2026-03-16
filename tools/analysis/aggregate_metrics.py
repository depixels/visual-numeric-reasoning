#!/usr/bin/env python3
"""Aggregate joined per-sample metrics by arbitrary metadata fields."""

import argparse
from typing import Any, Dict, List

from common import bucket_order_for_field, group_rows, load_rows, write_csv


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate joined metrics")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--group_by", nargs="+", required=True)
    p.add_argument("--split", default=None)
    return p.parse_args()


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    return {
        "n": total,
        "exact_acc": sum(1 for row in rows if row.get("is_exact") is True) / total if total else 0.0,
        "tol1_acc": sum(1 for row in rows if row.get("tol_1") is True) / total if total else 0.0,
        "tol5_acc": sum(1 for row in rows if row.get("tol_5") is True) / total if total else 0.0,
        "hour_acc": sum(1 for row in rows if row.get("hour_correct") is True) / total if total else 0.0,
        "minute_acc": sum(1 for row in rows if row.get("minute_correct") is True) / total if total else 0.0,
        "mae": sum(float(row["abs_err_minutes"]) for row in rows if isinstance(row.get("abs_err_minutes"), (int, float)))
        / max(1, sum(1 for row in rows if isinstance(row.get("abs_err_minutes"), (int, float)))),
    }


def main() -> None:
    args = _parse_args()
    rows = load_rows(args.input_jsonl)
    if args.split:
        rows = [row for row in rows if row.get("split") == args.split]

    grouped = group_rows(rows, args.group_by)
    out_rows: List[Dict[str, Any]] = []
    for key, group in grouped.items():
        agg = _aggregate_rows(group)
        out_row = {field: value for field, value in zip(args.group_by, key)}
        out_row.update(agg)
        out_rows.append(out_row)

    if len(args.group_by) == 1:
        order = bucket_order_for_field(args.group_by[0])
        if order:
            rank = {name: idx for idx, name in enumerate(order)}
            out_rows.sort(key=lambda row: (rank.get(str(row.get(args.group_by[0])), 10**6), str(row.get(args.group_by[0]))))
        else:
            out_rows.sort(key=lambda row: str(row.get(args.group_by[0])))
    else:
        out_rows.sort(key=lambda row: tuple(str(row.get(field)) for field in args.group_by))

    write_csv(args.output_csv, out_rows)
    print(f"groups={len(out_rows)}")
    print(f"output={args.output_csv}")


if __name__ == "__main__":
    main()
