"""Evaluate pairwise delta predictions."""

import argparse
import json
import math
from typing import Dict

from parse_time import parse_delta_minutes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pairwise delta predictions")
    parser.add_argument("--gt_jsonl", required=True)
    parser.add_argument("--pred_jsonl", required=True)
    return parser.parse_args()


def _load_preds(path: str) -> Dict[str, str]:
    preds: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            preds[row["id"]] = row.get("output", "")
    return preds


def main() -> None:
    args = _parse_args()
    preds = _load_preds(args.pred_jsonl)

    total = 0
    exact = 0
    abs_err_sum = 0.0
    parsed = 0

    with open(args.gt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            gt = row["label"]["delta_minutes"]
            pred_text = preds.get(row["id"], "")
            pred = parse_delta_minutes(pred_text)
            if pred is None:
                continue
            parsed += 1
            err = abs(pred - gt)
            abs_err_sum += err
            if err == 0:
                exact += 1

    mae = abs_err_sum / parsed if parsed else math.nan
    print(f"total={total}")
    print(f"parsed={parsed}")
    print(f"exact_delta={exact} ({exact / total:.3f})")
    print(f"mae_delta={mae:.3f}")


if __name__ == "__main__":
    main()
