"""Evaluate single-image clock readout predictions."""

import argparse
import json
import math
from typing import Dict, Optional

from parse_time import parse_hhmm


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-image predictions")
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
    tol_1 = 0
    tol_5 = 0
    abs_err_sum = 0.0
    parsed = 0

    with open(args.gt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            gt = row["label"]["time_minutes"]
            pred_text = preds.get(row["id"], "")
            pred = parse_hhmm(pred_text)
            if pred is None:
                continue
            parsed += 1
            err = abs(pred - gt)
            abs_err_sum += err
            if err == 0:
                exact += 1
            if err <= 1:
                tol_1 += 1
            if err <= 5:
                tol_5 += 1

    mae = abs_err_sum / parsed if parsed else math.nan
    print(f"total={total}")
    print(f"parsed={parsed}")
    print(f"exact={exact} ({exact / total:.3f})")
    print(f"tol_1min={tol_1} ({tol_1 / total:.3f})")
    print(f"tol_5min={tol_5} ({tol_5 / total:.3f})")
    print(f"mae={mae:.3f}")


if __name__ == "__main__":
    main()
