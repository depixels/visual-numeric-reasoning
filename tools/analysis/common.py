"""Common helpers for analysis and plotting scripts."""

import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


TILT_BUCKET_ORDER = ["Front", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"]
SPECULAR_BUCKET_ORDER = ["0.0", "0.0-0.1", "0.1-0.3", "0.3-0.6", "0.6+"]
BLUR_BUCKET_ORDER = ["0.0", "0.0-0.05", "0.05-0.15", "0.15-0.30", "0.30+"]


def load_rows(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return payload["rows"]
    raise ValueError(f"Unsupported input format: {path}")


def write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(val, ensure_ascii=False)
                    if isinstance(val, (list, dict))
                    else val
                    for key, val in row.items()
                }
            )


def get_nested(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for token in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(token)
        if cur is None:
            return default
    return cur


def abs_yaw_bucket(yaw: Optional[float]) -> str:
    if yaw is None:
        return "unknown"
    value = abs(float(yaw))
    if value < 10:
        return "Front"
    if value < 20:
        return "10-20"
    if value < 30:
        return "20-30"
    if value < 40:
        return "30-40"
    if value < 50:
        return "40-50"
    if value < 60:
        return "50-60"
    if value < 70:
        return "60-70"
    return "70+"


def specular_bucket(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    if value <= 0.0:
        return "0.0"
    if value < 0.1:
        return "0.0-0.1"
    if value < 0.3:
        return "0.1-0.3"
    if value < 0.6:
        return "0.3-0.6"
    return "0.6+"


def blur_bucket(motion_blur: Optional[float], defocus: Optional[float]) -> str:
    blur_level = max(float(motion_blur or 0.0), float(defocus or 0.0))
    if blur_level <= 0.0:
        return "0.0"
    if blur_level < 0.05:
        return "0.0-0.05"
    if blur_level < 0.15:
        return "0.05-0.15"
    if blur_level < 0.30:
        return "0.15-0.30"
    return "0.30+"


def bucket_order_for_field(field: str) -> Optional[List[str]]:
    mapping = {
        "tilt_bucket": TILT_BUCKET_ORDER,
        "specular_bucket": SPECULAR_BUCKET_ORDER,
        "blur_bucket": BLUR_BUCKET_ORDER,
    }
    return mapping.get(field)


def metric_mean(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float, bool))]
    return sum(vals) / len(rows) if rows else 0.0


def group_rows(rows: List[Dict[str, Any]], fields: List[str]) -> Dict[tuple, List[Dict[str, Any]]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        buckets[key].append(row)
    return buckets


def maybe_filter_split(rows: List[Dict[str, Any]], split: Optional[str]) -> List[Dict[str, Any]]:
    if not split:
        return rows
    return [row for row in rows if row.get("split") == split]


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)
