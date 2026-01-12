"""Validate rege-bench annotations and report dataset stats."""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Tuple

TIME_RE = re.compile(r"^(?:0?[1-9]|1[0-2]):[0-5]\d$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate rege-bench annotations")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--images_root", default=None)
    parser.add_argument("--type", choices=["sample", "pair"], default=None)
    return parser.parse_args()


def _bucket(value: float) -> str:
    if value < 0.33:
        return "low"
    if value < 0.66:
        return "mid"
    return "high"


def _check_time_hhmm(time_hhmm: str, time_minutes: int, errors: list) -> None:
    if not TIME_RE.match(time_hhmm):
        errors.append(f"invalid time_hhmm {time_hhmm}")
        return
    hh, mm = map(int, time_hhmm.split(":"))
    if hh == 12:
        hh = 0
    if hh * 60 + mm != time_minutes:
        errors.append(f"time_hhmm {time_hhmm} does not match time_minutes {time_minutes}")


def _validate_sample(row: Dict[str, Any], images_root: str, errors: list) -> Tuple[str, Dict[str, str]]:
    image_path = os.path.join(images_root, row.get("image", ""))
    if not os.path.exists(image_path):
        errors.append(f"missing image {image_path}")
    label = row.get("label", {})
    _check_time_hhmm(label.get("time_hhmm", ""), label.get("time_minutes", -1), errors)
    if not (0 <= label.get("time_minutes", -1) <= 719):
        errors.append(f"time_minutes out of range {label.get('time_minutes')}")
    meta = row.get("meta", {})
    style_id = meta.get("style_id", "unknown")
    degradation = meta.get("degradation", {})
    buckets = {
        name: _bucket(float(degradation.get(name, 0.0)))
        for name in ["specular", "motion_blur", "defocus", "noise", "occlusion"]
    }
    return style_id, buckets


def _validate_pair(row: Dict[str, Any], images_root: str, errors: list) -> Tuple[str, str]:
    image_a = os.path.join(images_root, row.get("image_a", ""))
    image_b = os.path.join(images_root, row.get("image_b", ""))
    if not os.path.exists(image_a):
        errors.append(f"missing image {image_a}")
    if not os.path.exists(image_b):
        errors.append(f"missing image {image_b}")
    label = row.get("label", {})
    time_a_hhmm = label.get("time_a_hhmm", "")
    time_b_hhmm = label.get("time_b_hhmm", "")
    if not TIME_RE.match(time_a_hhmm):
        errors.append(f"invalid time_a_hhmm {time_a_hhmm}")
    if not TIME_RE.match(time_b_hhmm):
        errors.append(f"invalid time_b_hhmm {time_b_hhmm}")
    delta = label.get("delta_minutes")
    if not isinstance(delta, int) or delta < -719 or delta > 719:
        errors.append(f"delta_minutes out of range {delta}")
    meta = row.get("meta", {})
    return meta.get("style_id_a", "unknown"), meta.get("style_id_b", "unknown")


def _infer_type(row: Dict[str, Any]) -> str:
    if "image_a" in row:
        return "pair"
    return "sample"


def main() -> None:
    args = _parse_args()
    errors: list = []
    style_counts = Counter()
    style_pair_counts = Counter()
    degradation_counts = defaultdict(Counter)

    images_root = args.images_root or os.path.dirname(args.jsonl)
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_type = args.type or _infer_type(row)
            before = len(errors)
            if row_type == "sample":
                style_id, buckets = _validate_sample(row, images_root, errors)
                style_counts[style_id] += 1
                for name, bucket in buckets.items():
                    degradation_counts[name][bucket] += 1
            else:
                style_a, style_b = _validate_pair(row, images_root, errors)
                style_pair_counts[(style_a, style_b)] += 1
            for i in range(before, len(errors)):
                errors[i] = f"line {line_num}: {errors[i]}"

    if errors:
        print("Validation errors:")
        for err in errors:
            print(f"- {err}")
    else:
        print("No validation errors.")

    if style_counts:
        print("\nStyle counts:")
        for style_id, count in style_counts.most_common():
            print(f"- {style_id}: {count}")

    if style_pair_counts:
        print("\nStyle pair counts:")
        for (style_a, style_b), count in style_pair_counts.most_common():
            print(f"- {style_a} / {style_b}: {count}")

    if degradation_counts:
        print("\nDegradation buckets:")
        for name, buckets in degradation_counts.items():
            bucket_str = ", ".join(f"{k}={v}" for k, v in buckets.items())
            print(f"- {name}: {bucket_str}")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
