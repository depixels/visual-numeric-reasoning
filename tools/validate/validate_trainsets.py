"""Validate training datasets for Qwen3-VL stages."""

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trainsets")
    parser.add_argument("--stage", choices=["stage1", "stage2_single", "stage2_pair", "stage3"], required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--images_root", required=True)
    return parser.parse_args()


def _load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _extract_answer(text: str) -> str:
    match = ANSWER_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = map(int, hhmm.split(":"))
    if hh == 12:
        hh = 0
    return hh * 60 + mm


def _validate_stage1(row: Dict[str, Any], images_root: str, errors: list) -> None:
    for key in ("anchor", "positive", "negative"):
        path = os.path.join(images_root, row.get(key, ""))
        if not os.path.exists(path):
            errors.append(f"missing {key} {path}")
    label = row.get("label", {})
    delta_minutes = label.get("negative_delta")
    bucket = row.get("meta", {}).get("delta_bucket")
    is_gauge = "anchor_gauge_value" in label
    if bucket == "hard" and not (1 <= abs(delta_minutes) <= 5):
        errors.append("hard delta out of range")
    if bucket == "medium" and not (6 <= abs(delta_minutes) <= 20):
        errors.append("medium delta out of range")
    easy_min, easy_max = (21, 60) if is_gauge else (30, 180)
    if bucket == "easy" and not (easy_min <= abs(delta_minutes) <= easy_max):
        errors.append("easy delta out of range")


def _validate_stage2_single(row: Dict[str, Any], images_root: str, errors: list) -> None:
    image = row.get("images", [""])[0]
    path = os.path.join(images_root, image)
    if not os.path.exists(path):
        errors.append(f"missing image {path}")
    target_text = row.get("messages", [{}, {}])[1].get("content", [{}])[0].get("text", "")
    answer = _extract_answer(target_text) or target_text.strip()
    label = row.get("label", {})
    if "gauge_value" in label:
        expected = str(int(label["gauge_value"]))
        if answer != expected:
            errors.append(f"gauge answer mismatch {answer} != {expected}")
        return
    expected = label.get("time_hhmmss") if label.get("seconds") is not None and label.get("time_hhmmss") else label.get("time_hhmm")
    if answer != expected:
        errors.append(f"answer mismatch {answer} != {label.get('time_hhmm')}")


def _validate_stage2_pair(row: Dict[str, Any], images_root: str, errors: list) -> None:
    for image in row.get("images", []):
        path = os.path.join(images_root, image)
        if not os.path.exists(path):
            errors.append(f"missing image {path}")
    target_text = row.get("messages", [{}, {}])[1].get("content", [{}])[0].get("text", "")
    answer = _extract_answer(target_text) or target_text.strip()
    label = row.get("label", {})
    try:
        if int(answer) != int(label.get("delta_minutes")):
            errors.append(f"delta mismatch {answer} != {label.get('delta_minutes')}")
    except Exception:
        errors.append("delta answer not int")


def _validate_stage3(row: Dict[str, Any], images_root: str, errors: list) -> None:
    for image in row.get("images", []):
        path = os.path.join(images_root, image)
        if not os.path.exists(path):
            errors.append(f"missing image {path}")
    chosen = _extract_answer(row.get("chosen", ""))
    rejected = _extract_answer(row.get("rejected", ""))
    label = row.get("label", {})
    if "delta_minutes" in label:
        try:
            if int(chosen) != int(label.get("delta_minutes")):
                errors.append("chosen delta incorrect")
            if int(rejected) == int(label.get("delta_minutes")):
                errors.append("rejected delta should be wrong")
        except Exception:
            errors.append("delta answer format invalid")
    else:
        if chosen != label.get("time_hhmm"):
            errors.append("chosen time incorrect")
        if rejected == label.get("time_hhmm"):
            errors.append("rejected time should be wrong")


def main() -> None:
    args = _parse_args()
    errors = []
    counts = Counter()

    for idx, row in enumerate(_load_jsonl(args.jsonl), 1):
        if args.stage == "stage1":
            _validate_stage1(row, args.images_root, errors)
        elif args.stage == "stage2_single":
            _validate_stage2_single(row, args.images_root, errors)
        elif args.stage == "stage2_pair":
            _validate_stage2_pair(row, args.images_root, errors)
        else:
            _validate_stage3(row, args.images_root, errors)
        if errors:
            errors[-1] = f"line {idx}: {errors[-1]}"
        counts["rows"] += 1

    for err in errors:
        print(f"- {err}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
