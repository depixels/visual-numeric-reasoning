"""Shared utilities for clock evaluation outputs and metrics."""

import csv
import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List, Optional


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
        for key in row.keys():
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
                    if isinstance(val, (dict, list))
                    else val
                    for key, val in row.items()
                }
            )


def extract_image_relpath(row: Dict[str, Any]) -> str:
    if isinstance(row.get("image"), str):
        return row["image"]
    images = row.get("images")
    if isinstance(images, list) and images:
        if isinstance(images[0], str):
            return images[0]
    messages = row.get("messages", [])
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for item in msg.get("content", []):
            if item.get("type") == "image" and isinstance(item.get("image"), str):
                return item["image"]
    raise ValueError(f"Cannot find image path in row id={row.get('id')}")


def infer_split(row: Dict[str, Any], gt_jsonl: Optional[str] = None) -> str:
    if isinstance(row.get("split"), str) and row["split"]:
        return row["split"]
    meta = row.get("meta", {})
    for key in ("split", "source", "domain"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    if gt_jsonl:
        return os.path.basename(os.path.dirname(os.path.abspath(gt_jsonl))) or "unknown"
    return "unknown"


def extract_gt_label(row: Dict[str, Any]) -> Dict[str, Any]:
    label = row.get("label", {})
    minutes = label.get("time_minutes")
    hhmm = label.get("time_hhmm")
    seconds = label.get("seconds")
    seconds_total = label.get("time_seconds_total")
    hhmmss = label.get("time_hhmmss")

    if minutes is None and isinstance(hhmm, str) and ":" in hhmm:
        hh, mm = hhmm.split(":")
        hh_i = int(hh)
        mm_i = int(mm)
        if hh_i == 12:
            hh_i = 0
        minutes = hh_i * 60 + mm_i

    if seconds_total is None and minutes is not None and seconds is not None:
        seconds_total = (int(minutes) * 60 + int(seconds)) % (12 * 3600)

    if minutes is None:
        raise ValueError(f"Cannot find time label in row id={row.get('id')}")

    return {
        "time_minutes": int(minutes),
        "time_hhmm": hhmm if isinstance(hhmm, str) else minutes_to_hhmm(int(minutes)),
        "seconds": None if seconds is None else int(seconds),
        "time_seconds_total": None if seconds_total is None else int(seconds_total),
        "time_hhmmss": hhmmss if isinstance(hhmmss, str) else None,
    }


def minutes_to_hhmm(total_min: int) -> str:
    hour = (total_min // 60) % 12
    minute = total_min % 60
    if hour == 0:
        hour = 12
    return f"{hour:02d}:{minute:02d}"


def seconds_to_hhmmss(total_seconds: int) -> str:
    total_seconds %= 12 * 3600
    total_minutes, second = divmod(total_seconds, 60)
    return f"{minutes_to_hhmm(total_minutes)}:{second:02d}"


def minutes_to_parts(total_min: int) -> tuple[int, int]:
    hour = (total_min // 60) % 12
    minute = total_min % 60
    return hour, minute


def seconds_to_parts(total_seconds: int) -> tuple[int, int, int]:
    total_seconds %= 12 * 3600
    total_minutes, second = divmod(total_seconds, 60)
    hour, minute = minutes_to_parts(total_minutes)
    return hour, minute, second


def circular_minute_error(gt_minutes: int, pred_minutes: int) -> int:
    diff = abs(int(pred_minutes) - int(gt_minutes)) % 720
    return min(diff, 720 - diff)


def circular_second_error(gt_seconds: int, pred_seconds: int) -> int:
    diff = abs(int(pred_seconds) - int(gt_seconds)) % (12 * 3600)
    return min(diff, 12 * 3600 - diff)


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def finalize_prediction_row(
    sample_id: str,
    split: str,
    image_rel: str,
    gt_label: Dict[str, Any],
    raw_output: str,
    pred_minutes: Optional[int],
    pred_seconds_total: Optional[int],
    latency_sec: Optional[float] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    gt_minutes = int(gt_label["time_minutes"])
    gt_h, gt_m = minutes_to_parts(gt_minutes)
    gt_seconds_total = gt_label.get("time_seconds_total")
    gt_seconds = gt_label.get("seconds")

    hour_correct = None
    minute_correct = None
    second_correct = None
    is_exact = False
    tol_1 = False
    tol_5 = False
    abs_err_minutes = None

    pred_hhmm = None if pred_minutes is None else minutes_to_hhmm(pred_minutes)
    pred_hhmmss = None if pred_seconds_total is None else seconds_to_hhmmss(pred_seconds_total)

    if pred_minutes is not None:
        pr_h, pr_m = minutes_to_parts(pred_minutes)
        hour_correct = pr_h == gt_h
        minute_correct = pr_m == gt_m
        is_exact = bool(hour_correct and minute_correct)
        abs_err_minutes = circular_minute_error(gt_minutes, pred_minutes)
        tol_1 = abs_err_minutes <= 1
        tol_5 = abs_err_minutes <= 5

    if gt_seconds_total is not None and pred_seconds_total is not None:
        _, _, gt_s = seconds_to_parts(gt_seconds_total)
        _, _, pr_s = seconds_to_parts(pred_seconds_total)
        second_correct = gt_s == pr_s

    row = {
        "id": sample_id,
        "split": split,
        "image": image_rel,
        "gt_time_hhmm": gt_label.get("time_hhmm"),
        "gt_time_minutes": gt_minutes,
        "gt_time_hhmmss": gt_label.get("time_hhmmss"),
        "gt_time_seconds_total": gt_seconds_total,
        "gt_seconds": gt_seconds,
        "pred_time_hhmm": pred_hhmm,
        "pred_time_minutes": pred_minutes,
        "pred_time_hhmmss": pred_hhmmss,
        "pred_time_seconds_total": pred_seconds_total,
        "parsed_ok": pred_minutes is not None,
        "is_exact": is_exact,
        "tol_1": tol_1,
        "tol_5": tol_5,
        "abs_err_minutes": abs_err_minutes,
        "hour_correct": hour_correct,
        "minute_correct": minute_correct,
        "second_correct": second_correct,
        "raw_output": raw_output,
        "latency_sec": latency_sec,
        "error": error,
    }
    return row


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    parsed_rows = [row for row in rows if row.get("parsed_ok")]
    abs_errors = [
        float(row["abs_err_minutes"])
        for row in parsed_rows
        if isinstance(row.get("abs_err_minutes"), (int, float))
    ]
    hour_vals = [row["hour_correct"] for row in rows if isinstance(row.get("hour_correct"), bool)]
    minute_vals = [row["minute_correct"] for row in rows if isinstance(row.get("minute_correct"), bool)]
    second_vals = [row["second_correct"] for row in rows if isinstance(row.get("second_correct"), bool)]

    exact = sum(1 for row in rows if row.get("is_exact") is True)
    tol_1 = sum(1 for row in rows if row.get("tol_1") is True)
    tol_5 = sum(1 for row in rows if row.get("tol_5") is True)

    metrics = {
        "total": total,
        "parsed": len(parsed_rows),
        "parsed_rate": safe_div(len(parsed_rows), total),
        "exact_acc": safe_div(exact, total),
        "tol1_acc": safe_div(tol_1, total),
        "tol5_acc": safe_div(tol_5, total),
        "hour_acc": safe_div(sum(1 for v in hour_vals if v), total),
        "minute_acc": safe_div(sum(1 for v in minute_vals if v), total),
        "second_acc": safe_div(sum(1 for v in second_vals if v), total) if second_vals else None,
        "mae": statistics.fmean(abs_errors) if abs_errors else math.nan,
        "median_abs_error_minutes": statistics.median(abs_errors) if abs_errors else math.nan,
    }

    metrics["exact_hhmm_acc"] = metrics["exact_acc"]
    metrics["tol_1min_acc"] = metrics["tol1_acc"]
    metrics["tol_5min_acc"] = metrics["tol5_acc"]
    metrics["mae_minutes"] = metrics["mae"]
    metrics["minute_given_hour_acc"] = safe_div(exact, sum(1 for v in hour_vals if v))
    return metrics
