#!/usr/bin/env python3
"""Join per-sample predictions with GT labels and metadata."""

import argparse
import os
from typing import Any, Dict, List

from common import abs_yaw_bucket, blur_bucket, get_nested, load_rows, specular_bucket, write_jsonl


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Join GT jsonl with per-sample predictions")
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--pred_path", required=True, help="per-sample jsonl/json from eval scripts")
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--split", default=None)
    return p.parse_args()


def _infer_source(gt_row: Dict[str, Any], pred_row: Dict[str, Any], gt_jsonl: str) -> str:
    for value in (
        pred_row.get("split"),
        gt_row.get("split"),
        get_nested(gt_row, "meta.source"),
        get_nested(gt_row, "meta.domain"),
    ):
        if isinstance(value, str) and value:
            return value
    return os.path.basename(os.path.dirname(os.path.abspath(gt_jsonl))) or "unknown"


def main() -> None:
    args = _parse_args()
    gt_rows = {row["id"]: row for row in load_rows(args.gt_jsonl)}
    pred_rows = {row["id"]: row for row in load_rows(args.pred_path)}

    joined: List[Dict[str, Any]] = []
    for sample_id, pred_row in pred_rows.items():
        gt_row = gt_rows.get(sample_id)
        if gt_row is None:
            continue

        meta = gt_row.get("meta", {})
        view = meta.get("view", {}) or {}
        pose = meta.get("pose", {}) or {}
        degradation = meta.get("degradation", {}) or {}
        lighting = meta.get("lighting", {}) or {}

        yaw = view.get("yaw", pose.get("yaw"))
        pitch = view.get("pitch", pose.get("pitch"))
        roll = view.get("roll", pose.get("roll"))
        motion_blur = degradation.get("motion_blur")
        defocus = degradation.get("defocus")
        specular = degradation.get("specular")

        split = args.split or pred_row.get("split") or gt_row.get("split")
        source = _infer_source(gt_row, pred_row, args.gt_jsonl)
        joined.append(
            {
                **pred_row,
                "split": split or pred_row.get("split"),
                "source": source,
                "task": gt_row.get("task"),
                "style_id": meta.get("style_id"),
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "view_bucket": meta.get("view_bucket"),
                "tilt_bucket": abs_yaw_bucket(yaw),
                "specular": specular,
                "specular_bucket": specular_bucket(specular),
                "motion_blur": motion_blur,
                "defocus": defocus,
                "blur_bucket": blur_bucket(motion_blur, defocus),
                "lighting_env_id": lighting.get("env_id"),
                "gt_meta": meta,
                "gt_label": gt_row.get("label", {}),
            }
        )

    write_jsonl(args.output_jsonl, joined)
    print(f"joined_rows={len(joined)}")
    print(f"output={args.output_jsonl}")


if __name__ == "__main__":
    main()
