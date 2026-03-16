#!/usr/bin/env python3
"""Plot paper-ready accuracy-vs-tilt curves from joined per-sample jsonl files."""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANALYSIS_DIR = os.path.join(ROOT_DIR, "tools", "analysis")
import sys

if ANALYSIS_DIR not in sys.path:
    sys.path.insert(0, ANALYSIS_DIR)

from common import TILT_BUCKET_ORDER, load_rows, maybe_filter_split  # noqa: E402


STYLE_MAP = {
    "baseline": {"color": "#4C78A8", "marker": "o"},
    "ours stage1+2": {"color": "#F58518", "marker": "s"},
    "ours full": {"color": "#54A24B", "marker": "^"},
    "molmo": {"color": "#E45756", "marker": "D"},
}
METRIC_TO_FIELD = {"exact_acc": "is_exact", "tol1_acc": "tol_1", "tol5_acc": "tol_5"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot accuracy vs tilt curve")
    p.add_argument("--input", action="append", required=True, help="label=/path/to/joined.jsonl")
    p.add_argument("--output_prefix", required=True)
    p.add_argument("--metric", choices=["exact_acc", "tol1_acc", "tol5_acc"], default="exact_acc")
    p.add_argument("--split", default=None, help="Optional split filter, e.g. clean or noisy")
    p.add_argument("--title", default=None)
    return p.parse_args()


def _parse_inputs(items: List[str]) -> List[Tuple[str, str]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected label=path, got: {item}")
        label, path = item.split("=", 1)
        parsed.append((label, path))
    return parsed


def _bucket_metric(rows: List[Dict], metric_field: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for bucket in TILT_BUCKET_ORDER:
        bucket_rows = [row for row in rows if row.get("tilt_bucket") == bucket]
        if not bucket_rows:
            continue
        out[bucket] = sum(1 for row in bucket_rows if row.get(metric_field) is True) / len(bucket_rows)
    return out


def main() -> None:
    args = _parse_args()
    metric_field = METRIC_TO_FIELD[args.metric]
    series = _parse_inputs(args.input)

    plt.figure(figsize=(8.2, 5.2), dpi=300)
    ax = plt.gca()

    for label, path in series:
        rows = maybe_filter_split(load_rows(path), args.split)
        bucket_scores = _bucket_metric(rows, metric_field)
        x_labels = [bucket for bucket in TILT_BUCKET_ORDER if bucket in bucket_scores]
        xs = [TILT_BUCKET_ORDER.index(bucket) for bucket in x_labels]
        ys = [bucket_scores[bucket] * 100.0 for bucket in x_labels]
        style = STYLE_MAP.get(label.lower(), {"color": None, "marker": "o"})

        ax.plot(
            xs,
            ys,
            label=label,
            linewidth=2.4,
            markersize=7,
            marker=style["marker"],
            color=style["color"],
        )
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    title = args.title
    if not title:
        split_title = f" ({args.split})" if args.split else ""
        title = f"Accuracy vs Viewpoint Yaw{split_title}"
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Yaw bucket (absolute degrees)", fontsize=12)
    ax.set_ylabel(args.metric.replace("_", " ").upper() + " (%)", fontsize=12)
    ax.set_xticks(range(len(TILT_BUCKET_ORDER)))
    ax.set_xticklabels(TILT_BUCKET_ORDER, rotation=0, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout()

    png_path = args.output_prefix + ".png"
    pdf_path = args.output_prefix + ".pdf"

    out_dir = os.path.dirname(os.path.abspath(png_path))
    os.makedirs(out_dir, exist_ok=True)  # 自动创建目录

    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")



if __name__ == "__main__":
    main()
