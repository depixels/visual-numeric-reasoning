#!/usr/bin/env python3
"""Plot robustness curves for specular or blur buckets."""

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from common import BLUR_BUCKET_ORDER, SPECULAR_BUCKET_ORDER, load_rows, maybe_filter_split


STYLE_MAP = {
    "baseline": {"color": "#4C78A8", "marker": "o"},
    "ours": {"color": "#F58518", "marker": "s"},
    "ours full": {"color": "#54A24B", "marker": "^"},
}
METRIC_TO_FIELD = {"exact_acc": "is_exact", "tol1_acc": "tol_1", "tol5_acc": "tol_5"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot photometric robustness curve")
    p.add_argument("--input", action="append", required=True, help="label=/path/to/joined.jsonl")
    p.add_argument("--field", choices=["specular_bucket", "blur_bucket"], required=True)
    p.add_argument("--output_prefix", required=True)
    p.add_argument("--metric", choices=["exact_acc", "tol1_acc", "tol5_acc"], default="exact_acc")
    p.add_argument("--split", default=None)
    return p.parse_args()


def _parse_inputs(items: List[str]) -> List[Tuple[str, str]]:
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected label=path, got: {item}")
        label, path = item.split("=", 1)
        parsed.append((label, path))
    return parsed


def _bucket_order(field: str) -> List[str]:
    return SPECULAR_BUCKET_ORDER if field == "specular_bucket" else BLUR_BUCKET_ORDER


def _compute_curve(rows: List[Dict], field: str, metric_field: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for bucket in _bucket_order(field):
        bucket_rows = [row for row in rows if row.get(field) == bucket]
        if bucket_rows:
            out[bucket] = sum(1 for row in bucket_rows if row.get(metric_field) is True) / len(bucket_rows)
    return out


def main() -> None:
    args = _parse_args()
    metric_field = METRIC_TO_FIELD[args.metric]
    series = _parse_inputs(args.input)
    bucket_order = _bucket_order(args.field)

    plt.figure(figsize=(8.0, 5.0), dpi=300)
    ax = plt.gca()

    for label, path in series:
        rows = maybe_filter_split(load_rows(path), args.split)
        curve = _compute_curve(rows, args.field, metric_field)
        x_labels = [bucket for bucket in bucket_order if bucket in curve]
        xs = [bucket_order.index(bucket) for bucket in x_labels]
        ys = [curve[bucket] * 100.0 for bucket in x_labels]
        style = STYLE_MAP.get(label.lower(), {"color": None, "marker": "o"})
        ax.plot(xs, ys, label=label, linewidth=2.4, markersize=7, marker=style["marker"], color=style["color"])
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    ax.set_title(f"{args.metric.replace('_', ' ').upper()} vs {args.field}", fontsize=14, pad=10)
    ax.set_xlabel(args.field, fontsize=12)
    ax.set_ylabel(args.metric.replace("_", " ").upper() + " (%)", fontsize=12)
    ax.set_xticks(range(len(bucket_order)))
    ax.set_xticklabels(bucket_order, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout()

    png_path = args.output_prefix + ".png"
    pdf_path = args.output_prefix + ".pdf"
    plt.savefig(png_path, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"saved_png={png_path}")
    print(f"saved_pdf={pdf_path}")


if __name__ == "__main__":
    main()
