"""Render synthetic semicircular analog gauges with matplotlib."""

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge


THETA_MIN = -120.0
THETA_MAX = 120.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render matplotlib gauge dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--id_prefix", default="matplot_gauge")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _style_bank() -> List[Dict[str, Any]]:
    return [
        {
            "style_id": "matplot_gauge_amber",
            "face": "#25272b",
            "inner": "#444750",
            "tick": "#f0f2f4",
            "minor": "#d3d7db",
            "accent": "#ffb347",
            "pointer": "#e85d3f",
            "text": "#f7f7f7",
            "title": ["SPEED", "RPM x100", "PRESSURE"],
        },
        {
            "style_id": "matplot_gauge_blue",
            "face": "#21242b",
            "inner": "#39485a",
            "tick": "#8fd7ff",
            "minor": "#d7f4ff",
            "accent": "#4cb3ff",
            "pointer": "#ff8b3d",
            "text": "#eef8ff",
            "title": ["BOOST", "RPM", "TEMP"],
        },
        {
            "style_id": "matplot_gauge_daylight",
            "face": "#d5d8dd",
            "inner": "#eff2f5",
            "tick": "#2f3338",
            "minor": "#5c6268",
            "accent": "#ff9b21",
            "pointer": "#d4492f",
            "text": "#202327",
            "title": ["SPEED", "LOAD", "FUEL"],
        },
    ]


def _pointer_angle(value: int) -> float:
    return THETA_MIN + (value / 100.0) * (THETA_MAX - THETA_MIN)


def _polar(theta_deg: float, radius: float) -> tuple[float, float]:
    theta = math.radians(theta_deg)
    return radius * math.cos(theta), radius * math.sin(theta)


def _sample_value(idx: int) -> int:
    anchors = [2, 7, 11, 18, 24, 31, 38, 44, 52, 59, 66, 73, 81, 88, 94, 99]
    base = anchors[idx % len(anchors)]
    jitter = random.randint(-2, 2)
    return max(0, min(100, base + jitter))


def _draw_gauge(path: str, value: int, style: Dict[str, Any], resolution: int) -> None:
    dpi = 100
    fig = plt.figure(figsize=(resolution / dpi, resolution / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.05, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.add_patch(Wedge((0, -0.12), 1.08, 0, 180, facecolor=style["face"], edgecolor="none"))
    ax.add_patch(Wedge((0, -0.12), 0.92, 0, 180, facecolor=style["inner"], edgecolor="none"))

    for tick_value in range(0, 101):
        theta = 180.0 - (tick_value / 100.0) * 180.0
        theta = theta - 90.0
        if tick_value % 10 == 0:
            r1, r2, lw, color = 0.78, 1.00, 2.0, style["tick"]
        elif tick_value % 5 == 0:
            r1, r2, lw, color = 0.84, 0.98, 1.3, style["minor"]
        else:
            x, y = _polar(theta, 0.90)
            ax.add_patch(Circle((x, y - 0.12), 0.010, color=style["minor"]))
            continue
        x1, y1 = _polar(theta, r1)
        x2, y2 = _polar(theta, r2)
        ax.plot([x1, x2], [y1 - 0.12, y2 - 0.12], color=color, linewidth=lw, solid_capstyle="round")

    for numeral in range(0, 101, 10):
        theta = 180.0 - (numeral / 100.0) * 180.0 - 90.0
        radius = 0.70 if numeral % 20 == 0 else 0.63
        x, y = _polar(theta, radius)
        fontsize = 12 if numeral % 20 == 0 else 8
        ax.text(x, y - 0.12, str(numeral), color=style["text"], fontsize=fontsize, ha="center", va="center", fontweight="bold" if numeral % 20 == 0 else "normal")

    for start, end, color in [(0, 70, style["accent"]), (70, 90, "#ffcf40"), (90, 100, "#e04a3a")]:
        theta1 = 180.0 - (end / 100.0) * 180.0
        theta2 = 180.0 - (start / 100.0) * 180.0
        ax.add_patch(Wedge((0, -0.12), 1.06, theta1, theta2, width=0.045, facecolor=color, edgecolor="none", alpha=0.9))

    angle = 180.0 - (value / 100.0) * 180.0 - 90.0
    x_tip, y_tip = _polar(angle, 0.82)
    x_l, y_l = _polar(angle + 90, 0.03)
    x_r, y_r = _polar(angle - 90, 0.03)
    verts = [
        (-x_l * 0.55, -0.12 - y_l * 0.55),
        (x_r * 0.55, -0.12 + y_r * 0.55),
        (x_tip + x_r * 0.12, y_tip - 0.12 + y_r * 0.12),
        (x_tip + x_l * 0.12, y_tip - 0.12 - y_l * 0.12),
    ]
    ax.fill([v[0] for v in verts], [v[1] for v in verts], color=style["pointer"], zorder=10)
    ax.add_patch(Circle((0, -0.12), 0.05, color="#d3d6da", zorder=11))

    ax.text(0, 0.18, random.choice(style["title"]), color=style["text"], fontsize=14, ha="center", va="center", fontweight="bold")

    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)
    out_root = os.path.abspath(args.out_dir)
    split_root = os.path.join(out_root, "gauge_clean_matplot")
    images_dir = os.path.join(split_root, "images")
    os.makedirs(images_dir, exist_ok=True)
    annotations_path = os.path.join(split_root, "annotations.jsonl")

    start_index = 0
    if args.resume and os.path.exists(annotations_path):
        start_index = len(_load_jsonl(annotations_path))
        if start_index >= args.n:
            return

    styles = _style_bank()
    mode = "a" if args.resume else "w"
    with open(annotations_path, mode, encoding="utf-8") as f:
        for idx in range(start_index, args.n):
            random.seed(args.seed + idx)
            style = rng.choice(styles)
            value = _sample_value(idx)
            angle = _pointer_angle(value)
            filename = f"sample_{idx:05d}.png"
            rel_image = os.path.join("images", filename)
            _draw_gauge(os.path.join(split_root, rel_image), value, style, args.resolution)
            row = {
                "id": f"{args.id_prefix}_{idx:06d}",
                "image": rel_image,
                "task": "analog_gauge_readout",
                "label": {
                    "gauge_value": value,
                    "pointer_angle_deg": round(angle, 4),
                    "value_norm": round(value / 100.0, 4),
                },
                "meta": {
                    "source": "matplot_gauge",
                    "style_id": style["style_id"],
                    "benchmark_split": "clean",
                },
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
