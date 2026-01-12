"""Generate synthetic analog clocks with matplotlib."""

import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.colors import to_rgb

STYLE_PATH = os.path.join(os.path.dirname(__file__), "assets", "matplot_styles.json")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render matplotlib clock dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--id_prefix", default="matplot")
    parser.add_argument("--second_hand_prob", type=float, default=0.5)
    parser.add_argument("--alarm_hand_prob", type=float, default=0.2)
    parser.add_argument("--time_mode", choices=["hm", "hms", "random"], default="random")
    parser.add_argument("--force_hand_config", choices=["2", "3", "4"], default=None)
    parser.add_argument("--max_seconds", type=int, default=59)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _time_hhmm(time_minutes: int) -> str:
    hh = (time_minutes // 60) % 12
    mm = time_minutes % 60
    if hh == 0:
        hh = 12
    return f"{hh:02d}:{mm:02d}"


def _time_hhmmss(time_minutes: int, seconds: int) -> str:
    hh = (time_minutes // 60) % 12
    mm = time_minutes % 60
    if hh == 0:
        hh = 12
    return f"{hh:02d}:{mm:02d}:{seconds:02d}"


def _load_styles(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        styles = json.load(f)
    if not isinstance(styles, list):
        raise ValueError("matplot_styles.json must be a list")
    return styles


def _luminance(color: Any) -> float:
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ensure_min_numerals(style: Dict[str, Any]) -> Dict[str, Any]:
    numerals = dict(style.get("numerals", {}))
    dial = style.get("dial", {})
    if dial.get("double_ring"):
        if numerals.get("type") in (None, "none", ""):
            numerals["type"] = "double_ring"
    if numerals.get("type") in (None, "none", ""):
        numerals["type"] = "arabic_quadrants"
        numerals.setdefault("fontsize", 16)
        numerals.setdefault("color", style.get("ticks", {}).get("color", "#222222"))
    updated = dict(style)
    updated["numerals"] = numerals
    return updated


def _ensure_contrast(style: Dict[str, Any]) -> Dict[str, Any]:
    dial = style.get("dial", {})
    if dial.get("shape", "circle") != "square":
        return style
    face_color = dial.get("face_color", dial.get("color", "#f0f0f0"))
    dial_lum = _luminance(face_color)
    contrast_color = "#111111" if dial_lum >= 0.5 else "#f7f7f7"
    updated = dict(style)

    numerals = dict(updated.get("numerals", {}))
    if numerals.get("type") not in (None, "none", ""):
        numerals["color"] = contrast_color
    updated["numerals"] = numerals

    ticks = dict(updated.get("ticks", {}))
    if ticks.get("type") not in (None, "none", ""):
        ticks["color"] = contrast_color
    updated["ticks"] = ticks
    return updated


def _build_time_label(time_minutes: int, seconds: int | None) -> Dict[str, Any]:
    label = {"time_hhmm": _time_hhmm(time_minutes), "time_minutes": time_minutes, "seconds": None}
    if seconds is None:
        return label
    total_seconds = (time_minutes * 60 + seconds) % (12 * 3600)
    label["time_hhmmss"] = _time_hhmmss(time_minutes, seconds)
    label["time_seconds_total"] = total_seconds
    label["seconds"] = seconds
    return label


def _clock_angle(minutes: int) -> float:
    return 2.0 * math.pi * (minutes / 60.0)


def _polar(angle: float, radius: float) -> Tuple[float, float]:
    return math.sin(angle) * radius, math.cos(angle) * radius


def _draw_dial(ax: Any, style: Dict[str, Any]) -> None:
    dial = style.get("dial", {})
    face_color = dial.get("face_color", "#f0f0f0")
    border_color = dial.get("border_color", "#222222")
    border_width = float(dial.get("border_width", 1.5))
    border_type = dial.get("border_type", "single")
    shape = dial.get("shape", "circle")
    double_ring = dial.get("double_ring", False)
    ring_color = dial.get("ring_color", border_color)

    if shape == "square":
        face = Rectangle((-1.0, -1.0), 2.0, 2.0, facecolor=face_color, edgecolor="none", zorder=1)
        ax.add_patch(face)
        outer = Rectangle((-1.0, -1.0), 2.0, 2.0, facecolor="none", edgecolor=border_color, linewidth=border_width, zorder=3)
        ax.add_patch(outer)
        if border_type == "double":
            inner = Rectangle((-0.94, -0.94), 1.88, 1.88, facecolor="none", edgecolor=border_color, linewidth=max(1.0, border_width * 0.7), zorder=3)
            ax.add_patch(inner)
        if double_ring:
            ax.add_patch(Rectangle((-0.72, -0.72), 1.44, 1.44, facecolor="none", edgecolor=ring_color, linewidth=max(1.0, border_width * 0.6), zorder=3))
    else:
        face = Circle((0, 0), 1.0, facecolor=face_color, edgecolor="none", zorder=1)
        ax.add_patch(face)
        outer = Circle((0, 0), 1.0, facecolor="none", edgecolor=border_color, linewidth=border_width, zorder=3)
        ax.add_patch(outer)
        if border_type == "double":
            inner = Circle((0, 0), 0.94, facecolor="none", edgecolor=border_color, linewidth=max(1.0, border_width * 0.7), zorder=3)
            ax.add_patch(inner)
        if double_ring:
            ax.add_patch(Circle((0, 0), 0.72, facecolor="none", edgecolor=ring_color, linewidth=max(1.0, border_width * 0.6), zorder=3))


def _draw_ticks(ax: Any, style: Dict[str, Any]) -> None:
    ticks = style.get("ticks", {})
    tick_type = ticks.get("type", "major_minor")
    color = ticks.get("color", "#222222")
    if style.get("dial", {}).get("shape", "circle") == "square":
        face_color = style.get("dial", {}).get("face_color", style.get("dial", {}).get("color", "#f0f0f0"))
        color = "#111111" if _luminance(face_color) >= 0.5 else "#f7f7f7"
    width_major = float(ticks.get("width_major", 2.0))
    width_minor = float(ticks.get("width_minor", 1.0))
    length_major = float(ticks.get("length_major", 0.11))
    length_minor = float(ticks.get("length_minor", 0.06))
    dot_size = float(ticks.get("dot_size", 18.0))

    if tick_type == "dots":
        xs = []
        ys = []
        for i in range(12):
            angle = math.radians(i * 30.0)
            x, y = _polar(angle, 0.86)
            xs.append(x)
            ys.append(y)
        ax.scatter(xs, ys, s=dot_size, c=color, zorder=4)
        return

    if tick_type == "major_only":
        indices = range(0, 60, 5)
    elif tick_type == "sparse_5min":
        indices = range(0, 60, 5)
    else:
        indices = range(60)

    for i in indices:
        angle = math.radians(i * 6.0)
        is_major = i % 5 == 0
        if tick_type == "major_only" and not is_major:
            continue
        if tick_type == "sparse_5min" and not is_major:
            continue
        length = length_major if is_major else length_minor
        width = width_major if is_major else width_minor
        r_outer = 0.98
        r_inner = r_outer - length
        x1, y1 = _polar(angle, r_inner)
        x2, y2 = _polar(angle, r_outer)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=4, solid_capstyle="round")


def _roman_numeral(value: int) -> str:
    mapping = {
        1: "I",
        2: "II",
        3: "III",
        4: "IV",
        5: "V",
        6: "VI",
        7: "VII",
        8: "VIII",
        9: "IX",
        10: "X",
        11: "XI",
        12: "XII",
    }
    return mapping.get(value, str(value))


def _draw_numerals(ax: Any, style: Dict[str, Any]) -> None:
    numerals = style.get("numerals", {})
    numerals_type = numerals.get("type", "none")
    if numerals_type == "none":
        return
    color = numerals.get("color", "#222222")
    if style.get("dial", {}).get("shape", "circle") == "square":
        face_color = style.get("dial", {}).get("face_color", style.get("dial", {}).get("color", "#f0f0f0"))
        color = "#111111" if _luminance(face_color) >= 0.5 else "#f7f7f7"
    fontsize = float(numerals.get("fontsize", 16))
    fontfamily = numerals.get("fontfamily", "DejaVu Sans")

    if numerals_type == "double_ring":
        inner = [(12, "12"), (3, "3"), (6, "6"), (9, "9")]
        outer = [(3, "15"), (6, "30"), (9, "45")]
        dial_shape = style.get("dial", {}).get("shape", "circle")
        if dial_shape == "square":
            fontsize *= 0.9
        inner_r = 0.5 if dial_shape == "square" else 0.55
        outer_r = 0.82 if dial_shape == "square" else 0.88
        for i, label in inner:
            angle = math.radians(i * 30.0)
            x, y = _polar(angle, inner_r)
            ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color=color, fontfamily=fontfamily, zorder=5)
        for i, label in outer:
            angle = math.radians(i * 30.0)
            x, y = _polar(angle, outer_r)
            ax.text(x, y, label, ha="center", va="center", fontsize=fontsize * 0.9, color=color, fontfamily=fontfamily, zorder=5)
        return
    if numerals_type == "arabic_full":
        indices = list(range(1, 13))
        labels = {i: str(i) for i in indices}
    elif numerals_type == "arabic_quadrants":
        indices = [12, 3, 6, 9]
        labels = {i: str(i) for i in indices}
    elif numerals_type == "roman_full":
        indices = list(range(1, 13))
        labels = {i: _roman_numeral(i) for i in indices}
    else:
        return

    for i in indices:
        angle = math.radians(i * 30.0)
        x, y = _polar(angle, 0.7)
        ax.text(
            x,
            y,
            labels[i],
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontfamily=fontfamily,
            zorder=5,
        )


def _hand_polygon(angle: float, length: float, width: float, back: float) -> List[Tuple[float, float]]:
    dir_x, dir_y = _polar(angle, 1.0)
    perp_x, perp_y = dir_y, -dir_x
    tip = (dir_x * length, dir_y * length)
    base_left = (-dir_x * back + perp_x * width / 2.0, -dir_y * back + perp_y * width / 2.0)
    base_right = (-dir_x * back - perp_x * width / 2.0, -dir_y * back - perp_y * width / 2.0)
    return [base_left, tip, base_right]


def _draw_hand(ax: Any, hand_cfg: Dict[str, Any], angle: float) -> None:
    hand_type = hand_cfg.get("type", "rect")
    length = float(hand_cfg.get("length", 0.7))
    width = float(hand_cfg.get("width", 0.04))
    color = hand_cfg.get("color", "#111111")

    if hand_type == "triangle":
        points = _hand_polygon(angle, length, width, back=0.08)
        patch = Polygon(points, closed=True, facecolor=color, edgecolor=color, zorder=6)
        ax.add_patch(patch)
        return

    x, y = _polar(angle, length)
    ax.plot([0, x], [0, y], color=color, linewidth=width * 100, zorder=6, solid_capstyle="round")


def _select_hand_config(force: str | None, second_prob: float, alarm_prob: float) -> Tuple[int, bool, bool]:
    if force:
        hand_config = int(force)
        return hand_config, hand_config >= 3, hand_config >= 4
    has_second = random.random() < second_prob
    has_alarm = has_second and (random.random() < alarm_prob)
    hand_config = 4 if has_alarm else (3 if has_second else 2)
    return hand_config, has_second, has_alarm


def _sample_time(time_mode: str, hand_config: int, max_seconds: int) -> Tuple[int, int]:
    time_minutes = random.randint(0, 719)
    if time_mode == "hm":
        return time_minutes, 0
    if time_mode == "hms":
        return time_minutes, random.randint(0, max(0, max_seconds))
    if hand_config >= 3:
        return time_minutes, random.randint(0, max(0, max_seconds))
    return time_minutes, 0


def _apply_hand_config(style: Dict[str, Any], hand_config: int, has_second: bool, has_alarm: bool, seconds: int) -> Dict[str, Any]:
    updated = dict(style)
    hands = dict(updated.get("hands", {}))
    second = dict(hands.get("second", {}))
    alarm = dict(hands.get("alarm", {}))
    second.setdefault("length", 0.95)
    second.setdefault("width", 0.015)
    second.setdefault("color", "#cc2222")
    second.setdefault("type", "rect")
    alarm.setdefault("length", 0.35)
    alarm.setdefault("width", 0.03)
    alarm.setdefault("color", "#333333")
    alarm.setdefault("type", "triangle")
    second["enabled"] = bool(hand_config >= 3 and has_second)
    alarm["enabled"] = bool(hand_config >= 4 and has_alarm)
    hands["second"] = second
    hands["alarm"] = alarm
    updated["hands"] = hands
    updated["_seconds"] = seconds
    return updated


def draw_clock_matplotlib(time_minutes: int, style: Dict[str, Any], out_path: str, resolution: int) -> None:
    """Draw a single clock image using matplotlib."""
    background = style.get("background", {}).get("color", "#ffffff")

    dpi = 100
    size = resolution / dpi
    fig = plt.figure(figsize=(size, size), dpi=dpi, facecolor=background)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(background)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    _draw_dial(ax, style)
    _draw_ticks(ax, style)
    _draw_numerals(ax, style)

    minute = time_minutes % 60
    hour = (time_minutes // 60) % 12
    seconds = int(style.get("_seconds", 0))
    minute_angle = _clock_angle(minute + seconds / 60.0)
    hour_angle = _clock_angle(hour * 5 + minute / 12.0 + seconds / 720.0)

    hands = style.get("hands", {})
    _draw_hand(ax, hands.get("hour", {}), hour_angle)
    _draw_hand(ax, hands.get("minute", {}), minute_angle)

    second_cfg = hands.get("second", {})
    if second_cfg.get("enabled", False):
        second_angle = _clock_angle(seconds)
        _draw_hand(ax, second_cfg, second_angle)

    alarm_cfg = hands.get("alarm", {})
    if alarm_cfg.get("enabled", False):
        total_seconds = (time_minutes * 60 + seconds) % 3600
        alarm_angle = _clock_angle(total_seconds / 60.0)
        _draw_hand(ax, alarm_cfg, alarm_angle)

    ax.add_patch(Circle((0, 0), 0.02, facecolor=hands.get("minute", {}).get("color", "#111111"), edgecolor="none", zorder=7))

    fig.savefig(out_path, dpi=dpi, facecolor=background)
    plt.close(fig)


def _ensure_dirs(out_dir: str) -> str:
    images_dir = os.path.join(out_dir, "rege_clean_matplot", "images")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def main() -> None:
    args = _parse_args()
    rng = random.Random(args.seed)

    styles = _load_styles(STYLE_PATH)
    if len(styles) != 20:
        raise ValueError(f"Expected 20 styles, found {len(styles)}")

    images_dir = _ensure_dirs(args.out_dir)
    annotations_path = os.path.join(args.out_dir, "rege_clean_matplot", "annotations.jsonl")

    repeats = math.ceil(args.n / len(styles))
    style_pool = (styles * repeats)[: args.n]
    rng.shuffle(style_pool)
    shape_pool = ["square" if rng.random() < 0.25 else "circle" for _ in range(args.n)]
    ring_pool = [rng.random() < 0.2 for _ in range(args.n)]

    start_index = 0
    if args.resume and os.path.exists(annotations_path):
        with open(annotations_path, "r", encoding="utf-8") as f:
            start_index = sum(1 for _ in f if _.strip())
    mode = "a" if args.resume else "w"

    time_list = []
    seconds_list = []
    hand_configs = []
    for _ in range(args.n):
        hand_config, has_second, has_alarm = _select_hand_config(args.force_hand_config, args.second_hand_prob, args.alarm_hand_prob)
        time_minutes, seconds = _sample_time(args.time_mode, hand_config, args.max_seconds)
        time_list.append(time_minutes)
        seconds_list.append(seconds if has_second else 0)
        hand_configs.append((hand_config, has_second, has_alarm))

    with open(annotations_path, mode, encoding="utf-8") as f:
        for idx in range(start_index, args.n):
            style = style_pool[idx]
            dial = style.get("dial", {})
            if "shape" not in dial or "double_ring" not in dial:
                dial = dict(dial)
                dial.setdefault("shape", shape_pool[idx])
                dial.setdefault("double_ring", ring_pool[idx])
                if dial.get("double_ring"):
                    numerals = dict(style.get("numerals", {}))
                    numerals.setdefault("type", "double_ring")
                    style = dict(style)
                    style["numerals"] = numerals
                style = dict(style)
                style["dial"] = dial
            style = _ensure_min_numerals(style)
            style = _ensure_contrast(style)
            time_minutes = time_list[idx]
            hand_config, has_second, has_alarm = hand_configs[idx]
            seconds = seconds_list[idx]
            style = _apply_hand_config(style, hand_config, has_second, has_alarm, seconds)
            filename = f"{idx + 1:06d}.png"
            image_rel = os.path.join("images", filename)
            out_path = os.path.join(images_dir, filename)
            if not (args.resume and os.path.exists(out_path)):
                draw_clock_matplotlib(time_minutes, style, out_path, args.resolution)

            row = {
                "id": f"{args.id_prefix}_{idx + 1:06d}",
                "image": image_rel,
                "task": "single_readout",
                "label": _build_time_label(time_minutes, seconds if has_second else None),
                "meta": {
                    "domain": "matplot",
                    "style_id": style.get("style_id", "unknown"),
                    "render": {"resolution": [args.resolution, args.resolution]},
                    "hand_config": hand_config,
                    "has_second": has_second,
                    "has_alarm": has_alarm,
                },
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
