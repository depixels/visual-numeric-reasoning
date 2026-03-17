"""Render a minimal automotive-style analog gauge smoke dataset with Blender."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import render_batch as base  # type: ignore

THETA_MIN = -120.0
THETA_MAX = 120.0
OUTER_RADIUS = 1.95
INNER_RADIUS = 1.62
CENTER_Y = -0.92


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render automotive-style gauge smoke data")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--split", choices=["clean", "noisy"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--id_prefix", default="gauge_smoke")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _gauge_point(theta_deg: float, radius: float) -> Tuple[float, float, float]:
    theta = math.radians(theta_deg)
    return (radius * math.sin(theta), CENTER_Y + radius * math.cos(theta), 0.0)


def _bucket_tilt(yaw: float) -> str:
    yaw = abs(float(yaw))
    if yaw < 10:
        return "Front"
    if yaw < 20:
        return "10-20"
    if yaw < 30:
        return "20-30"
    if yaw < 40:
        return "30-40"
    if yaw < 50:
        return "40-50"
    if yaw < 60:
        return "50-60"
    if yaw < 70:
        return "60-70"
    return "70+"


def _bucket_specular(value: float) -> str:
    if value <= 0.0:
        return "0.0"
    if value < 0.1:
        return "0.0-0.1"
    if value < 0.3:
        return "0.1-0.3"
    if value < 0.6:
        return "0.3-0.6"
    return "0.6+"


def _bucket_blur(motion_blur: float, defocus: float) -> str:
    value = max(float(motion_blur), float(defocus))
    if value <= 0.0:
        return "0.0"
    if value < 0.05:
        return "0.0-0.05"
    if value < 0.15:
        return "0.05-0.15"
    if value < 0.30:
        return "0.15-0.30"
    return "0.30+"


def _make_half_disc(name: str, radius: float, depth: float, color: Tuple[float, float, float], roughness: float, metallic: float, specular: float) -> Any:
    verts: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []
    steps = 48
    for idx in range(steps + 1):
        theta = THETA_MIN + (THETA_MAX - THETA_MIN) * idx / steps
        verts.append(_gauge_point(theta, radius))
    faces.append(list(range(len(verts))))

    mesh = base.bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = base.bpy.data.objects.new(name, mesh)
    base.bpy.context.collection.objects.link(obj)
    solidify = obj.modifiers.new(name="Solidify", type="SOLIDIFY")
    solidify.thickness = depth
    solidify.offset = 0.0
    base.bpy.context.view_layer.objects.active = obj
    base.bpy.ops.object.modifier_apply(modifier=solidify.name)
    mat = base._make_material(name + "Mat", color, roughness=roughness, metallic=metallic, specular=specular)
    obj.data.materials.append(mat)
    return obj


def _add_tick(theta_deg: float, radius_inner: float, radius_outer: float, width: float, depth: float, color: Tuple[float, float, float], name: str) -> Any:
    x_mid, y_mid, _ = _gauge_point(theta_deg, (radius_inner + radius_outer) * 0.5)
    length = radius_outer - radius_inner
    base.bpy.ops.mesh.primitive_cube_add(size=1.0, location=(x_mid, y_mid, depth))
    obj = base.bpy.context.active_object
    obj.name = name
    obj.scale = (width, length * 0.5, 0.015)
    obj.rotation_euler[2] = -math.radians(theta_deg)
    obj.data.materials.append(base._make_material(name + "Mat", color, roughness=0.4, metallic=0.0, specular=0.2))
    return obj


def _add_tick_dot(theta_deg: float, radius: float, size: float, depth: float, color: Tuple[float, float, float], name: str) -> Any:
    x, y, _ = _gauge_point(theta_deg, radius)
    base.bpy.ops.mesh.primitive_uv_sphere_add(radius=size, location=(x, y, depth), segments=12, ring_count=8)
    obj = base.bpy.context.active_object
    obj.name = name
    obj.scale = (1.0, 1.0, 0.45)
    obj.data.materials.append(base._make_material(name + "Mat", color, roughness=0.45, metallic=0.0, specular=0.18))
    return obj


def _add_numeral(text: str, theta_deg: float, radius: float, size: float, color: Tuple[float, float, float], depth: float, name: str) -> Any:
    x, y, _ = _gauge_point(theta_deg, radius)
    base.bpy.ops.object.text_add(location=(x, y, depth))
    obj = base.bpy.context.active_object
    obj.name = name
    obj.data.body = text
    obj.data.size = size
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.rotation_euler = (0.0, 0.0, 0.0)
    obj.data.extrude = 0.01
    obj.data.materials.append(base._make_material(name + "Mat", color, roughness=0.5, metallic=0.0, specular=0.1))
    return obj


def _build_pointer(pointer_angle_deg: float, style: Dict[str, Any]) -> Any:
    pointer_length = style["pointer_length"]
    pointer_width = style["pointer_width"]
    pointer_tip_width = style["pointer_tip_width"]
    pointer_color = style["pointer_color"]
    theta = math.radians(pointer_angle_deg)
    tip_x = math.sin(theta) * pointer_length
    tip_y = CENTER_Y + math.cos(theta) * pointer_length
    perp_x = math.cos(theta)
    perp_y = -math.sin(theta)

    root_half = pointer_width * 0.5
    tip_half = pointer_tip_width * 0.5
    back_offset = 0.08

    verts = [
        (-perp_x * root_half, CENTER_Y - perp_y * root_half, 0.115),
        (perp_x * root_half, CENTER_Y + perp_y * root_half, 0.115),
        (tip_x + perp_x * tip_half, tip_y + perp_y * tip_half, 0.115),
        (tip_x - perp_x * tip_half, tip_y - perp_y * tip_half, 0.115),
        (-perp_x * root_half * 0.7, CENTER_Y - back_offset - perp_y * root_half * 0.7, 0.115),
        (perp_x * root_half * 0.7, CENTER_Y - back_offset + perp_y * root_half * 0.7, 0.115),
    ]
    faces = [[4, 5, 1, 0], [0, 1, 2, 3]]
    mesh = base.bpy.data.meshes.new("PointerMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    pointer = base.bpy.data.objects.new("Pointer", mesh)
    base.bpy.context.collection.objects.link(pointer)
    solidify = pointer.modifiers.new(name="Solidify", type="SOLIDIFY")
    solidify.thickness = 0.03
    solidify.offset = 0.0
    base.bpy.context.view_layer.objects.active = pointer
    base.bpy.ops.object.modifier_apply(modifier=solidify.name)
    pointer.data.materials.append(base._make_material("PointerMat", pointer_color, roughness=0.22, metallic=0.0, specular=0.28))

    base.bpy.ops.mesh.primitive_cylinder_add(radius=0.08, depth=0.06, location=(0.0, CENTER_Y, 0.13))
    cap = base.bpy.context.active_object
    cap.data.materials.append(base._make_material("HubMat", style["hub_color"], roughness=0.3, metallic=0.7, specular=0.4))

    return pointer


def _build_gauge(value: int, pointer_angle_deg: float, noisy: bool) -> Tuple[Any, Dict[str, Any]]:
    style_bank = [
        {
            "style_id": "gauge_amber_classic",
            "face_color": (0.06, 0.06, 0.07),
            "inner_color": (0.11, 0.11, 0.12),
            "tick_color": (0.86, 0.87, 0.88),
            "text_color": (0.95, 0.95, 0.96),
            "pointer_color": (0.92, 0.18, 0.14),
            "hub_color": (0.62, 0.64, 0.68),
            "accent_color": (1.0, 0.56, 0.08),
            "pointer_length": 1.54,
            "pointer_width": 0.036,
            "pointer_tip_width": 0.010,
            "title_choices": ["SPEED", "RPM x100", "PRESSURE"],
        },
        {
            "style_id": "gauge_blue_modern",
            "face_color": (0.08, 0.09, 0.11),
            "inner_color": (0.12, 0.13, 0.16),
            "tick_color": (0.70, 0.88, 1.0),
            "text_color": (0.92, 0.96, 1.0),
            "pointer_color": (0.88, 0.33, 0.05),
            "hub_color": (0.55, 0.58, 0.64),
            "accent_color": (0.22, 0.65, 1.0),
            "pointer_length": 1.48,
            "pointer_width": 0.038,
            "pointer_tip_width": 0.012,
            "title_choices": ["RPM x100", "BOOST", "TURBO"],
        },
        {
            "style_id": "gauge_redline_sport",
            "face_color": (0.05, 0.05, 0.06),
            "inner_color": (0.10, 0.10, 0.11),
            "tick_color": (0.90, 0.91, 0.92),
            "text_color": (0.98, 0.98, 0.98),
            "pointer_color": (0.95, 0.16, 0.12),
            "hub_color": (0.64, 0.64, 0.66),
            "accent_color": (0.92, 0.12, 0.10),
            "pointer_length": 1.56,
            "pointer_width": 0.034,
            "pointer_tip_width": 0.009,
            "title_choices": ["RPM", "REV", "SPORT"],
        },
        {
            "style_id": "gauge_green_utility",
            "face_color": (0.07, 0.08, 0.07),
            "inner_color": (0.12, 0.14, 0.12),
            "tick_color": (0.88, 0.90, 0.86),
            "text_color": (0.93, 0.96, 0.91),
            "pointer_color": (0.98, 0.73, 0.18),
            "hub_color": (0.58, 0.61, 0.56),
            "accent_color": (0.38, 0.86, 0.44),
            "pointer_length": 1.50,
            "pointer_width": 0.036,
            "pointer_tip_width": 0.010,
            "title_choices": ["PRESSURE", "TEMP", "LOAD"],
        },
        {
            "style_id": "gauge_white_daylight",
            "face_color": (0.74, 0.76, 0.79),
            "inner_color": (0.86, 0.87, 0.89),
            "tick_color": (0.24, 0.26, 0.28),
            "text_color": (0.16, 0.17, 0.18),
            "pointer_color": (0.92, 0.28, 0.10),
            "hub_color": (0.52, 0.54, 0.58),
            "accent_color": (0.98, 0.58, 0.12),
            "pointer_length": 1.46,
            "pointer_width": 0.034,
            "pointer_tip_width": 0.010,
            "title_choices": ["SPEED", "FUEL", "TEMP"],
        },
    ]
    style = random.choice(style_bank)

    root = base.bpy.data.objects.new("GaugeRoot", None)
    base.bpy.context.collection.objects.link(root)

    back = _make_half_disc("GaugeBack", OUTER_RADIUS, 0.10, style["face_color"], roughness=0.45, metallic=0.3, specular=0.25)
    back.parent = root
    face = _make_half_disc("GaugeFace", INNER_RADIUS, 0.06, style["inner_color"], roughness=0.55, metallic=0.0, specular=0.15)
    face.location.z = 0.03
    face.parent = root

    for tick_value in range(0, 101):
        theta = THETA_MIN + (tick_value / 100.0) * (THETA_MAX - THETA_MIN)
        if tick_value % 10 == 0:
            is_major = tick_value % 20 == 0
            radius_outer = 1.58
            radius_inner = 1.26 if is_major else 1.34
            width = 0.02 if is_major else 0.013
            tick = _add_tick(theta, radius_inner, radius_outer, width, 0.085, style["tick_color"], f"Tick_{tick_value:03d}")
            tick.parent = root
        elif tick_value % 5 == 0:
            tick = _add_tick(theta, 1.40, 1.54, 0.010, 0.085, style["tick_color"], f"TickMid_{tick_value:03d}")
            tick.parent = root
        else:
            dot = _add_tick_dot(theta, 1.48, 0.018, 0.085, style["tick_color"], f"TickDot_{tick_value:03d}")
            dot.parent = root

    for numeral in range(0, 101, 10):
        theta = THETA_MIN + (numeral / 100.0) * (THETA_MAX - THETA_MIN)
        radius = 1.15 if numeral % 20 == 0 else 1.09
        size = 0.18 if numeral % 20 == 0 else 0.13
        obj = _add_numeral(str(numeral), theta, radius, size, style["text_color"], 0.10, f"Num_{numeral:03d}")
        obj.parent = root

    for seg_start, seg_end, color in [(0, 70, style["accent_color"]), (70, 90, (1.0, 0.68, 0.08)), (90, 100, (0.90, 0.12, 0.10))]:
        for step in range(seg_start, seg_end, 5):
            theta = THETA_MIN + (step / 100.0) * (THETA_MAX - THETA_MIN)
            band = _add_tick(theta, 1.62, 1.82, 0.04, 0.09, color, f"Band_{step:03d}")
            band.parent = root

    pointer = _build_pointer(pointer_angle_deg, style)
    pointer.parent = root

    base.bpy.ops.object.text_add(location=(0.0, CENTER_Y + 0.34, 0.11))
    title = base.bpy.context.active_object
    title.data.body = random.choice(style["title_choices"])
    title.data.size = 0.22
    title.data.align_x = "CENTER"
    title.data.extrude = 0.01
    title.data.materials.append(base._make_material("TitleMat", style["text_color"], roughness=0.5, metallic=0.0, specular=0.1))
    title.parent = root

    if noisy:
        glass = _make_half_disc(
            "GaugeGlass",
            OUTER_RADIUS * 0.985,
            0.045,
            (1.0, 1.0, 1.0),
            roughness=0.03,
            metallic=0.0,
            specular=0.0,
        )
        glass.location.z = 0.17
        glass.data.materials.clear()
        glass.data.materials.append(
            base._make_glass_material(
                "GaugeGlassMat",
                roughness=0.03,
                ior=1.46,
                specular=0.85,
                tint=(1.0, 1.0, 1.0),
                bump=0.03,
            )
        )
        glass.parent = root

    return root, style


def _sample_view(noisy: bool) -> Tuple[Dict[str, float], float]:
    if noisy:
        return {
            "yaw": random.uniform(-32.0, 32.0),
            "pitch": random.uniform(44.0, 68.0),
        }, random.uniform(-6.0, 6.0)
    return {
        "yaw": random.uniform(-14.0, 14.0),
        "pitch": random.uniform(60.0, 82.0),
    }, random.uniform(-3.0, 3.0)


def _sample_degradation(noisy: bool) -> Dict[str, float]:
    if noisy:
        return {
            "specular": random.uniform(0.35, 0.95),
            "motion_blur": random.uniform(0.08, 0.35),
            "defocus": random.uniform(0.02, 0.25),
        }
    return {"specular": random.uniform(0.0, 0.10), "motion_blur": 0.0, "defocus": 0.0}


def _sample_value(idx: int, n: int) -> int:
    anchors = [5, 12, 18, 27, 35, 42, 55, 63, 74, 82, 91, 98]
    base_value = anchors[idx % len(anchors)]
    jitter = random.randint(-3, 3)
    return max(0, min(100, base_value + jitter))


def _collect_bbox_points(root: Any) -> List[Any]:
    points = []
    for obj in [root] + list(root.children_recursive):
        if getattr(obj, "type", None) not in {"MESH", "CURVE", "FONT", "SURFACE", "META"}:
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ base.Vector(corner))
    return points


def _project_bbox(scene: Any, camera: Any, root: Any) -> Tuple[float, float, float, float]:
    bbox_points = _collect_bbox_points(root)
    base.bpy.context.view_layer.update()
    xs = []
    ys = []
    for point in bbox_points:
        view = base.world_to_camera_view(scene, camera, point)
        xs.append(float(view.x))
        ys.append(float(view.y))
    return min(xs), max(xs), min(ys), max(ys)


def _tighten_camera_framing(scene: Any, camera: Any, root: Any, target_span_x: float, target_span_y: float) -> None:
    bbox_points = _collect_bbox_points(root)
    if not bbox_points:
        return

    def _measure() -> Tuple[float, float]:
        xs = []
        ys = []
        base.bpy.context.view_layer.update()
        for point in bbox_points:
            view = base.world_to_camera_view(scene, camera, point)
            xs.append(float(view.x))
            ys.append(float(view.y))
        return max(xs) - min(xs), max(ys) - min(ys)

    span_x, span_y = _measure()
    if span_x <= 0 or span_y <= 0:
        return

    # For orthographic cameras, a smaller ortho_scale makes the object larger in frame.
    scale_factor = max(span_x / target_span_x, span_y / target_span_y)
    camera.data.ortho_scale *= scale_factor

    # Safety pass: if any side is still too close to the border, back off slightly.
    base.bpy.context.view_layer.update()
    xs = []
    ys = []
    for point in bbox_points:
        view = base.world_to_camera_view(scene, camera, point)
        xs.append(float(view.x))
        ys.append(float(view.y))
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x < 0.08 or max_x > 0.92 or min_y < 0.08 or max_y > 0.92:
        camera.data.ortho_scale *= 1.15  # 增加缩放系数


def _ensure_dirs(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)


def _render_one(out_dir: str, idx: int, split: str, resolution: int, seed: int, id_prefix: str) -> Dict[str, Any]:
    noisy = split == "noisy"
    random.seed(seed)
    base._clear_scene()
    base._set_cycles(base.bpy.context.scene)

    value = _sample_value(idx, 1)
    pointer_angle_deg = THETA_MIN + (value / 100.0) * (THETA_MAX - THETA_MIN)
    root, style = _build_gauge(value, pointer_angle_deg, noisy)

    env_id = random.choice(
        ["studio_softbox", "studio_softbox_round", "top_light", "window_side"]
        if noisy
        else ["studio_softbox", "top_light"]
    )
    lighting = base._setup_studio_environment(env_id, noisy=noisy)
    view_cfg, view_roll = _sample_view(noisy)
    cam, view_angles, _ = base._setup_camera(resolution, view_cfg, view_roll)
    base._fit_ortho_scale(base.bpy.context.scene, cam, root, resolution)
    # cam.data.ortho_scale *= 1.2  # 增加20%的视野范围
    # _tighten_camera_framing(
    #     base.bpy.context.scene,
    #     cam,
    #     root,
    #     target_span_x=0.75,  # 降低这个值,给更多边距
    #     target_span_y=0.60,  # 降低这个值,给更多边距
    # )

    degradation = _sample_degradation(noisy)
    base._set_render_samples(noisy)
    base._apply_motion_blur(degradation["motion_blur"])

    filename = f"sample_{idx:05d}.png"
    image_path = os.path.join(out_dir, "images", filename)
    base._render(image_path)
    bbox_view = _project_bbox(base.bpy.context.scene, cam, root)

    return {
        "id": f"{id_prefix}_{idx:06d}",
        "image": os.path.join("images", filename),
        "task": "analog_gauge_readout",
        "label": {
            "gauge_value": value,
            "pointer_angle_deg": round(pointer_angle_deg, 4),
        },
        "meta": {
            "source": "blender_gauge",
            "benchmark_split": split,
            "render": {
                "resolution": [resolution, resolution],
                "seed": seed,
                "camera": "ortho",
            },
            "view": view_angles,
            "lighting": lighting,
            "degradation": degradation,
            "style_id": style["style_id"],
            "crop_bbox_view": [round(v, 6) for v in bbox_view],
            "tilt_bucket": _bucket_tilt(view_angles["yaw"]),
            "specular_bucket": _bucket_specular(degradation["specular"]),
            "blur_bucket": _bucket_blur(degradation["motion_blur"], degradation["defocus"]),
        },
    }


def main() -> None:
    args = _parse_args()
    out_dir = os.path.join(os.path.abspath(args.out_dir), args.split)
    _ensure_dirs(out_dir)
    out_jsonl = os.path.join(out_dir, "samples.jsonl")
    start_index = 0
    if args.resume and os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            start_index = sum(1 for line in f if line.strip())
    mode = "a" if args.resume else "w"
    with open(out_jsonl, mode, encoding="utf-8") as f:
        for idx in range(start_index, args.n):
            row = _render_one(out_dir, idx, args.split, args.resolution, args.seed + idx, args.id_prefix)
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            if idx % 8 == 0:
                f.flush()


if __name__ == "__main__":
    main()
