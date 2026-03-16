"""Batch render synthetic analog clocks with Blender.

Usage example:
  blender -b -P render_batch.py -- \
    --out_dir /tmp/rege --n 10 --split clean --seed 123 \
    --resolution 512 --style_bank_dir /path/to/styles --difficulty medium
"""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

try:
    import bpy  # type: ignore
    import bmesh  # type: ignore
    from mathutils import Vector  # type: ignore
    from bpy_extras.object_utils import world_to_camera_view  # type: ignore
except Exception:
    bpy = None


DIAL_RADIUS = 1.0
DEFAULT_STYLE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "assets", "styles")
)


def _parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Render analog clock dataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--split", choices=["clean", "noisy", "pair"], default="clean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--style_bank_dir", default=None)
    parser.add_argument("--difficulty", default="medium")
    parser.add_argument("--second_hand_prob", type=float, default=0.5)
    parser.add_argument("--alarm_hand_prob", type=float, default=0.2)
    parser.add_argument("--time_mode", choices=["hm", "hms", "random"], default="random")
    parser.add_argument("--force_hand_config", choices=["2", "3", "4"], default=None)
    parser.add_argument("--max_seconds", type=int, default=59)
    parser.add_argument("--clean_view_mode", choices=["front", "mild"], default="mild")
    parser.add_argument("--view_yaw_min", type=float, default=None)
    parser.add_argument("--view_yaw_max", type=float, default=None)
    parser.add_argument("--view_pitch_min", type=float, default=None)
    parser.add_argument("--view_pitch_max", type=float, default=None)
    parser.add_argument("--view_roll_min", type=float, default=None)
    parser.add_argument("--view_roll_max", type=float, default=None)
    parser.add_argument("--pose_yaw_max", type=float, default=None)
    parser.add_argument("--pose_pitch_max", type=float, default=None)
    parser.add_argument("--pose_roll_max", type=float, default=None)
    parser.add_argument("--pose_x_max", type=float, default=None)
    parser.add_argument("--pose_y_max", type=float, default=None)
    parser.add_argument("--specular_min", type=float, default=None)
    parser.add_argument("--specular_max", type=float, default=None)
    parser.add_argument("--motion_blur_min", type=float, default=None)
    parser.add_argument("--motion_blur_max", type=float, default=None)
    parser.add_argument("--defocus_min", type=float, default=None)
    parser.add_argument("--defocus_max", type=float, default=None)
    parser.add_argument("--env_id_choices", default=None, help="Comma-separated lighting env ids")
    parser.add_argument("--spotcheck", action="store_true")
    parser.add_argument("--spotcheck_split", choices=["clean", "noisy"], default="noisy")
    parser.add_argument("--pair_quota_json", default=None)
    parser.add_argument("--delta_quota_json", default=None)
    parser.add_argument("--id_prefix", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sample_range(
    min_value: Optional[float],
    max_value: Optional[float],
    default_min: float,
    default_max: float,
) -> float:
    low = default_min if min_value is None else min_value
    high = default_max if max_value is None else max_value
    return random.uniform(low, high)


def _color_tuple(values: Optional[List[float]], fallback: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if not values or len(values) != 3:
        return fallback
    return tuple(_clamp01(float(v)) for v in values)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _default_style(style_id: str) -> Dict[str, Any]:
    return {
        "style_id": style_id,
        "dial": {"color": [0.92, 0.92, 0.92], "roughness": 0.55},
        "scale": {"type": "major_minor", "color": [0.2, 0.2, 0.2]},
        "numerals": {"type": "arabic", "subset": "quadrants", "color": [0.15, 0.15, 0.15], "size": 0.18},
        "hands": {
            "hour": {"length": 0.5, "width": 0.08, "color": [0.1, 0.1, 0.1], "shape": "rect"},
            "minute": {"length": 0.8, "width": 0.05, "color": [0.1, 0.1, 0.1], "shape": "rect"},
            "second": {"enabled": False, "length": 0.95, "width": 0.015, "color": [0.8, 0.1, 0.1], "shape": "rect"},
            "alarm": {"enabled": False, "length": 0.35, "width": 0.03, "color": [0.2, 0.2, 0.2], "shape": "triangle"},
        },
        "bezel": {"enabled": False, "color": [0.05, 0.05, 0.05], "thickness": 0.08},
        "glass": {"enabled": False, "roughness": 0.08},
        "background": {"color": [0.96, 0.96, 0.96]},
    }


def _normalize_style(entry: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    style_id = entry.get("style_id") or fallback_id
    merged = _merge_dict(_default_style(style_id), entry)
    merged["style_id"] = style_id
    return merged


def _vary_color(color: Tuple[float, float, float], delta: float) -> Tuple[float, float, float]:
    return tuple(_clamp01(c + random.uniform(-delta, delta)) for c in color)


def _apply_style_variation(style_cfg: Dict[str, Any], noisy: bool) -> Dict[str, Any]:
    jitter = 0.06 if noisy else 0.03
    scale_jitter = 0.15 if noisy else 0.08

    updated = json.loads(json.dumps(style_cfg))
    dial = updated.get("dial", {})
    base_color = _color_tuple(dial.get("face_color") or dial.get("color"), (0.92, 0.92, 0.92))
    dial["face_color"] = _vary_color(base_color, jitter)
    dial["roughness"] = _clamp01(float(dial.get("roughness", 0.55)) * random.uniform(0.9, 1.1))
    if "shape" not in dial:
        dial["shape"] = "square" if random.random() < (0.25 if noisy else 0.15) else "circle"
    if "double_ring" not in dial:
        dial["double_ring"] = random.random() < (0.2 if noisy else 0.1)
    updated["dial"] = dial

    ticks = updated.get("ticks", {})
    if ticks.get("type") not in (None, "none"):
        for key in ("major_len", "minor_len", "major_w", "minor_w"):
            if key in ticks:
                ticks[key] = float(ticks[key]) * random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
    updated["ticks"] = ticks

    hands = updated.get("hands", {})
    for hand_key in ("hour", "minute", "second", "alarm"):
        hand = hands.get(hand_key, {})
        if "length" in hand:
            hand["length"] = float(hand["length"]) * random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        if "width" in hand:
            hand["width"] = float(hand["width"]) * random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
        if "color" in hand:
            hand["color"] = _vary_color(_color_tuple(hand["color"], (0.1, 0.1, 0.1)), jitter)
        hands[hand_key] = hand
    updated["hands"] = hands

    numerals = updated.get("numerals", {})
    if "size" in numerals:
        numerals["size"] = float(numerals["size"]) * random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
    updated["numerals"] = numerals

    bezel = updated.get("bezel", {})
    if "thickness" in bezel:
        bezel["thickness"] = float(bezel["thickness"]) * random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter)
    updated["bezel"] = bezel

    return updated


def _load_style_bank(style_bank_dir: Optional[str]) -> List[Dict[str, Any]]:
    styles: List[Dict[str, Any]] = []
    search_dir = style_bank_dir or DEFAULT_STYLE_DIR
    if search_dir and os.path.isdir(search_dir):
        for name in sorted(os.listdir(search_dir)):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(search_dir, name)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else [data]
            for entry in entries:
                styles.append(_normalize_style(entry, os.path.splitext(name)[0]))
    if not styles:
        styles = [_default_style("default")]
    return styles


def _has_ticks(style_cfg: Dict[str, Any]) -> bool:
    ticks_cfg = style_cfg.get("ticks", {})
    tick_type = ticks_cfg.get("type")
    if tick_type is None:
        tick_type = style_cfg.get("scale", {}).get("type")
    if not tick_type:
        tick_type = "major_minor"
    return tick_type != "none"


def _set_cycles(scene: Any) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.use_nodes = False
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.border_min_x = 0.0
    scene.render.border_min_y = 0.0
    scene.render.border_max_x = 1.0
    scene.render.border_max_y = 1.0


def _make_material(name: str, color: Tuple[float, float, float], specular: float = 0.5, roughness: float = 0.5, metallic: float = 0.0) -> Any:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled = nodes.get("Principled BSDF")

    if principled:
        principled.inputs["Base Color"].default_value = (*color, 1.0)
        blender_version = bpy.app.version
        if blender_version >= (4, 0, 0):
            if "Specular IOR Level" in principled.inputs:
                principled.inputs["Specular IOR Level"].default_value = specular
        else:
            if "Specular" in principled.inputs:
                principled.inputs["Specular"].default_value = specular
        principled.inputs["Roughness"].default_value = roughness
        principled.inputs["Metallic"].default_value = metallic

    return mat


def _make_emissive_material(name: str, color: Tuple[float, float, float], strength: float) -> Any:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = strength
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def _make_glass_material(name: str, roughness: float, ior: float, specular: float, tint: Tuple[float, float, float], bump: float) -> Any:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (*tint, 1.0)
        if "Transmission" in principled.inputs:
            principled.inputs["Transmission"].default_value = 1.0
        elif "Transmission Weight" in principled.inputs:
            principled.inputs["Transmission Weight"].default_value = 1.0
        principled.inputs["Roughness"].default_value = roughness
        if "IOR" in principled.inputs:
            principled.inputs["IOR"].default_value = ior
        blender_version = bpy.app.version
        if blender_version >= (4, 0, 0):
            if "Specular IOR Level" in principled.inputs:
                principled.inputs["Specular IOR Level"].default_value = specular
        else:
            if "Specular" in principled.inputs:
                principled.inputs["Specular"].default_value = specular
    if bump > 0.0:
        noise = nodes.new(type="ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 200.0
        bump_node = nodes.new(type="ShaderNodeBump")
        bump_node.inputs["Strength"].default_value = bump
        links.new(noise.outputs["Fac"], bump_node.inputs["Height"])
        links.new(bump_node.outputs["Normal"], principled.inputs["Normal"])
    return mat


def _clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _build_dial(style_cfg: Dict[str, Any], root: Any, specular_boost: float) -> Any:
    """Create the dial mesh and material."""
    dial_cfg = style_cfg.get("dial", {})
    dial_color = _color_tuple(dial_cfg.get("face_color") or dial_cfg.get("color"), (0.92, 0.92, 0.92))
    roughness = float(dial_cfg.get("roughness", 0.55))
    shape = dial_cfg.get("shape", "circle")
    if shape == "square":
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
        dial = bpy.context.active_object
        dial.scale = (DIAL_RADIUS, DIAL_RADIUS, 0.05)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=DIAL_RADIUS, depth=0.1, location=(0, 0, 0))
        dial = bpy.context.active_object
    dial.name = "Dial"
    dial.parent = root
    dial_mat = _make_material(
        "DialMat",
        dial_color,
        roughness=max(0.05, roughness - 0.2 * specular_boost),
        metallic=0.0,
        specular=0.25 + 0.4 * specular_boost,
    )
    dial.data.materials.append(dial_mat)

    if dial_cfg.get("double_ring", False):
        ring_color = _color_tuple(dial_cfg.get("ring_color"), (0.75, 0.75, 0.75))
        ring_mat = _make_material("RingMat", ring_color, roughness=0.6, metallic=0.0, specular=0.1)
        if shape == "square":
            for scale in (0.98, 0.72):
                bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0.06))
                ring = bpy.context.active_object
                ring.scale = (DIAL_RADIUS * scale, DIAL_RADIUS * scale, 0.01)
                ring.data.materials.append(ring_mat)
                ring.parent = root
        else:
            for radius in (DIAL_RADIUS * 0.98, DIAL_RADIUS * 0.72):
                bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=0.01, location=(0, 0, 0.06))
                ring = bpy.context.active_object
                ring.data.materials.append(ring_mat)
                ring.parent = root
    return dial


def _build_scale(style_cfg: Dict[str, Any], root: Any) -> None:
    """Create tick marks based on scale type."""
    scale_cfg = style_cfg.get("ticks", {})
    scale_type = scale_cfg.get("type") or style_cfg.get("scale", {}).get("type", "major_minor")
    scale_color = _color_tuple(scale_cfg.get("color") or style_cfg.get("scale", {}).get("color"), (0.2, 0.2, 0.2))
    if style_cfg.get("dial", {}).get("shape", "circle") == "square":
        scale_color = _contrast_color_for_square(style_cfg)
        tick_mat = _make_emissive_material("TickMat", scale_color, strength=1.5)
    else:
        tick_mat = _make_material("TickMat", scale_color, roughness=0.5, metallic=0.0, specular=0.1)

    def add_tick(angle: float, length: float, width: float) -> None:
        x = math.sin(angle) * (DIAL_RADIUS * 0.85)
        y = math.cos(angle) * (DIAL_RADIUS * 0.85)
        bpy.ops.mesh.primitive_cube_add(size=1, location=(x, y, 0.07))
        tick = bpy.context.active_object
        tick.scale = (width, length, 0.01)
        tick.rotation_euler[2] = -angle
        tick.data.materials.append(tick_mat)
        tick.parent = root

    if scale_type == "none":
        return
    if scale_type == "dots":
        major_r = float(scale_cfg.get("dot_r_major", 0.03))
        for i in range(12):
            angle = math.radians(i * 30.0)
            x = math.sin(angle) * (DIAL_RADIUS * 0.85)
            y = math.cos(angle) * (DIAL_RADIUS * 0.85)
            bpy.ops.mesh.primitive_cylinder_add(radius=major_r, depth=0.02, location=(x, y, 0.07))
            dot = bpy.context.active_object
            dot.data.materials.append(tick_mat)
            dot.parent = root
        return
    if scale_type == "sparse_5min":
        for i in range(12):
            angle = math.radians(i * 30.0)
            length = float(scale_cfg.get("major_len", 0.12))
            width = float(scale_cfg.get("major_w", 0.02))
            add_tick(angle, length=length, width=width)
        return

    length_major = float(scale_cfg.get("major_len", scale_cfg.get("length_major", 0.12)))
    length_minor = float(scale_cfg.get("minor_len", scale_cfg.get("length_minor", 0.06)))
    width_major = float(scale_cfg.get("major_w", scale_cfg.get("width_major", 0.025)))
    width_minor = float(scale_cfg.get("minor_w", scale_cfg.get("width_minor", 0.015)))

    for i in range(60):
        angle = math.radians(i * 6.0)
        is_major = i % 5 == 0
        if scale_type == "major_only" and not is_major:
            continue
        length = length_major if is_major else length_minor
        width = width_major if is_major else width_minor
        add_tick(angle, length=length, width=width)


def _numeral_label(numeral_type: str, value: int) -> str:
    if numeral_type == "roman":
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
    return str(value)


def _build_numerals(style_cfg: Dict[str, Any], root: Any) -> None:
    """Create numeral text objects around the dial."""
    numerals_cfg = style_cfg.get("numerals", {})
    numeral_type = numerals_cfg.get("type", "none")
    if numeral_type == "none":
        return

    dial_shape = style_cfg.get("dial", {}).get("shape", "circle")
    if numeral_type == "double_ring":
        color = _color_tuple(numerals_cfg.get("color"), (0.15, 0.15, 0.15))
        if dial_shape == "square":
            color = _contrast_color_for_square(style_cfg)
        size = float(numerals_cfg.get("size", 0.18))
        if dial_shape == "square":
            size = max(size, 0.22)
            text_mat = _make_emissive_material("NumeralMat", color, strength=1.5)
        else:
            text_mat = _make_material("NumeralMat", color, roughness=0.6, metallic=0.0, specular=0.1)
        inner = [(12, "12"), (3, "3"), (6, "6"), (9, "9")]
        outer = [(3, "15"), (6, "30"), (9, "45")]
        if dial_shape == "square":
            size *= 0.9
        inner_r = 0.5 if dial_shape == "square" else 0.55
        outer_r = 0.82 if dial_shape == "square" else 0.88
        for i, label in inner:
            angle = math.radians(i * 30.0)
            x = math.sin(angle) * (DIAL_RADIUS * inner_r)
            y = math.cos(angle) * (DIAL_RADIUS * inner_r)
            bpy.ops.object.text_add(location=(x, y, 0.1))
            text_obj = bpy.context.active_object
            text_obj.data.body = label
            text_obj.data.align_x = "CENTER"
            text_obj.data.align_y = "CENTER"
            text_obj.scale = (size, size, size)
            text_obj.data.materials.append(text_mat)
            text_obj.parent = root
        for i, label in outer:
            angle = math.radians(i * 30.0)
            x = math.sin(angle) * (DIAL_RADIUS * outer_r)
            y = math.cos(angle) * (DIAL_RADIUS * outer_r)
            bpy.ops.object.text_add(location=(x, y, 0.1))
            text_obj = bpy.context.active_object
            text_obj.data.body = label
            text_obj.data.align_x = "CENTER"
            text_obj.data.align_y = "CENTER"
            text_obj.scale = (size * 0.9, size * 0.9, size * 0.9)
            text_obj.data.materials.append(text_mat)
            text_obj.parent = root
        return

    color = _color_tuple(numerals_cfg.get("color"), (0.15, 0.15, 0.15))
    if dial_shape == "square":
        color = _contrast_color_for_square(style_cfg)
    size = float(numerals_cfg.get("size", 0.18))
    font = None
    font_path = numerals_cfg.get("font")
    if font_path and os.path.exists(font_path):
        font = bpy.data.fonts.load(font_path)

    subset = numerals_cfg.get("subset", "full")
    if subset == "quadrants":
        indices = [12, 3, 6, 9]
    else:
        indices = list(range(1, 13))

    if dial_shape == "square":
        size = max(size, 0.24)
        text_mat = _make_emissive_material("NumeralMat", color, strength=1.5)
    else:
        text_mat = _make_material("NumeralMat", color, roughness=0.6, metallic=0.0, specular=0.1)
    for i in indices:
        angle = math.radians(i * 30.0)
        x = math.sin(angle) * (DIAL_RADIUS * 0.68)
        y = math.cos(angle) * (DIAL_RADIUS * 0.68)
        bpy.ops.object.text_add(location=(x, y, 0.1))
        text_obj = bpy.context.active_object
        text_obj.data.body = _numeral_label(numeral_type, i)
        text_obj.data.align_x = "CENTER"
        text_obj.data.align_y = "CENTER"
        if font:
            text_obj.data.font = font
        text_obj.scale = (size, size, size)
        text_obj.data.materials.append(text_mat)
        text_obj.parent = root


def _build_hand_rect(name: str, length: float, width: float, z: float, root: Any, material: Any) -> Any:
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, z))
    hand = bpy.context.active_object
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.transform.translate(value=(0, 0.5, 0))
    bpy.ops.object.mode_set(mode="OBJECT")
    hand.scale = (width, length, 0.02)
    hand.name = name
    hand.data.materials.append(material)
    hand.parent = root
    return hand


def _build_hand_triangle(name: str, length: float, width: float, z: float, root: Any, material: Any) -> Any:
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    hand = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(hand)

    bm = bmesh.new()
    v0 = bm.verts.new((0.0, 0.0, 0.0))
    v1 = bm.verts.new((width / 2.0, 0.0, 0.0))
    v2 = bm.verts.new((-width / 2.0, 0.0, 0.0))
    v3 = bm.verts.new((0.0, length, 0.0))
    bm.faces.new([v0, v1, v3])
    bm.faces.new([v0, v3, v2])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    hand.location = (0.0, 0.0, z)
    hand.data.materials.append(material)
    hand.parent = root
    return hand


def _build_hands(style_cfg: Dict[str, Any], root: Any, time_minutes: int, seconds: Optional[int], alarm_seconds: Optional[int]) -> Dict[str, Any]:
    """Create and rotate hour/minute/second/alarm hands for a given time."""
    hands_cfg = style_cfg.get("hands", {})
    hour_cfg = hands_cfg.get("hour", {})
    minute_cfg = hands_cfg.get("minute", {})
    second_cfg = hands_cfg.get("second", {})
    alarm_cfg = hands_cfg.get("alarm", {})
    hour_color = _color_tuple(hour_cfg.get("color") or minute_cfg.get("color"), (0.1, 0.1, 0.1))
    minute_color = _color_tuple(minute_cfg.get("color") or hour_cfg.get("color"), (0.1, 0.1, 0.1))

    hour_length = float(hour_cfg.get("length", 0.5)) * DIAL_RADIUS
    minute_length = float(minute_cfg.get("length", 0.8)) * DIAL_RADIUS
    hour_width = float(hour_cfg.get("width", 0.08))
    minute_width = float(minute_cfg.get("width", 0.05))
    hour_shape = hour_cfg.get("shape", hour_cfg.get("type", "rect"))
    minute_shape = minute_cfg.get("shape", minute_cfg.get("type", "rect"))

    second_enabled = bool(second_cfg.get("enabled", False)) and seconds is not None
    alarm_enabled = bool(alarm_cfg.get("enabled", False)) and alarm_seconds is not None
    second_color = _color_tuple(second_cfg.get("color"), (0.8, 0.1, 0.1))
    alarm_color = _color_tuple(alarm_cfg.get("color"), (0.2, 0.2, 0.2))
    second_length = float(second_cfg.get("length", 0.95)) * DIAL_RADIUS
    alarm_length = float(alarm_cfg.get("length", 0.35)) * DIAL_RADIUS
    second_width = float(second_cfg.get("width", 0.015))
    alarm_width = float(alarm_cfg.get("width", 0.03))
    second_shape = second_cfg.get("shape", second_cfg.get("type", "rect"))
    alarm_shape = alarm_cfg.get("shape", alarm_cfg.get("type", "triangle"))

    max_minute = DIAL_RADIUS * 0.9
    max_second = DIAL_RADIUS * 0.95
    max_alarm = DIAL_RADIUS * 0.6
    minute_length = min(minute_length, max_minute)
    second_length = min(second_length, max_second)
    alarm_length = min(alarm_length, max_alarm)

    if second_enabled and second_length <= minute_length:
        second_length = min(max_second, minute_length * 1.05)
    if second_enabled and second_width >= minute_width:
        second_width = minute_width * 0.5
    if alarm_enabled and alarm_length >= hour_length:
        alarm_length = hour_length * 0.7

    hour_mat = _make_material("HourHandMat", hour_color, roughness=0.35, metallic=0.0, specular=0.25)
    minute_mat = _make_material("MinuteHandMat", minute_color, roughness=0.35, metallic=0.0, specular=0.25)
    second_mat = _make_material("SecondHandMat", second_color, roughness=0.2, metallic=0.0, specular=0.3)
    alarm_mat = _make_material("AlarmHandMat", alarm_color, roughness=0.4, metallic=0.0, specular=0.2)

    if hour_shape == "triangle":
        hour_hand = _build_hand_triangle("HourHand", hour_length, hour_width, 0.12, root, hour_mat)
    else:
        hour_hand = _build_hand_rect("HourHand", hour_length, hour_width, 0.12, root, hour_mat)

    if minute_shape == "triangle":
        minute_hand = _build_hand_triangle("MinuteHand", minute_length, minute_width, 0.135, root, minute_mat)
    else:
        minute_hand = _build_hand_rect("MinuteHand", minute_length, minute_width, 0.135, root, minute_mat)

    second_hand = None
    if second_enabled:
        if second_shape == "triangle":
            second_hand = _build_hand_triangle("SecondHand", second_length, second_width, 0.155, root, second_mat)
        else:
            second_hand = _build_hand_rect("SecondHand", second_length, second_width, 0.155, root, second_mat)

    alarm_hand = None
    if alarm_enabled:
        if alarm_shape == "rect":
            alarm_hand = _build_hand_rect("AlarmHand", alarm_length, alarm_width, 0.11, root, alarm_mat)
        else:
            alarm_hand = _build_hand_triangle("AlarmHand", alarm_length, alarm_width, 0.11, root, alarm_mat)

    minute = time_minutes % 60
    hour = (time_minutes // 60) % 12
    sec = seconds or 0
    minute_angle = math.radians(((minute + sec / 60.0) / 60.0) * 360.0)
    hour_angle = math.radians(((hour + minute / 60.0 + sec / 3600.0) / 12.0) * 360.0)
    minute_hand.rotation_euler[2] = -minute_angle
    hour_hand.rotation_euler[2] = -hour_angle
    if second_hand:
        second_angle = math.radians((seconds / 60.0) * 360.0)
        second_hand.rotation_euler[2] = -second_angle
    if alarm_hand:
        alarm_angle = math.radians(((alarm_seconds % 3600) / 3600.0) * 360.0)
        alarm_hand.rotation_euler[2] = -alarm_angle

    bpy.ops.mesh.primitive_cylinder_add(radius=0.035, depth=0.02, location=(0, 0, 0.16))
    cap = bpy.context.active_object
    cap.data.materials.append(_make_material("CapMat", minute_color, roughness=0.4, metallic=0.0, specular=0.2))
    cap.parent = root

    return {"hour": hour_hand, "minute": minute_hand, "second": second_hand, "alarm": alarm_hand}


def _build_bezel(style_cfg: Dict[str, Any], root: Any) -> None:
    """Create bezel and optional glass cover."""
    bezel_cfg = style_cfg.get("bezel", {})
    if not bezel_cfg.get("enabled", False):
        return
    material_cfg = bezel_cfg.get("material", {})
    bezel_color = _color_tuple(material_cfg.get("color") or bezel_cfg.get("color"), (0.05, 0.05, 0.05))
    thickness = float(bezel_cfg.get("thickness", 0.08))
    metallic = float(material_cfg.get("metallic", 0.6))
    roughness = float(material_cfg.get("roughness", 0.3))

    dial_shape = style_cfg.get("dial", {}).get("shape", "circle")
    if dial_shape == "square":
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0.02))
        bezel = bpy.context.active_object
        bezel.scale = (DIAL_RADIUS + thickness, DIAL_RADIUS + thickness, 0.06)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=DIAL_RADIUS + thickness, depth=0.08, location=(0, 0, 0.02))
        bezel = bpy.context.active_object
    bezel.name = "Bezel"
    if dial_shape != "square":
        bezel.scale = (1.0, 1.0, 0.6)
    bezel.data.materials.append(_make_material("BezelMat", bezel_color, roughness=roughness, metallic=metallic, specular=0.5))
    bezel.parent = root


def _build_glass(style_cfg: Dict[str, Any], root: Any, roughness_override: Optional[float], enabled_override: Optional[bool]) -> None:
    glass_cfg = style_cfg.get("glass", {})
    enabled = glass_cfg.get("enabled", False)
    if enabled_override is not None:
        enabled = enabled_override
    if not enabled:
        return
    roughness = float(glass_cfg.get("roughness", 0.08))
    if roughness_override is not None:
        roughness = roughness_override
    ior = float(glass_cfg.get("ior", 1.45))
    specular = float(glass_cfg.get("specular", 0.8))
    tint = _color_tuple(glass_cfg.get("tint"), (1.0, 1.0, 1.0))
    bump = float(glass_cfg.get("bump", 0.0))

    dial_shape = style_cfg.get("dial", {}).get("shape", "circle")
    if dial_shape == "square":
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0.16))
        glass = bpy.context.active_object
        glass.scale = (DIAL_RADIUS * 0.99, DIAL_RADIUS * 0.99, 0.005)
    else:
        bpy.ops.mesh.primitive_cylinder_add(radius=DIAL_RADIUS * 0.99, depth=0.01, location=(0, 0, 0.16))
        glass = bpy.context.active_object
    glass.name = "Glass"
    glass.data.materials.append(_make_glass_material("GlassMat", roughness, ior, specular, tint, bump))
    glass.parent = root


def _build_clock(
    style_cfg: Dict[str, Any],
    time_minutes: int,
    specular_boost: float,
    pose_jitter: Dict[str, float],
    glass_roughness: Optional[float],
    glass_enabled: Optional[bool],
    seconds: Optional[int],
    alarm_seconds: Optional[int],
) -> Any:
    """Assemble a full clock and return the root object."""
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0, 0, 0))
    root = bpy.context.active_object
    root.name = "ClockRoot"

    root.location = (pose_jitter["x"], pose_jitter["y"], 0.15)
    root.rotation_euler = (
        math.radians(pose_jitter["pitch"]),
        math.radians(pose_jitter["roll"]),
        math.radians(pose_jitter["yaw"]),
    )

    _build_dial(style_cfg, root, specular_boost)
    _build_scale(style_cfg, root)
    _build_numerals(style_cfg, root)
    _build_hands(style_cfg, root, time_minutes, seconds, alarm_seconds)
    _build_bezel(style_cfg, root)
    _build_glass(style_cfg, root, glass_roughness, glass_enabled)

    return root


def _collect_bbox_points(root: Any) -> List[Vector]:
    points: List[Vector] = []
    for obj in [root] + list(root.children_recursive):
        if obj.type not in {"MESH", "CURVE", "FONT", "SURFACE", "META"}:
            continue
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    return points


def _compute_ortho_scale(camera: Any, bbox_points: List[Vector], resolution: int) -> float:
    if not bbox_points:
        return 3.0
    cam_inv = camera.matrix_world.inverted()
    coords = [cam_inv @ pt for pt in bbox_points]
    xs = [c.x for c in coords]
    ys = [c.y for c in coords]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    aspect = resolution / resolution
    base = max(width, height * aspect)
    return base * 1.25 * 1.10


def _fit_ortho_scale(scene: Any, camera: Any, root: Any, resolution: int) -> None:
    bpy.context.view_layer.update()
    bbox_points = _collect_bbox_points(root)
    if not bbox_points:
        return
    camera.data.ortho_scale = _compute_ortho_scale(camera, bbox_points, resolution)

    for _ in range(3):
        outside = False
        bpy.context.view_layer.update()
        for point in bbox_points:
            view = world_to_camera_view(scene, camera, point)
            if view.x < 0.05 or view.x > 0.95 or view.y < 0.05 or view.y > 0.95:
                outside = True
                break
        if not outside:
            return
        camera.data.ortho_scale *= 1.1


def _view_bucket(yaw: float, pitch: float, roll: float) -> str:
    yaw_abs = abs(yaw)
    roll_abs = abs(roll)
    if yaw_abs <= 20:
        yaw_bucket = "front"
    elif yaw_abs <= 40:
        yaw_bucket = "angled"
    else:
        yaw_bucket = "side"

    if pitch <= 40:
        pitch_bucket = "low"
    elif pitch <= 60:
        pitch_bucket = "mid"
    else:
        pitch_bucket = "high"

    roll_bucket = "level" if roll_abs <= 4 else "tilted"
    return f"{yaw_bucket}_{pitch_bucket}_{roll_bucket}"


def _setup_camera(resolution: int, view_cfg: Dict[str, float], roll: float) -> Tuple[Any, Dict[str, float], str]:
    cam = bpy.context.scene.camera
    if cam is None:
        bpy.ops.object.camera_add(location=(0, -2.8, 2.6))
        cam = bpy.context.active_object

    yaw = view_cfg["yaw"]
    pitch = view_cfg["pitch"]

    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    radius = 4.0
    cam.location = (
        radius * math.sin(yaw_rad) * math.cos(pitch_rad),
        -radius * math.cos(yaw_rad) * math.cos(pitch_rad),
        radius * math.sin(pitch_rad),
    )

    cam.data.type = "ORTHO"
    bpy.context.scene.camera = cam
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution

    target = bpy.data.objects.get("ClockRoot")
    if target is not None:
        if cam.constraints.get("TRACK_TO") is None:
            constraint = cam.constraints.new(type="TRACK_TO")
            constraint.name = "TRACK_TO"
        else:
            constraint = cam.constraints.get("TRACK_TO")
        constraint.target = target
        constraint.track_axis = "TRACK_NEGATIVE_Z"
        constraint.up_axis = "UP_Y"
    cam.rotation_euler[2] = math.radians(roll)

    view_bucket = _view_bucket(yaw, pitch, roll)
    return cam, {"yaw": yaw, "pitch": pitch, "roll": roll}, view_bucket


def _add_light(light_type: str, location: Tuple[float, float, float], energy: float, color: Tuple[float, float, float], shape: Optional[str] = None, size: Optional[float] = None, size_y: Optional[float] = None) -> Dict[str, Any]:
    bpy.ops.object.light_add(type=light_type, location=location)
    obj = bpy.context.active_object
    obj.data.energy = energy
    obj.data.color = color
    if light_type == "AREA" and shape:
        obj.data.shape = shape
        if size is not None:
            obj.data.size = size
        if size_y is not None:
            obj.data.size_y = size_y
    return {
        "type": light_type,
        "location": [float(v) for v in location],
        "energy": float(energy),
        "color": [float(v) for v in color],
        "shape": shape,
        "size": float(size) if size is not None else None,
        "size_y": float(size_y) if size_y is not None else None,
    }


def _setup_world(strength: float, color: Tuple[float, float, float]) -> Dict[str, Any]:
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Color"].default_value = (*color, 1.0)
    background.inputs["Strength"].default_value = strength
    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(background.outputs["Background"], output.inputs["Surface"])
    return {"strength": float(strength), "color": [float(c) for c in color]}


def _setup_ground_plane(color: Tuple[float, float, float]) -> None:
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, -0.6))
    plane = bpy.context.active_object
    plane_mat = _make_material("GroundMat", color, roughness=0.85, metallic=0.0, specular=0.0)
    plane.data.materials.append(plane_mat)


def _setup_studio_environment(env_id: str, noisy: bool) -> Dict[str, Any]:
    lights: List[Dict[str, Any]] = []
    strip_size = random.uniform(3.0, 4.8) if noisy else 2.0
    strip_size_y = random.uniform(0.18, 0.35) if noisy else 1.2
    brightness = random.uniform(0.8, 1.25) if noisy else random.uniform(0.9, 1.2)
    world_strength = random.uniform(0.35, 0.75) if noisy else random.uniform(0.55, 0.85)
    shape = "RECTANGLE"
    if env_id in {"studio_softbox_round", "top_light_round"}:
        shape = "DISK"
    if env_id in {"studio_softbox_ellipse", "top_light_ellipse"}:
        shape = "ELLIPSE"

    if env_id in {"studio_softbox", "studio_softbox_round", "studio_softbox_ellipse"}:
        world = _setup_world(world_strength, (0.6, 0.6, 0.6))
        _setup_ground_plane((0.6, 0.6, 0.6))
        lights.append(_add_light("AREA", (2.2, -1.8, 3.0), 220.0 * brightness, (1.0, 1.0, 1.0), shape=shape, size=strip_size, size_y=strip_size_y))
        lights.append(_add_light("AREA", (-2.0, -2.4, 2.0), 120.0 * brightness, (1.0, 1.0, 1.0), shape=shape, size=strip_size * 0.8, size_y=strip_size_y))
        lights.append(_add_light("AREA", (0.0, 2.5, 2.8), 90.0 * brightness, (1.0, 1.0, 1.0), shape=shape, size=strip_size * 0.7, size_y=strip_size_y))
    elif env_id in {"top_light", "top_light_round", "top_light_ellipse"}:
        world = _setup_world(world_strength, (0.5, 0.5, 0.5))
        _setup_ground_plane((0.5, 0.5, 0.5))
        lights.append(_add_light("AREA", (0.0, -0.2, 4.0), 260.0 * brightness, (1.0, 1.0, 1.0), shape=shape, size=strip_size, size_y=strip_size_y))
        lights.append(_add_light("AREA", (-1.5, -2.0, 1.6), 70.0 * brightness, (1.0, 1.0, 1.0), shape=shape, size=strip_size * 0.6, size_y=strip_size_y))
    else:
        world = _setup_world(world_strength * 0.8, (0.35, 0.35, 0.35))
        _setup_ground_plane((0.35, 0.35, 0.35))
        lights.append(_add_light("AREA", (2.8, -1.6, 2.8), 240.0 * brightness, (1.0, 1.0, 1.0), shape="RECTANGLE", size=strip_size * 1.2, size_y=strip_size_y))
        lights.append(_add_light("AREA", (-1.2, -2.4, 1.4), 60.0 * brightness, (1.0, 1.0, 1.0), shape="RECTANGLE", size=strip_size * 0.6, size_y=strip_size_y))

    return {"env_id": env_id, "lights": lights, "world": world}


def _render(output_path: str) -> None:
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def _time_hhmm(time_minutes: int) -> str:
    hh = (time_minutes // 60) % 12
    mm = time_minutes % 60
    if hh == 0:
        hh = 12
    return f"{hh:02d}:{mm:02d}"


def _random_degradation(noisy: bool, args: argparse.Namespace) -> Dict[str, float]:
    if not noisy and all(
        value is None
        for value in (
            args.specular_min,
            args.specular_max,
            args.motion_blur_min,
            args.motion_blur_max,
            args.defocus_min,
            args.defocus_max,
        )
    ):
        return {"specular": 0.0, "motion_blur": 0.0, "defocus": 0.0}
    return {
        "specular": _clamp01(_sample_range(args.specular_min, args.specular_max, 0.3 if noisy else 0.0, 1.0 if noisy else 0.0)),
        "motion_blur": _clamp01(_sample_range(args.motion_blur_min, args.motion_blur_max, 0.0, 1.0 if noisy else 0.0)),
        "defocus": _clamp01(_sample_range(args.defocus_min, args.defocus_max, 0.0, 1.0 if noisy else 0.0)),
    }


def _delta_from_difficulty(name: str) -> int:
    if name == "easy":
        return random.choice([random.randint(60, 360), -random.randint(60, 360)])
    if name == "hard":
        return random.choice([random.randint(1, 30), -random.randint(1, 30)])
    return random.choice([random.randint(10, 180), -random.randint(10, 180)])


def _build_pair_type_list(n: int, pair_quota_json: Optional[str]) -> List[str]:
    if not pair_quota_json:
        return []
    quotas = json.loads(pair_quota_json)
    if not isinstance(quotas, dict):
        raise ValueError("pair_quota_json must be a JSON object")
    pair_types: List[str] = []
    for key, count in quotas.items():
        pair_types.extend([key] * int(count))
    if len(pair_types) != n:
        raise ValueError(f"pair_quota_json total {len(pair_types)} does not match n={n}")
    random.shuffle(pair_types)
    return pair_types


def _build_delta_list(n: int, delta_quota_json: Optional[str]) -> Tuple[List[int], Optional[int]]:
    if not delta_quota_json:
        return [], None
    quotas = json.loads(delta_quota_json)
    if not isinstance(quotas, dict):
        raise ValueError("delta_quota_json must be a JSON object")
    hard_n = int(quotas.get("hard_n", 0))
    easy_n = int(quotas.get("easy_n", 0))
    hard_min = int(quotas.get("hard_min", 1))
    hard_max = int(quotas.get("hard_max", 5))
    easy_min = int(quotas.get("easy_min", 30))
    easy_max = int(quotas.get("easy_max", 180))
    deltas: List[int] = []
    for _ in range(hard_n):
        delta = random.randint(hard_min, hard_max)
        deltas.append(delta if random.random() < 0.5 else -delta)
    for _ in range(easy_n):
        delta = random.randint(easy_min, easy_max)
        deltas.append(delta if random.random() < 0.5 else -delta)
    if len(deltas) != n:
        raise ValueError(f"delta_quota_json total {len(deltas)} does not match n={n}")
    random.shuffle(deltas)
    return deltas, hard_max


def _apply_motion_blur(level: float) -> None:
    scene = bpy.context.scene
    scene.render.use_motion_blur = level > 0.05
    scene.render.motion_blur_shutter = 0.2 + level * 0.6


def _set_render_samples(noisy: bool) -> None:
    scene = bpy.context.scene
    scene.cycles.samples = 96 if noisy else 128


def _pose_jitter(noisy: bool, args: argparse.Namespace) -> Dict[str, float]:
    if noisy or any(
        value is not None
        for value in (args.pose_yaw_max, args.pose_pitch_max, args.pose_roll_max, args.pose_x_max, args.pose_y_max)
    ):
        return {
            "yaw": random.uniform(-(args.pose_yaw_max or 15.0), args.pose_yaw_max or 15.0),
            "pitch": random.uniform(-(args.pose_pitch_max or 15.0), args.pose_pitch_max or 15.0),
            "roll": random.uniform(-(args.pose_roll_max or 15.0), args.pose_roll_max or 15.0),
            "x": random.uniform(-(args.pose_x_max or 0.05), args.pose_x_max or 0.05),
            "y": random.uniform(-(args.pose_y_max or 0.05), args.pose_y_max or 0.05),
        }
    return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "x": 0.0, "y": 0.0}


def _ensure_min_numerals(style_cfg: Dict[str, Any]) -> Dict[str, Any]:
    numerals = dict(style_cfg.get("numerals", {}))
    if style_cfg.get("dial", {}).get("double_ring"):
        if numerals.get("type") in (None, "none"):
            numerals["type"] = "double_ring"
    if numerals.get("type") == "none" or not numerals.get("type"):
        numerals["type"] = "arabic"
        numerals["subset"] = "quadrants"
        numerals["size"] = numerals.get("size", 0.2)
        numerals["color"] = numerals.get("color") or style_cfg.get("ticks", {}).get("color") or [0.2, 0.2, 0.2]
    updated = dict(style_cfg)
    updated["numerals"] = numerals
    return updated


def _ensure_contrast(style_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dial_cfg = style_cfg.get("dial", {})
    if dial_cfg.get("shape", "circle") != "square":
        return style_cfg
    dial_color = _color_tuple(dial_cfg.get("face_color") or dial_cfg.get("color"), (0.92, 0.92, 0.92))
    dial_lum = 0.2126 * dial_color[0] + 0.7152 * dial_color[1] + 0.0722 * dial_color[2]
    contrast_color = [0.1, 0.1, 0.1] if dial_lum >= 0.5 else [0.95, 0.95, 0.95]

    def adjust_color(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
        if key not in cfg:
            return cfg
        cfg[key] = contrast_color
        return cfg

    updated = dict(style_cfg)
    numerals = dict(updated.get("numerals", {}))
    if numerals.get("type") not in (None, "", "none"):
        numerals = adjust_color(numerals, "color")
    updated["numerals"] = numerals
    ticks = dict(updated.get("ticks", {}))
    if ticks.get("type") not in (None, "", "none"):
        ticks = adjust_color(ticks, "color")
    updated["ticks"] = ticks
    return updated


def _contrast_color_for_square(style_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    dial_cfg = style_cfg.get("dial", {})
    dial_color = _color_tuple(dial_cfg.get("face_color") or dial_cfg.get("color"), (0.92, 0.92, 0.92))
    dial_lum = 0.2126 * dial_color[0] + 0.7152 * dial_color[1] + 0.0722 * dial_color[2]
    return (0.08, 0.08, 0.08) if dial_lum >= 0.5 else (0.95, 0.95, 0.95)


def _ensure_square_legibility(style_cfg: Dict[str, Any]) -> Dict[str, Any]:
    if style_cfg.get("dial", {}).get("shape", "circle") != "square":
        return style_cfg
    updated = dict(style_cfg)
    contrast = _contrast_color_for_square(style_cfg)

    numerals = dict(updated.get("numerals", {}))
    if numerals.get("type") in (None, "", "none"):
        numerals["type"] = "arabic"
        numerals["subset"] = "quadrants"
    numerals["color"] = contrast
    numerals["size"] = max(float(numerals.get("size", 0.2)), 0.26)
    updated["numerals"] = numerals

    ticks = dict(updated.get("ticks", {}))
    if ticks.get("type") in (None, "", "none"):
        ticks["type"] = "major_only"
        ticks.setdefault("major_len", 0.12)
        ticks.setdefault("major_w", 0.02)
    ticks["color"] = contrast
    updated["ticks"] = ticks

    hands = dict(updated.get("hands", {}))
    for key in ("hour", "minute"):
        hand = dict(hands.get(key, {}))
        hand["color"] = contrast
        hands[key] = hand
    updated["hands"] = hands
    return updated


def _ensure_hand_contrast(style_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dial_cfg = style_cfg.get("dial", {})
    dial_color = _color_tuple(dial_cfg.get("face_color") or dial_cfg.get("color"), (0.92, 0.92, 0.92))
    dial_lum = 0.2126 * dial_color[0] + 0.7152 * dial_color[1] + 0.0722 * dial_color[2]
    contrast = [0.08, 0.08, 0.08] if dial_lum >= 0.5 else [0.95, 0.95, 0.95]

    def normalize_hand(cfg: Dict[str, Any], min_len: float, min_w: float) -> Dict[str, Any]:
        color = _color_tuple(cfg.get("color"), contrast)
        lum = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
        if abs(lum - dial_lum) < 0.35:
            cfg["color"] = contrast
        cfg["length"] = max(float(cfg.get("length", min_len)), min_len)
        cfg["width"] = max(float(cfg.get("width", min_w)), min_w)
        return cfg

    updated = dict(style_cfg)
    hands = dict(updated.get("hands", {}))
    hands["hour"] = normalize_hand(dict(hands.get("hour", {})), min_len=0.45, min_w=0.06)
    hands["minute"] = normalize_hand(dict(hands.get("minute", {})), min_len=0.7, min_w=0.035)
    hands["second"] = normalize_hand(dict(hands.get("second", {})), min_len=0.88, min_w=0.012)
    hands["alarm"] = normalize_hand(dict(hands.get("alarm", {})), min_len=0.3, min_w=0.025)
    updated["hands"] = hands
    return updated


def _apply_hand_config(style_cfg: Dict[str, Any], hand_config: int, has_second: bool, has_alarm: bool) -> Dict[str, Any]:
    updated = dict(style_cfg)
    hands = dict(updated.get("hands", {}))
    second_cfg = dict(hands.get("second", {}))
    alarm_cfg = dict(hands.get("alarm", {}))
    second_cfg["enabled"] = bool(hand_config >= 3 and has_second)
    alarm_cfg["enabled"] = bool(hand_config >= 4 and has_alarm)
    hands["second"] = second_cfg
    hands["alarm"] = alarm_cfg
    updated["hands"] = hands
    return updated


def _build_time_label(time_minutes: int, seconds: Optional[int]) -> Dict[str, Any]:
    label = {"time_hhmm": _time_hhmm(time_minutes), "time_minutes": time_minutes, "seconds": None}
    if seconds is None:
        return label
    total_seconds = (time_minutes * 60 + seconds) % (12 * 3600)
    label["time_hhmmss"] = _time_hhmmss(time_minutes, seconds)
    label["time_seconds_total"] = total_seconds
    label["seconds"] = seconds
    return label


def _view_config(noisy: bool, clean_view_mode: str, args: argparse.Namespace) -> Tuple[Dict[str, float], float]:
    if any(
        value is not None
        for value in (
            args.view_yaw_min,
            args.view_yaw_max,
            args.view_pitch_min,
            args.view_pitch_max,
            args.view_roll_min,
            args.view_roll_max,
        )
    ):
        return {
            "yaw": _sample_range(args.view_yaw_min, args.view_yaw_max, -60.0 if noisy else -25.0, 60.0 if noisy else 25.0),
            "pitch": _sample_range(args.view_pitch_min, args.view_pitch_max, 25.0 if noisy else 30.0, 75.0 if noisy else 60.0),
        }, _sample_range(args.view_roll_min, args.view_roll_max, -8.0 if noisy else -2.0, 8.0 if noisy else 2.0)
    if noisy:
        return {
            "yaw": random.uniform(-60.0, 60.0),
            "pitch": random.uniform(25.0, 75.0),
        }, random.uniform(-8.0, 8.0)
    if clean_view_mode == "front":
        return {"yaw": 0.0, "pitch": 90.0}, 0.0
    return {"yaw": random.uniform(-25.0, 25.0), "pitch": random.uniform(30.0, 60.0)}, random.uniform(-2.0, 2.0)


def _select_hand_config(force: Optional[str], noisy: bool, second_prob: float, alarm_prob: float) -> Tuple[int, bool, bool]:
    if force:
        hand_config = int(force)
        return hand_config, hand_config >= 3, hand_config >= 4
    has_second = random.random() < second_prob
    has_alarm = has_second and (random.random() < alarm_prob)
    if not noisy:
        has_alarm = has_alarm and random.random() < 0.5
    hand_config = 4 if has_alarm else (3 if has_second else 2)
    return hand_config, has_second, has_alarm


def _sample_time(time_mode: str, hand_config: int, max_seconds: int) -> Tuple[int, int]:
    time_minutes = random.randint(0, 719)
    if time_mode == "hm":
        return time_minutes, 0
    if time_mode == "hms":
        seconds = random.randint(0, max(0, max_seconds))
        return time_minutes, seconds
    if hand_config >= 3:
        seconds = random.randint(0, max(0, max_seconds))
        return time_minutes, seconds
    return time_minutes, 0


def _time_hhmmss(time_minutes: int, seconds: int) -> str:
    hh = (time_minutes // 60) % 12
    mm = time_minutes % 60
    if hh == 0:
        hh = 12
    return f"{hh:02d}:{mm:02d}:{seconds:02d}"


def _render_single(
    out_dir: str,
    idx: int,
    style_cfg: Dict[str, Any],
    resolution: int,
    time_minutes: Optional[int],
    seconds: Optional[int],
    seed: int,
    noisy: bool,
    id_prefix: str,
    second_hand_prob: float,
    alarm_hand_prob: float,
    time_mode: str,
    force_hand_config: Optional[str],
    max_seconds: int,
    clean_view_mode: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    _clear_scene()
    _set_cycles(bpy.context.scene)
    bpy.context.scene.cycles.seed = seed

    degradation = _random_degradation(noisy, args)
    hand_config, has_second, has_alarm = _select_hand_config(force_hand_config, noisy, second_hand_prob, alarm_hand_prob)
    if time_minutes is None:
        time_minutes, seconds = _sample_time(time_mode, hand_config, max_seconds)
    if seconds is None:
        seconds = _sample_time(time_mode, hand_config, max_seconds)[1]
    style_cfg = _apply_style_variation(style_cfg, noisy=noisy)
    style_cfg = _ensure_min_numerals(style_cfg)
    style_cfg = _ensure_contrast(style_cfg)
    style_cfg = _ensure_square_legibility(style_cfg)
    style_cfg = _ensure_hand_contrast(style_cfg)
    style_cfg = _apply_hand_config(style_cfg, hand_config, has_second, has_alarm)
    pose_jitter = _pose_jitter(noisy, args)
    view_cfg, view_roll = _view_config(noisy, clean_view_mode, args)

    glass_enabled = noisy
    if noisy:
        glass_variant = random.choice(
            [
                {"roughness": 0.02, "ior": 1.45, "specular": 0.9, "tint": [1.0, 1.0, 1.0], "bump": 0.05},
                {"roughness": 0.04, "ior": 1.5, "specular": 0.8, "tint": [0.95, 0.98, 1.0], "bump": 0.03},
                {"roughness": 0.06, "ior": 1.4, "specular": 0.7, "tint": [1.0, 0.98, 0.95], "bump": 0.02},
            ]
        )
        style_cfg = dict(style_cfg)
        glass_cfg = dict(style_cfg.get("glass", {}))
        glass_cfg.update(glass_variant)
        style_cfg["glass"] = glass_cfg
        glass_roughness = glass_cfg["roughness"]
    else:
        glass_roughness = 0.08
    root = _build_clock(
        style_cfg,
        time_minutes,
        specular_boost=degradation["specular"],
        pose_jitter=pose_jitter,
        glass_roughness=glass_roughness,
        glass_enabled=glass_enabled,
        seconds=seconds if has_second else None,
        alarm_seconds=(time_minutes * 60 + seconds) if has_alarm else None,
    )

    env_choices = [
        item.strip()
        for item in (args.env_id_choices or "").split(",")
        if item.strip()
    ]
    env_id = "studio_softbox" if not noisy else random.choice(
        env_choices
        or [
            "studio_softbox",
            "studio_softbox_round",
            "studio_softbox_ellipse",
            "top_light",
            "top_light_round",
            "top_light_ellipse",
            "window_side",
        ]
    )
    lighting = _setup_studio_environment(env_id, noisy=noisy)

    cam, view_angles, view_bucket = _setup_camera(resolution, view_cfg, view_roll)
    _fit_ortho_scale(bpy.context.scene, cam, root, resolution)

    _set_render_samples(noisy)
    _apply_motion_blur(degradation["motion_blur"])

    filename = f"sample_{idx:05d}.png"
    image_path = os.path.join(out_dir, "images", filename)
    _render(image_path)

    return {
        "id": f"{id_prefix}_{idx:06d}",
        "image": os.path.join("images", filename),
        "task": "clock_readout",
        "label": _build_time_label(time_minutes, seconds if has_second else None),
        "meta": {
            "domain": "synthetic",
            "style_id": style_cfg["style_id"],
            "clock_type": "analog",
            "render": {
                "resolution": [resolution, resolution],
                "seed": seed,
                "camera": "ortho",
            },
            "view": view_angles,
            "view_bucket": view_bucket,
            "lighting": lighting,
            "pose": pose_jitter,
            "degradation": degradation,
            "hand_config": hand_config,
            "has_second": has_second,
            "has_alarm": has_alarm,
        },
    }


def _render_pair(
    out_dir: str,
    idx: int,
    style_a: Dict[str, Any],
    style_b: Dict[str, Any],
    resolution: int,
    time_a: int,
    time_b: int,
    seed: int,
    pair_type: str,
    timezone_a: str,
    timezone_b: str,
    delta_minutes: int,
    nearby_times: bool,
    cross_timezone: bool,
    id_prefix: str,
    second_hand_prob: float,
    alarm_hand_prob: float,
    time_mode: str,
    force_hand_config: Optional[str],
    max_seconds: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    sample_a = _render_single(
        out_dir,
        idx * 2,
        style_a,
        resolution,
        time_a,
        None,
        seed,
        noisy=False,
        id_prefix=f"{id_prefix}_a",
        second_hand_prob=second_hand_prob,
        alarm_hand_prob=alarm_hand_prob,
        time_mode=time_mode,
        force_hand_config=force_hand_config,
        max_seconds=max_seconds,
        clean_view_mode="front",
        args=args,
    )
    sample_b = _render_single(
        out_dir,
        idx * 2 + 1,
        style_b,
        resolution,
        time_b,
        None,
        seed + 1,
        noisy=False,
        id_prefix=f"{id_prefix}_b",
        second_hand_prob=second_hand_prob,
        alarm_hand_prob=alarm_hand_prob,
        time_mode=time_mode,
        force_hand_config=force_hand_config,
        max_seconds=max_seconds,
        clean_view_mode="front",
        args=args,
    )

    image_a = sample_a["image"].replace("sample_", "pair_").replace(".png", "_a.png")
    image_b = sample_b["image"].replace("sample_", "pair_").replace(".png", "_b.png")

    os.rename(os.path.join(out_dir, sample_a["image"]), os.path.join(out_dir, image_a))
    os.rename(os.path.join(out_dir, sample_b["image"]), os.path.join(out_dir, image_b))

    return {
        "id": f"{id_prefix}_{idx:06d}",
        "image_a": image_a,
        "image_b": image_b,
        "task": "clock_delta",
        "label": {
            "time_a_hhmm": _time_hhmm(time_a),
            "time_b_hhmm": _time_hhmm(time_b),
            "delta_minutes": delta_minutes,
            "timezone_a": timezone_a,
            "timezone_b": timezone_b,
        },
        "meta": {
            "pair_type": pair_type,
            "style_id_a": style_a["style_id"],
            "style_id_b": style_b["style_id"],
            "difficulty": {
                "nearby_times": nearby_times,
                "cross_timezone": cross_timezone,
            },
        },
    }


def _ensure_dirs(out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)


def _spotcheck(
    out_dir: str,
    styles: List[Dict[str, Any]],
    resolution: int,
    seed: int,
    noisy: bool,
    clean_view_mode: str,
    args: argparse.Namespace,
) -> None:
    rng = random.Random(seed)
    envs = [
        "studio_softbox",
        "studio_softbox_round",
        "studio_softbox_ellipse",
        "top_light",
        "top_light_round",
        "top_light_ellipse",
        "window_side",
    ]
    images_dir = os.path.join(out_dir, "spotcheck", "images")
    os.makedirs(images_dir, exist_ok=True)
    annotations_path = os.path.join(out_dir, "spotcheck", "annotations.jsonl")

    with open(annotations_path, "w", encoding="utf-8") as f:
        idx = 0
        for env_id in envs:
            for _ in range(4):
                style_cfg = _apply_style_variation(rng.choice(styles), noisy=noisy)
                _clear_scene()
                _set_cycles(bpy.context.scene)
                bpy.context.scene.cycles.seed = seed + idx
                pose_jitter = _pose_jitter(noisy=noisy, args=args)
                view_cfg, view_roll = _view_config(noisy=noisy, clean_view_mode=clean_view_mode, args=args)
                style_cfg = _ensure_min_numerals(style_cfg)
                style_cfg = _ensure_contrast(style_cfg)
                style_cfg = _ensure_square_legibility(style_cfg)
                style_cfg = _ensure_hand_contrast(style_cfg)
                hand_config = [2, 3, 4][idx % 3]
                has_second = hand_config >= 3
                has_alarm = hand_config >= 4
                style_cfg = _apply_hand_config(style_cfg, hand_config, has_second, has_alarm)
                glass_roughness = None
                glass_enabled = False
                if noisy:
                    glass_variant = random.choice(
                        [
                            {"roughness": 0.02, "ior": 1.45, "specular": 0.9, "tint": [1.0, 1.0, 1.0], "bump": 0.05},
                            {"roughness": 0.04, "ior": 1.5, "specular": 0.8, "tint": [0.95, 0.98, 1.0], "bump": 0.03},
                            {"roughness": 0.06, "ior": 1.4, "specular": 0.7, "tint": [1.0, 0.98, 0.95], "bump": 0.02},
                        ]
                    )
                    style_cfg = dict(style_cfg)
                    glass_cfg = dict(style_cfg.get("glass", {}))
                    glass_cfg.update(glass_variant)
                    style_cfg["glass"] = glass_cfg
                    glass_roughness = glass_cfg["roughness"]
                    glass_enabled = True
                root = _build_clock(
                    style_cfg,
                    10 * 60 + 10,
                    specular_boost=0.5 if noisy else 0.0,
                    pose_jitter=pose_jitter,
                    glass_roughness=glass_roughness,
                    glass_enabled=glass_enabled,
                    seconds=10 if has_second else None,
                    alarm_seconds=(10 * 60 + 10) if has_alarm else None,
                )
                lighting = _setup_studio_environment(env_id, noisy=noisy)
                cam, view_angles, view_bucket = _setup_camera(resolution, view_cfg, view_roll)
                _fit_ortho_scale(bpy.context.scene, cam, root, resolution)
                _set_render_samples(noisy)
                _apply_motion_blur(0.0)

                filename = f"spot_{idx:03d}.png"
                image_path = os.path.join(images_dir, filename)
                _render(image_path)

                row = {
                    "id": f"spot_{idx:03d}",
                    "image": os.path.join("images", filename),
                    "task": "clock_readout",
                    "label": _build_time_label(10 * 60 + 10, 10 if has_second else None),
                    "meta": {
                        "domain": "synthetic",
                        "style_id": style_cfg["style_id"],
                        "render": {"resolution": [resolution, resolution], "seed": seed + idx},
                        "view": view_angles,
                        "view_bucket": view_bucket,
                        "lighting": lighting,
                        "pose": pose_jitter,
                        "hand_config": hand_config,
                        "has_second": has_second,
                        "has_alarm": has_alarm,
                    },
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                idx += 1


def main() -> None:
    if bpy is None:
        raise RuntimeError("This script must be run with Blender's Python (bpy unavailable).")

    args = _parse_args()
    random.seed(args.seed)

    out_dir = os.path.join(args.out_dir, args.split)
    _ensure_dirs(out_dir)

    styles = [s for s in _load_style_bank(args.style_bank_dir) if _has_ticks(s)]

    if args.spotcheck:
        _spotcheck(
            args.out_dir,
            styles,
            args.resolution,
            args.seed,
            noisy=args.spotcheck_split == "noisy",
            clean_view_mode=args.clean_view_mode,
            args=args,
        )
        return

    if args.n <= 0:
        raise ValueError("--n must be > 0 when not using --spotcheck")

    annotations: List[Dict[str, Any]] = []

    if args.split in {"clean", "noisy"}:
        second_prob = args.second_hand_prob
        alarm_prob = args.alarm_hand_prob
        if args.split == "clean":
            alarm_prob = min(alarm_prob, 0.1)
        out_jsonl = os.path.join(out_dir, "samples.jsonl")
        start_index = 0
        if args.resume and os.path.exists(out_jsonl):
            with open(out_jsonl, "r", encoding="utf-8") as f:
                start_index = sum(1 for _ in f if _.strip())
        mode = "a" if args.resume else "w"
        with open(out_jsonl, mode, encoding="utf-8") as f:
            for idx in range(start_index, args.n):
                random.seed(args.seed + idx)
                style_cfg = _apply_style_variation(random.choice(styles), noisy=args.split == "noisy")
                row = _render_single(
                    out_dir,
                    idx,
                    style_cfg,
                    args.resolution,
                    None,
                    None,
                    args.seed + idx,
                    noisy=args.split == "noisy",
                    id_prefix=args.id_prefix or f"rege_{args.split}_blender",
                    second_hand_prob=second_prob,
                    alarm_hand_prob=alarm_prob,
                    time_mode=args.time_mode,
                    force_hand_config=args.force_hand_config,
                    max_seconds=args.max_seconds,
                    clean_view_mode=args.clean_view_mode,
                    args=args,
                )
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
                if idx % 50 == 0:
                    f.flush()
        return

    pair_types = [
        "same_style_same_tz",
        "cross_style_same_tz",
        "same_style_cross_tz",
        "cross_style_cross_tz",
    ]
    timezones = ["UTC-8", "UTC-5", "UTC+0", "UTC+1", "UTC+3", "UTC+8"]

    pair_type_list = _build_pair_type_list(args.n, args.pair_quota_json)
    delta_list, hard_max = _build_delta_list(args.n, args.delta_quota_json)

    for idx in range(args.n):
        pair_type = pair_type_list[idx] if pair_type_list else random.choice(pair_types)
        same_style = "same_style" in pair_type
        cross_tz = "cross_tz" in pair_type
        style_a = random.choice(styles)
        if same_style:
            style_b = style_a
        else:
            style_b = random.choice(styles)
            if style_b["style_id"] == style_a["style_id"] and len(styles) > 1:
                style_b = random.choice([s for s in styles if s["style_id"] != style_a["style_id"]])

        if delta_list:
            delta = delta_list[idx]
            nearby_times = abs(delta) <= (hard_max or 15)
        else:
            delta = _delta_from_difficulty(args.difficulty)
            nearby_times = abs(delta) <= 15
        time_a = random.randint(0, 719)
        time_b = (time_a + delta) % 720
        timezone_a = random.choice(timezones)
        timezone_b = timezone_a if not cross_tz else random.choice([tz for tz in timezones if tz != timezone_a])

        annotations.append(
            _render_pair(
                out_dir,
                idx,
                style_a,
                style_b,
                args.resolution,
                time_a,
                time_b,
                args.seed + idx,
                pair_type,
                timezone_a,
                timezone_b,
                delta,
                nearby_times=nearby_times,
                cross_timezone=cross_tz,
                id_prefix=args.id_prefix or "rege_pair_blender",
                second_hand_prob=args.second_hand_prob,
                alarm_hand_prob=min(args.alarm_hand_prob, 0.1),
                time_mode="hm",
                force_hand_config="2",
                max_seconds=args.max_seconds,
                args=args,
            )
        )

    out_jsonl = os.path.join(out_dir, "pairs.jsonl")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in annotations:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
