"""Fast wrapper for render_batch.py (GPU + low samples).

Usage example:
  blender -b -P render_batch_fast.py -- \
    --out_dir /tmp/rege --n 100 --split clean --seed 1 --resolution 512
"""

import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import render_batch as base  # type: ignore


def _enable_gpu(scene: Any) -> str:
    try:
        prefs = base.bpy.context.preferences.addons["cycles"].preferences
    except Exception:
        scene.cycles.device = "CPU"
        return "CPU"

    device_type = "NONE"
    for candidate in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = candidate
            device_type = candidate
            break
        except Exception:
            continue

    if device_type == "NONE":
        scene.cycles.device = "CPU"
        return "CPU"

    for device in prefs.devices:
        device.use = True
    scene.cycles.device = "GPU"
    return device_type


def _fast_set_cycles(scene: Any) -> None:
    scene.render.engine = "CYCLES"
    scene.use_nodes = False
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.border_min_x = 0.0
    scene.render.border_min_y = 0.0
    scene.render.border_max_x = 1.0
    scene.render.border_max_y = 1.0
    scene.render.use_persistent_data = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.transparent_max_bounces = 4
    _enable_gpu(scene)


def _fast_set_render_samples(noisy: bool) -> None:
    scene = base.bpy.context.scene
    scene.cycles.samples = 32 if noisy else 24
    scene.cycles.use_denoising = True
    if hasattr(scene.cycles, "denoiser"):
        scene.cycles.denoiser = "OPTIX"


def main() -> None:
    base._set_cycles = _fast_set_cycles  # type: ignore
    base._set_render_samples = _fast_set_render_samples  # type: ignore
    base.main()


if __name__ == "__main__":
    main()
