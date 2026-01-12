# rege-bench

rege-bench is a publishable benchmark repo for precise analog clock readout and pairwise delta reasoning. It includes synthetic data generators (Blender and Matplotlib), JSONL annotation schemas, validation utilities, and evaluation scripts.

## Repository layout

```
rege-bench/
  schemas/
  data/
  tools/
    generate/
      blender/
        render_batch.py
        assets/styles/
      matplot/
        render_matplot_batch.py
        assets/matplot_styles.json
    validate/
      validate_annotations.py
    eval/
      parse_time.py
      eval_single.py
      eval_pair.py
    prompts/
```

## Annotation schemas

- `schemas/sample_v1.json`
  - Single-image clock readout
  - `label.time_hhmm` + `label.time_minutes`
  - `meta` includes style id, render settings, and degradation knobs

- `schemas/pair_v1.json`
  - Pairwise clock delta reasoning
  - `label.time_a_hhmm`, `label.time_b_hhmm`, `label.delta_minutes`, `timezone_a`, `timezone_b`
  - `meta` includes pair type, style ids, and difficulty flags

## Blender generator

`tools/generate/blender/render_batch.py` builds a simple analog clock with:
- Dial, ticks, numerals, hands, bezel, and optional glass
- PBR materials (opaque dial, metallic bezel, glass)
- ORTHO camera with auto-framing and safety checks (prevents cropping)
- Fixed studio environments (no floating objects)
- Clean vs noisy policy:
  - Clean: fixed studio environment, mild camera angles, no blur/noise
  - Noisy: multiple environments, wider view angles, optional defocus/motion blur, stronger specular via glass

### Lighting environments (PBR)
- `studio_softbox` / `studio_softbox_round` / `studio_softbox_ellipse`
- `top_light` / `top_light_round` / `top_light_ellipse`
- `window_side`

Each render records `meta.lighting.env_id` and light parameters.

### Style bank

Blender styles live at:

```
rege-bench/tools/generate/blender/assets/styles/
```

Styles define dial, ticks, numerals, hands, bezel, and glass. Unknown fields are ignored with defaults. Styles with `ticks.type = none` are skipped to ensure minimum readability.

### Commands

Clean split:

```
blender -b -P rege-bench/tools/generate/blender/render_batch.py -- \
  --out_dir /tmp/rege --n 100 --split clean --seed 1 --resolution 512
```

Noisy split:

```
blender -b -P rege-bench/tools/generate/blender/render_batch.py -- \
  --out_dir /tmp/rege --n 100 --split noisy --seed 1 --resolution 512
```

Spotcheck (12 images of time 10:10 across envs/views):

```
blender -b -P rege-bench/tools/generate/blender/render_batch.py -- \
  --out_dir /tmp/rege --n 1 --split clean --seed 1 --resolution 512 --spotcheck
```

## Matplotlib generator

`tools/generate/matplot/render_matplot_batch.py` provides a pure‑python generator:
- 20 styles defined in `tools/generate/matplot/assets/matplot_styles.json`
- Uniform style distribution by default
- Outputs PNGs + `annotations.jsonl` under `rege_clean_matplot/`

Example:

```
python rege-bench/tools/generate/matplot/render_matplot_batch.py \
  --out_dir /tmp/rege --n 100 --seed 123 --resolution 512
```

## Validation

`tools/validate/validate_annotations.py` checks:
- Image files exist
- `time_hhmm` matches `time_minutes`
- Valid ranges for time and delta
- Reports counts per style and degradation bucket

Example:

```
python rege-bench/tools/validate/validate_annotations.py \
  --jsonl /tmp/rege/clean/samples.jsonl --images_root /tmp/rege/clean
```

## Evaluation

- `tools/eval/parse_time.py`: robust regex parsing for HH:MM and delta minutes
- `tools/eval/eval_single.py`: exact / tol_1min / tol_5min / MAE
- `tools/eval/eval_pair.py`: exact_delta / MAE_delta

Example:

```
python rege-bench/tools/eval/eval_single.py \
  --gt_jsonl /tmp/rege/clean/samples.jsonl --pred_jsonl /path/to/preds.jsonl
```

```
python rege-bench/tools/eval/eval_pair.py \
  --gt_jsonl /tmp/rege/pair/pairs.jsonl --pred_jsonl /path/to/preds.jsonl
```
