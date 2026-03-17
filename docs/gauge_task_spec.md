# Gauge Task Spec

## Task Definition

- Task name: `analog_gauge_readout`
- Input: a single image of an automotive-style analog gauge
- Output: an integer value in `[0, 100]`

This task is intentionally different from analog clock reading:
- the dial is a half-circle rather than a full circular clock face
- there is only one pointer
- the label is a scalar value rather than a time string
- the visual design should resemble a speedometer / tachometer / pressure gauge, not a clock

## Geometry Definition

- Dial shape: half-circle with a straight lower chord
- Value range: `0` to `100`
- Pointer count: `1`
- Arc range:
  - `theta_min = -120 deg`
  - `theta_max = 120 deg`
- Angle convention:
  - angles are measured around the dial center
  - `theta = 0 deg` points to the top of the gauge
  - negative angles sweep toward the lower-left
  - positive angles sweep toward the lower-right

## Value Mapping

The integer gauge value is linearly mapped to pointer angle:

```text
pointer_angle_deg = theta_min + (gauge_value / 100) * (theta_max - theta_min)
```

Equivalently:

```text
gauge_value = round((pointer_angle_deg - theta_min) / (theta_max - theta_min) * 100)
```

with clamping to `[0, 100]`.

## Label Definition

Each sample should contain:

- `label.gauge_value`
- `label.pointer_angle_deg`

Recommended task field:

- `task = "analog_gauge_readout"`

## Smoke Test Splits

Two minimal splits are required:

### `clean`

- moderate viewpoint change
- clear dial face
- mild or no visible glass reflection
- no or near-zero blur

### `noisy`

- larger viewpoint shift
- stronger glass reflection / glare
- optional visible motion blur / defocus
- harder lighting

## Visual Design Principles

The gauge should look more like an automotive analog instrument than a clock.

Must satisfy:
- half-circle silhouette
- single pointer
- sparse numeric labels such as `0, 20, 40, 60, 80, 100`
- tick layout aligned to a semicircular instrument arc
- dark dashboard-like face and metallic / glass feel are preferred

Must avoid:
- full circular clock layout
- 12-clock-style major ticks
- `HH:MM` semantics
- clock numerals or Roman numerals
- multiple hands
- decorative clock center / bezel choices that make it look like a clock variant

## Smoke Output Metadata

Each smoke sample should include at least:

- `id`
- `image`
- `label.gauge_value`
- `label.pointer_angle_deg`
- `meta.source = "blender_gauge"`
- `meta.benchmark_split`
- `meta.view`
- `meta.degradation`
- `meta.lighting`
- `meta.tilt_bucket`
- `meta.specular_bucket`
- `meta.blur_bucket`
