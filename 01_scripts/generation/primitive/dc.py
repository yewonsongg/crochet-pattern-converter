from __future__ import annotations

import math
from typing import Any

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg.bar_stem import build_bar_stem_svg


CLASS_NAME = "dc"


def generate_dc(spec: ClassSpec, sample: SampledParameters, config: GenerationConfig) -> GeneratedObject:
  """Generate SVG for one sampled double-crochet primitive."""
  values = sample.as_dict()
  bar_stem_ratio = _finite_number(values.get("bar_stem_ratio"), "bar_stem_ratio")
  cross_bar_ratio = _finite_number(values.get("cross_bar_ratio"), "cross_bar_ratio")
  cross_bar_y = _finite_number(values.get("cross_bar_y"), "cross_bar_y")
  cross_bar_angle_deg = _finite_number(values.get("cross_bar_angle_deg"), "cross_bar_angle_deg")
  stroke_width = _finite_number(values.get("stroke_width", config.stroke_width_normalized), "stroke_width")
  if bar_stem_ratio <= 0 or cross_bar_ratio <= 0 or stroke_width <= 0:
    raise ValueError("dc ratios and stroke_width must be positive.")
  if not 0.0 <= cross_bar_y <= 1.0:
    raise ValueError("dc cross_bar_y must be in the range [0, 1].")
  if not -45.0 <= cross_bar_angle_deg <= 45.0:
    raise ValueError("dc cross_bar_angle_deg must be in the range [-45, 45].")

  geometry = build_bar_stem_svg(
    config,
    bar_stem_ratio=bar_stem_ratio,
    cross_bar_ratio=cross_bar_ratio,
    cross_bar_y=cross_bar_y,
    cross_bar_angle_deg=cross_bar_angle_deg,
    cross_bar_count=1,
    stroke_width=stroke_width,
  )
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "bar_stem_ratio": bar_stem_ratio,
    "cross_bar_ratio": cross_bar_ratio,
    "cross_bar_y": cross_bar_y,
    "cross_bar_angle_deg": cross_bar_angle_deg,
    "stem_length_px": geometry.stem_length_px,
    "bar_length_px": geometry.bar_length_px,
    "cross_bar_length_px": geometry.cross_bar_length_px,
    "cross_bar_y_positions": list(geometry.cross_bar_y_positions),
    "stroke_width": stroke_width,
    "canvas": {"width_px": config.canvas_width_px, "height_px": config.canvas_height_px},
    "target_visible_px": config.target_visible_px,
    "visual_rotation_deg": config.rotation_deg,
  }
  return GeneratedObject(
    class_id=spec.class_id, class_name=spec.class_name, variant_id=None,
    svg=geometry.svg, metadata=metadata, obb_pixels=None,
    obb_normalized=None, yolo_label=None, sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )


def _finite_number(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"dc {name} must be a finite number.")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"dc {name} must be a finite number.")
  return result
