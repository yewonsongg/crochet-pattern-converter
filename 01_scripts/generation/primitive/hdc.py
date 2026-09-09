from __future__ import annotations

import math
from typing import Any

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg.bar_stem import build_bar_stem_svg


CLASS_NAME = "hdc"


def generate_hdc(spec: ClassSpec, sample: SampledParameters, config: GenerationConfig) -> GeneratedObject:
  """Generate SVG for one sampled half-double-crochet primitive."""
  values = sample.as_dict()
  bar_stem_ratio = _finite_number(values.get("bar_stem_ratio"), "bar_stem_ratio")
  stroke_width = _finite_number(values.get("stroke_width", config.stroke_width_normalized), "stroke_width")
  if bar_stem_ratio <= 0 or stroke_width <= 0:
    raise ValueError("hdc bar_stem_ratio and stroke_width must be positive.")

  geometry = build_bar_stem_svg(config, bar_stem_ratio=bar_stem_ratio, stroke_width=stroke_width)
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "bar_stem_ratio": bar_stem_ratio,
    "stem_length_px": geometry.stem_length_px,
    "bar_length_px": geometry.bar_length_px,
    "canvas": {"width_px": config.canvas_width_px, "height_px": config.canvas_height_px},
    "target_visible_px": config.target_visible_px,
    "visual_rotation_deg": config.rotation_deg,
    "stroke_width": stroke_width,
  }
  return GeneratedObject(
    class_id=spec.class_id, class_name=spec.class_name, variant_id=None,
    svg=geometry.svg, metadata=metadata, obb_pixels=None,
    obb_normalized=None, yolo_label=None, sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )


def _finite_number(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"hdc {name} must be a finite number.")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"hdc {name} must be a finite number.")
  return result
