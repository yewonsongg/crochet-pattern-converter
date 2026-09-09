from __future__ import annotations

import math
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec


CLASS_NAME = "dc"
SVG_NS = "http://www.w3.org/2000/svg"


def _build_dc_svg(
  config: GenerationConfig,
  bar_stem_ratio: float,
  cross_bar_ratio: float,
  cross_bar_y: float,
  cross_bar_angle_deg: float,
  stroke_width: float,
) -> str:
  """Build a stem, attached top bar, and one centered crossbar."""
  stem_length_px = config.target_visible_px
  bar_length_px = stem_length_px * bar_stem_ratio
  cross_bar_length_px = stem_length_px * cross_bar_ratio

  stem_half_height = 50.0 * stem_length_px / config.canvas_height_px
  bar_half_width = 50.0 * bar_length_px / config.canvas_width_px
  stem_top = 50.0 - stem_half_height
  stem_bottom = 50.0 + stem_half_height
  cross_bar_position = stem_bottom - (cross_bar_y * stem_length_px * 100.0 / config.canvas_height_px)
  cross_bar_half_length_px = cross_bar_length_px / 2.0
  angle_rad = math.radians(cross_bar_angle_deg)
  cross_bar_dx = 100.0 * cross_bar_half_length_px * math.cos(angle_rad) / config.canvas_width_px
  # SVG's y axis points down, so positive angles remain visually counterclockwise.
  cross_bar_dy = 100.0 * cross_bar_half_length_px * math.sin(angle_rad) / config.canvas_height_px

  svg = Element("svg", {
    "xmlns": SVG_NS,
    "width": f"{config.canvas_width_px}px",
    "height": f"{config.canvas_height_px}px",
    "viewBox": "0 0 100 100",
  })
  group = SubElement(svg, "g", {
    "fill": "none",
    "stroke": "black",
    "stroke-width": str(stroke_width),
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "transform": f"rotate({config.rotation_deg} 50 50)",
  })
  SubElement(group, "line", {
    "x1": "50",
    "y1": f"{stem_top:.8f}",
    "x2": "50",
    "y2": f"{stem_bottom:.8f}",
  })
  SubElement(group, "line", {
    "x1": f"{50.0 - bar_half_width:.8f}",
    "y1": f"{stem_top:.8f}",
    "x2": f"{50.0 + bar_half_width:.8f}",
    "y2": f"{stem_top:.8f}",
  })
  SubElement(group, "line", {
    "x1": f"{50.0 - cross_bar_dx:.8f}",
    "y1": f"{cross_bar_position + cross_bar_dy:.8f}",
    "x2": f"{50.0 + cross_bar_dx:.8f}",
    "y2": f"{cross_bar_position - cross_bar_dy:.8f}",
  })
  return tostring(svg, encoding="unicode")


def generate_dc(
  spec: ClassSpec,
  sample: SampledParameters,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate SVG for one sampled double-crochet primitive."""
  values = sample.as_dict()
  bar_stem_ratio = values.get("bar_stem_ratio")
  cross_bar_ratio = values.get("cross_bar_ratio")
  cross_bar_y = values.get("cross_bar_y")
  cross_bar_angle_deg = values.get("cross_bar_angle_deg")
  stroke_width = values.get("stroke_width", config.stroke_width_normalized)

  bar_stem_ratio = _finite_number(bar_stem_ratio, "bar_stem_ratio")
  cross_bar_ratio = _finite_number(cross_bar_ratio, "cross_bar_ratio")
  cross_bar_y = _finite_number(cross_bar_y, "cross_bar_y")
  cross_bar_angle_deg = _finite_number(cross_bar_angle_deg, "cross_bar_angle_deg")
  stroke_width = _finite_number(stroke_width, "stroke_width")
  if bar_stem_ratio <= 0 or cross_bar_ratio <= 0:
    raise ValueError("dc bar and crossbar ratios must be positive.")
  if not 0.0 <= cross_bar_y <= 1.0:
    raise ValueError("dc cross_bar_y must be in the range [0, 1].")
  if not -45.0 <= cross_bar_angle_deg <= 45.0:
    raise ValueError("dc cross_bar_angle_deg must be in the range [-45, 45].")
  if stroke_width <= 0:
    raise ValueError("dc stroke_width must be positive.")

  svg = _build_dc_svg(
    config,
    bar_stem_ratio,
    cross_bar_ratio,
    cross_bar_y,
    cross_bar_angle_deg,
    stroke_width,
  )
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "bar_stem_ratio": bar_stem_ratio,
    "cross_bar_ratio": cross_bar_ratio,
    "cross_bar_y": cross_bar_y,
    "cross_bar_angle_deg": cross_bar_angle_deg,
    "stem_length_px": config.target_visible_px,
    "bar_length_px": config.target_visible_px * bar_stem_ratio,
    "cross_bar_length_px": config.target_visible_px * cross_bar_ratio,
    "stroke_width": stroke_width,
    "canvas": {
      "width_px": config.canvas_width_px,
      "height_px": config.canvas_height_px,
    },
    "target_visible_px": config.target_visible_px,
    "visual_rotation_deg": config.rotation_deg,
  }
  return GeneratedObject(
    class_id=spec.class_id,
    class_name=spec.class_name,
    variant_id=None,
    svg=svg,
    metadata=metadata,
    obb_pixels=None,
    obb_normalized=None,
    yolo_label=None,
    sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )


def _finite_number(value: Any, name: str) -> float:
  """Narrow one sampled value to a finite numeric value."""
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"dc {name} must be a finite number.")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"dc {name} must be a finite number.")
  return result
