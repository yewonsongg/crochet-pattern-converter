from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec


CLASS_NAME = "hdc"
SVG_NS = "http://www.w3.org/2000/svg"


def _build_hdc_svg(config: GenerationConfig, bar_stem_ratio: float) -> str:
  """Build an hdc from a centered vertical stem and attached top bar."""
  if bar_stem_ratio <= 0:
    raise ValueError("hdc bar_stem_ratio must be positive.")

  stem_length_px = config.target_visible_px
  bar_length_px = stem_length_px * bar_stem_ratio

  stem_half_height = 50.0 * stem_length_px / config.canvas_height_px
  bar_half_width = 50.0 * bar_length_px / config.canvas_width_px
  stem_top = 50.0 - stem_half_height
  stem_bottom = 50.0 + stem_half_height

  svg = Element("svg", {
    "xmlns": SVG_NS,
    "width": f"{config.canvas_width_px}px",
    "height": f"{config.canvas_height_px}px",
    "viewBox": "0 0 100 100",
  })
  group = SubElement(svg, "g", {
    "fill": "none",
    "stroke": "black",
    "stroke-width": str(config.stroke_width_normalized),
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
  return tostring(svg, encoding="unicode")


def generate_hdc(
  spec: ClassSpec,
  sample: SampledParameters,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate SVG for one sampled half-double-crochet primitive."""
  values = sample.as_dict()
  bar_stem_ratio = values.get("bar_stem_ratio")
  if isinstance(bar_stem_ratio, bool) or not isinstance(bar_stem_ratio, (int, float)):
    raise ValueError("hdc bar_stem_ratio must be numeric.")
  bar_stem_ratio = float(bar_stem_ratio)
  if bar_stem_ratio <= 0:
    raise ValueError("hdc bar_stem_ratio must be positive.")

  svg = _build_hdc_svg(config, bar_stem_ratio)
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "bar_stem_ratio": bar_stem_ratio,
    "stem_length_px": config.target_visible_px,
    "bar_length_px": config.target_visible_px * bar_stem_ratio,
    "canvas": {
      "width_px": config.canvas_width_px,
      "height_px": config.canvas_height_px,
    },
    "target_visible_px": config.target_visible_px,
    "visual_rotation_deg": config.rotation_deg,
    "stroke_width_normalized": config.stroke_width_normalized,
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
