from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec


CLASS_NAME = "sc"
SVG_NS = "http://www.w3.org/2000/svg"


def _build_sc_svg(config: GenerationConfig, asymmetry: float) -> str:
  """Build a centered plus/cross with equal or unequal opposing arms."""
  if not 0.0 <= asymmetry < 1.0:
    raise ValueError("sc asymmetry must be in the range [0, 1).")

  half_span_x = 50.0 * config.target_visible_px / config.canvas_width_px
  half_span_y = 50.0 * config.target_visible_px / config.canvas_height_px

  long_x = half_span_x * (1.0 + asymmetry)
  short_x = half_span_x * (1.0 - asymmetry)
  long_y = half_span_y * (1.0 + asymmetry)
  short_y = half_span_y * (1.0 - asymmetry)

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
    "x1": f"{50.0 - long_x:.8f}",
    "y1": "50",
    "x2": f"{50.0 + short_x:.8f}",
    "y2": "50",
  })
  SubElement(group, "line", {
    "x1": "50",
    "y1": f"{50.0 - long_y:.8f}",
    "x2": "50",
    "y2": f"{50.0 + short_y:.8f}",
  })
  return tostring(svg, encoding="unicode")


def generate_sc(
  spec: ClassSpec,
  sample: SampledParameters,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate SVG for one sampled single-crochet plus/cross primitive."""
  values = sample.as_dict()
  shape = values.get("shape")
  asymmetry = values.get("asymmetry")
  if shape not in {"symmetric", "asymmetric"}:
    raise ValueError(f"Unsupported sc shape: {shape!r}.")
  if isinstance(asymmetry, bool) or not isinstance(asymmetry, (int, float)):
    raise ValueError("sc asymmetry must be numeric.")

  asymmetry = float(asymmetry)
  if shape == "symmetric" and asymmetry != 0.0:
    raise ValueError("A symmetric sc must have asymmetry=0.0.")
  if shape == "asymmetric" and asymmetry <= 0.0:
    raise ValueError("An asymmetric sc must have asymmetry greater than 0.0.")

  svg = _build_sc_svg(config, asymmetry)
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "shape": shape,
    "asymmetry": asymmetry,
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
