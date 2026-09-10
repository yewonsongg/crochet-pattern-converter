from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg.stroke import resolve_stroke_width


SVG_NS = "http://www.w3.org/2000/svg"


def _build_slst_svg(config: GenerationConfig, aspect_ratio: float, stroke_width: float) -> str:
  if aspect_ratio <= 0:
    raise ValueError("slst aspect_ratio must be positive.")

  width_px = config.target_visible_px
  height_px = config.target_visible_px
  if aspect_ratio >= 1.0:
    height_px /= aspect_ratio
  else:
    width_px *= aspect_ratio

  width_normalized = 100.0 * width_px / config.canvas_width_px
  height_normalized = 100.0 * height_px / config.canvas_height_px
  svg = Element("svg", {
    "xmlns": SVG_NS,
    "width": f"{config.canvas_width_px}px",
    "height": f"{config.canvas_height_px}px",
    "viewBox": "0 0 100 100",
  })
  group = SubElement(svg, "g", {
    "fill": "black",
    "stroke": "black",
    "stroke-width": str(stroke_width),
    "transform": f"rotate({config.rotation_deg} 50 50)",
  })
  SubElement(group, "ellipse", {
    "fill": "black",
    "cx": "50",
    "cy": "50",
    "rx": f"{width_normalized / 2:.8f}",
    "ry": f"{height_normalized / 2:.8f}",
  })
  return tostring(svg, encoding="unicode")


def generate_slst(
  spec: ClassSpec,
  sample: SampledParameters,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate SVG for one sampled slip-stitch primitive."""
  values = sample.as_dict()
  shape = values.get("shape")
  aspect_ratio = values.get("aspect_ratio")
  stroke_width = resolve_stroke_width(sample, config, class_name="slst")
  if shape not in {"oval", "circle"}:
    raise ValueError(f"Unsupported slst shape: {shape!r}.")
  if isinstance(aspect_ratio, bool) or not isinstance(aspect_ratio, (int, float)):
    raise ValueError("slst aspect_ratio must be numeric.")

  svg = _build_slst_svg(config, float(aspect_ratio), stroke_width)
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "shape": shape,
    "aspect_ratio": float(aspect_ratio),
    "canvas": {"width_px": config.canvas_width_px, "height_px": config.canvas_height_px},
    "target_visible_px": config.target_visible_px,
    "visual_rotation_deg": config.rotation_deg,
    "stroke_width": stroke_width,
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
