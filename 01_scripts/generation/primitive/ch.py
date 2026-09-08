from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.sampling.schema import ClassSpec
from ..core.models import SampledParameters, GeneratedObject, GenerationConfig


SVG_NS = "http://www.w3.org/2000/svg"


def generate_ch(
  spec: ClassSpec,
  sample: SampledParameters,
  config: GenerationConfig,
) -> GeneratedObject:
  values = sample.as_dict()
  shape = values.get("shape")
  aspect_ratio = values.get("aspect_ratio")
  if shape not in {"oval", "circle"}:
    raise ValueError(f"Unsupported ch shape: {shape!r}.")
  if isinstance(aspect_ratio, bool) or not isinstance(aspect_ratio, (int, float)):
    raise ValueError("ch aspect_ratio must be numeric.")

  svg = _build_ch_svg(config, float(aspect_ratio))
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "shape": shape,
    "aspect_ratio": float(aspect_ratio),
    "canvas": {"width_px": config.canvas_width_px, "height_px": config.canvas_height_px},
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


def _build_ch_svg(config: GenerationConfig, aspect_ratio: float) -> str:
  if aspect_ratio <= 0:
    raise ValueError("ch aspect_ratio must be positive.")

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
    "fill": "none",
    "stroke": "black",
    "stroke-width": str(config.stroke_width_normalized),
    "transform": f"rotate({config.rotation_deg} 50 50)",
  })
  SubElement(group, "ellipse", {
    "fill": "none",
    "cx": "50",
    "cy": "50",
    "rx": f"{width_normalized / 2:.8f}",
    "ry": f"{height_normalized / 2:.8f}",
  })
  return tostring(svg, encoding="unicode")
