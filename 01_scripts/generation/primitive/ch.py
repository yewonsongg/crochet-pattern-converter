from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.sampling.schema import ClassSpec
from ..core.models import SampledParameters, GeneratedObject, GenerationConfig
from ..core.svg.stroke import resolve_stroke_width


SVG_NS = "http://www.w3.org/2000/svg"


def generate_ch(
  spec: ClassSpec,
  sample: SampledParameters,
  config: GenerationConfig,
) -> GeneratedObject:
  values = sample.as_dict()
  shape = values.get("shape")
  aspect_ratio = values.get("aspect_ratio")
  stroke_width = resolve_stroke_width(sample, config, class_name="ch")
  if shape not in {"oval", "circle"}:
    raise ValueError(f"Unsupported ch shape: {shape!r}.")
  if isinstance(aspect_ratio, bool) or not isinstance(aspect_ratio, (int, float)):
    raise ValueError("ch aspect_ratio must be numeric.")

  svg = _build_ch_svg(config, float(aspect_ratio), stroke_width)
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


def _build_ch_svg(config: GenerationConfig, aspect_ratio: float, stroke_width: float) -> str:
  if aspect_ratio <= 0:
    raise ValueError("ch aspect_ratio must be positive.")

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
    "transform": f"rotate({config.rotation_deg} 50 50)",
  })
  append_ch_ellipse(
    group,
    config=config,
    center=(50.0, 50.0),
    target_visible_px=config.target_visible_px,
    aspect_ratio=aspect_ratio,
  )
  return tostring(svg, encoding="unicode")


def append_ch_ellipse(
  parent: Element,
  *,
  config: GenerationConfig,
  center: tuple[float, float],
  target_visible_px: float,
  aspect_ratio: float,
  rotation_deg: float = 0.0,
) -> Element:
  """Append reusable chain-stitch ellipse geometry to an SVG parent."""

  if target_visible_px <= 0:
    raise ValueError("ch target_visible_px must be positive.")
  if aspect_ratio <= 0:
    raise ValueError("ch aspect_ratio must be positive.")

  width_px = float(target_visible_px)
  height_px = float(target_visible_px)
  if aspect_ratio >= 1.0:
    height_px /= aspect_ratio
  else:
    width_px *= aspect_ratio

  cx, cy = center
  attributes = {
    "fill": "none",
    "cx": f"{cx:.8f}",
    "cy": f"{cy:.8f}",
    "rx": f"{50.0 * width_px / config.canvas_width_px:.8f}",
    "ry": f"{50.0 * height_px / config.canvas_height_px:.8f}",
  }
  if rotation_deg:
    attributes["transform"] = f"rotate({rotation_deg:.8f} {cx:.8f} {cy:.8f})"
  return SubElement(parent, "ellipse", attributes)
