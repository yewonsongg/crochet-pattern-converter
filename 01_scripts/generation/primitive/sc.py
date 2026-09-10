from __future__ import annotations

from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg.stroke import resolve_stroke_width


CLASS_NAME = "sc"
SVG_NS = "http://www.w3.org/2000/svg"


def _build_sc_svg(
  config: GenerationConfig,
  asymmetry: float,
  cross_bar_ratio: float,
  stroke_width: float,
) -> str:
  """Build a centered plus/cross with an optionally vertically offset bar."""
  if not -1.0 < asymmetry < 1.0:
    raise ValueError("sc asymmetry must be in the range (-1, 1).")
  if not 0.0 < cross_bar_ratio <= 1.0:
    raise ValueError("sc cross_bar_ratio must be in the range (0, 1].")

  half_span_y = 50.0 * config.target_visible_px / config.canvas_height_px
  # The vertical stroke establishes the target visible dimension. The
  # horizontal bar is scaled from that same pixel span, then converted into
  # the SVG viewBox's x-axis units for non-square canvases.
  horizontal_span_px = config.target_visible_px * cross_bar_ratio
  half_span_x = 50.0 * horizontal_span_px / config.canvas_width_px

  # ``sc`` asymmetry describes placement of the horizontal bar, not unequal
  # arm lengths. Keep the vertical stroke centered and preserve equal left /
  # right horizontal arms. Positive asymmetry moves the bar upward; negative
  # asymmetry moves it downward.
  horizontal_y = 50.0 - (asymmetry * half_span_y)

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
    "x1": f"{50.0 - half_span_x:.8f}",
    "y1": f"{horizontal_y:.8f}",
    "x2": f"{50.0 + half_span_x:.8f}",
    "y2": f"{horizontal_y:.8f}",
  })
  SubElement(group, "line", {
    "x1": "50",
    "y1": f"{50.0 - half_span_y:.8f}",
    "x2": "50",
    "y2": f"{50.0 + half_span_y:.8f}",
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
  cross_bar_ratio = values.get("cross_bar_ratio")
  stroke_width = resolve_stroke_width(sample, config, class_name="sc")
  if shape not in {"symmetric", "asymmetric"}:
    raise ValueError(f"Unsupported sc shape: {shape!r}.")
  if isinstance(asymmetry, bool) or not isinstance(asymmetry, (int, float)):
    raise ValueError("sc asymmetry must be numeric.")
  if isinstance(cross_bar_ratio, bool) or not isinstance(cross_bar_ratio, (int, float)):
    raise ValueError("sc cross_bar_ratio must be numeric.")

  asymmetry = float(asymmetry)
  cross_bar_ratio = float(cross_bar_ratio)
  if shape == "symmetric" and asymmetry != 0.0:
    raise ValueError("A symmetric sc must have asymmetry=0.0.")
  # The asymmetric sampling branch may legitimately include zero (for
  # example, its configured mean or a deterministic coverage probe). In that
  # case the geometry is simply the centered limit of the branch.

  svg = _build_sc_svg(config, asymmetry, cross_bar_ratio, stroke_width)
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "shape": shape,
    "asymmetry": asymmetry,
    "cross_bar_ratio": cross_bar_ratio,
    "canvas": {
      "width_px": config.canvas_width_px,
      "height_px": config.canvas_height_px,
    },
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
