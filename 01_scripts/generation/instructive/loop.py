"""Upper half-chain loop SVG generation."""

from __future__ import annotations

import math
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from ..compound.fan import expand_bounds, fit_unit_bounds
from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg import rendered_px_to_viewbox, resolve_stroke_width


CLASS_NAME = "loop"
SVG_NS = "http://www.w3.org/2000/svg"
Point = tuple[float, float]


def generate_loop(
  spec: ClassSpec,
  sample: SampledParameters,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate the upper half of a horizontally oriented chain ellipse."""

  if not isinstance(sample, SampledParameters):
    raise TypeError("loop generation requires SampledParameters.")
  if spec.class_group != "instructive" or spec.class_name != CLASS_NAME:
    raise ValueError("generate_loop requires the instructive.loop class specification.")

  values = sample.as_dict()
  aspect_ratio = _positive_finite(values.get("aspect_ratio"), "loop aspect_ratio")
  curvature = _curvature(values.get("curvature"))
  stroke_width = resolve_stroke_width(sample, config, class_name=CLASS_NAME)

  ellipse_width, ellipse_height = _unit_ellipse_dimensions(aspect_ratio)
  radius_x = ellipse_width / 2.0
  radius_y = ellipse_height / 2.0
  unit_points = _arc_points(
    center=(0.0, 0.0),
    radius_x=radius_x,
    radius_y=radius_y,
    curvature=curvature,
  )
  unit_bounds = (-radius_x, -radius_y, radius_x, 0.0)
  fit = fit_unit_bounds(
    unit_bounds,
    config=config,
    stroke_width=stroke_width,
    class_name=CLASS_NAME,
  )
  points = tuple(
    (
      fit.origin_px[0] + point[0] * fit.scale_px_per_unit,
      fit.origin_px[1] + point[1] * fit.scale_px_per_unit,
    )
    for point in unit_points
  )

  svg, group = _svg_root(config, stroke_width)
  _append_arc(group, config=config, points=points)
  centerline_bounds = (
    fit.origin_px[0] - radius_x * fit.scale_px_per_unit,
    fit.origin_px[1] - radius_y * fit.scale_px_per_unit,
    fit.origin_px[0] + radius_x * fit.scale_px_per_unit,
    fit.origin_px[1],
  )
  rendered_bounds = expand_bounds(centerline_bounds, fit.stroke_width_px / 2.0)
  start, control_1, control_2, apex, control_3, control_4, end = points

  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "phenotype": "upper_half_chain",
    "aspect_ratio": aspect_ratio,
    "curvature": curvature,
    "ellipse_width_px": ellipse_width * fit.scale_px_per_unit,
    "ellipse_height_px": ellipse_height * fit.scale_px_per_unit,
    "arc_height_px": radius_y * fit.scale_px_per_unit,
    "endpoints_px": [list(start), list(end)],
    "apex_px": list(apex),
    "control_points_px": [
      list(control_1),
      list(control_2),
      list(control_3),
      list(control_4),
    ],
    "centerline_bounds_px": list(centerline_bounds),
    "rendered_bounds_px": list(rendered_bounds),
    "stroke_width": stroke_width,
    "stroke_width_px": fit.stroke_width_px,
    "canvas": {
      "width_px": config.canvas_width_px,
      "height_px": config.canvas_height_px,
    },
    "target_visible_px": float(config.target_visible_px),
    "visual_rotation_deg": config.rotation_deg,
  }
  return GeneratedObject(
    class_id=spec.class_id,
    class_name=spec.class_name,
    variant_id=None,
    svg=tostring(svg, encoding="unicode"),
    metadata=metadata,
    obb_pixels=None,
    obb_normalized=None,
    yolo_label=None,
    sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )


def _unit_ellipse_dimensions(aspect_ratio: float) -> tuple[float, float]:
  width = 1.0
  height = 1.0
  if aspect_ratio >= 1.0:
    height /= aspect_ratio
  else:
    width *= aspect_ratio
  return width, height


def _arc_points(
  *,
  center: Point,
  radius_x: float,
  radius_y: float,
  curvature: float,
) -> tuple[Point, Point, Point, Point, Point, Point, Point]:
  center_x, center_y = center
  return (
    (center_x - radius_x, center_y),
    (center_x - radius_x, center_y - curvature * radius_y),
    (center_x - curvature * radius_x, center_y - radius_y),
    (center_x, center_y - radius_y),
    (center_x + curvature * radius_x, center_y - radius_y),
    (center_x + radius_x, center_y - curvature * radius_y),
    (center_x + radius_x, center_y),
  )


def _append_arc(
  parent: Element,
  *,
  config: GenerationConfig,
  points: tuple[Point, Point, Point, Point, Point, Point, Point],
) -> None:
  converted = tuple(rendered_px_to_viewbox(config, point) for point in points)
  start, control_1, control_2, apex, control_3, control_4, end = converted
  SubElement(parent, "path", {
    "d": (
      f"M {start[0]:.8f},{start[1]:.8f} "
      f"C {control_1[0]:.8f},{control_1[1]:.8f} "
      f"{control_2[0]:.8f},{control_2[1]:.8f} "
      f"{apex[0]:.8f},{apex[1]:.8f} "
      f"C {control_3[0]:.8f},{control_3[1]:.8f} "
      f"{control_4[0]:.8f},{control_4[1]:.8f} "
      f"{end[0]:.8f},{end[1]:.8f}"
    ),
  })


def _svg_root(
  config: GenerationConfig,
  stroke_width: float,
) -> tuple[Element, Element]:
  rendered_px_to_viewbox(config, (0.0, 0.0))
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
  return svg, group


def _curvature(value: Any) -> float:
  result = _finite(value, "loop curvature")
  if not 0.0 < result < 1.0:
    raise ValueError("loop curvature must be in the range (0, 1).")
  return result


def _finite(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a finite number.")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{name} must be a finite number.")
  return result


def _positive_finite(value: Any, name: str) -> float:
  result = _finite(value, name)
  if result <= 0.0:
    raise ValueError(f"{name} must be positive.")
  return result
