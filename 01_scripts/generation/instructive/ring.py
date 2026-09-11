"""Magic-ring and reusable-chain-prototype ring SVG generation."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import math
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.models import CompositeSample, GeneratedObject, GenerationConfig
from ..core.sampling.schema import ClassSpec
from ..core.svg.stroke import resolve_stroke_width
from ..primitive.ch import append_ch_ellipse


CLASS_NAME = "ring"
SVG_NS = "http://www.w3.org/2000/svg"
_MAGIC_TURNS = 1.9
_MAGIC_INNER_RADIUS = 25.0
_MAGIC_OUTER_RADIUS = 30.0
_MAGIC_SEGMENT_COUNT = 38
_MAGIC_BOUNDS_SAMPLES = 4096
Point = tuple[float, float]
CubicSegment = tuple[Point, Point, Point]


def generate_ring(
  spec: ClassSpec,
  composite: CompositeSample,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate one ring from a parent sample and reusable child prototypes."""

  if not isinstance(composite, CompositeSample):
    raise TypeError("ring generation requires a realized CompositeSample.")
  if spec.class_group != "instructive" or spec.class_name != CLASS_NAME:
    raise ValueError("generate_ring requires the instructive.ring class specification.")

  sample = composite.parent
  values = sample.as_dict()
  variant = values.get("variant")
  stroke_width = resolve_stroke_width(sample, config, class_name=CLASS_NAME)

  svg, group = _svg_root(config, stroke_width)
  geometry_metadata: dict[str, Any] = {}

  if variant == "magic":
    if composite.prototypes:
      raise ValueError("A magic ring must not contain component prototypes.")
    geometry_metadata.update(_append_magic_ring(group, config, stroke_width))
  elif variant == "chain":
    diameter = _finite_positive(values.get("diameter"), "diameter")
    geometry_metadata["diameter"] = diameter
    geometry_metadata.update(
      _append_chain_ring(group, composite, config, values, diameter, stroke_width)
    )
  else:
    raise ValueError(f"Unsupported ring variant: {variant!r}.")

  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "variant": variant,
    **geometry_metadata,
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
    variant_id=variant,
    svg=tostring(svg, encoding="unicode"),
    metadata=metadata,
    obb_pixels=None,
    obb_normalized=None,
    yolo_label=None,
    sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )


def _svg_root(
  config: GenerationConfig,
  stroke_width: float,
) -> tuple[Element, Element]:
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


def _append_magic_ring(
  group: Element,
  config: GenerationConfig,
  stroke_width: float,
) -> dict[str, Any]:
  """Append the canonical, outward-winding magic-ring spiral."""

  start_point, segments, bounds, endpoint_angles = _canonical_magic_geometry()
  min_x, max_x, min_y, max_y = bounds
  raw_width = max_x - min_x
  raw_height = max_y - min_y

  viewport_scale_px = min(config.canvas_width_px, config.canvas_height_px) / 100.0
  stroke_px = stroke_width * viewport_scale_px
  available_width = config.target_visible_px - stroke_px
  available_height = config.target_visible_px - stroke_px
  if available_width <= 0.0 or available_height <= 0.0:
    raise ValueError("magic ring target_visible_px must exceed its rendered stroke width.")
  scale_px = min(available_width / raw_width, available_height / raw_height)
  center_x = (min_x + max_x) / 2.0
  center_y = (min_y + max_y) / 2.0

  def svg_point(point: tuple[float, float]) -> tuple[float, float]:
    return (
      50.0 + scale_px * (point[0] - center_x) / viewport_scale_px,
      50.0 + scale_px * (point[1] - center_y) / viewport_scale_px,
    )

  start_svg = svg_point(start_point)
  commands = [f"M{start_svg[0]:.8f},{start_svg[1]:.8f}"]
  for control_1, control_2, point_1 in segments:
    control_1_svg = svg_point(control_1)
    control_2_svg = svg_point(control_2)
    point_1_svg = svg_point(point_1)
    commands.append(
      f"C{control_1_svg[0]:.8f},{control_1_svg[1]:.8f} "
      f"{control_2_svg[0]:.8f},{control_2_svg[1]:.8f} "
      f"{point_1_svg[0]:.8f},{point_1_svg[1]:.8f}"
    )

  SubElement(group, "path", {
    "d": " ".join(commands),
    "fill": "none",
    "stroke-linecap": "round",
  })
  centerline_width_px = raw_width * scale_px
  centerline_height_px = raw_height * scale_px
  return {
    "phenotype": "canonical_spiral",
    "turns": _MAGIC_TURNS,
    "winding": "clockwise_outward",
    "inner_to_outer_radius_ratio": _MAGIC_INNER_RADIUS / _MAGIC_OUTER_RADIUS,
    "endpoint_angles_deg": list(endpoint_angles),
    "centerline_width_px": centerline_width_px,
    "centerline_height_px": centerline_height_px,
    "rendered_width_px": centerline_width_px + stroke_px,
    "rendered_height_px": centerline_height_px + stroke_px,
  }


@lru_cache(maxsize=1)
def _canonical_magic_geometry(
) -> tuple[Point, tuple[CubicSegment, ...], tuple[float, float, float, float], tuple[float, float]]:
  """Return immutable Bézier geometry and sampled bounds for the fixed phenotype."""

  start_angle = -math.pi / 2.0 + math.pi * (2.0 - _MAGIC_TURNS)
  end_angle = start_angle + 2.0 * math.pi * _MAGIC_TURNS
  radial_growth = (
    (_MAGIC_OUTER_RADIUS - _MAGIC_INNER_RADIUS)
    / (end_angle - start_angle)
  )

  def point_and_derivative(theta: float) -> tuple[Point, Point]:
    radius = _MAGIC_INNER_RADIUS + radial_growth * (theta - start_angle)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return (
      (radius * cosine, radius * sine),
      (
        radial_growth * cosine - radius * sine,
        radial_growth * sine + radius * cosine,
      ),
    )

  sampled_points = tuple(
    point_and_derivative(
      start_angle + (end_angle - start_angle) * index / _MAGIC_BOUNDS_SAMPLES
    )[0]
    for index in range(_MAGIC_BOUNDS_SAMPLES + 1)
  )
  bounds = (
    min(point[0] for point in sampled_points),
    max(point[0] for point in sampled_points),
    min(point[1] for point in sampled_points),
    max(point[1] for point in sampled_points),
  )

  theta_step = (end_angle - start_angle) / _MAGIC_SEGMENT_COUNT
  segments: list[CubicSegment] = []
  for index in range(_MAGIC_SEGMENT_COUNT):
    theta_0 = start_angle + index * theta_step
    theta_1 = theta_0 + theta_step
    point_0, derivative_0 = point_and_derivative(theta_0)
    point_1, derivative_1 = point_and_derivative(theta_1)
    control_1 = (
      point_0[0] + theta_step * derivative_0[0] / 3.0,
      point_0[1] + theta_step * derivative_0[1] / 3.0,
    )
    control_2 = (
      point_1[0] - theta_step * derivative_1[0] / 3.0,
      point_1[1] - theta_step * derivative_1[1] / 3.0,
    )
    segments.append((control_1, control_2, point_1))

  endpoint_angles = tuple(
    round(((math.degrees(angle) + 180.0) % 360.0) - 180.0, 8)
    for angle in (start_angle, end_angle)
  )
  return sampled_points[0], tuple(segments), bounds, endpoint_angles


def _append_chain_ring(
  group: Element,
  composite: CompositeSample,
  config: GenerationConfig,
  values: dict[str, Any],
  diameter: float,
  stroke_width: float,
) -> dict[str, Any]:
  count = values.get("count")
  if isinstance(count, bool) or not isinstance(count, int) or count < 3:
    raise ValueError("ring count must be an integer of at least 3.")
  chain_pitch = _finite_positive(values.get("chain_pitch"), "chain_pitch")
  expected_diameter = count * chain_pitch / math.pi
  if not math.isclose(diameter, expected_diameter, rel_tol=1e-9, abs_tol=1e-9):
    raise ValueError("ring diameter must equal count * chain_pitch / pi.")
  if set(composite.prototypes) != {"chain"}:
    raise ValueError("A chain ring requires exactly one 'chain' component prototype.")

  prototype = composite.prototypes["chain"]
  if (prototype.class_group, prototype.class_name) != ("primitive", "ch"):
    raise ValueError("ring chain prototype must resolve to primitive.ch.")
  if prototype.occurrence_count != count:
    raise ValueError("ring chain prototype occurrence_count must equal ring count.")
  if prototype.arrangement != "ring":
    raise ValueError("ring chain prototype arrangement must be 'ring'.")
  if prototype.declaration.get("inherit_generator") is not True:
    raise ValueError("ring chain prototype must inherit its primitive generator.")
  inherited_stroke = prototype.inherited_parameters.get("stroke_width")
  if inherited_stroke != stroke_width:
    raise ValueError("ring chain prototype must inherit the parent stroke_width.")

  prototype_values = prototype.sample.as_dict()
  shape = prototype_values.get("shape")
  aspect_ratio = _finite_positive(prototype_values.get("aspect_ratio"), "chain aspect_ratio")
  if shape not in {"oval", "circle"}:
    raise ValueError(f"Unsupported ring chain prototype shape: {shape!r}.")

  construction_scale = config.target_visible_px / 10.0
  chain_pitch_px = chain_pitch * construction_scale
  diameter_px = diameter * construction_scale
  radius_px = diameter_px / 2.0
  for index in range(count):
    angle_deg = -90.0 + index * 360.0 / count
    angle_rad = math.radians(angle_deg)
    center = (
      50.0 + 100.0 * radius_px * math.cos(angle_rad) / config.canvas_width_px,
      50.0 + 100.0 * radius_px * math.sin(angle_rad) / config.canvas_height_px,
    )
    append_ch_ellipse(
      group,
      config=config,
      center=center,
      target_visible_px=chain_pitch_px,
      aspect_ratio=aspect_ratio,
      rotation_deg=angle_deg + 90.0,
    )

  chain_width_px, chain_height_px = _chain_dimensions(chain_pitch_px, aspect_ratio)
  provenance = prototype.sample.provenance
  return {
    "count": count,
    "chain_pitch": chain_pitch,
    "chain_pitch_px": chain_pitch_px,
    "construction_scale_px_per_unit": construction_scale,
    "centerline_diameter": diameter,
    "centerline_diameter_px": diameter_px,
    "centerline_radius_px": radius_px,
    "chain_prototype": {
      "role": prototype.role,
      "class_group": prototype.class_group,
      "class_name": prototype.class_name,
      "occurrence_count": prototype.occurrence_count,
      "arrangement": prototype.arrangement,
      "shape": shape,
      "aspect_ratio": aspect_ratio,
      "width_px": chain_width_px,
      "height_px": chain_height_px,
      "inherited_parameters": dict(prototype.inherited_parameters),
      "sampling_provenance": asdict(provenance) if provenance is not None else None,
    },
  }


def _chain_dimensions(target_visible_px: float, aspect_ratio: float) -> tuple[float, float]:
  width_px = target_visible_px
  height_px = target_visible_px
  if aspect_ratio >= 1.0:
    height_px /= aspect_ratio
  else:
    width_px *= aspect_ratio
  return width_px, height_px


def _finite_positive(value: Any, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"ring {name} must be a finite number.")
  result = float(value)
  if not math.isfinite(result) or result <= 0.0:
    raise ValueError(f"ring {name} must be positive and finite.")
  return result
