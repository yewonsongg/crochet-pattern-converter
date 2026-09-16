"""Front- and back-post compound generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping
from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.models import (
  ComponentPrototype,
  CompositeSample,
  GeneratedObject,
  GenerationConfig,
)
from ..core.sampling.schema import ClassSpec
from ..core.svg import (
  STITCH_GEOMETRY_REGISTRY,
  StitchGeometry,
  StitchPlacement,
  append_stitch_geometry,
  rendered_px_to_viewbox,
  resolve_stroke_width,
)
from .fan import expand_bounds, fit_unit_bounds, union_bounds


CLASS_NAME = "post"
SVG_NS = "http://www.w3.org/2000/svg"
_CIRCLE_HANDLE = 0.5522847498307936
Point = tuple[float, float]
Bounds = tuple[float, float, float, float]
CubicSegment = tuple[Point, Point, Point]


@dataclass(frozen=True)
class PostPathGeometry:
  """One post-hook path measured before global rotation."""

  start_px: Point
  segments: tuple[CubicSegment, ...]
  end_px: Point
  centerline_bounds_px: Bounds


def generate_post(
  spec: ClassSpec,
  composite: CompositeSample,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate one primitive stitch with a mirrored post hook at its base."""

  if not isinstance(composite, CompositeSample):
    raise TypeError("post generation requires a realized CompositeSample.")
  if spec.class_group != "compound" or spec.class_name != CLASS_NAME:
    raise ValueError("generate_post requires the compound.post class specification.")

  parent = composite.parent
  values = parent.as_dict()
  post_type = values.get("post_type")
  if post_type not in {"front", "back"}:
    raise ValueError(f"Unsupported post post_type: {post_type!r}.")
  variant = values.get("variant")
  if variant not in {"hook", "j_shape"}:
    raise ValueError(f"Unsupported post variant: {variant!r}.")
  hook_curvature = _positive_finite(
    values.get("hook_curvature"), "post hook_curvature"
  )
  stroke_width = resolve_stroke_width(parent, config, class_name=CLASS_NAME)
  prototype = _prototype(
    composite,
    stitch_class=values.get("stitch"),
    stroke_width=stroke_width,
  )
  stitch_class = prototype.class_name
  stitch_values = prototype.sample.as_dict()

  unit_placement = StitchPlacement(base_px=(0.0, 0.0), top_px=(0.0, -1.0))
  unit_group = Element("g")
  unit_stitch = append_stitch_geometry(
    unit_group,
    class_name=stitch_class,
    sampled_values=stitch_values,
    config=config,
    placement=unit_placement,
    stroke_width=stroke_width,
  )
  unit_path = _post_path(
    variant=variant,
    post_type=post_type,
    attachment=(0.0, 0.0),
    radius=hook_curvature,
  )
  unit_bounds = union_bounds(
    [unit_stitch.centerline_bounds_px, unit_path.centerline_bounds_px],
    class_name=CLASS_NAME,
  )
  fit = fit_unit_bounds(
    unit_bounds,
    config=config,
    stroke_width=stroke_width,
    class_name=CLASS_NAME,
  )

  attachment = fit.origin_px
  placement = StitchPlacement(
    base_px=attachment,
    top_px=(attachment[0], attachment[1] - fit.scale_px_per_unit),
  )
  svg, group = _svg_root(config, stroke_width)
  stitch = append_stitch_geometry(
    group,
    class_name=stitch_class,
    sampled_values=stitch_values,
    config=config,
    placement=placement,
    stroke_width=stroke_width,
  )
  path = _post_path(
    variant=variant,
    post_type=post_type,
    attachment=attachment,
    radius=hook_curvature * fit.scale_px_per_unit,
  )
  _append_path(group, config=config, geometry=path)
  centerline_bounds = union_bounds(
    [stitch.centerline_bounds_px, path.centerline_bounds_px],
    class_name=CLASS_NAME,
  )
  rendered_bounds = expand_bounds(centerline_bounds, fit.stroke_width_px / 2.0)
  opening_side = "left" if post_type == "front" else "right"
  opening_angle_interval = [90.0, 180.0] if post_type == "front" else [0.0, 90.0]
  hook_width = path.centerline_bounds_px[2] - path.centerline_bounds_px[0]
  hook_height = path.centerline_bounds_px[3] - path.centerline_bounds_px[1]

  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "post_type": post_type,
    "variant": variant,
    "stitch": stitch_class,
    "opening_side": opening_side,
    "opening_angle_interval_deg": opening_angle_interval,
    "attachment_angle_deg": 90.0 if variant == "hook" else None,
    "hook_curvature": hook_curvature,
    "stem_length_px": fit.scale_px_per_unit,
    "hook_radius_px": hook_curvature * fit.scale_px_per_unit,
    "hook_width_px": hook_width,
    "hook_height_px": hook_height,
    "attachment_px": list(attachment),
    "circle_center_px": (
      [attachment[0], attachment[1] + hook_curvature * fit.scale_px_per_unit]
      if variant == "hook"
      else None
    ),
    "stitch_placement": {
      "base_px": list(placement.base_px),
      "top_px": list(placement.top_px),
    },
    "path_start_px": list(path.start_px),
    "path_end_px": list(path.end_px),
    "path_segments": [
      {
        "control_1_px": list(control_1),
        "control_2_px": list(control_2),
        "end_px": list(end),
      }
      for control_1, control_2, end in path.segments
    ],
    "path_centerline_bounds_px": list(path.centerline_bounds_px),
    "centerline_bounds_px": list(centerline_bounds),
    "rendered_bounds_px": list(rendered_bounds),
    "stroke_width": stroke_width,
    "stroke_width_px": fit.stroke_width_px,
    "component_prototype": _prototype_metadata(prototype),
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
    variant_id=variant,
    svg=tostring(svg, encoding="unicode"),
    metadata=metadata,
    obb_pixels=None,
    obb_normalized=None,
    yolo_label=None,
    sampled_parameters=dict(values),
    sampling_provenance=parent.provenance,
  )


def _post_path(
  *,
  variant: str,
  post_type: str,
  attachment: Point,
  radius: float,
) -> PostPathGeometry:
  radius = _positive_finite(radius, "post hook radius")
  direction = -1.0 if post_type == "front" else 1.0
  if variant == "hook":
    start, segments = _hook_segments(attachment, radius, direction)
  elif variant == "j_shape":
    start, segments = _j_segments(attachment, radius, direction)
  else:
    raise ValueError(f"Unsupported post variant: {variant!r}.")
  bounds = _cubic_path_bounds(start, segments)
  return PostPathGeometry(
    start_px=start,
    segments=segments,
    end_px=segments[-1][2],
    centerline_bounds_px=bounds,
  )


def _hook_segments(
  attachment: Point,
  radius: float,
  direction: float,
) -> tuple[Point, tuple[CubicSegment, CubicSegment, CubicSegment]]:
  x, y = attachment

  def point(relative_x: float, relative_y: float) -> Point:
    return (x + direction * relative_x * radius, y + relative_y * radius)

  start = point(0.0, 0.0)
  segments = (
    (
      point(-_CIRCLE_HANDLE, 0.0),
      point(-1.0, 1.0 - _CIRCLE_HANDLE),
      point(-1.0, 1.0),
    ),
    (
      point(-1.0, 1.0 + _CIRCLE_HANDLE),
      point(-_CIRCLE_HANDLE, 2.0),
      point(0.0, 2.0),
    ),
    (
      point(_CIRCLE_HANDLE, 2.0),
      point(1.0, 1.0 + _CIRCLE_HANDLE),
      point(1.0, 1.0),
    ),
  )
  return start, segments


def _j_segments(
  attachment: Point,
  radius: float,
  direction: float,
) -> tuple[Point, tuple[CubicSegment, CubicSegment]]:
  x, y = attachment

  def point(relative_x: float, relative_y: float) -> Point:
    return (x + direction * relative_x * radius, y + relative_y * radius)

  # Two ellipse-like quarter arcs form a broad, rounded bowl.  Keeping the
  # shared point at the bowl's lowest point gives both cubics the same
  # horizontal tangent, while the first and last tangents remain vertical.
  half_width = 1.15 / 2.0
  bowl_depth = 1.60
  terminal_height = 0.85
  return_rise = bowl_depth - terminal_height
  start = point(0.0, 0.0)
  segments = (
    (
      point(0.0, _CIRCLE_HANDLE * bowl_depth),
      point(half_width * (1.0 - _CIRCLE_HANDLE), bowl_depth),
      point(half_width, bowl_depth),
    ),
    (
      point(half_width * (1.0 + _CIRCLE_HANDLE), bowl_depth),
      point(1.15, terminal_height + _CIRCLE_HANDLE * return_rise),
      point(1.15, terminal_height),
    ),
  )
  return start, segments


def _append_path(
  parent: Element,
  *,
  config: GenerationConfig,
  geometry: PostPathGeometry,
) -> None:
  start = rendered_px_to_viewbox(config, geometry.start_px)
  commands = [f"M {start[0]:.8f},{start[1]:.8f}"]
  for control_1, control_2, end in geometry.segments:
    control_1_svg = rendered_px_to_viewbox(config, control_1)
    control_2_svg = rendered_px_to_viewbox(config, control_2)
    end_svg = rendered_px_to_viewbox(config, end)
    commands.append(
      f"C {control_1_svg[0]:.8f},{control_1_svg[1]:.8f} "
      f"{control_2_svg[0]:.8f},{control_2_svg[1]:.8f} "
      f"{end_svg[0]:.8f},{end_svg[1]:.8f}"
    )
  SubElement(parent, "path", {"d": " ".join(commands)})


def _cubic_path_bounds(start: Point, segments: tuple[CubicSegment, ...]) -> Bounds:
  points = [start]
  segment_start = start
  for control_1, control_2, end in segments:
    candidates = {0.0, 1.0}
    candidates.update(_cubic_extrema_parameters(
      segment_start[0], control_1[0], control_2[0], end[0]
    ))
    candidates.update(_cubic_extrema_parameters(
      segment_start[1], control_1[1], control_2[1], end[1]
    ))
    points.extend(
      _cubic_point(segment_start, control_1, control_2, end, parameter)
      for parameter in candidates
    )
    segment_start = end
  return (
    min(point[0] for point in points),
    min(point[1] for point in points),
    max(point[0] for point in points),
    max(point[1] for point in points),
  )


def _cubic_extrema_parameters(
  start: float,
  control_1: float,
  control_2: float,
  end: float,
) -> set[float]:
  quadratic = -start + 3.0 * control_1 - 3.0 * control_2 + end
  linear = 2.0 * (start - 2.0 * control_1 + control_2)
  constant = control_1 - start
  epsilon = 1e-12
  if abs(quadratic) < epsilon:
    if abs(linear) < epsilon:
      return set()
    root = -constant / linear
    return {root} if 0.0 < root < 1.0 else set()
  discriminant = linear * linear - 4.0 * quadratic * constant
  if discriminant < -epsilon:
    return set()
  discriminant = max(discriminant, 0.0)
  square_root = math.sqrt(discriminant)
  roots = {
    (-linear - square_root) / (2.0 * quadratic),
    (-linear + square_root) / (2.0 * quadratic),
  }
  return {root for root in roots if 0.0 < root < 1.0}


def _cubic_point(
  start: Point,
  control_1: Point,
  control_2: Point,
  end: Point,
  parameter: float,
) -> Point:
  inverse = 1.0 - parameter
  return (
    inverse ** 3 * start[0]
    + 3.0 * inverse ** 2 * parameter * control_1[0]
    + 3.0 * inverse * parameter ** 2 * control_2[0]
    + parameter ** 3 * end[0],
    inverse ** 3 * start[1]
    + 3.0 * inverse ** 2 * parameter * control_1[1]
    + 3.0 * inverse * parameter ** 2 * control_2[1]
    + parameter ** 3 * end[1],
  )


def _prototype(
  composite: CompositeSample,
  *,
  stitch_class: Any,
  stroke_width: float,
) -> ComponentPrototype:
  if set(composite.prototypes) != {"stitch"}:
    raise ValueError("post requires exactly one 'stitch' component prototype.")
  if not isinstance(stitch_class, str) or stitch_class not in STITCH_GEOMETRY_REGISTRY:
    raise ValueError(f"Unsupported post stitch class: {stitch_class!r}.")
  prototype = composite.prototypes["stitch"]
  if prototype.role != "stitch":
    raise ValueError("post stitch prototype role must be 'stitch'.")
  if (prototype.class_group, prototype.class_name) != ("primitive", stitch_class):
    raise ValueError(
      "post stitch prototype must resolve to the sampled primitive stitch class."
    )
  if prototype.occurrence_count is not None:
    raise ValueError("post stitch prototype occurrence_count must be unspecified.")
  if prototype.arrangement is not None:
    raise ValueError("post stitch prototype arrangement must be unspecified.")
  if prototype.declaration.get("inherit_generator") is not True:
    raise ValueError("post stitch prototype must inherit its primitive generator.")
  if not _same_number(prototype.inherited_parameters.get("stroke_width"), stroke_width):
    raise ValueError("post stitch prototype must inherit the parent stroke_width.")
  if not _same_number(prototype.sample.as_dict().get("stroke_width"), stroke_width):
    raise ValueError("post child sample stroke_width must equal the parent stroke_width.")
  return prototype


def _prototype_metadata(prototype: ComponentPrototype) -> dict[str, Any]:
  provenance = prototype.sample.provenance
  return {
    "role": prototype.role,
    "class_group": prototype.class_group,
    "class_name": prototype.class_name,
    "occurrence_count": prototype.occurrence_count,
    "arrangement": prototype.arrangement,
    "sampled_parameters": dict(prototype.sample.as_dict()),
    "inherited_parameters": dict(prototype.inherited_parameters),
    "sampling_policy": dict(prototype.sampling_policy),
    "replaced_parameters": list(prototype.replaced_parameters),
    "sampling_provenance": asdict(provenance) if provenance is not None else None,
  }


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


def _same_number(left: Any, right: float) -> bool:
  try:
    left_value = _finite(left, "stroke_width")
  except ValueError:
    return False
  return math.isclose(left_value, right, rel_tol=1e-12, abs_tol=1e-12)
