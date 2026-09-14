"""Joined-top compound stitch generation."""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Iterable
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
  append_stitch_geometry,
  rendered_px_to_viewbox,
  resolve_stroke_width,
)
from .fan import SymmetricFanLayout, symmetric_fan_layout


CLASS_NAME = "together"
SVG_NS = "http://www.w3.org/2000/svg"
Bounds = tuple[float, float, float, float]
Line = tuple[tuple[float, float], tuple[float, float]]


def generate_together(
  spec: ClassSpec,
  composite: CompositeSample,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate one symmetric joined-top compound from a reusable prototype."""

  if not isinstance(composite, CompositeSample):
    raise TypeError("together generation requires a realized CompositeSample.")
  if spec.class_group != "compound" or spec.class_name != CLASS_NAME:
    raise ValueError("generate_together requires the compound.together class specification.")

  parent = composite.parent
  values = parent.as_dict()
  stitch_class = values.get("stitch")
  count = _count(values.get("count"))
  spread_angle_deg = _spread(values.get("spread_angle_deg"))
  connector_type, connector_ratio = _connector(values)
  stroke_width = resolve_stroke_width(parent, config, class_name=CLASS_NAME)
  prototype = _prototype(
    composite,
    stitch_class=stitch_class,
    count=count,
    stroke_width=stroke_width,
  )
  _topology(parent.topology)

  prototype_values = prototype.sample.as_dict()
  unit_layout = symmetric_fan_layout(
    count=count,
    spread_angle_deg=spread_angle_deg,
    stem_length_px=1.0,
    join_px=(0.0, 0.0),
    joined_at="top",
  )
  unit_geometries = _append_stitches(
    Element("g"),
    layout=unit_layout,
    class_name=stitch_class,
    sampled_values=prototype_values,
    config=config,
    stroke_width=stroke_width,
  )
  unit_connector = _connector_line(
    join_px=(0.0, 0.0),
    connector_type=connector_type,
    connector_length_px=connector_ratio,
  )
  unit_bounds = _combined_centerline_bounds(unit_geometries, unit_connector)

  viewport_scale = _viewport_scale(config)
  stroke_width_px = stroke_width * viewport_scale
  target_visible_px = _positive_finite(
    config.target_visible_px, "together target_visible_px"
  )
  available_centerline_px = target_visible_px - stroke_width_px
  if available_centerline_px <= 0.0:
    raise ValueError(
      "together target_visible_px must exceed its rendered stroke width."
    )
  unit_span = max(
    unit_bounds[2] - unit_bounds[0],
    unit_bounds[3] - unit_bounds[1],
  )
  if unit_span <= 0.0:
    raise ValueError("together unit geometry must have positive visible extent.")
  stem_length_px = available_centerline_px / unit_span

  unit_center = (
    (unit_bounds[0] + unit_bounds[2]) / 2.0,
    (unit_bounds[1] + unit_bounds[3]) / 2.0,
  )
  canvas_center = (
    config.canvas_width_px / 2.0,
    config.canvas_height_px / 2.0,
  )
  join_px = (
    canvas_center[0] - stem_length_px * unit_center[0],
    canvas_center[1] - stem_length_px * unit_center[1],
  )
  final_layout = symmetric_fan_layout(
    count=count,
    spread_angle_deg=spread_angle_deg,
    stem_length_px=stem_length_px,
    join_px=join_px,
    joined_at="top",
  )

  svg, group = _svg_root(config, stroke_width)
  geometries = _append_stitches(
    group,
    layout=final_layout,
    class_name=stitch_class,
    sampled_values=prototype_values,
    config=config,
    stroke_width=stroke_width,
  )
  connector = _connector_line(
    join_px=join_px,
    connector_type=connector_type,
    connector_length_px=connector_ratio * stem_length_px,
  )
  if connector is not None:
    _append_pixel_line(group, config, connector)

  centerline_bounds = _combined_centerline_bounds(geometries, connector)
  stroke_radius_px = stroke_width_px / 2.0
  rendered_bounds = _expand_bounds(centerline_bounds, stroke_radius_px)
  placements_metadata = [
    {
      "axis_angle_deg": angle,
      "base_px": list(placement.base_px),
      "top_px": list(placement.top_px),
    }
    for angle, placement in zip(
      final_layout.axis_angles_deg,
      final_layout.placements,
      strict=True,
    )
  ]
  child_provenance = prototype.sample.provenance
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "stitch": stitch_class,
    "count": count,
    "spread_angle_deg": spread_angle_deg,
    "connector_type": connector_type,
    "connector_length": connector_ratio,
    "stem_length_px": stem_length_px,
    "connector_length_px": connector_ratio * stem_length_px,
    "axis_angles_deg": list(final_layout.axis_angles_deg),
    "placements": placements_metadata,
    "centerline_bounds_px": list(centerline_bounds),
    "rendered_bounds_px": list(rendered_bounds),
    "stroke_width": stroke_width,
    "stroke_width_px": stroke_width_px,
    "component_prototype": {
      "role": prototype.role,
      "class_group": prototype.class_group,
      "class_name": prototype.class_name,
      "occurrence_count": prototype.occurrence_count,
      "arrangement": prototype.arrangement,
      "sampled_parameters": dict(prototype_values),
      "inherited_parameters": dict(prototype.inherited_parameters),
      "sampling_policy": dict(prototype.sampling_policy),
      "replaced_parameters": list(prototype.replaced_parameters),
      "sampling_provenance": (
        asdict(child_provenance) if child_provenance is not None else None
      ),
    },
    "canvas": {
      "width_px": config.canvas_width_px,
      "height_px": config.canvas_height_px,
    },
    "target_visible_px": target_visible_px,
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
    sampling_provenance=parent.provenance,
  )


def _append_stitches(
  group: Element,
  *,
  layout: SymmetricFanLayout,
  class_name: str,
  sampled_values: dict[str, Any],
  config: GenerationConfig,
  stroke_width: float,
) -> tuple[StitchGeometry, ...]:
  return tuple(
    append_stitch_geometry(
      group,
      class_name=class_name,
      sampled_values=sampled_values,
      config=config,
      placement=placement,
      stroke_width=stroke_width,
      include_top_bar=False,
    )
    for placement in layout.placements
  )


def _prototype(
  composite: CompositeSample,
  *,
  stitch_class: Any,
  count: int,
  stroke_width: float,
) -> ComponentPrototype:
  if set(composite.prototypes) != {"stitch"}:
    raise ValueError("together requires exactly one 'stitch' component prototype.")
  if not isinstance(stitch_class, str) or stitch_class not in STITCH_GEOMETRY_REGISTRY:
    raise ValueError(f"Unsupported together stitch class: {stitch_class!r}.")
  prototype = composite.prototypes["stitch"]
  if prototype.role != "stitch":
    raise ValueError("together stitch prototype role must be 'stitch'.")
  if (prototype.class_group, prototype.class_name) != ("primitive", stitch_class):
    raise ValueError(
      "together stitch prototype must resolve to the sampled primitive stitch class."
    )
  if prototype.occurrence_count != count:
    raise ValueError("together stitch prototype occurrence_count must equal count.")
  if prototype.arrangement is not None:
    raise ValueError("together stitch prototype arrangement must be unspecified.")
  if prototype.declaration.get("inherit_generator") is not True:
    raise ValueError("together stitch prototype must inherit its primitive generator.")
  inherited_stroke = prototype.inherited_parameters.get("stroke_width")
  if not _same_number(inherited_stroke, stroke_width):
    raise ValueError("together stitch prototype must inherit the parent stroke_width.")
  child_stroke = prototype.sample.as_dict().get("stroke_width")
  if not _same_number(child_stroke, stroke_width):
    raise ValueError("together child sample stroke_width must equal the parent stroke_width.")
  return prototype


def _topology(topology: dict[str, Any]) -> None:
  if not isinstance(topology, dict):
    raise TypeError("together topology must be a mapping.")
  if topology.get("base_relation") != "separate":
    raise ValueError("together topology base_relation must be 'separate'.")
  if topology.get("top_relation") != "joined":
    raise ValueError("together topology top_relation must be 'joined'.")


def _connector(values: dict[str, Any]) -> tuple[str, float]:
  connector_type = values.get("connector_type")
  ratio = _finite(values.get("connector_length"), "together connector_length")
  stitch_class = values.get("stitch")
  if stitch_class == "sc":
    if connector_type != "none" or ratio != 0.0:
      raise ValueError(
        "together sc requires connector_type='none' and connector_length=0."
      )
  elif connector_type != "bar" or ratio <= 0.0:
    raise ValueError(
      "together taller stitches require connector_type='bar' and a positive connector_length."
    )
  return connector_type, ratio


def _connector_line(
  *,
  join_px: tuple[float, float],
  connector_type: str,
  connector_length_px: float,
) -> Line | None:
  if connector_type == "none":
    return None
  half = connector_length_px / 2.0
  return (
    (join_px[0] - half, join_px[1]),
    (join_px[0] + half, join_px[1]),
  )


def _append_pixel_line(group: Element, config: GenerationConfig, line: Line) -> None:
  start = rendered_px_to_viewbox(config, line[0])
  end = rendered_px_to_viewbox(config, line[1])
  SubElement(group, "line", {
    "x1": f"{start[0]:.8f}",
    "y1": f"{start[1]:.8f}",
    "x2": f"{end[0]:.8f}",
    "y2": f"{end[1]:.8f}",
  })


def _svg_root(
  config: GenerationConfig,
  stroke_width: float,
) -> tuple[Element, Element]:
  # Validate the viewport through the shared conversion before serializing it.
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


def _combined_centerline_bounds(
  geometries: Iterable[StitchGeometry],
  connector: Line | None,
) -> Bounds:
  bounds = [geometry.centerline_bounds_px for geometry in geometries]
  if connector is not None:
    bounds.append((
      min(connector[0][0], connector[1][0]),
      min(connector[0][1], connector[1][1]),
      max(connector[0][0], connector[1][0]),
      max(connector[0][1], connector[1][1]),
    ))
  if not bounds:
    raise ValueError("together geometry must contain at least one line.")
  return (
    min(item[0] for item in bounds),
    min(item[1] for item in bounds),
    max(item[2] for item in bounds),
    max(item[3] for item in bounds),
  )


def _expand_bounds(bounds: Bounds, amount: float) -> Bounds:
  return (
    bounds[0] - amount,
    bounds[1] - amount,
    bounds[2] + amount,
    bounds[3] + amount,
  )


def _viewport_scale(config: GenerationConfig) -> float:
  rendered_px_to_viewbox(config, (0.0, 0.0))
  return min(config.canvas_width_px, config.canvas_height_px) / 100.0


def _count(value: Any) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < 2:
    raise ValueError("together count must be an integer of at least 2.")
  return value


def _spread(value: Any) -> float:
  result = _finite(value, "together spread_angle_deg")
  if not 0.0 < result < 180.0:
    raise ValueError("together spread_angle_deg must be in the range (0, 180).")
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


def _same_number(left: Any, right: float) -> bool:
  try:
    left_value = _finite(left, "stroke_width")
  except ValueError:
    return False
  return math.isclose(left_value, right, rel_tol=1e-12, abs_tol=1e-12)
