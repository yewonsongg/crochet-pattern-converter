"""Three-chain picot generation from reusable component prototypes."""

from __future__ import annotations

from dataclasses import asdict
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
  ChainGeometry,
  ChainPlacement,
  SlipStitchGeometry,
  SlipStitchPlacement,
  append_chain_geometry,
  append_slip_stitch_geometry,
  rendered_px_to_viewbox,
  resolve_stroke_width,
)
from .fan import expand_bounds, fit_unit_bounds, union_bounds


CLASS_NAME = "ch3picot"
SVG_NS = "http://www.w3.org/2000/svg"
CHAIN_SPAN_UNITS = 1.0
CLOSURE_SPAN_UNITS = 0.30
Point = tuple[float, float]


def generate_ch3picot(
  spec: ClassSpec,
  composite: CompositeSample,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate an open three-chain loop with a slip-stitch closure."""

  if not isinstance(composite, CompositeSample):
    raise TypeError("ch3picot generation requires a realized CompositeSample.")
  if spec.class_group != "compound" or spec.class_name != CLASS_NAME:
    raise ValueError(
      "generate_ch3picot requires the compound.ch3picot class specification."
    )

  parent = composite.parent
  values = parent.as_dict()
  obsolete_parameters = {
    "radius", "opening_angle_deg", "frame_span", "frame_aspect_ratio",
  }.intersection(values)
  if obsolete_parameters:
    raise ValueError(
      "ch3picot no longer accepts obsolete parameters: "
      f"{sorted(obsolete_parameters)!r}."
    )
  junction_gap_ratio = _positive_finite(
    values.get("junction_gap_ratio"), "ch3picot junction_gap_ratio"
  )
  closure_offset = _closure_offset(values.get("closure_offset"))
  stroke_width = resolve_stroke_width(parent, config, class_name=CLASS_NAME)
  chains_prototype, closure_prototype = _prototypes(
    composite,
    stroke_width=stroke_width,
  )
  chain_values = chains_prototype.sample.as_dict()
  closure_values = closure_prototype.sample.as_dict()
  chain_aspect_ratio, closure_aspect_ratio = _prototype_values(
    chain_values, closure_values
  )

  chain_minor_span = CHAIN_SPAN_UNITS / chain_aspect_ratio
  closure_width = (
    CLOSURE_SPAN_UNITS
    if closure_aspect_ratio >= 1.0
    else CLOSURE_SPAN_UNITS * closure_aspect_ratio
  )
  minimum_base_half_span = (
    chain_minor_span / 2.0
    + closure_width / 2.0
    + junction_gap_ratio
  ) / (1.0 - abs(closure_offset))
  base_half_span = max(
    CHAIN_SPAN_UNITS / 2.0 + junction_gap_ratio / math.sqrt(2.0),
    minimum_base_half_span,
  )
  basal_y = CHAIN_SPAN_UNITS / 2.0 + junction_gap_ratio
  construction_centers: tuple[Point, Point, Point] = (
    (0.0, 0.0),
    (-base_half_span, basal_y),
    (base_half_span, basal_y),
  )
  chain_rotations: tuple[float, float, float] = (0.0, 90.0, 90.0)
  construction_closure_center = (
    closure_offset * base_half_span,
    basal_y,
  )
  junction_gap_units = math.hypot(
    base_half_span - CHAIN_SPAN_UNITS / 2.0,
    junction_gap_ratio,
  )
  left_closure_gap_units = (
    base_half_span + construction_closure_center[0]
    - chain_minor_span / 2.0 - closure_width / 2.0
  )
  right_closure_gap_units = (
    base_half_span - construction_closure_center[0]
    - chain_minor_span / 2.0 - closure_width / 2.0
  )

  probe_group = Element("g")
  probe_chains = _append_chains(
    probe_group,
    centers=construction_centers,
    rotations_deg=chain_rotations,
    visible_span_px=CHAIN_SPAN_UNITS,
    sampled_values=chain_values,
    config=config,
    stroke_width=stroke_width,
  )
  probe_closure = append_slip_stitch_geometry(
    probe_group,
    sampled_values=closure_values,
    config=config,
    placement=SlipStitchPlacement(
      center_px=construction_closure_center,
      visible_span_px=CLOSURE_SPAN_UNITS,
    ),
    stroke_width=stroke_width,
  )
  unit_bounds = union_bounds(
    [
      *(geometry.centerline_bounds_px for geometry in probe_chains),
      probe_closure.centerline_bounds_px,
    ],
    class_name=CLASS_NAME,
  )
  fit = fit_unit_bounds(
    unit_bounds,
    config=config,
    stroke_width=stroke_width,
    class_name=CLASS_NAME,
  )

  centers = tuple(
    _transform_point(point, fit.origin_px, fit.scale_px_per_unit)
    for point in construction_centers
  )
  closure_center = _transform_point(
    construction_closure_center,
    fit.origin_px,
    fit.scale_px_per_unit,
  )
  svg, group = _svg_root(config, stroke_width)
  chains = _append_chains(
    group,
    centers=centers,
    rotations_deg=chain_rotations,
    visible_span_px=CHAIN_SPAN_UNITS * fit.scale_px_per_unit,
    sampled_values=chain_values,
    config=config,
    stroke_width=stroke_width,
  )
  closure = append_slip_stitch_geometry(
    group,
    sampled_values=closure_values,
    config=config,
    placement=SlipStitchPlacement(
      center_px=closure_center,
      visible_span_px=CLOSURE_SPAN_UNITS * fit.scale_px_per_unit,
    ),
    stroke_width=stroke_width,
  )
  centerline_bounds = union_bounds(
    [
      *(geometry.centerline_bounds_px for geometry in chains),
      closure.centerline_bounds_px,
    ],
    class_name=CLASS_NAME,
  )
  rendered_bounds = expand_bounds(centerline_bounds, fit.stroke_width_px / 2.0)

  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "junction_gap_ratio": junction_gap_ratio,
    "closure_offset": closure_offset,
    "construction_base_span": base_half_span * 2.0,
    "construction_trace_height": basal_y,
    "construction_junction_gap": junction_gap_units,
    "construction_closure_gaps": [
      left_closure_gap_units, right_closure_gap_units,
    ],
    "construction_scale_px_per_unit": fit.scale_px_per_unit,
    "base_span_px": base_half_span * 2.0 * fit.scale_px_per_unit,
    "trace_height_px": basal_y * fit.scale_px_per_unit,
    "junction_gap_px": junction_gap_units * fit.scale_px_per_unit,
    "closure_gaps_px": [
      left_closure_gap_units * fit.scale_px_per_unit,
      right_closure_gap_units * fit.scale_px_per_unit,
    ],
    "chain_span_px": CHAIN_SPAN_UNITS * fit.scale_px_per_unit,
    "closure_span_px": CLOSURE_SPAN_UNITS * fit.scale_px_per_unit,
    "chain_placements": [
      _chain_metadata(position, geometry)
      for position, geometry in zip(
        ("top", "lower_left", "lower_right"), chains, strict=True
      )
    ],
    "closure_placement": _closure_metadata(closure),
    "centerline_bounds_px": list(centerline_bounds),
    "rendered_bounds_px": list(rendered_bounds),
    "stroke_width": stroke_width,
    "stroke_width_px": fit.stroke_width_px,
    "component_prototypes": {
      "chains": _prototype_metadata(chains_prototype),
      "closure": _prototype_metadata(closure_prototype),
    },
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
    sampling_provenance=parent.provenance,
  )


def _append_chains(
  group: Element,
  *,
  centers: tuple[Point, Point, Point],
  rotations_deg: tuple[float, float, float],
  visible_span_px: float,
  sampled_values: Mapping[str, Any],
  config: GenerationConfig,
  stroke_width: float,
) -> tuple[ChainGeometry, ChainGeometry, ChainGeometry]:
  geometries: list[ChainGeometry] = []
  for center, rotation in zip(centers, rotations_deg, strict=True):
    geometries.append(append_chain_geometry(
      group,
      sampled_values=sampled_values,
      config=config,
      placement=ChainPlacement(
        center_px=center,
        visible_span_px=visible_span_px,
        rotation_deg=rotation,
      ),
      stroke_width=stroke_width,
    ))
  return geometries[0], geometries[1], geometries[2]


def _prototypes(
  composite: CompositeSample,
  *,
  stroke_width: float,
) -> tuple[ComponentPrototype, ComponentPrototype]:
  if set(composite.prototypes) != {"chains", "closure"}:
    raise ValueError(
      "ch3picot requires exactly 'chains' and 'closure' component prototypes."
    )
  chains = composite.prototypes["chains"]
  closure = composite.prototypes["closure"]

  if chains.role != "chains" or (
    chains.class_group, chains.class_name
  ) != ("primitive", "ch"):
    raise ValueError("ch3picot chains prototype must resolve to primitive.ch.")
  if chains.occurrence_count != 3:
    raise ValueError("ch3picot chains occurrence_count must equal 3.")
  if chains.arrangement != "triangular":
    raise ValueError("ch3picot chains arrangement must be 'triangular'.")
  _prototype_inheritance(chains, stroke_width, role="chains", child_has_stroke=True)

  if closure.role != "closure" or (
    closure.class_group, closure.class_name
  ) != ("primitive", "slst"):
    raise ValueError("ch3picot closure prototype must resolve to primitive.slst.")
  if closure.occurrence_count != 1:
    raise ValueError("ch3picot closure occurrence_count must equal 1.")
  if closure.arrangement is not None:
    raise ValueError("ch3picot closure arrangement must be unspecified.")
  _prototype_inheritance(
    closure, stroke_width, role="closure", child_has_stroke=False
  )
  return chains, closure


def _prototype_inheritance(
  prototype: ComponentPrototype,
  stroke_width: float,
  *,
  role: str,
  child_has_stroke: bool,
) -> None:
  if prototype.declaration.get("inherit_generator") is not True:
    raise ValueError(f"ch3picot {role} prototype must inherit its generator.")
  if not _same_number(prototype.inherited_parameters.get("stroke_width"), stroke_width):
    raise ValueError(f"ch3picot {role} prototype must inherit parent stroke_width.")
  child_stroke = prototype.sample.as_dict().get("stroke_width")
  if child_has_stroke and not _same_number(child_stroke, stroke_width):
    raise ValueError(
      f"ch3picot {role} child stroke_width must equal parent stroke_width."
    )
  if not child_has_stroke and child_stroke is not None:
    raise ValueError(f"ch3picot {role} child must not sample stroke_width.")


def _prototype_values(
  chain_values: Mapping[str, Any],
  closure_values: Mapping[str, Any],
) -> tuple[float, float]:
  if chain_values.get("shape") != "oval":
    raise ValueError("ch3picot chains prototype shape must be 'oval'.")
  chain_aspect = _positive_finite(
    chain_values.get("aspect_ratio"), "ch3picot chain aspect_ratio"
  )
  if chain_aspect <= 1.0:
    raise ValueError("ch3picot chain aspect_ratio must exceed 1.")
  if closure_values.get("shape") != "circle":
    raise ValueError("ch3picot closure prototype shape must be 'circle'.")
  closure_aspect = _positive_finite(
    closure_values.get("aspect_ratio"), "ch3picot closure aspect_ratio"
  )
  return chain_aspect, closure_aspect


def _chain_metadata(position: str, geometry: ChainGeometry) -> dict[str, Any]:
  return {
    "position": position,
    "center_px": list(geometry.center_px),
    "visible_span_px": geometry.visible_span_px,
    "width_px": geometry.width_px,
    "height_px": geometry.height_px,
    "rotation_deg": geometry.rotation_deg,
    "centerline_bounds_px": list(geometry.centerline_bounds_px),
    "rendered_bounds_px": list(geometry.rendered_bounds_px),
  }


def _closure_metadata(geometry: SlipStitchGeometry) -> dict[str, Any]:
  return {
    "center_px": list(geometry.center_px),
    "visible_span_px": geometry.visible_span_px,
    "width_px": geometry.width_px,
    "height_px": geometry.height_px,
    "rotation_deg": geometry.rotation_deg,
    "centerline_bounds_px": list(geometry.centerline_bounds_px),
    "rendered_bounds_px": list(geometry.rendered_bounds_px),
  }


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


def _transform_point(point: Point, origin: Point, scale: float) -> Point:
  return origin[0] + scale * point[0], origin[1] + scale * point[1]


def _closure_offset(value: Any) -> float:
  offset = _finite(value, "ch3picot closure_offset")
  if not -1.0 < offset < 1.0:
    raise ValueError("ch3picot closure_offset must be in the range (-1, 1).")
  return offset


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
