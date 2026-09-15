"""Joined-base increase generation with optional reusable chain geometry."""

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
  STITCH_GEOMETRY_REGISTRY,
  ChainGeometry,
  ChainPlacement,
  StitchGeometry,
  append_chain_geometry,
  append_stitch_geometry,
  rendered_px_to_viewbox,
  resolve_stroke_width,
)
from .fan import (
  SymmetricFanLayout,
  expand_bounds,
  fit_unit_bounds,
  symmetric_fan_layout,
  union_bounds,
)


CLASS_NAME = "increase"
SVG_NS = "http://www.w3.org/2000/svg"


def generate_increase(
  spec: ClassSpec,
  composite: CompositeSample,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate one symmetric joined-base fan from reusable child prototypes."""

  if not isinstance(composite, CompositeSample):
    raise TypeError("increase generation requires a realized CompositeSample.")
  if spec.class_group != "compound" or spec.class_name != CLASS_NAME:
    raise ValueError("generate_increase requires the compound.increase class specification.")

  parent = composite.parent
  values = parent.as_dict()
  stitch_class = values.get("stitch")
  count = _positive_count(values.get("count"), "increase count", minimum=2)
  chain_count = _positive_count(values.get("chain_count"), "increase chain_count")
  if chain_count != count - 1:
    raise ValueError("increase chain_count must equal count - 1.")
  chain_presence = values.get("chain_presence")
  if not isinstance(chain_presence, bool):
    raise ValueError("increase chain_presence must be a boolean.")
  chain_gap_ratio = _chain_gap_ratio(
    values.get("chain_gap_ratio"), chain_presence=chain_presence
  )
  spread_angle_deg = _spread(values.get("spread_angle_deg"))
  stroke_width = resolve_stroke_width(parent, config, class_name=CLASS_NAME)
  _topology(parent.topology)

  stitch_prototype = _stitch_prototype(
    composite,
    stitch_class=stitch_class,
    count=count,
    stroke_width=stroke_width,
    chain_presence=chain_presence,
  )
  stitch_class = stitch_prototype.class_name
  chain_prototype = _chain_prototype(
    composite,
    chain_presence=chain_presence,
    chain_count=chain_count,
    stroke_width=stroke_width,
  )
  stitch_values = stitch_prototype.sample.as_dict()
  chain_values = chain_prototype.sample.as_dict() if chain_prototype is not None else None

  unit_layout = symmetric_fan_layout(
    count=count,
    spread_angle_deg=spread_angle_deg,
    stem_length_px=1.0,
    join_px=(0.0, 0.0),
    joined_at="base",
  )
  unit_group = Element("g")
  unit_stitches = _append_stitches(
    unit_group,
    layout=unit_layout,
    class_name=stitch_class,
    sampled_values=stitch_values,
    config=config,
    stroke_width=stroke_width,
  )
  unit_chains = _append_chains(
    unit_group,
    layout=unit_layout,
    sampled_values=chain_values,
    gap_ratio=chain_gap_ratio,
    config=config,
    stroke_width=stroke_width,
  )
  unit_bounds = union_bounds(
    [geometry.centerline_bounds_px for geometry in (*unit_stitches, *unit_chains)],
    class_name=CLASS_NAME,
  )
  fit = fit_unit_bounds(
    unit_bounds,
    config=config,
    stroke_width=stroke_width,
    class_name=CLASS_NAME,
  )

  layout = symmetric_fan_layout(
    count=count,
    spread_angle_deg=spread_angle_deg,
    stem_length_px=fit.scale_px_per_unit,
    join_px=fit.origin_px,
    joined_at="base",
  )
  svg, group = _svg_root(config, stroke_width)
  stitches = _append_stitches(
    group,
    layout=layout,
    class_name=stitch_class,
    sampled_values=stitch_values,
    config=config,
    stroke_width=stroke_width,
  )
  chains = _append_chains(
    group,
    layout=layout,
    sampled_values=chain_values,
    gap_ratio=chain_gap_ratio,
    config=config,
    stroke_width=stroke_width,
  )
  centerline_bounds = union_bounds(
    [geometry.centerline_bounds_px for geometry in (*stitches, *chains)],
    class_name=CLASS_NAME,
  )
  rendered_bounds = expand_bounds(centerline_bounds, fit.stroke_width_px / 2.0)

  placements_metadata = [
    {
      "axis_angle_deg": angle,
      "base_px": list(placement.base_px),
      "top_px": list(placement.top_px),
    }
    for angle, placement in zip(
      layout.axis_angles_deg,
      layout.placements,
      strict=True,
    )
  ]
  chain_placements_metadata = [
    {
      "center_px": list(geometry.center_px),
      "visible_span_px": geometry.visible_span_px,
      "width_px": geometry.width_px,
      "height_px": geometry.height_px,
      "rotation_deg": geometry.rotation_deg,
      "centerline_bounds_px": list(geometry.centerline_bounds_px),
      "rendered_bounds_px": list(geometry.rendered_bounds_px),
    }
    for geometry in chains
  ]
  prototypes_metadata = {
    "stitch": _prototype_metadata(stitch_prototype),
  }
  if chain_prototype is not None:
    prototypes_metadata["chains"] = _prototype_metadata(chain_prototype)

  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "stitch": stitch_class,
    "count": count,
    "chain_presence": chain_presence,
    "chain_count": chain_count if chain_presence else 0,
    "chain_gap_ratio": chain_gap_ratio,
    "spread_angle_deg": spread_angle_deg,
    "stem_length_px": fit.scale_px_per_unit,
    "axis_angles_deg": list(layout.axis_angles_deg),
    "placements": placements_metadata,
    "chain_placements": chain_placements_metadata,
    "centerline_bounds_px": list(centerline_bounds),
    "rendered_bounds_px": list(rendered_bounds),
    "stroke_width": stroke_width,
    "stroke_width_px": fit.stroke_width_px,
    "component_prototypes": prototypes_metadata,
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


def _append_stitches(
  group: Element,
  *,
  layout: SymmetricFanLayout,
  class_name: str,
  sampled_values: Mapping[str, Any],
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
      include_top_bar=True,
    )
    for placement in layout.placements
  )


def _append_chains(
  group: Element,
  *,
  layout: SymmetricFanLayout,
  sampled_values: Mapping[str, Any] | None,
  gap_ratio: float,
  config: GenerationConfig,
  stroke_width: float,
) -> tuple[ChainGeometry, ...]:
  if sampled_values is None:
    return ()

  geometries: list[ChainGeometry] = []
  for left, right in zip(layout.placements, layout.placements[1:]):
    start = left.top_px
    end = right.top_px
    chord_length = math.dist(start, end)
    if chord_length <= 0.0:
      raise ValueError("increase adjacent stitch tops must be distinct.")
    center = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    rotation = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
    geometries.append(append_chain_geometry(
      group,
      sampled_values=sampled_values,
      config=config,
      placement=ChainPlacement(
        center_px=center,
        visible_span_px=gap_ratio * chord_length,
        rotation_deg=rotation,
      ),
      stroke_width=stroke_width,
    ))
  return tuple(geometries)


def _stitch_prototype(
  composite: CompositeSample,
  *,
  stitch_class: Any,
  count: int,
  stroke_width: float,
  chain_presence: bool,
) -> ComponentPrototype:
  expected_roles = {"stitch", "chains"} if chain_presence else {"stitch"}
  if set(composite.prototypes) != expected_roles:
    raise ValueError(
      f"increase requires component prototype roles {sorted(expected_roles)!r}."
    )
  if not isinstance(stitch_class, str) or stitch_class not in STITCH_GEOMETRY_REGISTRY:
    raise ValueError(f"Unsupported increase stitch class: {stitch_class!r}.")
  prototype = composite.prototypes["stitch"]
  if prototype.role != "stitch":
    raise ValueError("increase stitch prototype role must be 'stitch'.")
  if (prototype.class_group, prototype.class_name) != ("primitive", stitch_class):
    raise ValueError(
      "increase stitch prototype must resolve to the sampled primitive stitch class."
    )
  if prototype.occurrence_count != count:
    raise ValueError("increase stitch prototype occurrence_count must equal count.")
  if prototype.arrangement is not None:
    raise ValueError("increase stitch prototype arrangement must be unspecified.")
  _prototype_inheritance(prototype, stroke_width, role="stitch")
  return prototype


def _chain_prototype(
  composite: CompositeSample,
  *,
  chain_presence: bool,
  chain_count: int,
  stroke_width: float,
) -> ComponentPrototype | None:
  if not chain_presence:
    return None
  prototype = composite.prototypes["chains"]
  if prototype.role != "chains":
    raise ValueError("increase chain prototype role must be 'chains'.")
  if (prototype.class_group, prototype.class_name) != ("primitive", "ch"):
    raise ValueError("increase chains prototype must resolve to primitive.ch.")
  if prototype.occurrence_count != chain_count:
    raise ValueError("increase chains occurrence_count must equal chain_count.")
  if prototype.arrangement != "between_adjacent_tops":
    raise ValueError(
      "increase chains arrangement must be 'between_adjacent_tops'."
    )
  activation = prototype.declaration.get("when")
  if activation != {"parameter": "chain_presence", "equals": True}:
    raise ValueError("increase chains prototype must use the chain_presence activation.")
  _prototype_inheritance(prototype, stroke_width, role="chains")
  values = prototype.sample.as_dict()
  if values.get("shape") != "oval":
    raise ValueError("increase chains prototype shape must be 'oval'.")
  aspect_ratio = _positive_finite(
    values.get("aspect_ratio"), "increase chain aspect_ratio"
  )
  if aspect_ratio <= 1.0:
    raise ValueError(
      "increase chain aspect_ratio must exceed 1 so its major axis follows the gap."
    )
  return prototype


def _prototype_inheritance(
  prototype: ComponentPrototype,
  stroke_width: float,
  *,
  role: str,
) -> None:
  if prototype.declaration.get("inherit_generator") is not True:
    raise ValueError(
      f"increase {role} prototype must inherit its primitive generator."
    )
  if not _same_number(prototype.inherited_parameters.get("stroke_width"), stroke_width):
    raise ValueError(f"increase {role} prototype must inherit the parent stroke_width.")
  if not _same_number(prototype.sample.as_dict().get("stroke_width"), stroke_width):
    raise ValueError(
      f"increase {role} child sample stroke_width must equal the parent stroke_width."
    )


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


def _topology(topology: Mapping[str, Any]) -> None:
  if not isinstance(topology, Mapping):
    raise TypeError("increase topology must be a mapping.")
  if topology.get("base_relation") != "joined":
    raise ValueError("increase topology base_relation must be 'joined'.")
  if topology.get("top_relation") != "separate":
    raise ValueError("increase topology top_relation must be 'separate'.")
  connector = topology.get("connector")
  if not isinstance(connector, Mapping) or connector.get("enabled") is not False:
    raise ValueError("increase topology connector must be disabled.")


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


def _chain_gap_ratio(value: Any, *, chain_presence: bool) -> float:
  ratio = _finite(value, "increase chain_gap_ratio")
  if chain_presence:
    if not 0.0 < ratio < 1.0:
      raise ValueError(
        "increase chain_gap_ratio must be in the range (0, 1) when chains are present."
      )
  elif ratio != 0.0:
    raise ValueError(
      "increase chain_gap_ratio must be 0 when chains are absent."
    )
  return ratio


def _spread(value: Any) -> float:
  spread = _finite(value, "increase spread_angle_deg")
  if not 0.0 < spread < 180.0:
    raise ValueError("increase spread_angle_deg must be in the range (0, 180).")
  return spread


def _positive_count(value: Any, name: str, *, minimum: int = 1) -> int:
  if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
    raise ValueError(f"{name} must be an integer of at least {minimum}.")
  return value


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
