"""Symmetric crossed-stitch generation from reusable child prototypes."""

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
  ChainGeometry,
  ChainPlacement,
  StitchGeometry,
  StitchPlacement,
  append_chain_geometry,
  append_stitch_geometry,
  rendered_px_to_viewbox,
  resolve_stroke_width,
)
from .fan import expand_bounds, fit_unit_bounds, union_bounds


CLASS_NAME = "crossed"
SVG_NS = "http://www.w3.org/2000/svg"
TOP_BAR_ANGLE_DEG = 0.0
Point = tuple[float, float]


@dataclass(frozen=True)
class CrossedLayout:
  """Two equal-length placements crossing at one shared midpoint."""

  axis_angles_deg: tuple[float, float]
  placements: tuple[StitchPlacement, StitchPlacement]
  crossing_px: Point
  top_chord_length_px: float


def generate_crossed(
  spec: ClassSpec,
  composite: CompositeSample,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate a symmetric X with an optional chain between its tops."""

  if not isinstance(composite, CompositeSample):
    raise TypeError("crossed generation requires a realized CompositeSample.")
  if spec.class_group != "compound" or spec.class_name != CLASS_NAME:
    raise ValueError(
      "generate_crossed requires the compound.crossed class specification."
    )

  parent = composite.parent
  values = parent.as_dict()
  stitch_class = values.get("stitch")
  chain_presence = values.get("chain_presence")
  if not isinstance(chain_presence, bool):
    raise ValueError("crossed chain_presence must be a boolean.")
  crossing_angle_deg = _crossing_angle(values.get("crossing_angle_deg"))
  chain_gap_ratio = _chain_gap_ratio(
    values.get("chain_gap_ratio"), chain_presence=chain_presence
  )
  stroke_width = resolve_stroke_width(parent, config, class_name=CLASS_NAME)
  _topology(parent.topology)

  stitch_prototype = _stitch_prototype(
    composite,
    stitch_class=stitch_class,
    chain_presence=chain_presence,
    stroke_width=stroke_width,
  )
  stitch_class = stitch_prototype.class_name
  chain_prototype = _chain_prototype(
    composite,
    chain_presence=chain_presence,
    stroke_width=stroke_width,
  )
  stitch_values = stitch_prototype.sample.as_dict()
  chain_values = (
    chain_prototype.sample.as_dict() if chain_prototype is not None else None
  )

  unit_layout = _crossed_layout(
    crossing_angle_deg=crossing_angle_deg,
    stem_length_px=1.0,
    crossing_px=(0.0, 0.0),
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
  unit_chain = _append_chain(
    unit_group,
    layout=unit_layout,
    sampled_values=chain_values,
    gap_ratio=chain_gap_ratio,
    config=config,
    stroke_width=stroke_width,
  )
  unit_bounds = union_bounds(
    [
      *(geometry.centerline_bounds_px for geometry in unit_stitches),
      *(() if unit_chain is None else (unit_chain.centerline_bounds_px,)),
    ],
    class_name=CLASS_NAME,
  )
  fit = fit_unit_bounds(
    unit_bounds,
    config=config,
    stroke_width=stroke_width,
    class_name=CLASS_NAME,
  )

  layout = _crossed_layout(
    crossing_angle_deg=crossing_angle_deg,
    stem_length_px=fit.scale_px_per_unit,
    crossing_px=fit.origin_px,
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
  chain = _append_chain(
    group,
    layout=layout,
    sampled_values=chain_values,
    gap_ratio=chain_gap_ratio,
    config=config,
    stroke_width=stroke_width,
  )
  centerline_bounds = union_bounds(
    [
      *(geometry.centerline_bounds_px for geometry in stitches),
      *(() if chain is None else (chain.centerline_bounds_px,)),
    ],
    class_name=CLASS_NAME,
  )
  rendered_bounds = expand_bounds(centerline_bounds, fit.stroke_width_px / 2.0)

  prototypes_metadata = {
    "stitch": _prototype_metadata(stitch_prototype),
  }
  if chain_prototype is not None:
    prototypes_metadata["chains"] = _prototype_metadata(chain_prototype)

  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "stitch": stitch_class,
    "chain_presence": chain_presence,
    "crossing_angle_deg": crossing_angle_deg,
    "chain_gap_ratio": chain_gap_ratio,
    "stem_length_px": fit.scale_px_per_unit,
    "top_bar_angle_deg": TOP_BAR_ANGLE_DEG,
    "crossing_point_px": list(layout.crossing_px),
    "top_chord_length_px": layout.top_chord_length_px,
    "axis_angles_deg": list(layout.axis_angles_deg),
    "placements": [
      {
        "position": position,
        "axis_angle_deg": angle,
        "base_px": list(placement.base_px),
        "top_px": list(placement.top_px),
      }
      for position, angle, placement in zip(
        ("left_base", "right_base"),
        layout.axis_angles_deg,
        layout.placements,
        strict=True,
      )
    ],
    "chain_placements": [] if chain is None else [_chain_metadata(chain)],
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


def _crossed_layout(
  *,
  crossing_angle_deg: float,
  stem_length_px: float,
  crossing_px: Point,
) -> CrossedLayout:
  angle = _crossing_angle(crossing_angle_deg)
  stem_length = _positive_finite(stem_length_px, "crossed stem_length_px")
  crossing = _point(crossing_px, "crossed crossing_px")
  half_angle = math.radians(angle / 2.0)
  half_horizontal = stem_length * math.sin(half_angle) / 2.0
  half_vertical = stem_length * math.cos(half_angle) / 2.0

  left_base = (
    crossing[0] - half_horizontal,
    crossing[1] + half_vertical,
  )
  right_base = (
    crossing[0] + half_horizontal,
    crossing[1] + half_vertical,
  )
  right_top = (
    crossing[0] + half_horizontal,
    crossing[1] - half_vertical,
  )
  left_top = (
    crossing[0] - half_horizontal,
    crossing[1] - half_vertical,
  )
  return CrossedLayout(
    axis_angles_deg=(angle / 2.0, -angle / 2.0),
    placements=(
      StitchPlacement(base_px=left_base, top_px=right_top),
      StitchPlacement(base_px=right_base, top_px=left_top),
    ),
    crossing_px=crossing,
    top_chord_length_px=math.dist(left_top, right_top),
  )


def _append_stitches(
  group: Element,
  *,
  layout: CrossedLayout,
  class_name: str,
  sampled_values: Mapping[str, Any],
  config: GenerationConfig,
  stroke_width: float,
) -> tuple[StitchGeometry, StitchGeometry]:
  geometries = [
    append_stitch_geometry(
      group,
      class_name=class_name,
      sampled_values=sampled_values,
      config=config,
      placement=placement,
      stroke_width=stroke_width,
      include_top_bar=True,
      top_bar_angle_deg=TOP_BAR_ANGLE_DEG,
    )
    for placement in layout.placements
  ]
  return (geometries[0], geometries[1])


def _append_chain(
  group: Element,
  *,
  layout: CrossedLayout,
  sampled_values: Mapping[str, Any] | None,
  gap_ratio: float,
  config: GenerationConfig,
  stroke_width: float,
) -> ChainGeometry | None:
  if sampled_values is None:
    return None
  top_a = layout.placements[0].top_px
  top_b = layout.placements[1].top_px
  center = (
    (top_a[0] + top_b[0]) / 2.0,
    (top_a[1] + top_b[1]) / 2.0,
  )
  return append_chain_geometry(
    group,
    sampled_values=sampled_values,
    config=config,
    placement=ChainPlacement(
      center_px=center,
      visible_span_px=gap_ratio * layout.top_chord_length_px,
      rotation_deg=0.0,
    ),
    stroke_width=stroke_width,
  )


def _stitch_prototype(
  composite: CompositeSample,
  *,
  stitch_class: Any,
  chain_presence: bool,
  stroke_width: float,
) -> ComponentPrototype:
  expected_roles = {"stitch", "chains"} if chain_presence else {"stitch"}
  if set(composite.prototypes) != expected_roles:
    raise ValueError(
      f"crossed requires component prototype roles {sorted(expected_roles)!r}."
    )
  if (
    not isinstance(stitch_class, str)
    or stitch_class not in {"hdc", "dc", "tr", "dtr"}
    or stitch_class not in STITCH_GEOMETRY_REGISTRY
  ):
    raise ValueError(f"Unsupported crossed stitch class: {stitch_class!r}.")
  prototype = composite.prototypes["stitch"]
  if prototype.role != "stitch":
    raise ValueError("crossed stitch prototype role must be 'stitch'.")
  if (prototype.class_group, prototype.class_name) != ("primitive", stitch_class):
    raise ValueError(
      "crossed stitch prototype must resolve to the sampled primitive stitch class."
    )
  if prototype.occurrence_count != 2:
    raise ValueError("crossed stitch prototype occurrence_count must equal 2.")
  if prototype.arrangement != "crossed":
    raise ValueError("crossed stitch prototype arrangement must be 'crossed'.")
  _prototype_inheritance(prototype, stroke_width, role="stitch")
  return prototype


def _chain_prototype(
  composite: CompositeSample,
  *,
  chain_presence: bool,
  stroke_width: float,
) -> ComponentPrototype | None:
  if not chain_presence:
    return None
  prototype = composite.prototypes["chains"]
  if prototype.role != "chains":
    raise ValueError("crossed chain prototype role must be 'chains'.")
  if (prototype.class_group, prototype.class_name) != ("primitive", "ch"):
    raise ValueError("crossed chains prototype must resolve to primitive.ch.")
  if prototype.occurrence_count != 1:
    raise ValueError("crossed chains occurrence_count must equal 1.")
  if prototype.arrangement != "between_adjacent_tops":
    raise ValueError(
      "crossed chains arrangement must be 'between_adjacent_tops'."
    )
  if prototype.declaration.get("when") != {
    "parameter": "chain_presence",
    "equals": True,
  }:
    raise ValueError("crossed chains prototype must use the chain_presence activation.")
  _prototype_inheritance(prototype, stroke_width, role="chains")
  chain_values = prototype.sample.as_dict()
  if chain_values.get("shape") != "oval":
    raise ValueError("crossed chains prototype shape must be 'oval'.")
  if _positive_finite(
    chain_values.get("aspect_ratio"), "crossed chain aspect_ratio"
  ) <= 1.0:
    raise ValueError(
      "crossed chain aspect_ratio must exceed 1 so its major axis follows the gap."
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
      f"crossed {role} prototype must inherit its primitive generator."
    )
  if not _same_number(prototype.inherited_parameters.get("stroke_width"), stroke_width):
    raise ValueError(f"crossed {role} prototype must inherit the parent stroke_width.")
  if not _same_number(prototype.sample.as_dict().get("stroke_width"), stroke_width):
    raise ValueError(
      f"crossed {role} child sample stroke_width must equal the parent stroke_width."
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


def _chain_metadata(chain: ChainGeometry) -> dict[str, Any]:
  return {
    "center_px": list(chain.center_px),
    "visible_span_px": chain.visible_span_px,
    "width_px": chain.width_px,
    "height_px": chain.height_px,
    "rotation_deg": chain.rotation_deg,
    "centerline_bounds_px": list(chain.centerline_bounds_px),
    "rendered_bounds_px": list(chain.rendered_bounds_px),
  }


def _topology(topology: Mapping[str, Any]) -> None:
  if not isinstance(topology, Mapping):
    raise TypeError("crossed topology must be a mapping.")
  if topology.get("base_relation") != "separate":
    raise ValueError("crossed topology base_relation must be 'separate'.")
  if topology.get("top_relation") != "separate":
    raise ValueError("crossed topology top_relation must be 'separate'.")
  connector = topology.get("connector")
  if not isinstance(connector, Mapping) or connector.get("enabled") is not False:
    raise ValueError("crossed topology connector must be disabled.")


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


def _crossing_angle(value: Any) -> float:
  angle = _finite(value, "crossed crossing_angle_deg")
  if not 0.0 < angle < 90.0:
    raise ValueError("crossed crossing_angle_deg must be acute and in (0, 90).")
  return angle


def _chain_gap_ratio(value: Any, *, chain_presence: bool) -> float:
  ratio = _finite(value, "crossed chain_gap_ratio")
  if chain_presence:
    if not 0.0 < ratio < 1.0:
      raise ValueError(
        "crossed chain_gap_ratio must be in the range (0, 1) when chains are present."
      )
  elif ratio != 0.0:
    raise ValueError("crossed chain_gap_ratio must be 0 when chains are absent.")
  return ratio


def _point(value: Any, name: str) -> Point:
  if not isinstance(value, tuple) or len(value) != 2:
    raise TypeError(f"{name} must be a two-item tuple.")
  return (_finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]"))


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
