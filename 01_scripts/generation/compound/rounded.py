"""Rounded cluster and popcorn compound generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping
from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.models import (
  ComponentPrototype,
  CompositeSample,
  GeneratedObject,
  GenerationConfig,
)
from ..core.sampling.schema import ClassSpec
from ..core.svg import rendered_px_to_viewbox, resolve_stroke_width
from .fan import expand_bounds, fit_unit_bounds, union_bounds


CLASS_NAME = "rounded"
SVG_NS = "http://www.w3.org/2000/svg"
SUPPORTED_STITCHES = frozenset({"hdc", "dc", "tr", "dtr"})
CROSSBAR_COUNTS = {"hdc": 0, "dc": 1, "tr": 2, "dtr": 3}
PARABOLIC_BULGE_RATIO = 0.24
Point = tuple[float, float]
Bounds = tuple[float, float, float, float]
Line = tuple[Point, Point]


@dataclass(frozen=True)
class _Curve:
  start: Point
  midpoint: Point
  end: Point
  quadratic_control: Point | None
  center: Point | None
  radius: float | None
  radius_y: float | None
  start_angle_rad: float | None
  sweep_rad: float
  length: float
  bounds: Bounds


@dataclass(frozen=True)
class _Crossbar:
  path_position: float
  center: Point
  tangent: Point
  sampled_angle_deg: float
  angle_deg: float
  length: float
  line: Line


@dataclass(frozen=True)
class _CurvedStitch:
  rank: float
  base: Point
  top: Point
  outward_midpoint: Point
  curve: _Curve
  crossbars: tuple[_Crossbar, ...]
  centerline_bounds: Bounds


@dataclass(frozen=True)
class _PopcornStitch:
  rank: float
  base: Point
  top: Point
  outward_midpoint: Point
  curve: _Curve
  crossbars: tuple[_Crossbar, ...]
  centerline_bounds: Bounds
  provisional_top: Point
  full_curve: _Curve
  trim_parameter: float
  discarded_length: float


@dataclass(frozen=True)
class _Connector:
  connector_type: str
  width: float
  line: Line | None
  curve: _Curve | None
  tails: tuple[_Curve, ...]
  centerline_bounds: Bounds


@dataclass(frozen=True)
class _RoundedLayout:
  base_connector_width: float
  connector_width: float
  bottom_span: float
  nominal_height: float
  effective_stitch_spacing_ratio: float
  minimum_top_anchor_gap: float
  minimum_top_anchor_horizontal_gap: float
  top_anchor_spacing: float | None
  connector_overhang: float | None
  ranks: tuple[float, ...]
  stitches: tuple[_CurvedStitch, ...]
  connector: _Connector
  centerline_bounds: Bounds


@dataclass(frozen=True)
class _PopcornLayout:
  nominal_height: float
  minimum_gap: float
  pitch: float
  effective_stitch_spacing_ratio: float
  top_span: float
  bottom_span: float
  connector_overhang: float
  connector_width: float
  connector_half_width: float
  connector_depth: float
  connector_circle_center: Point
  connector_circle_radius: float
  ranks: tuple[float, ...]
  stitches: tuple[_PopcornStitch, ...]
  connector: _Connector
  centerline_bounds: Bounds


def generate_rounded(
  spec: ClassSpec,
  composite: CompositeSample,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate a rounded cluster or popcorn from one reusable stitch prototype."""

  if not isinstance(composite, CompositeSample):
    raise TypeError("rounded generation requires a realized CompositeSample.")
  if spec.class_group != "compound" or spec.class_name != CLASS_NAME:
    raise ValueError("generate_rounded requires the compound.rounded class specification.")

  parent = composite.parent
  values = parent.as_dict()
  variant = values.get("variant")
  if variant not in {"cluster", "popcorn"}:
    raise ValueError(f"Unsupported rounded variant: {variant!r}.")
  stitch_class = values.get("stitch")
  count = _variant_count(values.get("count"), variant=variant)
  connector_type = _connector_type(values.get("connector_type"), variant=variant)
  bundle_width_ratio = (
    _open_unit_interval(
      values.get("bundle_width_ratio"), "rounded bundle_width_ratio"
    )
    if variant == "cluster"
    else _positive_finite(
      values.get("bundle_width_ratio"), "rounded bundle_width_ratio"
    )
  )
  stitch_spacing_ratio = _positive_finite(
    values.get("stitch_spacing_ratio"), "rounded stitch_spacing_ratio"
  )
  minimum_stitch_gap_ratio = _positive_finite(
    values.get("minimum_stitch_gap_ratio"),
    "rounded minimum_stitch_gap_ratio",
  )
  connector_overhang_ratio = _positive_finite(
    values.get("connector_overhang_ratio"),
    "rounded connector_overhang_ratio",
  )
  connector_depth_ratio = (
    _positive_finite(
      values.get("connector_depth_ratio"),
      "rounded connector_depth_ratio",
    )
    if variant == "popcorn"
    else 0.0
  )
  stroke_width = resolve_stroke_width(parent, config, class_name=CLASS_NAME)
  _topology(parent.topology)
  prototype = _prototype(
    composite,
    stitch_class=stitch_class,
    count=count,
    stroke_width=stroke_width,
  )
  stitch_class = prototype.class_name
  sampled_values = prototype.sample.as_dict()
  _validate_stitch_values(stitch_class, sampled_values)

  unit_layout = _layout(
    variant=variant,
    connector_type=connector_type,
    stitch_class=stitch_class,
    count=count,
    bundle_width_ratio=bundle_width_ratio,
    stitch_spacing_ratio=stitch_spacing_ratio,
    minimum_stitch_gap_ratio=minimum_stitch_gap_ratio,
    connector_overhang_ratio=connector_overhang_ratio,
    connector_depth_ratio=connector_depth_ratio,
    sampled_values=sampled_values,
    nominal_height=1.0,
    origin=(0.0, 0.0),
  )
  fit = fit_unit_bounds(
    unit_layout.centerline_bounds,
    config=config,
    stroke_width=stroke_width,
    class_name=CLASS_NAME,
  )
  layout = _layout(
    variant=variant,
    connector_type=connector_type,
    stitch_class=stitch_class,
    count=count,
    bundle_width_ratio=bundle_width_ratio,
    stitch_spacing_ratio=stitch_spacing_ratio,
    minimum_stitch_gap_ratio=minimum_stitch_gap_ratio,
    connector_overhang_ratio=connector_overhang_ratio,
    connector_depth_ratio=connector_depth_ratio,
    sampled_values=sampled_values,
    nominal_height=fit.scale_px_per_unit,
    origin=fit.origin_px,
  )

  svg, group = _svg_root(config, stroke_width)
  for stitch in layout.stitches:
    _append_curve(group, config=config, curve=stitch.curve)
    for crossbar in stitch.crossbars:
      _append_line(group, config=config, line=crossbar.line)
  _append_connector(group, config=config, connector=layout.connector)

  rendered_bounds = expand_bounds(
    layout.centerline_bounds, fit.stroke_width_px / 2.0
  )
  if variant == "cluster":
    if not isinstance(layout, _RoundedLayout):
      raise TypeError("rounded cluster must produce the quarantined cluster layout.")
    metadata = {
      "class_id": spec.class_id,
      "class_name": spec.class_name,
      "variant": variant,
      "stitch": stitch_class,
      "count": count,
      "connector_type": connector_type,
      "bundle_width_ratio": bundle_width_ratio,
      "stitch_spacing_ratio": stitch_spacing_ratio,
      "effective_stitch_spacing_ratio": layout.effective_stitch_spacing_ratio,
      "minimum_stitch_gap_ratio": minimum_stitch_gap_ratio,
      "minimum_top_anchor_gap_px": layout.minimum_top_anchor_gap,
      "minimum_top_anchor_horizontal_gap_px": (
        layout.minimum_top_anchor_horizontal_gap
      ),
      "connector_overhang_ratio": connector_overhang_ratio,
      "base_connector_width_px": layout.base_connector_width,
      "connector_width_px": layout.connector_width,
      "top_span_px": (
        layout.top_anchor_spacing * (count - 1)
        if layout.top_anchor_spacing is not None else None
      ),
      "bottom_span_px": layout.bottom_span,
      "nominal_height_px": layout.nominal_height,
      "top_anchor_spacing_px": layout.top_anchor_spacing,
      "connector_overhang_px": layout.connector_overhang,
      "ranks": list(layout.ranks),
      "stitches": [_stitch_metadata(stitch) for stitch in layout.stitches],
      "connector": _connector_metadata(layout.connector),
      "centerline_bounds_px": list(layout.centerline_bounds),
      "rendered_bounds_px": list(rendered_bounds),
      "stroke_width": stroke_width,
      "stroke_width_px": fit.stroke_width_px,
      "component_prototype": _prototype_metadata(prototype),
      "canvas": {
        "width_px": config.canvas_width_px,
        "height_px": config.canvas_height_px,
      },
      "target_visible_px": float(config.target_visible_px),
    }
  else:
    if not isinstance(layout, _PopcornLayout):
      raise TypeError("rounded popcorn must produce a stitch-first popcorn layout.")
    metadata = {
      "class_id": spec.class_id,
      "class_name": spec.class_name,
      "variant": variant,
      "stitch": stitch_class,
      "count": count,
      "connector_type": connector_type,
      "bundle_width_ratio": bundle_width_ratio,
      "stitch_spacing_ratio": stitch_spacing_ratio,
      "effective_stitch_spacing_ratio": layout.effective_stitch_spacing_ratio,
      "minimum_stitch_gap_ratio": minimum_stitch_gap_ratio,
      "minimum_stitch_gap_px": layout.minimum_gap,
      "stitch_pitch_px": layout.pitch,
      "top_span_px": layout.top_span,
      "bottom_span_px": layout.bottom_span,
      "connector_overhang_ratio": connector_overhang_ratio,
      "connector_overhang_px": layout.connector_overhang,
      "connector_curve_type": "circular_segment",
      "connector_depth_ratio": connector_depth_ratio,
      "connector_width_px": layout.connector_width,
      "connector_half_width_px": layout.connector_half_width,
      "connector_depth_px": layout.connector_depth,
      "connector_circle_center_px": list(layout.connector_circle_center),
      "connector_circle_radius_px": layout.connector_circle_radius,
      "nominal_height_px": layout.nominal_height,
      "ranks": list(layout.ranks),
      "stitches": [
        _popcorn_stitch_metadata(stitch) for stitch in layout.stitches
      ],
      "connector": _connector_metadata(layout.connector),
      "centerline_bounds_px": list(layout.centerline_bounds),
      "rendered_bounds_px": list(rendered_bounds),
      "stroke_width": stroke_width,
      "stroke_width_px": fit.stroke_width_px,
      "component_prototype": _prototype_metadata(prototype),
      "canvas": {
        "width_px": config.canvas_width_px,
        "height_px": config.canvas_height_px,
      },
      "target_visible_px": float(config.target_visible_px),
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


def _effective_spacing_ratio(
  *,
  variant: str,
  count: int,
  sampled_ratio: float,
  connector_overhang_ratio: float,
) -> tuple[float, float]:
  """Keep anchors on the connector and convert excess spacing into width."""

  if variant == "cluster":
    geometric_limit = (
      count - 1 + 2.0 * connector_overhang_ratio
    ) / (count - 1)
  elif variant == "popcorn":
    geometric_limit = (count + 1) / (count - 1)
  else:
    raise ValueError(f"Unsupported rounded variant: {variant!r}.")

  safe_limit = 0.98 * geometric_limit
  effective = min(sampled_ratio, safe_limit)
  overflow_scale = max(1.0, sampled_ratio / effective)
  return effective, overflow_scale


def _top_horizontal_gap_per_connector_width(
  *,
  variant: str,
  count: int,
  spacing_ratio: float,
  connector_overhang_ratio: float,
) -> float:
  """Return the smallest adjacent horizontal gap for a unit connector."""

  if variant == "cluster":
    gap = spacing_ratio / (
      count - 1 + 2.0 * connector_overhang_ratio
    )
  elif variant == "popcorn":
    radius_x = 0.5
    anchor_xs: list[float] = []
    for index in range(count):
      nominal_fraction = (index + 1) / (count + 1)
      fraction = 0.5 + (nominal_fraction - 0.5) * spacing_ratio
      angle = math.pi - math.pi * fraction
      anchor_xs.append(radius_x * math.cos(angle))
    gap = min(
      abs(right - left)
      for left, right in zip(anchor_xs, anchor_xs[1:])
    )
  else:
    raise ValueError(f"Unsupported rounded variant: {variant!r}.")

  if not math.isfinite(gap) or gap <= 0.0:
    raise ValueError(
      "rounded stitch spacing produced a degenerate horizontal anchor gap."
    )
  return gap


def _layout(
  *,
  variant: str,
  connector_type: str,
  stitch_class: str,
  count: int,
  bundle_width_ratio: float,
  stitch_spacing_ratio: float,
  minimum_stitch_gap_ratio: float,
  connector_overhang_ratio: float,
  connector_depth_ratio: float,
  sampled_values: Mapping[str, Any],
  nominal_height: float,
  origin: Point,
) -> _RoundedLayout | _PopcornLayout:
  if variant == "cluster":
    return _cluster_layout(
      connector_type=connector_type,
      stitch_class=stitch_class,
      count=count,
      bundle_width_ratio=bundle_width_ratio,
      stitch_spacing_ratio=stitch_spacing_ratio,
      minimum_stitch_gap_ratio=minimum_stitch_gap_ratio,
      connector_overhang_ratio=connector_overhang_ratio,
      sampled_values=sampled_values,
      nominal_height=nominal_height,
      origin=origin,
    )
  if variant == "popcorn":
    return _popcorn_layout(
      connector_type=connector_type,
      stitch_class=stitch_class,
      count=count,
      bundle_width_ratio=bundle_width_ratio,
      stitch_spacing_ratio=stitch_spacing_ratio,
      minimum_stitch_gap_ratio=minimum_stitch_gap_ratio,
      connector_overhang_ratio=connector_overhang_ratio,
      connector_depth_ratio=connector_depth_ratio,
      sampled_values=sampled_values,
      nominal_height=nominal_height,
      origin=origin,
    )
  raise ValueError(f"Unsupported rounded variant: {variant!r}.")


def _cluster_layout(
  *,
  connector_type: str,
  stitch_class: str,
  count: int,
  bundle_width_ratio: float,
  stitch_spacing_ratio: float,
  minimum_stitch_gap_ratio: float,
  connector_overhang_ratio: float,
  sampled_values: Mapping[str, Any],
  nominal_height: float,
  origin: Point,
) -> _RoundedLayout:
  """Frozen cluster construction, isolated from popcorn geometry."""

  height = _positive_finite(nominal_height, "rounded nominal height")
  origin = _point(origin, "rounded origin")
  bar_stem_ratio = _ratio(
    sampled_values.get("bar_stem_ratio"), "rounded child bar_stem_ratio"
  )
  base_connector_width = bar_stem_ratio * height
  effective_spacing_ratio, overflow_scale = _effective_spacing_ratio(
    variant="cluster",
    count=count,
    sampled_ratio=stitch_spacing_ratio,
    connector_overhang_ratio=connector_overhang_ratio,
  )
  gap_per_connector_width = _top_horizontal_gap_per_connector_width(
    variant="cluster",
    count=count,
    spacing_ratio=effective_spacing_ratio,
    connector_overhang_ratio=connector_overhang_ratio,
  )
  required_gap = minimum_stitch_gap_ratio * height
  connector_width = max(
    base_connector_width * overflow_scale,
    required_gap / gap_per_connector_width,
  )
  ranks = tuple(-1.0 + 2.0 * index / (count - 1) for index in range(count))
  top_baseline = origin[1] - height / 2.0
  base_baseline = origin[1] + height / 2.0
  nominal_top_step = connector_width / (
    count - 1 + 2.0 * connector_overhang_ratio
  )
  top_step = nominal_top_step * effective_spacing_ratio
  occupied_top_span = (count - 1) * top_step
  bottom_span = bundle_width_ratio * occupied_top_span
  edge_overhang = (connector_width - occupied_top_span) / 2.0
  connector_left = origin[0] - connector_width / 2.0
  top_anchors = [
    (connector_left + edge_overhang + index * top_step, top_baseline)
    for index in range(count)
  ]
  base_anchors = [
    (origin[0] + rank * bottom_span / 2.0, base_baseline)
    for rank in ranks
  ]
  connector_line = (
    (origin[0] - connector_width / 2.0, top_baseline),
    (origin[0] + connector_width / 2.0, top_baseline),
  )
  connector = _Connector(
    connector_type=connector_type,
    width=connector_width,
    line=connector_line,
    curve=None,
    tails=(),
    centerline_bounds=_line_bounds(connector_line),
  )
  minimum_top_anchor_horizontal_gap = _positive_finite(
    min(
      abs(right[0] - left[0])
      for left, right in zip(top_anchors, top_anchors[1:])
    ),
    "rounded minimum horizontal top-anchor gap",
  )
  stitches = tuple(
    _cluster_curved_stitch(
      rank=rank,
      base=base,
      top=top,
      connector_width=base_connector_width,
      stitch_class=stitch_class,
      sampled_values=sampled_values,
    )
    for rank, base, top in zip(ranks, base_anchors, top_anchors, strict=True)
  )
  bounds = union_bounds(
    [
      *(stitch.centerline_bounds for stitch in stitches),
      connector.centerline_bounds,
    ],
    class_name=CLASS_NAME,
  )
  return _RoundedLayout(
    base_connector_width=base_connector_width,
    connector_width=connector_width,
    bottom_span=bottom_span,
    nominal_height=height,
    effective_stitch_spacing_ratio=effective_spacing_ratio,
    minimum_top_anchor_gap=top_step,
    minimum_top_anchor_horizontal_gap=minimum_top_anchor_horizontal_gap,
    top_anchor_spacing=top_step,
    connector_overhang=edge_overhang,
    ranks=ranks,
    stitches=stitches,
    connector=connector,
    centerline_bounds=bounds,
  )


def _popcorn_layout(
  *,
  connector_type: str,
  stitch_class: str,
  count: int,
  bundle_width_ratio: float,
  stitch_spacing_ratio: float,
  minimum_stitch_gap_ratio: float,
  connector_overhang_ratio: float,
  connector_depth_ratio: float,
  sampled_values: Mapping[str, Any],
  nominal_height: float,
  origin: Point,
) -> _PopcornLayout:
  height = _positive_finite(nominal_height, "rounded nominal height")
  origin = _point(origin, "rounded origin")
  minimum_gap = minimum_stitch_gap_ratio * height
  effective_spacing_ratio = max(1.0, stitch_spacing_ratio)
  pitch = minimum_gap * effective_spacing_ratio
  top_span = (count - 1) * pitch
  bottom_span = bundle_width_ratio * top_span
  connector_overhang = connector_overhang_ratio * pitch
  connector_width = top_span + 2.0 * connector_overhang
  connector_half_width = connector_width / 2.0
  connector_depth = connector_depth_ratio * pitch
  if connector_depth >= connector_half_width:
    raise ValueError(
      "rounded popcorn connector depth must be less than its half-width."
    )
  ranks = tuple(-1.0 + 2.0 * index / (count - 1) for index in range(count))
  top_baseline = origin[1] - height / 2.0
  base_baseline = origin[1] + height / 2.0
  provisional_tops = tuple(
    (origin[0] + rank * top_span / 2.0, top_baseline)
    for rank in ranks
  )
  base_anchors = tuple(
    (origin[0] + rank * bottom_span / 2.0, base_baseline)
    for rank in ranks
  )
  connector_center = (origin[0], top_baseline)
  connector_curve = _circle_curve(
    (
      connector_center[0] - connector_half_width,
      connector_center[1],
    ),
    (
      connector_center[0],
      connector_center[1] + connector_depth,
    ),
    (
      connector_center[0] + connector_half_width,
      connector_center[1],
    ),
  )
  if connector_curve.center is None or connector_curve.radius is None:
    raise ValueError("rounded popcorn connector must resolve to a circular arc.")
  connector = _Connector(
    connector_type=connector_type,
    width=connector_width,
    line=None,
    curve=connector_curve,
    tails=(),
    centerline_bounds=connector_curve.bounds,
  )
  stitches = tuple(
    _popcorn_curved_stitch(
      rank=rank,
      base=base,
      provisional_top=provisional_top,
      pitch=pitch,
      connector_chord_center=connector_center,
      connector_half_width=connector_half_width,
      connector_circle_center=connector_curve.center,
      connector_circle_radius=connector_curve.radius,
      stitch_class=stitch_class,
      sampled_values=sampled_values,
    )
    for rank, base, provisional_top in zip(
      ranks, base_anchors, provisional_tops, strict=True
    )
  )
  bounds = union_bounds(
    [
      *(stitch.centerline_bounds for stitch in stitches),
      connector.centerline_bounds,
    ],
    class_name=CLASS_NAME,
  )
  return _PopcornLayout(
    nominal_height=height,
    minimum_gap=minimum_gap,
    pitch=pitch,
    effective_stitch_spacing_ratio=effective_spacing_ratio,
    top_span=top_span,
    bottom_span=bottom_span,
    connector_overhang=connector_overhang,
    connector_width=connector_width,
    connector_half_width=connector_half_width,
    connector_depth=connector_depth,
    connector_circle_center=connector_curve.center,
    connector_circle_radius=connector_curve.radius,
    ranks=ranks,
    stitches=stitches,
    connector=connector,
    centerline_bounds=bounds,
  )


def _cluster_curved_stitch(
  *,
  rank: float,
  base: Point,
  top: Point,
  connector_width: float,
  stitch_class: str,
  sampled_values: Mapping[str, Any],
) -> _CurvedStitch:
  """Frozen cluster parabola construction."""

  chord_midpoint = ((base[0] + top[0]) / 2.0, (base[1] + top[1]) / 2.0)
  if math.isclose(rank, 0.0, abs_tol=1e-12):
    outward_midpoint = chord_midpoint
    curve = _straight_curve(base, top)
  else:
    nominal_control_x = (
      chord_midpoint[0]
      + 2.0 * PARABOLIC_BULGE_RATIO * connector_width * rank
    )
    minimum_outward_control_x = (
      top[0] + PARABOLIC_BULGE_RATIO * connector_width * rank
    )
    control_x = (
      min(nominal_control_x, minimum_outward_control_x)
      if rank < 0.0
      else max(nominal_control_x, minimum_outward_control_x)
    )
    control = (control_x, chord_midpoint[1])
    outward_midpoint = (
      (base[0] + 2.0 * control[0] + top[0]) / 4.0,
      (base[1] + 2.0 * control[1] + top[1]) / 4.0,
    )
    curve = _parabolic_curve(base, outward_midpoint, top)
    if curve.quadratic_control is None:
      raise ValueError("rounded parabolic stitch is missing its control point.")
    top_to_control_x = curve.quadratic_control[0] - top[0]
    if top_to_control_x * rank <= 0.0:
      raise ValueError(
        "rounded stitch control point must lie outward of its top anchor."
      )
  crossbars = _crossbars(
    curve=curve,
    stitch_class=stitch_class,
    sampled_values=sampled_values,
  )
  bounds = union_bounds(
    [curve.bounds, *(_line_bounds(crossbar.line) for crossbar in crossbars)],
    class_name=CLASS_NAME,
  )
  return _CurvedStitch(
    rank=rank,
    base=base,
    top=top,
    outward_midpoint=outward_midpoint,
    curve=curve,
    crossbars=crossbars,
    centerline_bounds=bounds,
  )


def _popcorn_curved_stitch(
  *,
  rank: float,
  base: Point,
  provisional_top: Point,
  pitch: float,
  connector_chord_center: Point,
  connector_half_width: float,
  connector_circle_center: Point,
  connector_circle_radius: float,
  stitch_class: str,
  sampled_values: Mapping[str, Any],
) -> _PopcornStitch:
  chord_midpoint = (
    (base[0] + provisional_top[0]) / 2.0,
    (base[1] + provisional_top[1]) / 2.0,
  )
  if math.isclose(rank, 0.0, abs_tol=1e-12):
    prescribed_middle = chord_midpoint
  else:
    bow = 0.65 * pitch * (0.5 + 0.5 * abs(rank))
    middle_x = (
      min(base[0], provisional_top[0]) - bow
      if rank < 0.0
      else max(base[0], provisional_top[0]) + bow
    )
    prescribed_middle = (middle_x, chord_midpoint[1])

  full_curve = _parabolic_curve(base, prescribed_middle, provisional_top)
  trim_parameter = _popcorn_trim_parameter(
    curve=full_curve,
    connector_chord_center=connector_chord_center,
    connector_half_width=connector_half_width,
    connector_circle_center=connector_circle_center,
    connector_circle_radius=connector_circle_radius,
  )
  curve, discarded_curve = _split_quadratic_curve(full_curve, trim_parameter)
  if trim_parameter <= 0.5 + 1e-10:
    raise ValueError(
      "rounded popcorn connector must trim above the prescribed middle point."
    )
  discarded_length = _positive_finite(
    discarded_curve.length,
    "rounded popcorn discarded curve length",
  )
  crossbars = _crossbars(
    curve=curve,
    stitch_class=stitch_class,
    sampled_values=sampled_values,
  )
  bounds = union_bounds(
    [curve.bounds, *(_line_bounds(crossbar.line) for crossbar in crossbars)],
    class_name=CLASS_NAME,
  )
  return _PopcornStitch(
    rank=rank,
    base=base,
    top=curve.end,
    outward_midpoint=prescribed_middle,
    curve=curve,
    crossbars=crossbars,
    centerline_bounds=bounds,
    provisional_top=provisional_top,
    full_curve=full_curve,
    trim_parameter=trim_parameter,
    discarded_length=discarded_length,
  )


def _popcorn_trim_parameter(
  *,
  curve: _Curve,
  connector_chord_center: Point,
  connector_half_width: float,
  connector_circle_center: Point,
  connector_circle_radius: float,
) -> float:
  control = curve.quadratic_control
  if control is None:
    raise ValueError("rounded popcorn full stitch must be a quadratic parabola.")

  def signed_distance(parameter: float) -> float | None:
    point = _quadratic_point(curve.start, control, curve.end, parameter)
    horizontal_offset = point[0] - connector_chord_center[0]
    if abs(horizontal_offset) > connector_half_width + 1e-10:
      return None
    circle_offset = point[0] - connector_circle_center[0]
    radicand = (
      connector_circle_radius ** 2 - circle_offset ** 2
    )
    if radicand < -1e-10:
      return None
    connector_y = connector_circle_center[1] + math.sqrt(max(0.0, radicand))
    return point[1] - connector_y

  top_distance = signed_distance(1.0)
  if top_distance is None or top_distance >= 0.0:
    raise ValueError(
      "rounded popcorn provisional top must lie above the connector."
    )

  brackets: list[tuple[float, float]] = []
  previous_parameter: float | None = None
  previous_distance: float | None = None
  for index in range(129):
    parameter = 0.5 + 0.5 * index / 128.0
    distance = signed_distance(parameter)
    if distance is None:
      previous_parameter = None
      previous_distance = None
      continue
    if distance == 0.0:
      brackets.append((parameter, parameter))
    elif (
      previous_parameter is not None
      and previous_distance is not None
      and previous_distance * distance < 0.0
    ):
      brackets.append((previous_parameter, parameter))
    previous_parameter = parameter
    previous_distance = distance

  if len(brackets) != 1:
    raise ValueError(
      "rounded popcorn parabola must have exactly one upper connector intersection."
    )
  lower, upper = brackets[0]
  if lower == upper:
    return lower
  lower_distance = signed_distance(lower)
  if lower_distance is None:
    raise ValueError("rounded popcorn connector intersection bracket is invalid.")
  for _ in range(80):
    midpoint = (lower + upper) / 2.0
    distance = signed_distance(midpoint)
    if distance is None:
      raise ValueError("rounded popcorn connector intersection left its domain.")
    if abs(distance) <= 1e-12:
      return midpoint
    if lower_distance * distance > 0.0:
      lower = midpoint
      lower_distance = distance
    else:
      upper = midpoint
  return (lower + upper) / 2.0


def _split_quadratic_curve(
  curve: _Curve,
  parameter: float,
) -> tuple[_Curve, _Curve]:
  control = curve.quadratic_control
  if control is None:
    raise ValueError("rounded popcorn curve splitting requires a quadratic curve.")
  if not 0.0 < parameter < 1.0:
    raise ValueError("rounded popcorn trim parameter must lie strictly inside (0, 1).")

  start_to_control = _lerp_point(curve.start, control, parameter)
  control_to_end = _lerp_point(control, curve.end, parameter)
  intersection = _lerp_point(start_to_control, control_to_end, parameter)
  return (
    _quadratic_curve_from_control(curve.start, start_to_control, intersection),
    _quadratic_curve_from_control(intersection, control_to_end, curve.end),
  )


def _quadratic_curve_from_control(
  start: Point,
  control: Point,
  end: Point,
) -> _Curve:
  length = _quadratic_segment_length(start, control, end)
  if not math.isfinite(length) or length <= 0.0:
    raise ValueError("rounded popcorn quadratic segment must have positive length.")
  return _Curve(
    start=start,
    midpoint=_quadratic_point(start, control, end, 0.5),
    end=end,
    quadratic_control=control,
    center=None,
    radius=None,
    radius_y=None,
    start_angle_rad=None,
    sweep_rad=0.0,
    length=length,
    bounds=_quadratic_segment_bounds(start, control, end),
  )


def _lerp_point(start: Point, end: Point, parameter: float) -> Point:
  return (
    start[0] + (end[0] - start[0]) * parameter,
    start[1] + (end[1] - start[1]) * parameter,
  )


def _crossbars(
  *,
  curve: _Curve,
  stitch_class: str,
  sampled_values: Mapping[str, Any],
) -> tuple[_Crossbar, ...]:
  count = CROSSBAR_COUNTS[stitch_class]
  if count == 0:
    return ()
  ratio = _ratio(
    sampled_values.get("cross_bar_ratio"), "rounded child cross_bar_ratio"
  )
  center_position = _closed_unit_interval(
    sampled_values.get("cross_bar_y"), "rounded child cross_bar_y"
  )
  sampled_angle_deg = _finite(
    sampled_values.get("cross_bar_angle_deg"),
    "rounded child cross_bar_angle_deg",
  )
  if not -45.0 <= sampled_angle_deg <= 45.0:
    raise ValueError("rounded child cross_bar_angle_deg must be in [-45, 45].")
  angle_deg = sampled_angle_deg
  positions = tuple(
    center_position + (index - (count - 1) / 2.0) * 0.08
    for index in range(count)
  )
  if any(not 0.0 <= position <= 1.0 for position in positions):
    raise ValueError("rounded child crossbar positions must stay within the stem.")
  length = ratio * curve.length
  angle_rad = math.radians(angle_deg)
  result: list[_Crossbar] = []
  for position in positions:
    center, tangent = _curve_point_tangent(curve, position)
    direction = (
      math.cos(angle_rad),
      math.sin(angle_rad),
    )
    line = _centered_line(center, direction, length)
    result.append(_Crossbar(
      path_position=position,
      center=center,
      tangent=tangent,
      sampled_angle_deg=sampled_angle_deg,
      angle_deg=angle_deg,
      length=length,
      line=line,
    ))
  return tuple(result)


def _straight_curve(start: Point, end: Point) -> _Curve:
  length = math.dist(start, end)
  if length <= 0.0:
    raise ValueError("rounded stitch anchors must be distinct.")
  midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
  return _Curve(
    start=start,
    midpoint=midpoint,
    end=end,
    quadratic_control=None,
    center=None,
    radius=None,
    radius_y=None,
    start_angle_rad=None,
    sweep_rad=0.0,
    length=length,
    bounds=_line_bounds((start, end)),
  )


def _parabolic_curve(start: Point, midpoint: Point, end: Point) -> _Curve:
  chord_midpoint = (
    (start[0] + end[0]) / 2.0,
    (start[1] + end[1]) / 2.0,
  )
  control = (
    2.0 * midpoint[0] - chord_midpoint[0],
    2.0 * midpoint[1] - chord_midpoint[1],
  )
  length = _quadratic_segment_length(start, control, end)
  if not math.isfinite(length) or length <= 0.0:
    raise ValueError("rounded cluster parabola length must be positive and finite.")
  return _Curve(
    start=start,
    midpoint=midpoint,
    end=end,
    quadratic_control=control,
    center=None,
    radius=None,
    radius_y=None,
    start_angle_rad=None,
    sweep_rad=0.0,
    length=length,
    bounds=_quadratic_segment_bounds(start, control, end),
  )


def _circle_curve(start: Point, through: Point, end: Point) -> _Curve:
  ax, ay = start
  bx, by = through
  cx, cy = end
  determinant = 2.0 * (
    ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)
  )
  scale = max(math.dist(start, end), math.dist(start, through), 1.0)
  if abs(determinant) <= 1e-12 * scale * scale:
    raise ValueError("rounded stitch arc construction is degenerate.")
  a_squared = ax * ax + ay * ay
  b_squared = bx * bx + by * by
  c_squared = cx * cx + cy * cy
  center = (
    (
      a_squared * (by - cy)
      + b_squared * (cy - ay)
      + c_squared * (ay - by)
    ) / determinant,
    (
      a_squared * (cx - bx)
      + b_squared * (ax - cx)
      + c_squared * (bx - ax)
    ) / determinant,
  )
  radius = math.dist(center, start)
  if not math.isfinite(radius) or radius <= 0.0:
    raise ValueError("rounded stitch arc radius must be positive and finite.")
  start_angle = math.atan2(ay - center[1], ax - center[0])
  through_angle = math.atan2(by - center[1], bx - center[0])
  end_angle = math.atan2(cy - center[1], cx - center[0])
  positive_sweep = _positive_angle(end_angle - start_angle)
  through_from_start = _positive_angle(through_angle - start_angle)
  sweep = (
    positive_sweep
    if through_from_start <= positive_sweep + 1e-10
    else positive_sweep - 2.0 * math.pi
  )
  if abs(sweep) > math.pi + 1e-8:
    raise ValueError("rounded stitch construction must select the minor circular arc.")
  length = radius * abs(sweep)
  if not math.isfinite(length) or length <= 0.0:
    raise ValueError("rounded stitch arc length must be positive and finite.")
  bounds = _arc_bounds(center, radius, start_angle, sweep)
  return _Curve(
    start=start,
    midpoint=through,
    end=end,
    quadratic_control=None,
    center=center,
    radius=radius,
    radius_y=radius,
    start_angle_rad=start_angle,
    sweep_rad=sweep,
    length=length,
    bounds=bounds,
  )


def _ellipse_arc(
  *,
  center: Point,
  radius_x: float,
  radius_y: float,
  start_angle: float,
  sweep: float,
) -> _Curve:
  radius_x = _positive_finite(radius_x, "rounded connector horizontal radius")
  radius_y = _positive_finite(radius_y, "rounded connector vertical radius")
  start_angle = _finite(start_angle, "rounded connector start angle")
  sweep = _finite(sweep, "rounded connector sweep")
  if sweep == 0.0 or abs(sweep) > 2.0 * math.pi:
    raise ValueError("rounded connector sweep must be in [-2pi, 2pi] excluding 0.")

  def point(angle: float) -> Point:
    return (
      center[0] + radius_x * math.cos(angle),
      center[1] + radius_y * math.sin(angle),
    )

  start = point(start_angle)
  midpoint = point(start_angle + sweep / 2.0)
  end = point(start_angle + sweep)
  direction = 1.0 if sweep > 0.0 else -1.0
  length = _adaptive_simpson(
    lambda offset: math.hypot(
      radius_x * math.sin(start_angle + direction * offset),
      radius_y * math.cos(start_angle + direction * offset),
    ),
    0.0,
    abs(sweep),
    tolerance=1e-9,
  )
  return _Curve(
    start=start,
    midpoint=midpoint,
    end=end,
    quadratic_control=None,
    center=center,
    radius=radius_x,
    radius_y=radius_y,
    start_angle_rad=start_angle,
    sweep_rad=sweep,
    length=length,
    bounds=_ellipse_arc_bounds(
      center, radius_x, radius_y, start_angle, sweep
    ),
  )


def _curve_point_tangent(curve: _Curve, position: float) -> tuple[Point, Point]:
  position = _closed_unit_interval(position, "rounded curve position")
  if curve.quadratic_control is not None:
    parameter = _quadratic_parameter_at_length(
      curve.start,
      curve.quadratic_control,
      curve.end,
      position * curve.length,
      curve.length,
    )
    point = _quadratic_point(
      curve.start, curve.quadratic_control, curve.end, parameter
    )
    derivative = _quadratic_derivative(
      curve.start, curve.quadratic_control, curve.end, parameter
    )
    derivative_length = math.hypot(*derivative)
    if derivative_length <= 0.0:
      raise ValueError("rounded cluster parabola tangent is degenerate.")
    return point, (
      derivative[0] / derivative_length,
      derivative[1] / derivative_length,
    )
  if curve.center is None:
    dx = curve.end[0] - curve.start[0]
    dy = curve.end[1] - curve.start[1]
    point = (
      curve.start[0] + position * dx,
      curve.start[1] + position * dy,
    )
    return point, (dx / curve.length, dy / curve.length)
  assert curve.radius is not None
  radius_y = curve.radius_y if curve.radius_y is not None else curve.radius
  assert curve.start_angle_rad is not None
  angle = curve.start_angle_rad + curve.sweep_rad * position
  point = (
    curve.center[0] + curve.radius * math.cos(angle),
    curve.center[1] + radius_y * math.sin(angle),
  )
  direction = 1.0 if curve.sweep_rad > 0.0 else -1.0
  derivative = (
    -direction * curve.radius * math.sin(angle),
    direction * radius_y * math.cos(angle),
  )
  derivative_length = math.hypot(*derivative)
  tangent = (
    derivative[0] / derivative_length,
    derivative[1] / derivative_length,
  )
  return point, tangent


def _quadratic_point(
  start: Point,
  control: Point,
  end: Point,
  parameter: float,
) -> Point:
  inverse = 1.0 - parameter
  return (
    inverse ** 2 * start[0]
    + 2.0 * inverse * parameter * control[0]
    + parameter ** 2 * end[0],
    inverse ** 2 * start[1]
    + 2.0 * inverse * parameter * control[1]
    + parameter ** 2 * end[1],
  )


def _quadratic_derivative(
  start: Point,
  control: Point,
  end: Point,
  parameter: float,
) -> Point:
  inverse = 1.0 - parameter
  return (
    2.0 * inverse * (control[0] - start[0])
    + 2.0 * parameter * (end[0] - control[0]),
    2.0 * inverse * (control[1] - start[1])
    + 2.0 * parameter * (end[1] - control[1]),
  )


def _quadratic_segment_length(
  start: Point,
  control: Point,
  end: Point,
  maximum_parameter: float = 1.0,
) -> float:
  maximum_parameter = min(max(maximum_parameter, 0.0), 1.0)
  if maximum_parameter == 0.0:
    return 0.0

  def speed(parameter: float) -> float:
    return math.hypot(*_quadratic_derivative(
      start, control, end, parameter
    ))

  return _adaptive_simpson(speed, 0.0, maximum_parameter, tolerance=1e-9)


def _adaptive_simpson(
  function: Callable[[float], float],
  start: float,
  end: float,
  *,
  tolerance: float,
  maximum_depth: int = 14,
) -> float:
  midpoint = (start + end) / 2.0
  start_value = function(start)
  midpoint_value = function(midpoint)
  end_value = function(end)
  whole = (end - start) * (
    start_value + 4.0 * midpoint_value + end_value
  ) / 6.0

  def recurse(
    left: float,
    right: float,
    left_value: float,
    middle_value: float,
    right_value: float,
    estimate: float,
    remaining: int,
    local_tolerance: float,
  ) -> float:
    middle = (left + right) / 2.0
    left_middle = (left + middle) / 2.0
    right_middle = (middle + right) / 2.0
    left_middle_value = function(left_middle)
    right_middle_value = function(right_middle)
    left_estimate = (middle - left) * (
      left_value + 4.0 * left_middle_value + middle_value
    ) / 6.0
    right_estimate = (right - middle) * (
      middle_value + 4.0 * right_middle_value + right_value
    ) / 6.0
    refined = left_estimate + right_estimate
    if remaining == 0 or abs(refined - estimate) <= 15.0 * local_tolerance:
      return refined + (refined - estimate) / 15.0
    return recurse(
      left,
      middle,
      left_value,
      left_middle_value,
      middle_value,
      left_estimate,
      remaining - 1,
      local_tolerance / 2.0,
    ) + recurse(
      middle,
      right,
      middle_value,
      right_middle_value,
      right_value,
      right_estimate,
      remaining - 1,
      local_tolerance / 2.0,
    )

  return recurse(
    start,
    end,
    start_value,
    midpoint_value,
    end_value,
    whole,
    maximum_depth,
    tolerance,
  )


def _quadratic_parameter_at_length(
  start: Point,
  control: Point,
  end: Point,
  target_length: float,
  total_length: float,
) -> float:
  if target_length <= 0.0:
    return 0.0
  if target_length >= total_length:
    return 1.0
  lower = 0.0
  upper = 1.0
  for _ in range(32):
    midpoint = (lower + upper) / 2.0
    length = _quadratic_segment_length(
      start, control, end, midpoint
    )
    if length < target_length:
      lower = midpoint
    else:
      upper = midpoint
  return (lower + upper) / 2.0


def _quadratic_segment_bounds(
  start: Point,
  control: Point,
  end: Point,
) -> Bounds:
  candidates = {0.0, 1.0}
  for axis in (0, 1):
    denominator = start[axis] - 2.0 * control[axis] + end[axis]
    if abs(denominator) > 1e-12:
      parameter = (start[axis] - control[axis]) / denominator
      if 0.0 < parameter < 1.0:
        candidates.add(parameter)
  points = [
    _quadratic_point(start, control, end, parameter)
    for parameter in candidates
  ]
  return (
    min(point[0] for point in points),
    min(point[1] for point in points),
    max(point[0] for point in points),
    max(point[1] for point in points),
  )


def _arc_bounds(
  center: Point,
  radius: float,
  start_angle: float,
  sweep: float,
) -> Bounds:
  return _ellipse_arc_bounds(
    center, radius, radius, start_angle, sweep
  )


def _ellipse_arc_bounds(
  center: Point,
  radius_x: float,
  radius_y: float,
  start_angle: float,
  sweep: float,
) -> Bounds:
  angles = [start_angle, start_angle + sweep]
  for angle in (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0):
    if _angle_on_sweep(angle, start_angle, sweep):
      angles.append(angle)
  points = [
    (
      center[0] + radius_x * math.cos(angle),
      center[1] + radius_y * math.sin(angle),
    )
    for angle in angles
  ]
  return (
    min(point[0] for point in points),
    min(point[1] for point in points),
    max(point[0] for point in points),
    max(point[1] for point in points),
  )


def _angle_on_sweep(angle: float, start: float, sweep: float) -> bool:
  if sweep >= 0.0:
    return _positive_angle(angle - start) <= sweep + 1e-12
  return _positive_angle(start - angle) <= -sweep + 1e-12


def _positive_angle(angle: float) -> float:
  return angle % (2.0 * math.pi)


def _append_curve(parent: Element, *, config: GenerationConfig, curve: _Curve) -> None:
  if curve.quadratic_control is not None:
    start = rendered_px_to_viewbox(config, curve.start)
    control = rendered_px_to_viewbox(config, curve.quadratic_control)
    end = rendered_px_to_viewbox(config, curve.end)
    SubElement(parent, "path", {
      "d": (
        f"M {start[0]:.8f},{start[1]:.8f} "
        f"Q {control[0]:.8f},{control[1]:.8f} "
        f"{end[0]:.8f},{end[1]:.8f}"
      ),
    })
    return
  if curve.center is None:
    _append_line(parent, config=config, line=(curve.start, curve.end))
    return
  assert curve.radius is not None
  radius_y_px = curve.radius_y if curve.radius_y is not None else curve.radius
  start = rendered_px_to_viewbox(config, curve.start)
  end = rendered_px_to_viewbox(config, curve.end)
  center = rendered_px_to_viewbox(config, curve.center)
  edge = rendered_px_to_viewbox(
    config, (curve.center[0] + curve.radius, curve.center[1])
  )
  lower = rendered_px_to_viewbox(
    config, (curve.center[0], curve.center[1] + radius_y_px)
  )
  radius_x = abs(edge[0] - center[0])
  radius_y = abs(lower[1] - center[1])
  sweep_flag = "1" if curve.sweep_rad > 0.0 else "0"
  SubElement(parent, "path", {
    "d": (
      f"M {start[0]:.8f},{start[1]:.8f} "
      f"A {radius_x:.8f},{radius_y:.8f} 0 0 {sweep_flag} "
      f"{end[0]:.8f},{end[1]:.8f}"
    ),
  })


def _append_connector(
  parent: Element,
  *,
  config: GenerationConfig,
  connector: _Connector,
) -> None:
  if connector.tails:
    if connector.line is None or len(connector.tails) != 2:
      raise ValueError("rounded shelf connector requires two tails and one line.")
    left_tail, right_tail = connector.tails
    start = rendered_px_to_viewbox(config, left_tail.start)
    left_end = rendered_px_to_viewbox(config, left_tail.end)
    shelf_end = rendered_px_to_viewbox(config, connector.line[1])
    right_end = rendered_px_to_viewbox(config, right_tail.end)
    left_radius_x, left_radius_y = _curve_radii_viewbox(config, left_tail)
    right_radius_x, right_radius_y = _curve_radii_viewbox(config, right_tail)
    left_sweep = "1" if left_tail.sweep_rad > 0.0 else "0"
    right_sweep = "1" if right_tail.sweep_rad > 0.0 else "0"
    SubElement(parent, "path", {
      "d": (
        f"M {start[0]:.8f},{start[1]:.8f} "
        f"A {left_radius_x:.8f},{left_radius_y:.8f} 0 0 {left_sweep} "
        f"{left_end[0]:.8f},{left_end[1]:.8f} "
        f"L {shelf_end[0]:.8f},{shelf_end[1]:.8f} "
        f"A {right_radius_x:.8f},{right_radius_y:.8f} 0 0 {right_sweep} "
        f"{right_end[0]:.8f},{right_end[1]:.8f}"
      ),
    })
    return
  if connector.line is not None:
    _append_line(parent, config=config, line=connector.line)
    return
  if connector.curve is None:
    raise ValueError("rounded connector has no drawable geometry.")
  _append_curve(parent, config=config, curve=connector.curve)


def _curve_radii_viewbox(
  config: GenerationConfig,
  curve: _Curve,
) -> tuple[float, float]:
  if curve.center is None or curve.radius is None:
    raise ValueError("rounded connector tail must be an ellipse arc.")
  radius_y = curve.radius_y if curve.radius_y is not None else curve.radius
  center = rendered_px_to_viewbox(config, curve.center)
  horizontal = rendered_px_to_viewbox(
    config, (curve.center[0] + curve.radius, curve.center[1])
  )
  vertical = rendered_px_to_viewbox(
    config, (curve.center[0], curve.center[1] + radius_y)
  )
  return abs(horizontal[0] - center[0]), abs(vertical[1] - center[1])


def _append_line(parent: Element, *, config: GenerationConfig, line: Line) -> None:
  start = rendered_px_to_viewbox(config, line[0])
  end = rendered_px_to_viewbox(config, line[1])
  SubElement(parent, "line", {
    "x1": f"{start[0]:.8f}",
    "y1": f"{start[1]:.8f}",
    "x2": f"{end[0]:.8f}",
    "y2": f"{end[1]:.8f}",
  })


def _prototype(
  composite: CompositeSample,
  *,
  stitch_class: Any,
  count: int,
  stroke_width: float,
) -> ComponentPrototype:
  if set(composite.prototypes) != {"stitch"}:
    raise ValueError("rounded requires exactly one 'stitch' component prototype.")
  if not isinstance(stitch_class, str) or stitch_class not in SUPPORTED_STITCHES:
    raise ValueError(f"Unsupported rounded stitch class: {stitch_class!r}.")
  prototype = composite.prototypes["stitch"]
  if prototype.role != "stitch":
    raise ValueError("rounded stitch prototype role must be 'stitch'.")
  if (prototype.class_group, prototype.class_name) != ("primitive", stitch_class):
    raise ValueError(
      "rounded stitch prototype must resolve to the sampled primitive stitch class."
    )
  if prototype.occurrence_count != count:
    raise ValueError("rounded stitch prototype occurrence_count must equal count.")
  if prototype.arrangement is not None:
    raise ValueError("rounded stitch prototype arrangement must be unspecified.")
  if prototype.declaration.get("inherit_generator") is not True:
    raise ValueError("rounded stitch prototype must inherit its primitive generator.")
  if not _same_number(prototype.inherited_parameters.get("stroke_width"), stroke_width):
    raise ValueError("rounded stitch prototype must inherit the parent stroke_width.")
  if not _same_number(prototype.sample.as_dict().get("stroke_width"), stroke_width):
    raise ValueError("rounded child sample stroke_width must equal the parent stroke_width.")
  return prototype


def _validate_stitch_values(
  stitch_class: str,
  sampled_values: Mapping[str, Any],
) -> None:
  _ratio(sampled_values.get("bar_stem_ratio"), "rounded child bar_stem_ratio")
  if CROSSBAR_COUNTS[stitch_class]:
    _ratio(sampled_values.get("cross_bar_ratio"), "rounded child cross_bar_ratio")
    _closed_unit_interval(
      sampled_values.get("cross_bar_y"), "rounded child cross_bar_y"
    )
    angle = _finite(
      sampled_values.get("cross_bar_angle_deg"),
      "rounded child cross_bar_angle_deg",
    )
    if not -45.0 <= angle <= 45.0:
      raise ValueError("rounded child cross_bar_angle_deg must be in [-45, 45].")


def _topology(topology: Mapping[str, Any]) -> None:
  if not isinstance(topology, Mapping):
    raise TypeError("rounded topology must be a mapping.")
  if topology.get("base_relation") != "joined":
    raise ValueError("rounded topology base_relation must be 'joined'.")
  if topology.get("top_relation") != "joined":
    raise ValueError("rounded topology top_relation must be 'joined'.")
  connector = topology.get("connector")
  if not isinstance(connector, Mapping):
    raise ValueError("rounded topology connector must be a mapping.")
  if connector.get("enabled") is not True or connector.get("required") is not True:
    raise ValueError("rounded topology connector must be enabled and required.")


def _variant_count(value: Any, *, variant: str) -> int:
  if isinstance(value, bool) or not isinstance(value, int):
    raise ValueError("rounded count must be an integer.")
  allowed = {3, 4, 5} if variant == "cluster" else {4, 5, 6}
  if value not in allowed:
    raise ValueError(
      f"rounded {variant} count must be one of {sorted(allowed)}."
    )
  return value


def _connector_type(value: Any, *, variant: str) -> str:
  expected = "bar" if variant == "cluster" else "curved"
  if value != expected:
    raise ValueError(
      f"rounded {variant} connector_type must be {expected!r}."
    )
  return expected


def _stitch_metadata(stitch: _CurvedStitch) -> dict[str, Any]:
  curve = stitch.curve
  path_kind = (
    "quadratic_parabola"
    if curve.quadratic_control is not None
    else "line" if curve.center is None else "circular_arc"
  )
  return {
    "rank": stitch.rank,
    "base_px": list(stitch.base),
    "top_px": list(stitch.top),
    "outward_midpoint_px": list(stitch.outward_midpoint),
    "path_kind": path_kind,
    "quadratic_control_px": (
      list(curve.quadratic_control)
      if curve.quadratic_control is not None else None
    ),
    "circle_center_px": list(curve.center) if curve.center is not None else None,
    "radius_px": curve.radius,
    "sweep_deg": math.degrees(curve.sweep_rad),
    "arc_length_px": curve.length,
    "curve_bounds_px": list(curve.bounds),
    "crossbars": [
      {
        "path_position": crossbar.path_position,
        "center_px": list(crossbar.center),
        "tangent_px": list(crossbar.tangent),
        "sampled_angle_deg": crossbar.sampled_angle_deg,
        "angle_deg": crossbar.angle_deg,
        "length_px": crossbar.length,
        "endpoints_px": [list(crossbar.line[0]), list(crossbar.line[1])],
      }
      for crossbar in stitch.crossbars
    ],
    "centerline_bounds_px": list(stitch.centerline_bounds),
  }


def _popcorn_stitch_metadata(stitch: _PopcornStitch) -> dict[str, Any]:
  retained_control = stitch.curve.quadratic_control
  full_control = stitch.full_curve.quadratic_control
  if retained_control is None or full_control is None:
    raise ValueError("rounded popcorn metadata requires quadratic curves.")
  return {
    "rank": stitch.rank,
    "base_px": list(stitch.base),
    "provisional_top_px": list(stitch.provisional_top),
    "prescribed_middle_px": list(stitch.outward_midpoint),
    "full_control_px": list(full_control),
    "full_curve_length_px": stitch.full_curve.length,
    "full_curve_bounds_px": list(stitch.full_curve.bounds),
    "trim_parameter": stitch.trim_parameter,
    "connector_intersection_px": list(stitch.top),
    "retained_control_px": list(retained_control),
    "retained_midpoint_px": list(stitch.curve.midpoint),
    "retained_length_px": stitch.curve.length,
    "discarded_length_px": stitch.discarded_length,
    "retained_curve_bounds_px": list(stitch.curve.bounds),
    "crossbars": [
      {
        "path_position": crossbar.path_position,
        "center_px": list(crossbar.center),
        "tangent_px": list(crossbar.tangent),
        "sampled_angle_deg": crossbar.sampled_angle_deg,
        "angle_deg": crossbar.angle_deg,
        "length_px": crossbar.length,
        "endpoints_px": [list(crossbar.line[0]), list(crossbar.line[1])],
      }
      for crossbar in stitch.crossbars
    ],
    "centerline_bounds_px": list(stitch.centerline_bounds),
  }


def _connector_metadata(connector: _Connector) -> dict[str, Any]:
  curve = connector.curve
  return {
    "type": connector.connector_type,
    "width_px": connector.width,
    "line_endpoints_px": (
      [list(connector.line[0]), list(connector.line[1])]
      if connector.line is not None else None
    ),
    "start_px": list(curve.start) if curve is not None else None,
    "midpoint_px": list(curve.midpoint) if curve is not None else None,
    "end_px": list(curve.end) if curve is not None else None,
    "quadratic_control_px": (
      list(curve.quadratic_control)
      if curve is not None and curve.quadratic_control is not None else None
    ),
    "circle_center_px": (
      list(curve.center)
      if curve is not None and curve.center is not None else None
    ),
    "radius_px": curve.radius if curve is not None else None,
    "radius_x_px": curve.radius if curve is not None else None,
    "radius_y_px": curve.radius_y if curve is not None else None,
    "aspect_ratio": (
      curve.radius / curve.radius_y
      if curve is not None and curve.radius is not None and curve.radius_y is not None
      else None
    ),
    "tails": [
      {
        "start_px": list(tail.start),
        "midpoint_px": list(tail.midpoint),
        "end_px": list(tail.end),
        "ellipse_center_px": list(tail.center) if tail.center is not None else None,
        "radius_x_px": tail.radius,
        "radius_y_px": tail.radius_y,
        "sweep_deg": math.degrees(tail.sweep_rad),
        "arc_length_px": tail.length,
        "centerline_bounds_px": list(tail.bounds),
      }
      for tail in connector.tails
    ],
    "sweep_deg": math.degrees(curve.sweep_rad) if curve is not None else None,
    "arc_length_px": (
      curve.length
      if curve is not None
      else (
        math.dist(*connector.line) if connector.line is not None else 0.0
      ) + sum(tail.length for tail in connector.tails)
    ),
    "centerline_bounds_px": list(connector.centerline_bounds),
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
  })
  return svg, group


def _centered_line(center: Point, direction: Point, length: float) -> Line:
  half = length / 2.0
  return (
    (center[0] - direction[0] * half, center[1] - direction[1] * half),
    (center[0] + direction[0] * half, center[1] + direction[1] * half),
  )


def _line_bounds(line: Line) -> Bounds:
  return (
    min(line[0][0], line[1][0]),
    min(line[0][1], line[1][1]),
    max(line[0][0], line[1][0]),
    max(line[0][1], line[1][1]),
  )


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


def _open_unit_interval(value: Any, name: str) -> float:
  result = _finite(value, name)
  if not 0.0 < result < 1.0:
    raise ValueError(f"{name} must be in the range (0, 1).")
  return result


def _closed_unit_interval(value: Any, name: str) -> float:
  result = _finite(value, name)
  if not 0.0 <= result <= 1.0:
    raise ValueError(f"{name} must be in the range [0, 1].")
  return result


def _ratio(value: Any, name: str) -> float:
  result = _positive_finite(value, name)
  if result > 1.0:
    raise ValueError(f"{name} must be in the range (0, 1].")
  return result


def _same_number(left: Any, right: float) -> bool:
  try:
    left_value = _finite(left, "stroke_width")
  except ValueError:
    return False
  return math.isclose(left_value, right, rel_tol=1e-12, abs_tol=1e-12)
