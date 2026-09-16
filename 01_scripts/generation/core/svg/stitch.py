"""Reusable anchor-based geometry for crochet stitch primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping
from xml.etree.ElementTree import Element, SubElement, tostring

from ..models import GenerationConfig


Point = tuple[float, float]
Bounds = tuple[float, float, float, float]
StitchAppender = Callable[..., "StitchGeometry"]
SVG_NS = "http://www.w3.org/2000/svg"


@dataclass(frozen=True)
class StitchPlacement:
  """Rendered-pixel centerline anchors for one stitch."""

  base_px: Point
  top_px: Point


@dataclass(frozen=True)
class StitchGeometry:
  """Measurements and pre-rotation bounds for appended stitch geometry."""

  class_name: str
  base_px: Point
  top_px: Point
  stem_length_px: float
  top_bar_length_px: float
  cross_bar_length_px: float
  cross_bar_positions: tuple[float, ...]
  centerline_bounds_px: Bounds
  rendered_bounds_px: Bounds


@dataclass(frozen=True)
class _Viewport:
  scale_px_per_unit: float
  offset_x_px: float
  offset_y_px: float

  def to_viewbox(self, point: Point) -> Point:
    return (
      (point[0] - self.offset_x_px) / self.scale_px_per_unit,
      (point[1] - self.offset_y_px) / self.scale_px_per_unit,
    )


def append_stitch_geometry(
  parent: Element,
  *,
  class_name: str,
  sampled_values: Mapping[str, Any],
  config: GenerationConfig,
  placement: StitchPlacement,
  stroke_width: float,
  include_top_bar: bool = True,
  top_bar_angle_deg: float | None = None,
) -> StitchGeometry:
  """Append one supported stitch between absolute rendered-pixel anchors.

  ``top_bar_angle_deg`` is an absolute rendered-space angle. When omitted,
  taller stitches retain their usual top bar perpendicular to the stem.
  """

  if not isinstance(parent, Element):
    raise TypeError("stitch geometry parent must be an XML Element.")
  if not isinstance(sampled_values, Mapping):
    raise TypeError("stitch sampled_values must be a mapping.")
  if not isinstance(class_name, str) or not class_name:
    raise TypeError("stitch class_name must be a non-empty string.")
  if not isinstance(placement, StitchPlacement):
    raise TypeError("stitch placement must be a StitchPlacement.")
  if not isinstance(include_top_bar, bool):
    raise TypeError("include_top_bar must be a boolean.")
  if top_bar_angle_deg is not None:
    top_bar_angle_deg = _finite(top_bar_angle_deg, "top_bar_angle_deg")
  _positive_finite(stroke_width, "stroke_width")
  _placement_vectors(placement)
  _viewport(config)

  appender = STITCH_GEOMETRY_REGISTRY.get(class_name)
  if appender is None:
    raise KeyError(f"Unsupported reusable stitch class: {class_name!r}.")
  return appender(
    parent=parent,
    sampled_values=sampled_values,
    config=config,
    placement=placement,
    stroke_width=float(stroke_width),
    include_top_bar=include_top_bar,
    top_bar_angle_deg=top_bar_angle_deg,
  )


def rendered_px_to_viewbox(
  config: GenerationConfig,
  point_px: Point,
) -> Point:
  """Convert an absolute rendered-pixel point through centered SVG meet scaling."""

  return _viewport(config).to_viewbox(_point(point_px, "point_px"))


def build_centered_stitch_svg(
  *,
  class_name: str,
  sampled_values: Mapping[str, Any],
  config: GenerationConfig,
  stroke_width: float,
) -> tuple[str, StitchGeometry]:
  """Build a standalone primitive using centered vertical stitch anchors."""

  svg, group = _svg_root(config, stroke_width)
  center_x = config.canvas_width_px / 2.0
  center_y = config.canvas_height_px / 2.0
  half_length = config.target_visible_px / 2.0
  geometry = append_stitch_geometry(
    group,
    class_name=class_name,
    sampled_values=sampled_values,
    config=config,
    placement=StitchPlacement(
      base_px=(center_x, center_y + half_length),
      top_px=(center_x, center_y - half_length),
    ),
    stroke_width=stroke_width,
  )
  return tostring(svg, encoding="unicode"), geometry


def append_bar_stem_geometry(
  parent: Element,
  *,
  class_name: str,
  config: GenerationConfig,
  placement: StitchPlacement,
  bar_stem_ratio: float,
  cross_bar_ratio: float = 0.0,
  cross_bar_y: float | None = None,
  cross_bar_angle_deg: float = 0.0,
  cross_bar_count: int = 0,
  stroke_width: float,
  include_top_bar: bool = True,
  top_bar_angle_deg: float | None = None,
) -> StitchGeometry:
  """Append the shared bar/stem construction used by taller stitches."""

  bar_stem_ratio = _positive_finite(bar_stem_ratio, "bar_stem_ratio")
  stroke_width = _positive_finite(stroke_width, "stroke_width")
  if isinstance(cross_bar_count, bool) or not isinstance(cross_bar_count, int):
    raise ValueError("cross_bar_count must be a non-negative integer.")
  if cross_bar_count < 0:
    raise ValueError("cross_bar_count must be non-negative.")
  if not isinstance(include_top_bar, bool):
    raise TypeError("include_top_bar must be a boolean.")
  if top_bar_angle_deg is not None:
    top_bar_angle_deg = _finite(top_bar_angle_deg, "top_bar_angle_deg")
  cross_bar_angle_deg = _finite(cross_bar_angle_deg, "cross_bar_angle_deg")
  if not -45.0 <= cross_bar_angle_deg <= 45.0:
    raise ValueError("cross_bar_angle_deg must be in the range [-45, 45].")
  if cross_bar_y is not None:
    cross_bar_y = _finite(cross_bar_y, "cross_bar_y")
    if not 0.0 <= cross_bar_y <= 1.0:
      raise ValueError("cross_bar_y must be in the range [0, 1].")

  base, top, axis, perpendicular, stem_length = _placement_vectors(placement)
  bar_length = stem_length * bar_stem_ratio if include_top_bar else 0.0
  cross_bar_length = 0.0
  cross_positions: tuple[float, ...] = ()
  if cross_bar_count:
    cross_bar_ratio = _positive_finite(cross_bar_ratio, "cross_bar_ratio")
    if cross_bar_y is None:
      raise ValueError("cross_bar_y is required when crossbars are enabled.")
    spacing = 0.08
    cross_positions = tuple(
      cross_bar_y + (index - (cross_bar_count - 1) / 2.0) * spacing
      for index in range(cross_bar_count)
    )
    if any(not 0.0 <= position <= 1.0 for position in cross_positions):
      raise ValueError("crossbar positions derived from cross_bar_y must stay within the stem.")
    cross_bar_length = stem_length * cross_bar_ratio
  lines: list[tuple[Point, Point]] = [(top, base)]
  if include_top_bar:
    top_bar_direction = perpendicular
    if top_bar_angle_deg is not None:
      top_bar_angle_rad = math.radians(top_bar_angle_deg)
      top_bar_direction = (
        math.cos(top_bar_angle_rad),
        math.sin(top_bar_angle_rad),
      )
    lines.append(_centered_line(top, top_bar_direction, bar_length))

  angle_rad = math.radians(float(cross_bar_angle_deg))
  cross_direction = (
    math.cos(angle_rad) * perpendicular[0] + math.sin(angle_rad) * axis[0],
    math.cos(angle_rad) * perpendicular[1] + math.sin(angle_rad) * axis[1],
  )
  for position in cross_positions:
    center = (
      base[0] + axis[0] * stem_length * position,
      base[1] + axis[1] * stem_length * position,
    )
    lines.append(_centered_line(center, cross_direction, cross_bar_length))

  _append_lines(parent, config, lines)
  return _geometry(
    class_name=class_name,
    placement=placement,
    stem_length=stem_length,
    top_bar_length=bar_length,
    cross_bar_length=cross_bar_length,
    cross_positions=cross_positions,
    lines=lines,
    stroke_width=stroke_width,
    config=config,
  )


def _append_sc(
  *,
  parent: Element,
  sampled_values: Mapping[str, Any],
  config: GenerationConfig,
  placement: StitchPlacement,
  stroke_width: float,
  include_top_bar: bool,
  top_bar_angle_deg: float | None,
) -> StitchGeometry:
  del include_top_bar, top_bar_angle_deg
  shape = sampled_values.get("shape")
  if shape not in {"symmetric", "asymmetric"}:
    raise ValueError(f"Unsupported sc shape: {shape!r}.")
  asymmetry = _finite(sampled_values.get("asymmetry"), "sc asymmetry")
  cross_bar_ratio = _positive_finite(
    sampled_values.get("cross_bar_ratio"), "sc cross_bar_ratio"
  )
  if not -1.0 < asymmetry < 1.0:
    raise ValueError("sc asymmetry must be in the range (-1, 1).")
  if cross_bar_ratio > 1.0:
    raise ValueError("sc cross_bar_ratio must be in the range (0, 1].")
  if shape == "symmetric" and asymmetry != 0.0:
    raise ValueError("A symmetric sc must have asymmetry=0.0.")

  base, top, axis, perpendicular, stem_length = _placement_vectors(placement)
  position = (1.0 + asymmetry) / 2.0
  center = (
    base[0] + axis[0] * stem_length * position,
    base[1] + axis[1] * stem_length * position,
  )
  cross_bar_length = stem_length * cross_bar_ratio
  transverse = _centered_line(center, perpendicular, cross_bar_length)
  # Preserve the standalone primitive's transverse-then-axis element order.
  lines = [transverse, (top, base)]
  _append_lines(parent, config, lines)
  return _geometry(
    class_name="sc",
    placement=placement,
    stem_length=stem_length,
    top_bar_length=0.0,
    cross_bar_length=cross_bar_length,
    cross_positions=(position,),
    lines=lines,
    stroke_width=stroke_width,
    config=config,
  )


def _append_hdc(
  *, parent: Element, sampled_values: Mapping[str, Any], config: GenerationConfig,
  placement: StitchPlacement, stroke_width: float, include_top_bar: bool,
  top_bar_angle_deg: float | None,
) -> StitchGeometry:
  return _append_configured_bar_stem(
    class_name="hdc", cross_bar_count=0, parent=parent,
    sampled_values=sampled_values, config=config, placement=placement,
    stroke_width=stroke_width, include_top_bar=include_top_bar,
    top_bar_angle_deg=top_bar_angle_deg,
  )


def _append_dc(
  *, parent: Element, sampled_values: Mapping[str, Any], config: GenerationConfig,
  placement: StitchPlacement, stroke_width: float, include_top_bar: bool,
  top_bar_angle_deg: float | None,
) -> StitchGeometry:
  return _append_configured_bar_stem(
    class_name="dc", cross_bar_count=1, parent=parent,
    sampled_values=sampled_values, config=config, placement=placement,
    stroke_width=stroke_width, include_top_bar=include_top_bar,
    top_bar_angle_deg=top_bar_angle_deg,
  )


def _append_tr(
  *, parent: Element, sampled_values: Mapping[str, Any], config: GenerationConfig,
  placement: StitchPlacement, stroke_width: float, include_top_bar: bool,
  top_bar_angle_deg: float | None,
) -> StitchGeometry:
  return _append_configured_bar_stem(
    class_name="tr", cross_bar_count=2, parent=parent,
    sampled_values=sampled_values, config=config, placement=placement,
    stroke_width=stroke_width, include_top_bar=include_top_bar,
    top_bar_angle_deg=top_bar_angle_deg,
  )


def _append_dtr(
  *, parent: Element, sampled_values: Mapping[str, Any], config: GenerationConfig,
  placement: StitchPlacement, stroke_width: float, include_top_bar: bool,
  top_bar_angle_deg: float | None,
) -> StitchGeometry:
  return _append_configured_bar_stem(
    class_name="dtr", cross_bar_count=3, parent=parent,
    sampled_values=sampled_values, config=config, placement=placement,
    stroke_width=stroke_width, include_top_bar=include_top_bar,
    top_bar_angle_deg=top_bar_angle_deg,
  )


def _append_configured_bar_stem(
  *,
  class_name: str,
  cross_bar_count: int,
  parent: Element,
  sampled_values: Mapping[str, Any],
  config: GenerationConfig,
  placement: StitchPlacement,
  stroke_width: float,
  include_top_bar: bool,
  top_bar_angle_deg: float | None,
) -> StitchGeometry:
  return append_bar_stem_geometry(
    parent,
    class_name=class_name,
    config=config,
    placement=placement,
    bar_stem_ratio=sampled_values.get("bar_stem_ratio"),
    cross_bar_ratio=sampled_values.get("cross_bar_ratio", 0.0),
    cross_bar_y=sampled_values.get("cross_bar_y"),
    cross_bar_angle_deg=sampled_values.get("cross_bar_angle_deg", 0.0),
    cross_bar_count=cross_bar_count,
    stroke_width=stroke_width,
    include_top_bar=include_top_bar,
    top_bar_angle_deg=top_bar_angle_deg,
  )


def _svg_root(config: GenerationConfig, stroke_width: float) -> tuple[Element, Element]:
  _viewport(config)
  _positive_finite(config.target_visible_px, "target_visible_px")
  _positive_finite(stroke_width, "stroke_width")
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


def _viewport(config: GenerationConfig) -> _Viewport:
  width = config.canvas_width_px
  height = config.canvas_height_px
  if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
    raise ValueError("canvas_width_px must be a positive integer.")
  if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
    raise ValueError("canvas_height_px must be a positive integer.")
  scale = min(width, height) / 100.0
  return _Viewport(
    scale_px_per_unit=scale,
    offset_x_px=(width - 100.0 * scale) / 2.0,
    offset_y_px=(height - 100.0 * scale) / 2.0,
  )


def _placement_vectors(
  placement: StitchPlacement,
) -> tuple[Point, Point, Point, Point, float]:
  if not isinstance(placement, StitchPlacement):
    raise TypeError("stitch placement must be a StitchPlacement.")
  base = _point(placement.base_px, "base_px")
  top = _point(placement.top_px, "top_px")
  dx = top[0] - base[0]
  dy = top[1] - base[1]
  length = math.hypot(dx, dy)
  if length == 0.0:
    raise ValueError("stitch base_px and top_px must be distinct.")
  axis = (dx / length, dy / length)
  perpendicular = (-axis[1], axis[0])
  return base, top, axis, perpendicular, length


def _point(value: Any, name: str) -> Point:
  if not isinstance(value, tuple) or len(value) != 2:
    raise TypeError(f"stitch {name} must be a two-item tuple.")
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


def _centered_line(center: Point, direction: Point, length: float) -> tuple[Point, Point]:
  half = length / 2.0
  return (
    (center[0] - direction[0] * half, center[1] - direction[1] * half),
    (center[0] + direction[0] * half, center[1] + direction[1] * half),
  )


def _append_lines(
  parent: Element,
  config: GenerationConfig,
  lines: list[tuple[Point, Point]],
) -> None:
  viewport = _viewport(config)
  for start, end in lines:
    start_svg = viewport.to_viewbox(start)
    end_svg = viewport.to_viewbox(end)
    SubElement(parent, "line", {
      "x1": f"{start_svg[0]:.8f}",
      "y1": f"{start_svg[1]:.8f}",
      "x2": f"{end_svg[0]:.8f}",
      "y2": f"{end_svg[1]:.8f}",
    })


def _geometry(
  *,
  class_name: str,
  placement: StitchPlacement,
  stem_length: float,
  top_bar_length: float,
  cross_bar_length: float,
  cross_positions: tuple[float, ...],
  lines: list[tuple[Point, Point]],
  stroke_width: float,
  config: GenerationConfig,
) -> StitchGeometry:
  points = [point for line in lines for point in line]
  centerline_bounds = (
    min(point[0] for point in points),
    min(point[1] for point in points),
    max(point[0] for point in points),
    max(point[1] for point in points),
  )
  stroke_radius_px = stroke_width * _viewport(config).scale_px_per_unit / 2.0
  rendered_bounds = (
    centerline_bounds[0] - stroke_radius_px,
    centerline_bounds[1] - stroke_radius_px,
    centerline_bounds[2] + stroke_radius_px,
    centerline_bounds[3] + stroke_radius_px,
  )
  return StitchGeometry(
    class_name=class_name,
    base_px=_point(placement.base_px, "base_px"),
    top_px=_point(placement.top_px, "top_px"),
    stem_length_px=stem_length,
    top_bar_length_px=top_bar_length,
    cross_bar_length_px=cross_bar_length,
    cross_bar_positions=cross_positions,
    centerline_bounds_px=centerline_bounds,
    rendered_bounds_px=rendered_bounds,
  )


STITCH_GEOMETRY_REGISTRY: Mapping[str, StitchAppender] = MappingProxyType({
  "sc": _append_sc,
  "hdc": _append_hdc,
  "dc": _append_dc,
  "tr": _append_tr,
  "dtr": _append_dtr,
})
