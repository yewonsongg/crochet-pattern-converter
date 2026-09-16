"""Reusable rendered-pixel geometry for filled slip-stitch ellipses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping
from xml.etree.ElementTree import Element, SubElement

from ..models import GenerationConfig
from .stitch import rendered_px_to_viewbox


Point = tuple[float, float]
Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class SlipStitchPlacement:
  """Absolute rendered-pixel placement for one slip-stitch ellipse."""

  center_px: Point
  visible_span_px: float
  rotation_deg: float = 0.0


@dataclass(frozen=True)
class SlipStitchGeometry:
  """Pre-global-rotation measurements for one slip-stitch ellipse."""

  class_name: str
  center_px: Point
  visible_span_px: float
  width_px: float
  height_px: float
  rotation_deg: float
  centerline_bounds_px: Bounds
  rendered_bounds_px: Bounds


def append_slip_stitch_geometry(
  parent: Element,
  *,
  sampled_values: Mapping[str, Any],
  config: GenerationConfig,
  placement: SlipStitchPlacement,
  stroke_width: float,
) -> SlipStitchGeometry:
  """Append one sampled slip-stitch prototype at a pixel placement."""

  if not isinstance(parent, Element):
    raise TypeError("slip-stitch geometry parent must be an XML Element.")
  if not isinstance(sampled_values, Mapping):
    raise TypeError("slip-stitch sampled_values must be a mapping.")
  if not isinstance(placement, SlipStitchPlacement):
    raise TypeError("slip-stitch placement must be a SlipStitchPlacement.")

  shape = sampled_values.get("shape")
  if shape not in {"oval", "circle"}:
    raise ValueError(f"Unsupported slip-stitch shape: {shape!r}.")
  aspect_ratio = _positive_finite(
    sampled_values.get("aspect_ratio"), "slip-stitch aspect_ratio"
  )
  center = _point(placement.center_px, "slip-stitch center_px")
  visible_span = _positive_finite(
    placement.visible_span_px, "slip-stitch visible_span_px"
  )
  rotation = _finite(placement.rotation_deg, "slip-stitch rotation_deg")
  stroke = _positive_finite(stroke_width, "slip-stitch stroke_width")

  width_px = visible_span
  height_px = visible_span
  if aspect_ratio >= 1.0:
    height_px /= aspect_ratio
  else:
    width_px *= aspect_ratio

  center_svg = rendered_px_to_viewbox(config, center)
  viewport_scale = min(config.canvas_width_px, config.canvas_height_px) / 100.0
  attributes = {
    "fill": "black",
    "cx": f"{center_svg[0]:.8f}",
    "cy": f"{center_svg[1]:.8f}",
    "rx": f"{width_px / (2.0 * viewport_scale):.8f}",
    "ry": f"{height_px / (2.0 * viewport_scale):.8f}",
  }
  if rotation:
    attributes["transform"] = (
      f"rotate({rotation:.8f} {center_svg[0]:.8f} {center_svg[1]:.8f})"
    )
  SubElement(parent, "ellipse", attributes)

  angle = math.radians(rotation)
  half_width = math.sqrt(
    (width_px * math.cos(angle) / 2.0) ** 2
    + (height_px * math.sin(angle) / 2.0) ** 2
  )
  half_height = math.sqrt(
    (width_px * math.sin(angle) / 2.0) ** 2
    + (height_px * math.cos(angle) / 2.0) ** 2
  )
  centerline_bounds = (
    center[0] - half_width,
    center[1] - half_height,
    center[0] + half_width,
    center[1] + half_height,
  )
  stroke_radius_px = stroke * viewport_scale / 2.0
  rendered_bounds = (
    centerline_bounds[0] - stroke_radius_px,
    centerline_bounds[1] - stroke_radius_px,
    centerline_bounds[2] + stroke_radius_px,
    centerline_bounds[3] + stroke_radius_px,
  )
  return SlipStitchGeometry(
    class_name="slst",
    center_px=center,
    visible_span_px=visible_span,
    width_px=width_px,
    height_px=height_px,
    rotation_deg=rotation,
    centerline_bounds_px=centerline_bounds,
    rendered_bounds_px=rendered_bounds,
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
