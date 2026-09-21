"""Reusable symmetric fan placement for compound crochet symbols."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

from ..core.models import GenerationConfig
from ..core.svg import StitchPlacement, rendered_px_to_viewbox


Point = tuple[float, float]
Bounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class SymmetricFanLayout:
  """Equal-length placements distributed across a centered angular envelope."""

  axis_angles_deg: tuple[float, ...]
  placements: tuple[StitchPlacement, ...]


@dataclass(frozen=True)
class FanFit:
  """Uniform scale and translated unit origin for a centered fan."""

  scale_px_per_unit: float
  origin_px: Point
  stroke_width_px: float


def symmetric_fan_layout(
  *,
  count: int,
  spread_angle_deg: float,
  stem_length_px: float,
  join_px: Point,
  joined_at: Literal["base", "top"],
) -> SymmetricFanLayout:
  """Return left-to-right equal-length placements around one shared anchor."""

  if isinstance(count, bool) or not isinstance(count, int) or count < 2:
    raise ValueError("fan count must be an integer of at least 2.")
  spread = _finite(spread_angle_deg, "fan spread_angle_deg")
  if not 0.0 < spread < 180.0:
    raise ValueError("fan spread_angle_deg must be in the range (0, 180).")
  stem_length = _finite(stem_length_px, "fan stem_length_px")
  if stem_length <= 0.0:
    raise ValueError("fan stem_length_px must be positive.")
  join = _point(join_px, "fan join_px")
  if joined_at not in {"base", "top"}:
    raise ValueError("fan joined_at must be either 'base' or 'top'.")

  step = spread / (count - 1)
  ascending_angles = tuple(
    -spread / 2.0 + index * step for index in range(count)
  )
  # Placements are always emitted left-to-right. For a joined top, that means
  # the directed base-to-top axes run from positive to negative angles.
  angles = (
    tuple(reversed(ascending_angles))
    if joined_at == "top"
    else ascending_angles
  )
  placements: list[StitchPlacement] = []
  for angle_deg in angles:
    angle_rad = math.radians(angle_deg)
    horizontal = stem_length * math.sin(angle_rad)
    vertical = -stem_length * math.cos(angle_rad)
    if joined_at == "top":
      placements.append(StitchPlacement(
        base_px=(join[0] - horizontal, join[1] - vertical),
        top_px=join,
      ))
    else:
      placements.append(StitchPlacement(
        base_px=join,
        top_px=(join[0] + horizontal, join[1] + vertical),
      ))
  return SymmetricFanLayout(
    axis_angles_deg=angles,
    placements=tuple(placements),
  )


def fit_unit_bounds(
  unit_bounds: Bounds,
  *,
  config: GenerationConfig,
  stroke_width: float,
  class_name: str,
) -> FanFit:
  """Fit unit centerline bounds to the configured stroke-inclusive target."""

  bounds = _bounds(unit_bounds, f"{class_name} unit_bounds")
  rendered_px_to_viewbox(config, (0.0, 0.0))
  target = _finite(config.target_visible_px, f"{class_name} target_visible_px")
  if target <= 0.0:
    raise ValueError(f"{class_name} target_visible_px must be positive.")
  stroke = _finite(stroke_width, f"{class_name} stroke_width")
  if stroke <= 0.0:
    raise ValueError(f"{class_name} stroke_width must be positive.")
  viewport_scale = min(config.canvas_width_px, config.canvas_height_px) / 100.0
  stroke_width_px = stroke * viewport_scale
  available = target - stroke_width_px
  if available <= 0.0:
    raise ValueError(
      f"{class_name} target_visible_px must exceed its rendered stroke width."
    )
  span = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
  if span <= 0.0:
    raise ValueError(f"{class_name} unit geometry must have positive visible extent.")
  scale = available / span
  bounds_center = (
    (bounds[0] + bounds[2]) / 2.0,
    (bounds[1] + bounds[3]) / 2.0,
  )
  canvas_center = (
    config.canvas_width_px / 2.0,
    config.canvas_height_px / 2.0,
  )
  return FanFit(
    scale_px_per_unit=scale,
    origin_px=(
      canvas_center[0] - scale * bounds_center[0],
      canvas_center[1] - scale * bounds_center[1],
    ),
    stroke_width_px=stroke_width_px,
  )


def union_bounds(bounds: Iterable[Bounds], *, class_name: str) -> Bounds:
  """Return the smallest axis-aligned bounds containing every input bound."""

  items = tuple(_bounds(item, f"{class_name} bounds") for item in bounds)
  if not items:
    raise ValueError(f"{class_name} geometry must contain at least one bound.")
  return (
    min(item[0] for item in items),
    min(item[1] for item in items),
    max(item[2] for item in items),
    max(item[3] for item in items),
  )


def expand_bounds(bounds: Bounds, amount: float) -> Bounds:
  """Expand bounds equally in every direction."""

  item = _bounds(bounds, "bounds")
  expansion = _finite(amount, "bounds expansion")
  if expansion < 0.0:
    raise ValueError("bounds expansion must be non-negative.")
  return (
    item[0] - expansion,
    item[1] - expansion,
    item[2] + expansion,
    item[3] + expansion,
  )


def _point(value: object, name: str) -> Point:
  if not isinstance(value, tuple) or len(value) != 2:
    raise TypeError(f"{name} must be a two-item tuple.")
  return (_finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]"))


def _bounds(value: object, name: str) -> Bounds:
  if not isinstance(value, tuple) or len(value) != 4:
    raise TypeError(f"{name} must be a four-item tuple.")
  result = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
  if result[2] < result[0] or result[3] < result[1]:
    raise ValueError(f"{name} must use ordered minimum and maximum coordinates.")
  return result  # type: ignore[return-value]


def _finite(value: object, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a finite number.")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{name} must be a finite number.")
  return result
