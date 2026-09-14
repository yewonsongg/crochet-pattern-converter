"""Reusable symmetric fan placement for compound crochet symbols."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from ..core.svg import StitchPlacement


Point = tuple[float, float]


@dataclass(frozen=True)
class SymmetricFanLayout:
  """Equal-length placements distributed across a centered angular envelope."""

  axis_angles_deg: tuple[float, ...]
  placements: tuple[StitchPlacement, ...]


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


def _point(value: object, name: str) -> Point:
  if not isinstance(value, tuple) or len(value) != 2:
    raise TypeError(f"{name} must be a two-item tuple.")
  return (_finite(value[0], f"{name}[0]"), _finite(value[1], f"{name}[1]"))


def _finite(value: object, name: str) -> float:
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f"{name} must be a finite number.")
  result = float(value)
  if not math.isfinite(result):
    raise ValueError(f"{name} must be a finite number.")
  return result
