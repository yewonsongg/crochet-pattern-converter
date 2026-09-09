from __future__ import annotations

from dataclasses import dataclass
import math
from xml.etree.ElementTree import Element, SubElement, tostring

from ..models import GenerationConfig


@dataclass(frozen=True)
class BarStemSvg:
  """SVG and measurements for a stem with an attached bar and crossbars."""

  svg: str
  stem_length_px: float
  bar_length_px: float
  cross_bar_length_px: float
  cross_bar_y_positions: tuple[float, ...]


def build_bar_stem_svg(
  config: GenerationConfig,
  *,
  bar_stem_ratio: float,
  cross_bar_ratio: float = 0.0,
  cross_bar_y: float | None = None,
  cross_bar_angle_deg: float = 0.0,
  cross_bar_count: int = 0,
  stroke_width: float | None = None,
) -> BarStemSvg:
  """Build shared hdc/dc/tr/dtr bar-and-stem geometry."""
  if bar_stem_ratio <= 0:
    raise ValueError("bar_stem_ratio must be positive.")
  if cross_bar_count < 0:
    raise ValueError("cross_bar_count must be non-negative.")
  if cross_bar_count and cross_bar_ratio <= 0:
    raise ValueError("cross_bar_ratio must be positive when crossbars are enabled.")
  if cross_bar_count and cross_bar_y is None:
    raise ValueError("cross_bar_y is required when crossbars are enabled.")
  if cross_bar_y is not None and not 0.0 <= cross_bar_y <= 1.0:
    raise ValueError("cross_bar_y must be in the range [0, 1].")
  if not -45.0 <= cross_bar_angle_deg <= 45.0:
    raise ValueError("cross_bar_angle_deg must be in the range [-45, 45].")

  stem_length_px = float(config.target_visible_px)
  bar_length_px = stem_length_px * bar_stem_ratio
  cross_bar_length_px = stem_length_px * cross_bar_ratio
  stem_half_height = 50.0 * stem_length_px / config.canvas_height_px
  bar_half_width = 50.0 * bar_length_px / config.canvas_width_px
  stem_top = 50.0 - stem_half_height
  stem_bottom = 50.0 + stem_half_height

  if cross_bar_count:
    assert cross_bar_y is not None
    anchor = float(cross_bar_y)
    spacing = 0.08
    offsets = [(index - (cross_bar_count - 1) / 2.0) * spacing for index in range(cross_bar_count)]
    cross_bar_y_positions = tuple(anchor + offset for offset in offsets)
    if any(not 0.0 <= position <= 1.0 for position in cross_bar_y_positions):
      raise ValueError("crossbar positions derived from cross_bar_y must stay within the stem.")
  else:
    cross_bar_y_positions = ()

  stroke = config.stroke_width_normalized if stroke_width is None else stroke_width
  svg = Element("svg", {
    "xmlns": "http://www.w3.org/2000/svg",
    "width": f"{config.canvas_width_px}px",
    "height": f"{config.canvas_height_px}px",
    "viewBox": "0 0 100 100",
  })
  group = SubElement(svg, "g", {
    "fill": "none", "stroke": "black", "stroke-width": str(stroke),
    "stroke-linecap": "round", "stroke-linejoin": "round",
    "transform": f"rotate({config.rotation_deg} 50 50)",
  })
  SubElement(group, "line", {"x1": "50", "y1": f"{stem_top:.8f}", "x2": "50", "y2": f"{stem_bottom:.8f}"})
  SubElement(group, "line", {"x1": f"{50.0 - bar_half_width:.8f}", "y1": f"{stem_top:.8f}", "x2": f"{50.0 + bar_half_width:.8f}", "y2": f"{stem_top:.8f}"})

  half_crossbar_px = cross_bar_length_px / 2.0
  angle_rad = math.radians(cross_bar_angle_deg)
  cross_bar_dx = 100.0 * half_crossbar_px * math.cos(angle_rad) / config.canvas_width_px
  cross_bar_dy = 100.0 * half_crossbar_px * math.sin(angle_rad) / config.canvas_height_px
  for position in cross_bar_y_positions:
    center_y = stem_bottom - (position * stem_length_px * 100.0 / config.canvas_height_px)
    SubElement(group, "line", {
      "x1": f"{50.0 - cross_bar_dx:.8f}", "y1": f"{center_y + cross_bar_dy:.8f}",
      "x2": f"{50.0 + cross_bar_dx:.8f}", "y2": f"{center_y - cross_bar_dy:.8f}",
    })

  return BarStemSvg(tostring(svg, encoding="unicode"), stem_length_px, bar_length_px, cross_bar_length_px, cross_bar_y_positions)
