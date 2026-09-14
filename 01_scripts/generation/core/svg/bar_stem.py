"""Compatibility wrapper for standalone bar-and-stem SVG construction."""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree.ElementTree import Element, SubElement, tostring

from ..models import GenerationConfig
from .stitch import StitchPlacement, append_bar_stem_geometry


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
  """Build shared hdc/dc/tr/dtr geometry through the anchor-based API."""

  stroke = config.stroke_width_normalized if stroke_width is None else stroke_width
  svg = Element("svg", {
    "xmlns": "http://www.w3.org/2000/svg",
    "width": f"{config.canvas_width_px}px",
    "height": f"{config.canvas_height_px}px",
    "viewBox": "0 0 100 100",
  })
  group = SubElement(svg, "g", {
    "fill": "none",
    "stroke": "black",
    "stroke-width": str(stroke),
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "transform": f"rotate({config.rotation_deg} 50 50)",
  })
  center_x = config.canvas_width_px / 2.0
  center_y = config.canvas_height_px / 2.0
  half_length = config.target_visible_px / 2.0
  geometry = append_bar_stem_geometry(
    group,
    class_name="bar_stem",
    config=config,
    placement=StitchPlacement(
      base_px=(center_x, center_y + half_length),
      top_px=(center_x, center_y - half_length),
    ),
    bar_stem_ratio=bar_stem_ratio,
    cross_bar_ratio=cross_bar_ratio,
    cross_bar_y=cross_bar_y,
    cross_bar_angle_deg=cross_bar_angle_deg,
    cross_bar_count=cross_bar_count,
    stroke_width=stroke,
  )
  return BarStemSvg(
    svg=tostring(svg, encoding="unicode"),
    stem_length_px=geometry.stem_length_px,
    bar_length_px=geometry.top_bar_length_px,
    cross_bar_length_px=geometry.cross_bar_length_px,
    cross_bar_y_positions=geometry.cross_bar_positions,
  )
