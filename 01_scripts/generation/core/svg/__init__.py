"""Shared SVG construction helpers for class generators."""

from .bar_stem import BarStemSvg, build_bar_stem_svg
from .stitch import (
  STITCH_GEOMETRY_REGISTRY,
  StitchGeometry,
  StitchPlacement,
  append_stitch_geometry,
  rendered_px_to_viewbox,
)
from .stroke import resolve_stroke_width

__all__ = [
  "BarStemSvg",
  "STITCH_GEOMETRY_REGISTRY",
  "StitchGeometry",
  "StitchPlacement",
  "append_stitch_geometry",
  "build_bar_stem_svg",
  "rendered_px_to_viewbox",
  "resolve_stroke_width",
]
