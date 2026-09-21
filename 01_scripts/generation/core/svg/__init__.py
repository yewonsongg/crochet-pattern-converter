"""Shared SVG construction helpers for class generators."""

from .bar_stem import BarStemSvg, build_bar_stem_svg
from .chain import ChainGeometry, ChainPlacement, append_chain_geometry
from .slip_stitch import (
  SlipStitchGeometry,
  SlipStitchPlacement,
  append_slip_stitch_geometry,
)
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
  "ChainGeometry",
  "ChainPlacement",
  "SlipStitchGeometry",
  "SlipStitchPlacement",
  "STITCH_GEOMETRY_REGISTRY",
  "StitchGeometry",
  "StitchPlacement",
  "append_stitch_geometry",
  "append_chain_geometry",
  "append_slip_stitch_geometry",
  "build_bar_stem_svg",
  "rendered_px_to_viewbox",
  "resolve_stroke_width",
]
