"""Shared SVG construction helpers for class generators."""

from .bar_stem import BarStemSvg, build_bar_stem_svg
from .stroke import resolve_stroke_width

__all__ = ["BarStemSvg", "build_bar_stem_svg", "resolve_stroke_width"]
