from __future__ import annotations
from typing import Any, Mapping
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np

from ..core.models import GeneratedObject, GenerationConfig
from ..core.obb import format_yolo_obb_label, normalize_obb
from ..core.transforms import rotate_points


CLASS_ID = 00
CLASS_NAME = "sc"
PHENOTYPE = "square_symmetric_cross"
SVG_NS = "http://www.w3.org/2000/svg"


def _build_sc_obb(config: GenerationConfig, angle_deg: float = 0.0) -> np.ndarray:
  """Build SC's canonical square OBB; visual rotation is not encoded."""

  corners = np.array([
    [20.0, 20.0], [80.0, 20.0],
    [80.0, 80.0], [20.0, 80.0],
  ], dtype=np.float32)
  
  rotated = rotate_points(corners, angle_deg, center=(50.0, 50.0))
  rotated[:, 0] *= config.canvas_width_px / 100.0
  rotated[:, 1] *= config.canvas_height_px / 100.0
  return rotated


def _build_sc_svg(config: GenerationConfig) -> str:
  svg = Element("svg", {
    "xmlns": SVG_NS,
    "width": f"{config.canvas_width_px}px",
    "height": f"{config.canvas_height_px}px",
    "viewBox": "0 0 100 100",
  })
  group = SubElement(svg, "g", {
    "fill": "none",
    "stroke": "black",
    "stroke-width": str(config.stroke_width_normalized),
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "transform": f"rotate({config.rotation_deg} 50 50)",
  })
  SubElement(group, "line", {"x1": "22", "y1": "50", "x2": "78", "y2": "50"})
  SubElement(group, "line", {"x1": "50", "y1": "22", "x2": "50", "y2": "78"})
  return tostring(svg, encoding="unicode")


def generate_sc(config: GenerationConfig | None = None, sampled_parameters: Mapping[str, Any] | None = None) -> GeneratedObject:
  config = config or GenerationConfig()
  svg = _build_sc_svg(config)
  # SC is orientation-invariant for labeling purposes.
  obb_pixels = _build_sc_obb(config, angle_deg=0.0)
  obb_normalized = normalize_obb(
    obb_pixels, config.canvas_width_px, config.canvas_height_px
  )
  yolo_label = format_yolo_obb_label(
    CLASS_ID, obb_pixels, config.canvas_width_px, config.canvas_height_px
  )
  metadata = {
    "class_id": CLASS_ID,
    "class_name": CLASS_NAME,
    "phenotype": PHENOTYPE,
    "canvas": {
      "width_px": config.canvas_width_px,
      "height_px": config.canvas_height_px,
    },
    "target_visible_px": config.target_visible_px,
    "visual_rotation_deg": config.rotation_deg,
    "obb_angle_deg": 0.0,
    "orientation_policy": "canonical",
    "stroke_width_normalized": config.stroke_width_normalized,
    "geometry": {
      "approx_visible_bounds_normalized": [20, 20, 80, 80],
    },
    "obb": {
      "pixels": obb_pixels.tolist(),
      "normalized": obb_normalized.tolist(),
      "yolo_label": yolo_label,
    },
  }
  sampled_parameters = dict(sampled_parameters or {})
  return GeneratedObject(
    class_id=CLASS_ID,
    class_name=CLASS_NAME,
    phenotype=PHENOTYPE,
    svg=svg,
    metadata=metadata,
    obb_pixels=obb_pixels,
    obb_normalized=obb_normalized,
    yolo_label=yolo_label,
    sampled_parameters=sampled_parameters,
  )
