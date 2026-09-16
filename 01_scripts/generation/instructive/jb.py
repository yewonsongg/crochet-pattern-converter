"""Filled tapered-bar generation for the join-below symbol."""

from __future__ import annotations

import math
from typing import Any
from xml.etree.ElementTree import Element, SubElement, tostring

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg import rendered_px_to_viewbox


CLASS_NAME = "jb"
SVG_NS = "http://www.w3.org/2000/svg"
Point = tuple[float, float]
Bounds = tuple[float, float, float, float]
CubicSegment = tuple[Point, Point, Point]


def generate_jb(
  spec: ClassSpec,
  sample: SampledParameters,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate one centered, symmetric, filled tapered bar."""

  if not isinstance(sample, SampledParameters):
    raise TypeError("jb generation requires SampledParameters.")
  if spec.class_group != "instructive" or spec.class_name != CLASS_NAME:
    raise ValueError("generate_jb requires the instructive.jb class specification.")

  values = sample.as_dict()
  phenotype = values.get("phenotype")
  if phenotype != "canonical":
    raise ValueError(f"Unsupported jb phenotype: {phenotype!r}.")
  max_width_ratio = _open_unit_interval(
    values.get("max_width_ratio"), "jb max_width_ratio"
  )
  taper_extent = _unit_interval(
    values.get("taper_extent"), "jb taper_extent"
  )
  sampled_stroke_width = _positive_finite(
    values.get("stroke_width"), "jb stroke_width"
  )
  target_visible_px = _positive_finite(
    config.target_visible_px, "jb target_visible_px"
  )

  center = (config.canvas_width_px / 2.0, config.canvas_height_px / 2.0)
  half_height = target_visible_px / 2.0
  half_width = max_width_ratio * target_visible_px / 2.0
  top = (center[0], center[1] - half_height)
  right = (center[0] + half_width, center[1])
  bottom = (center[0], center[1] + half_height)
  left = (center[0] - half_width, center[1])
  segments: tuple[CubicSegment, ...] = (
    (
      (center[0] + taper_extent * half_width, top[1]),
      (right[0], center[1] - taper_extent * half_height),
      right,
    ),
    (
      (right[0], center[1] + taper_extent * half_height),
      (center[0] + taper_extent * half_width, bottom[1]),
      bottom,
    ),
    (
      (center[0] - taper_extent * half_width, bottom[1]),
      (left[0], center[1] + taper_extent * half_height),
      left,
    ),
    (
      (left[0], center[1] - taper_extent * half_height),
      (center[0] - taper_extent * half_width, top[1]),
      top,
    ),
  )
  bounds: Bounds = (
    center[0] - half_width,
    center[1] - half_height,
    center[0] + half_width,
    center[1] + half_height,
  )

  svg, group = _svg_root(config)
  _append_silhouette(group, config=config, start=top, segments=segments)
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "phenotype": phenotype,
    "max_width_ratio": max_width_ratio,
    "taper_extent": taper_extent,
    "height_px": target_visible_px,
    "max_width_px": 2.0 * half_width,
    "center_px": list(center),
    "top_tip_px": list(top),
    "bottom_tip_px": list(bottom),
    "center_extrema_px": [list(left), list(right)],
    "path_segments": [
      {
        "control_1_px": list(control_1),
        "control_2_px": list(control_2),
        "end_px": list(end),
      }
      for control_1, control_2, end in segments
    ],
    "rendered_bounds_px": list(bounds),
    "stroke_width": sampled_stroke_width,
    "stroke_width_applied": False,
    "canvas": {
      "width_px": config.canvas_width_px,
      "height_px": config.canvas_height_px,
    },
    "target_visible_px": target_visible_px,
    "visual_rotation_deg": config.rotation_deg,
  }
  return GeneratedObject(
    class_id=spec.class_id,
    class_name=spec.class_name,
    variant_id=None,
    svg=tostring(svg, encoding="unicode"),
    metadata=metadata,
    obb_pixels=None,
    obb_normalized=None,
    yolo_label=None,
    sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )


def _append_silhouette(
  parent: Element,
  *,
  config: GenerationConfig,
  start: Point,
  segments: tuple[CubicSegment, ...],
) -> None:
  start_svg = rendered_px_to_viewbox(config, start)
  commands = [f"M {start_svg[0]:.8f},{start_svg[1]:.8f}"]
  for control_1, control_2, end in segments:
    control_1_svg = rendered_px_to_viewbox(config, control_1)
    control_2_svg = rendered_px_to_viewbox(config, control_2)
    end_svg = rendered_px_to_viewbox(config, end)
    commands.append(
      f"C {control_1_svg[0]:.8f},{control_1_svg[1]:.8f} "
      f"{control_2_svg[0]:.8f},{control_2_svg[1]:.8f} "
      f"{end_svg[0]:.8f},{end_svg[1]:.8f}"
    )
  commands.append("Z")
  SubElement(parent, "path", {"d": " ".join(commands)})


def _svg_root(config: GenerationConfig) -> tuple[Element, Element]:
  rendered_px_to_viewbox(config, (0.0, 0.0))
  svg = Element("svg", {
    "xmlns": SVG_NS,
    "width": f"{config.canvas_width_px}px",
    "height": f"{config.canvas_height_px}px",
    "viewBox": "0 0 100 100",
  })
  group = SubElement(svg, "g", {
    "fill": "black",
    "stroke": "none",
    "transform": f"rotate({config.rotation_deg} 50 50)",
  })
  return svg, group


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


def _open_unit_interval(value: Any, name: str) -> float:
  result = _finite(value, name)
  if not 0.0 < result < 1.0:
    raise ValueError(f"{name} must be in the range (0, 1).")
  return result


def _unit_interval(value: Any, name: str) -> float:
  result = _finite(value, name)
  if not 0.0 < result <= 1.0:
    raise ValueError(f"{name} must be in the range (0, 1].")
  return result
