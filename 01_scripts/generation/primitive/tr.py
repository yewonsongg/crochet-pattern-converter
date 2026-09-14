from __future__ import annotations

from typing import Any

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg.stitch import build_centered_stitch_svg
from ..core.svg.stroke import resolve_stroke_width


CLASS_NAME = "tr"


def generate_tr(spec: ClassSpec, sample: SampledParameters, config: GenerationConfig) -> GeneratedObject:
  """Generate SVG for one sampled treble-crochet primitive."""
  values = sample.as_dict()
  bar_stem_ratio = values.get("bar_stem_ratio")
  cross_bar_ratio = values.get("cross_bar_ratio")
  cross_bar_y = values.get("cross_bar_y")
  cross_bar_angle_deg = values.get("cross_bar_angle_deg")
  stroke_width = resolve_stroke_width(sample, config, class_name="tr")
  svg, geometry = build_centered_stitch_svg(
    class_name=CLASS_NAME,
    sampled_values=values,
    config=config,
    stroke_width=stroke_width,
  )
  metadata = _metadata(spec, config, geometry, bar_stem_ratio, cross_bar_ratio, cross_bar_y, cross_bar_angle_deg, stroke_width)
  metadata["cross_bar_count"] = 2
  return GeneratedObject(
    class_id=spec.class_id, class_name=spec.class_name, variant_id=None,
    svg=svg, metadata=metadata, obb_pixels=None,
    obb_normalized=None, yolo_label=None, sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )


def _metadata(spec: ClassSpec, config: GenerationConfig, geometry, bar_stem_ratio: float, cross_bar_ratio: float, cross_bar_y: float, cross_bar_angle_deg: float, stroke_width: float) -> dict[str, Any]:
  return {
    "class_id": spec.class_id, "class_name": spec.class_name,
    "bar_stem_ratio": bar_stem_ratio, "cross_bar_ratio": cross_bar_ratio,
    "cross_bar_y": cross_bar_y, "cross_bar_angle_deg": cross_bar_angle_deg,
    "cross_bar_y_positions": list(geometry.cross_bar_positions),
    "stem_length_px": geometry.stem_length_px, "bar_length_px": geometry.top_bar_length_px,
    "cross_bar_length_px": geometry.cross_bar_length_px, "stroke_width": stroke_width,
    "canvas": {"width_px": config.canvas_width_px, "height_px": config.canvas_height_px},
    "target_visible_px": config.target_visible_px, "visual_rotation_deg": config.rotation_deg,
  }
