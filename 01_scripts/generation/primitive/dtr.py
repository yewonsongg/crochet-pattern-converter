from __future__ import annotations

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg.stitch import build_centered_stitch_svg
from ..core.svg.stroke import resolve_stroke_width


CLASS_NAME = "dtr"


def generate_dtr(spec: ClassSpec, sample: SampledParameters, config: GenerationConfig) -> GeneratedObject:
  """Generate SVG for one sampled double-treble-crochet primitive."""
  values = sample.as_dict()
  bar_stem_ratio = values.get("bar_stem_ratio")
  cross_bar_ratio = values.get("cross_bar_ratio")
  cross_bar_y = values.get("cross_bar_y")
  cross_bar_angle_deg = values.get("cross_bar_angle_deg")
  stroke_width = resolve_stroke_width(sample, config, class_name="dtr")
  svg, geometry = build_centered_stitch_svg(
    class_name=CLASS_NAME,
    sampled_values=values,
    config=config,
    stroke_width=stroke_width,
  )
  metadata = {
    "class_id": spec.class_id, "class_name": spec.class_name,
    "bar_stem_ratio": bar_stem_ratio, "cross_bar_ratio": cross_bar_ratio,
    "cross_bar_y": cross_bar_y, "cross_bar_angle_deg": cross_bar_angle_deg,
    "cross_bar_count": 3,
    "cross_bar_y_positions": list(geometry.cross_bar_positions),
    "stem_length_px": geometry.stem_length_px, "bar_length_px": geometry.top_bar_length_px,
    "cross_bar_length_px": geometry.cross_bar_length_px, "stroke_width": stroke_width,
    "canvas": {"width_px": config.canvas_width_px, "height_px": config.canvas_height_px},
    "target_visible_px": config.target_visible_px, "visual_rotation_deg": config.rotation_deg,
  }
  return GeneratedObject(
    class_id=spec.class_id, class_name=spec.class_name, variant_id=None,
    svg=svg, metadata=metadata, obb_pixels=None,
    obb_normalized=None, yolo_label=None, sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )
