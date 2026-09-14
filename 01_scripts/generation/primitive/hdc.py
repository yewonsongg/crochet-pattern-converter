from __future__ import annotations

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg.stitch import build_centered_stitch_svg
from ..core.svg.stroke import resolve_stroke_width


CLASS_NAME = "hdc"


def generate_hdc(spec: ClassSpec, sample: SampledParameters, config: GenerationConfig) -> GeneratedObject:
  """Generate SVG for one sampled half-double-crochet primitive."""
  values = sample.as_dict()
  bar_stem_ratio = values.get("bar_stem_ratio")
  stroke_width = resolve_stroke_width(sample, config, class_name="hdc")
  svg, geometry = build_centered_stitch_svg(
    class_name=CLASS_NAME,
    sampled_values=values,
    config=config,
    stroke_width=stroke_width,
  )
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "bar_stem_ratio": bar_stem_ratio,
    "stem_length_px": geometry.stem_length_px,
    "bar_length_px": geometry.top_bar_length_px,
    "canvas": {"width_px": config.canvas_width_px, "height_px": config.canvas_height_px},
    "target_visible_px": config.target_visible_px,
    "visual_rotation_deg": config.rotation_deg,
    "stroke_width": stroke_width,
  }
  return GeneratedObject(
    class_id=spec.class_id, class_name=spec.class_name, variant_id=None,
    svg=svg, metadata=metadata, obb_pixels=None,
    obb_normalized=None, yolo_label=None, sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )
