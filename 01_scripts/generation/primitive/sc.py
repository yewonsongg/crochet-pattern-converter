from __future__ import annotations

from ..core.models import GeneratedObject, GenerationConfig, SampledParameters
from ..core.sampling.schema import ClassSpec
from ..core.svg.stitch import build_centered_stitch_svg
from ..core.svg.stroke import resolve_stroke_width


CLASS_NAME = "sc"
def generate_sc(
  spec: ClassSpec,
  sample: SampledParameters,
  config: GenerationConfig,
) -> GeneratedObject:
  """Generate SVG for one sampled single-crochet plus/cross primitive."""
  values = sample.as_dict()
  shape = values.get("shape")
  asymmetry = values.get("asymmetry")
  cross_bar_ratio = values.get("cross_bar_ratio")
  stroke_width = resolve_stroke_width(sample, config, class_name="sc")
  if shape not in {"symmetric", "asymmetric"}:
    raise ValueError(f"Unsupported sc shape: {shape!r}.")
  if isinstance(asymmetry, bool) or not isinstance(asymmetry, (int, float)):
    raise ValueError("sc asymmetry must be numeric.")
  if isinstance(cross_bar_ratio, bool) or not isinstance(cross_bar_ratio, (int, float)):
    raise ValueError("sc cross_bar_ratio must be numeric.")

  asymmetry = float(asymmetry)
  cross_bar_ratio = float(cross_bar_ratio)
  if shape == "symmetric" and asymmetry != 0.0:
    raise ValueError("A symmetric sc must have asymmetry=0.0.")
  # The asymmetric sampling branch may legitimately include zero (for
  # example, its configured mean or a deterministic coverage probe). In that
  # case the geometry is simply the centered limit of the branch.

  svg, _ = build_centered_stitch_svg(
    class_name=CLASS_NAME,
    sampled_values=values,
    config=config,
    stroke_width=stroke_width,
  )
  metadata = {
    "class_id": spec.class_id,
    "class_name": spec.class_name,
    "shape": shape,
    "asymmetry": asymmetry,
    "cross_bar_ratio": cross_bar_ratio,
    "canvas": {
      "width_px": config.canvas_width_px,
      "height_px": config.canvas_height_px,
    },
    "target_visible_px": config.target_visible_px,
    "visual_rotation_deg": config.rotation_deg,
    "stroke_width": stroke_width,
  }
  return GeneratedObject(
    class_id=spec.class_id,
    class_name=spec.class_name,
    variant_id=None,
    svg=svg,
    metadata=metadata,
    obb_pixels=None,
    obb_normalized=None,
    yolo_label=None,
    sampled_parameters=dict(values),
    sampling_provenance=sample.provenance,
  )
