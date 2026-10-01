"""Combined upright rasterization and label measurement."""

from __future__ import annotations

from generation.core.models import GeneratedObject
from rasterization import downsample_png, rasterize_svg

from .obb import label_upright_png


def rasterize_and_label_upright_svg(
  generated: GeneratedObject,
  *,
  supersample_factor: int = 1,
  alpha_threshold: int = 0,
) -> bytes:
  """Return a logical-size preview after measuring a supersampled alpha mask."""
  if (
    isinstance(supersample_factor, bool)
    or not isinstance(supersample_factor, int)
    or supersample_factor <= 0
  ):
    raise ValueError("supersample_factor must be a positive integer.")
  width, height = generated.canvas_size_px
  sampled = rasterize_svg(
    generated.svg,
    output_width=width * supersample_factor,
    output_height=height * supersample_factor,
  )
  label_upright_png(
    generated,
    sampled,
    alpha_threshold=alpha_threshold,
    raster_scale=supersample_factor,
  )
  if supersample_factor == 1:
    return sampled
  return downsample_png(sampled, output_width=width, output_height=height)
