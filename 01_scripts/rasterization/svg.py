"""Rasterize SVG text without changing symbol geometry or labels."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image


def rasterize_svg(
  svg_text: str,
  *,
  output_width: int | None = None,
  output_height: int | None = None,
  background_color: str | None = None,
) -> bytes:
  """Return PNG bytes, preserving transparency unless a background is given."""
  try:
    import cairosvg
  except ImportError as exc:
    raise RuntimeError("SVG rasterization requires CairoSVG.") from exc

  options = {
    "output_width": output_width,
    "output_height": output_height,
    "background_color": background_color,
  }
  return cairosvg.svg2png(
    bytestring=svg_text.encode("utf-8"),
    **{key: value for key, value in options.items() if value is not None},
  )


def downsample_png(
  png_bytes: bytes,
  *,
  output_width: int,
  output_height: int,
) -> bytes:
  """Resize PNG bytes once with a fixed high-quality Lanczos filter."""
  if output_width <= 0 or output_height <= 0:
    raise ValueError("Output dimensions must be positive.")
  with Image.open(BytesIO(png_bytes)) as source:
    if source.format != "PNG":
      raise ValueError("Downsampling requires PNG input.")
    resized = source.resize(
      (output_width, output_height),
      resample=Image.Resampling.LANCZOS,
    )
    output = BytesIO()
    resized.save(output, format="PNG")
  return output.getvalue()


def rasterize_svg_supersampled(
  svg_text: str,
  *,
  output_width: int,
  output_height: int,
  supersample_factor: int = 1,
  background_color: str | None = None,
) -> bytes:
  """Rasterize vector geometry above target size and downsample exactly once."""
  if output_width <= 0 or output_height <= 0:
    raise ValueError("Output dimensions must be positive.")
  if isinstance(supersample_factor, bool) or not isinstance(supersample_factor, int):
    raise ValueError("supersample_factor must be a positive integer.")
  if supersample_factor <= 0:
    raise ValueError("supersample_factor must be a positive integer.")
  sampled = rasterize_svg(
    svg_text,
    output_width=output_width * supersample_factor,
    output_height=output_height * supersample_factor,
    background_color=background_color,
  )
  if supersample_factor == 1:
    return sampled
  return downsample_png(
    sampled,
    output_width=output_width,
    output_height=output_height,
  )


def render_png(
  svg_text: str,
  output_path: Path,
  *,
  output_width: int | None = None,
  output_height: int | None = None,
  background_color: str | None = None,
) -> None:
  """Write rasterized SVG to a PNG file."""
  output_path = Path(output_path)
  png_bytes = rasterize_svg(
    svg_text,
    output_width=output_width,
    output_height=output_height,
    background_color=background_color,
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_bytes(png_bytes)
