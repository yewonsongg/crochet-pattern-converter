"""Rasterize SVG text without changing symbol geometry or labels."""

from __future__ import annotations

from pathlib import Path


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
