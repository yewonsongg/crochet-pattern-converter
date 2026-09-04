from __future__ import annotations
from pathlib import Path


def render_png(
  svg_text: str,
  output_path: Path,
  *,
  output_width: int | None = None,
  output_height: int | None = None,
  background_color: str | None = None,
) -> None:
  """Render SVG text to PNG, optionally resizing and compositing a background."""

  try:
    import cairosvg
  except ImportError as exc:
    raise RuntimeError("PNG rendering requires CairoSVG.") from exc

  output_path.parent.mkdir(parents=True, exist_ok=True)
  options = {
    "output_width": output_width,
    "output_height": output_height,
    "background_color": background_color,
  }
  cairosvg.svg2png(
    bytestring=svg_text.encode("utf-8"),
    write_to=str(output_path),
    **{key: value for key, value in options.items() if value is not None},
  )
