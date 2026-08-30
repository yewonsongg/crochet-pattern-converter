from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring


SVG_NS = "http://www.w3.org/2000/svg"


def make_square_sc_svg(
  *,
  canvas_px: int = 25,
  target_visible_px: float = 15.0,
  rotation_deg: float = 0.0,
  stroke_width_normalized: float = 4.0,
  output_path: Path | None = None,
) -> dict:
  """
  Generate a square-form single-crochet symbol as a symmetric cross.

  Geometry is authored in normalized coordinates:
      viewBox = 0 0 100 100

  The visible cross is centered at (50, 50).
  Its visible geometry is approximately target_visible_px wide/high
  when rendered onto canvas_px x canvas_px.
  """

  if canvas_px <= 0:
    raise ValueError("canvas_px must be positive")

  if target_visible_px <= 0:
    raise ValueError("target_visible_px must be positive")

  # The visible geometry before stroke expansion is 60 normalized units:
  #
  # horizontal line: x=22..78
  # vertical line:   y=22..78
  #
  # With stroke width 4 normalized units, the approximate visible bounds
  # become x/y = 20..80, i.e. 60 normalized units.
  visible_span_normalized = 60.0

  # Since the SVG viewBox spans 100 units, this gives:
  #
  # visible_px = canvas_px * 60 / 100
  #
  # To hit target_visible_px, choose a canvas size accordingly.
  scale_from_normalized = target_visible_px / visible_span_normalized
  inferred_canvas_px = int(round(100 * scale_from_normalized))

  # Use the requested canvas size, but record the actual resulting estimate.
  actual_visible_px = canvas_px * visible_span_normalized / 100.0

  svg = Element(
    "svg",
    {
      "xmlns": SVG_NS,
      "width": f"{canvas_px}px",
      "height": f"{canvas_px}px",
      "viewBox": "0 0 100 100",
    },
  )

  group = SubElement(
    svg,
    "g",
    {
      "fill": "none",
      "stroke": "black",
      "stroke-width": str(stroke_width_normalized),
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "transform": f"rotate({rotation_deg} 50 50)",
    },
  )

  # Symmetric horizontal arm.
  SubElement(
    group,
    "line",
    {
      "x1": "22",
      "y1": "50",
      "x2": "78",
      "y2": "50",
    },
  )

  # Symmetric vertical arm.
  SubElement(
    group,
    "line",
    {
      "x1": "50",
      "y1": "22",
      "x2": "50",
      "y2": "78",
    },
  )

  svg_text = tostring(svg, encoding="unicode")

  metadata = {
    "class_name": "sc",
    "phenotype": "square_symmetric_cross",
    "canvas_px": canvas_px,
    "target_visible_px": target_visible_px,
    "estimated_visible_width_px": actual_visible_px,
    "estimated_visible_height_px": actual_visible_px,
    "viewbox": [0, 0, 100, 100],
    "rotation_deg": rotation_deg,
    "stroke_width_normalized": stroke_width_normalized,
    "geometry": {
      "horizontal_arm": {
        "x1": 22,
        "y1": 50,
        "x2": 78,
        "y2": 50,
      },
      "vertical_arm": {
        "x1": 50,
        "y1": 22,
        "x2": 50,
        "y2": 78,
      },
      "approx_visible_bounds_normalized": [20, 20, 80, 80],
    },
    "inferred_canvas_for_target_px": inferred_canvas_px,
  }

  if output_path is not None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg_text, encoding="utf-8")

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
      json.dumps(metadata, indent=2),
      encoding="utf-8",
    )

  return {
    "svg": svg_text,
    "metadata": metadata,
  }


def render_png(svg_text: str, output_path: Path) -> None:
  try:
    import cairosvg
  except ImportError as exc:
    raise RuntimeError(
      "PNG rendering requires CairoSVG. "
      "Install it in the active Colab Runtime with:\n"
      "%pip install cairosvg"
    ) from exc

  output_path.parent.mkdir(parents=True, exist_ok=True)

  try:
    cairosvg.svg2png(
      bytestring=svg_text.encode("utf-8"),
      write_to=str(output_path),
    )
  except Exception as exc:
    raise RuntimeError(
      f"CairoSVG failed to render PNG at {output_path}"
    ) from exc

  if not output_path.exists():
    raise RuntimeError(
      f"CairoSVG returned without creating {output_path}"
    )


def main() -> None:
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("smoke_outputs/sc"),
  )

  parser.add_argument(
    "--canvas-px",
    type=int,
    default=25,
  )

  parser.add_argument(
    "--target-visible-px",
    type=float,
    default=15.0,
  )

  parser.add_argument(
    "--rotation",
    type=float,
    default=0.0,
  )

  parser.add_argument(
    "--render-png",
    action="store_true",
    help="Render PNG using CairoSVG if installed.",
  )

  args = parser.parse_args()

  output_dir = args.output_dir
  output_dir.mkdir(parents=True, exist_ok=True)

  stem = (
    f"sc_square_cross_"
    f"canvas{args.canvas_px}_"
    f"rot{args.rotation:g}"
  )

  svg_path = output_dir / f"{stem}.svg"
  png_path = output_dir / f"{stem}.png"

  result = make_square_sc_svg(
    canvas_px=args.canvas_px,
    target_visible_px=args.target_visible_px,
    rotation_deg=args.rotation,
    output_path=svg_path,
  )

  rendered = False

  if args.render_png:
    render_png(
      result["svg"],
      png_path,
    )
    rendered = True

  print(json.dumps({
    "svg_path": str(svg_path),
    "png_path": str(png_path) if rendered else None,
    "metadata": result["metadata"],
  }, indent=2))


if __name__ == "__main__":
    main()