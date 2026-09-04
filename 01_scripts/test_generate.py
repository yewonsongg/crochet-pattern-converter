from __future__ import annotations

import argparse
import json
import math
import cv2
import numpy as np

from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring


SVG_NS = "http://www.w3.org/2000/svg"


def make_square_sc_svg(
  *,
  canvas_px: int = 25,
  target_visible_px: float = 15.0,
  rotation_deg: float = 0.0,
  stroke_width_normalized: float = 4.0,
  class_id: int = 0,
  output_path: Path | None = None,
) -> dict:
  """
  Generate a square-form single-crochet symbol.

  The visual symbol may be rotated, but because this phenotype is
  symmetric/square-like, its OBB label uses a canonical angle of 0 degrees.

  Writes:
    - SVG
    - JSON metadata
    - YOLO OBB TXT label

  Assumes these helpers already exist:
    make_sc_obb()
    normalize_obb()
    format_yolo_obb_label()
  """

  if canvas_px <= 0:
    raise ValueError("canvas_px must be positive")

  if target_visible_px <= 0:
    raise ValueError("target_visible_px must be positive")

  if not 0 <= class_id:
    raise ValueError("class_id must be non-negative")

  # The cross occupies approximately normalized coordinates 20..80,
  # giving an approximately 60/100 fraction of the SVG canvas.
  visible_span_normalized = 60.0

  estimated_visible_px = (
    canvas_px * visible_span_normalized / 100.0
  )

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
      # This rotates the visible symbol only.
      "transform": f"rotate({rotation_deg} 50 50)",
    },
  )

  # Horizontal arm.
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

  # Vertical arm.
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

  svg_text = tostring(
    svg,
    encoding="unicode",
  )

  if output_path is not None:
    output_path.parent.mkdir(
      parents=True,
      exist_ok=True,
    )

  # For this symmetric square phenotype, use a canonical OBB.
  #
  # Important:
  # - visual_rotation_deg describes the rendered symbol
  # - obb_angle_deg remains 0 because OBB orientation is not meaningful
  obb_angle_deg = 0.0

  obb_pixels = make_sc_obb(
    canvas_px=canvas_px,
    rotation_deg=obb_angle_deg,
  )

  obb_normalized = normalize_obb(
    obb_pixels,
    width_px=canvas_px,
    height_px=canvas_px,
  )

  yolo_label = format_yolo_obb_label(
    class_id=class_id,
    obb_pixels=obb_pixels,
    width_px=canvas_px,
    height_px=canvas_px,
  )

  metadata = {
    "class_id": class_id,
    "class_name": "sc",
    "phenotype": "square_symmetric_cross",

    "canvas": {
      "width_px": canvas_px,
      "height_px": canvas_px,
    },

    "target_visible_px": target_visible_px,
    "estimated_visible_width_px": estimated_visible_px,
    "estimated_visible_height_px": estimated_visible_px,

    "viewbox": [0, 0, 100, 100],

    "visual_rotation_deg": rotation_deg,
    "obb_angle_deg": obb_angle_deg,
    "orientation_policy": "canonical",

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
      "approx_visible_bounds_normalized": [
        20,
        20,
        80,
        80,
      ],
    },

    "obb": {
      "pixels": obb_pixels.tolist(),
      "normalized": obb_normalized.tolist(),
      "yolo_label": yolo_label,
    },
  }

  if output_path is not None:
    # SVG
    output_path.write_text(
      svg_text,
      encoding="utf-8",
    )

    # JSON metadata
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
      json.dumps(metadata, indent=2),
      encoding="utf-8",
    )

    # YOLO OBB label
    label_path = output_path.with_suffix(".txt")
    label_path.write_text(
      yolo_label + "\n",
      encoding="utf-8",
    )

  return {
    "svg": svg_text,
    "metadata": metadata,
    "obb_pixels": obb_pixels,
    "obb_normalized": obb_normalized,
    "yolo_label": yolo_label,
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


def rotate_points(
  points: np.ndarray,
  angle_deg: float,
  center: tuple[float, float],
) -> np.ndarray:
  """
  Rotate Nx2 points around center by angle_deg.
  """

  angle_rad = math.radians(angle_deg)

  rotation = np.array([
    [math.cos(angle_rad), -math.sin(angle_rad)],
    [math.sin(angle_rad), math.cos(angle_rad)],
  ])

  center_array = np.asarray(center, dtype=np.float32)

  return (
    (points - center_array) @ rotation.T
    + center_array
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


def make_sc_obb(
  *,
  canvas_px: int,
  rotation_deg: float,
  ) -> np.ndarray:
  """
  Return four OBB corners in pixel coordinates.

  Corner order:
    top-left, top-right, bottom-right, bottom-left
    before rotation, then rotated around center.
  """

  # Approximate visible geometry including stroke expansion.
  corners = np.array([
    [20.0, 20.0],
    [80.0, 20.0],
    [80.0, 80.0],
    [20.0, 80.0],
  ], dtype=np.float32)

  center = (50.0, 50.0)

  rotated = rotate_points(
    corners,
    angle_deg=rotation_deg,
    center=center,
  )

  rotated *= canvas_px / 100.0

  return rotated


def normalize_obb(
  obb_pixels: np.ndarray,
  width_px: int,
  height_px: int,
) -> np.ndarray:
  normalized = obb_pixels.copy()

  normalized[:, 0] /= width_px
  normalized[:, 1] /= height_px

  return normalized


def format_yolo_obb_label(
  class_id: int,
  obb_pixels: np.ndarray,
  width_px: int,
  height_px: int,
) -> str:
  normalized = normalize_obb(
    obb_pixels,
    width_px,
    height_px,
  )

  values = [class_id, *normalized.reshape(-1).tolist()]

  return " ".join(
    f"{value:.8f}" if index > 0 else str(value)
    for index, value in enumerate(values)
  )


def draw_obb_overlay(
  image_path: Path,
  obb_pixels: np.ndarray,
  output_path: Path,
  color: tuple[int, int, int] = (0, 0, 255),
) -> None:
  image = cv2.imread(str(image_path))

  if image is None:
    raise FileNotFoundError(image_path)

  points = np.round(obb_pixels).astype(np.int32)
  points = points.reshape((-1, 1, 2))

  cv2.polylines(
    image,
    [points],
    isClosed=True,
    color=color,
    thickness=1,
    lineType=cv2.LINE_AA,
  )

  # Draw corner indices for debugging.
  for index, point in enumerate(obb_pixels):
    x, y = np.round(point).astype(int)

    cv2.circle(
      image,
      (x, y),
      radius=2,
      color=(255, 0, 0),
      thickness=-1,
    )

    cv2.putText(
      image,
      str(index),
      (x + 3, y - 3),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.35,
      (255, 0, 0),
      1,
      cv2.LINE_AA,
    )

  output_path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(output_path), image)


if __name__ == "__main__":
    main()