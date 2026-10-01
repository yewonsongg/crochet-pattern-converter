"""Axis-aligned local boxes serialized as four-corner YOLO OBB labels."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from generation.core.models import GeneratedObject


def upright_obb_from_alpha(
  alpha: np.ndarray,
  *,
  alpha_threshold: int = 0,
) -> np.ndarray:
  """Fit an upright rectangle to visible pixels, using outer pixel edges."""
  if alpha.ndim != 2:
    raise ValueError("Alpha mask must be a two-dimensional array.")
  if alpha.size == 0:
    raise ValueError("Alpha mask must not be empty.")
  if not isinstance(alpha_threshold, int) or isinstance(alpha_threshold, bool) or not 0 <= alpha_threshold < 255:
    raise ValueError("alpha_threshold must be an integer from 0 to 254.")

  rows, columns = np.nonzero(alpha > alpha_threshold)
  if rows.size == 0:
    raise ValueError("Alpha mask contains no visible pixels.")

  left = float(columns.min())
  top = float(rows.min())
  right = float(columns.max() + 1)
  bottom = float(rows.max() + 1)
  return np.array([
    [left, top],
    [right, top],
    [right, bottom],
    [left, bottom],
  ], dtype=np.float64)


def normalize_obb(
  obb_pixels: np.ndarray,
  width_px: int,
  height_px: int,
) -> np.ndarray:
  """Normalize four pixel-space corners to image width and height."""
  if width_px <= 0 or height_px <= 0:
    raise ValueError("Image dimensions must be positive.")
  normalized = np.asarray(obb_pixels, dtype=np.float64).copy()
  if normalized.shape != (4, 2):
    raise ValueError("OBB corners must have shape (4, 2).")
  normalized[:, 0] /= width_px
  normalized[:, 1] /= height_px
  return normalized


def format_yolo_obb_label(
  class_id: int,
  obb_pixels: np.ndarray,
  width_px: int,
  height_px: int,
) -> str:
  """Serialize a four-corner box in YOLO OBB coordinate order."""
  normalized = normalize_obb(obb_pixels, width_px, height_px)
  values = [class_id, *normalized.reshape(-1).tolist()]
  return " ".join(
    f"{value:.8f}" if index else str(value)
    for index, value in enumerate(values)
  )


def label_upright_alpha(
  generated: GeneratedObject,
  alpha: np.ndarray,
  *,
  alpha_threshold: int = 0,
  raster_scale: int = 1,
) -> GeneratedObject:
  """Measure an alpha mask and store its OBB in logical SVG coordinates."""
  if alpha.ndim != 2:
    raise ValueError("Alpha mask must be a two-dimensional array.")
  if isinstance(raster_scale, bool) or not isinstance(raster_scale, int) or raster_scale <= 0:
    raise ValueError("raster_scale must be a positive integer.")
  height_px, width_px = alpha.shape
  logical_width, logical_height = generated.canvas_size_px
  if (width_px, height_px) != (
    logical_width * raster_scale,
    logical_height * raster_scale,
  ):
    raise ValueError("Alpha mask dimensions must match the scaled generated canvas.")

  obb_pixels = upright_obb_from_alpha(
    alpha, alpha_threshold=alpha_threshold,
  ) / raster_scale
  generated.obb_pixels = obb_pixels
  generated.obb_normalized = normalize_obb(obb_pixels, logical_width, logical_height)
  generated.yolo_label = format_yolo_obb_label(
    generated.class_id, obb_pixels, logical_width, logical_height,
  )
  return generated


def label_upright_png(
  generated: GeneratedObject,
  png_bytes: bytes,
  *,
  alpha_threshold: int = 0,
  raster_scale: int = 1,
) -> GeneratedObject:
  """Label a transparent PNG previously rasterized from the upright SVG."""
  with Image.open(BytesIO(png_bytes)) as image:
    if image.format != "PNG" or "A" not in image.getbands():
      raise ValueError("Labeling requires a PNG with an alpha channel.")
    alpha = np.asarray(image.getchannel("A"))
  return label_upright_alpha(
    generated,
    alpha,
    alpha_threshold=alpha_threshold,
    raster_scale=raster_scale,
  )


def write_yolo_label(generated: GeneratedObject, output_path: Path) -> None:
  """Write the completed YOLO OBB line for one generated object."""
  if generated.yolo_label is None:
    raise ValueError("Generated object has no OBB label.")
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(generated.yolo_label + "\n", encoding="utf-8")
  generated.label_path = output_path
