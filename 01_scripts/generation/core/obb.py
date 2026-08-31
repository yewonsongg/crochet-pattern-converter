from __future__ import annotations

import numpy as np


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
  normalized = normalize_obb(obb_pixels, width_px, height_px)
  values = [class_id, *normalized.reshape(-1).tolist()]
  return " ".join(
    f"{value:.8f}" if index else str(value)
    for index, value in enumerate(values)
  )
