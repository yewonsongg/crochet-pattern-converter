from __future__ import annotations

import math
import numpy as np


def rotate_points(
  points: np.ndarray,
  angle_deg: float,
  center: tuple[float, float],
) -> np.ndarray:
  """Rotate an Nx2 point array around a center."""

  angle_rad = math.radians(angle_deg)

  # this a standard counterclockwise 2D rotation matrix
  rotation = np.array([
    [math.cos(angle_rad), -math.sin(angle_rad)],
    [math.sin(angle_rad), math.cos(angle_rad)],
  ])
  center_array = np.asarray(center, dtype=np.float32)

  # apply rotation to every point, points translated to origin
  return (points - center_array) @ rotation.T + center_array
