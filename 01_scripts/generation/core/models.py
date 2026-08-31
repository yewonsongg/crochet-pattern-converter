from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GenerationConfig:
  """Shared canvas and sampling options passed to class generators."""

  canvas_width_px: int = 25
  canvas_height_px: int = 25
  target_visible_px: float = 15.0
  rotation_deg: float = 0.0
  stroke_width_normalized: float = 4.0


@dataclass
class GeneratedObject:
  """In-memory generated object for notebooks and scene orchestration.

  A generated object may be inspected without being written to disk. Paths
  are optional because downstream orchestration may compose objects directly
  or persist them later.
  """

  class_id: int
  class_name: str
  phenotype: str
  svg: str
  metadata: dict[str, Any]
  obb_pixels: np.ndarray
  obb_normalized: np.ndarray
  yolo_label: str
  svg_path: Path | None = None
  png_path: Path | None = None
  metadata_path: Path | None = None
  label_path: Path | None = None
  extras: dict[str, Any] = field(default_factory=dict)

  @property
  def canvas_size_px(self) -> tuple[int, int]:
    canvas = self.metadata.get("canvas", {})
    return canvas.get("width_px", 0), canvas.get("height_px", 0)
