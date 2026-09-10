from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class RenderedPage:
    """A rasterized PDF page ready for a detector implementation."""

    page_number: int
    width: int
    height: int
    image_path: Path | None


@dataclass(frozen=True, slots=True)
class RotatedBoundingBox:
    """Normalized OBB coordinates, with rotation measured in degrees."""

    center_x: float
    center_y: float
    width: float
    height: float
    rotation: float


@dataclass(frozen=True, slots=True)
class SymbolDetection:
    symbol_id: str
    class_name: str
    confidence: float
    bounding_box: RotatedBoundingBox
    page_number: int


class SymbolDetector(ABC):
    """Stable boundary implemented later by MockSymbolDetector or YoloSymbolDetector."""

    @abstractmethod
    def detect(self, page: RenderedPage) -> Sequence[SymbolDetection]:
        """Return normalized symbol detections for one rendered page."""
