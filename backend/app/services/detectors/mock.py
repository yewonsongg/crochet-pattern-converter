from collections.abc import Sequence

from .base import RenderedPage, RotatedBoundingBox, SymbolDetection, SymbolDetector


class MockSymbolDetector(SymbolDetector):
    """Fixture-only detector. It does not inspect the page or uploaded PDF."""

    def detect(self, page: RenderedPage) -> Sequence[SymbolDetection]:
        return (
            SymbolDetection(
                symbol_id="symbol-1",
                class_name="ch",
                confidence=0.96,
                bounding_box=RotatedBoundingBox(0.42, 0.31, 0.05, 0.08, -4.5),
                page_number=page.page_number,
            ),
            SymbolDetection(
                symbol_id="symbol-2",
                class_name="dc",
                confidence=0.93,
                bounding_box=RotatedBoundingBox(0.51, 0.38, 0.04, 0.12, 2.0),
                page_number=page.page_number,
            ),
            SymbolDetection(
                symbol_id="symbol-3",
                class_name="sc",
                confidence=0.91,
                bounding_box=RotatedBoundingBox(0.59, 0.46, 0.04, 0.04, 0.0),
                page_number=page.page_number,
            ),
        )
