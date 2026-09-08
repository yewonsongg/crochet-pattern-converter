from pathlib import Path

from app.schemas.conversion import (
    ConversionResponse,
    DetectedSymbol,
    OrientedBoundingBox,
    PatternInstruction,
    PdfPageResult,
)
from app.services.detectors import RenderedPage, SymbolDetector


class ConversionService:
    def __init__(self, detector: SymbolDetector) -> None:
        self._detector = detector

    def convert(self, pdf_path: Path, original_filename: str) -> ConversionResponse:
        # PDF rendering and chart interpretation deliberately belong here later.
        # The mock detector ignores image_path and returns a stable fixture instead.
        del pdf_path, original_filename
        page = RenderedPage(page_number=1, width=612, height=792, image_path=None)
        detections = list(self._detector.detect(page))
        symbols = [
            DetectedSymbol(
                id=detection.symbol_id,
                label=detection.class_name,
                confidence=detection.confidence,
                pageNumber=detection.page_number,
                boundingBox=OrientedBoundingBox(
                    centerX=detection.bounding_box.center_x,
                    centerY=detection.bounding_box.center_y,
                    width=detection.bounding_box.width,
                    height=detection.bounding_box.height,
                    rotation=detection.bounding_box.rotation,
                ),
            )
            for detection in detections
        ]
        average_confidence = round(
            sum(symbol.confidence for symbol in symbols) / len(symbols), 4
        ) if symbols else None

        return ConversionResponse(
            status="completed",
            mock=True,
            title="Simple Scalloped Coaster",
            instructions=_mock_instructions(),
            pages=[
                PdfPageResult(
                    pageNumber=page.page_number,
                    width=page.width,
                    height=page.height,
                    detectedSymbols=symbols,
                )
            ],
            averageConfidence=average_confidence,
            message=(
                "Mock conversion only: the uploaded PDF was validated but was not "
                "rendered, analyzed, or interpreted by a model."
            ),
        )


def _mock_instructions() -> list[PatternInstruction]:
    rows = (
        "Make a magic ring. Ch 3 (counts as dc), work 11 dc into ring. Join with sl st to top of ch-3. (12 sts)",
        "Ch 3, dc in same st, 2 dc in each st around. Join with sl st. (24 sts)",
        "Ch 3, 2 dc in next st, *dc in next st, 2 dc in next st; repeat from * around. Join with sl st. (36 sts)",
        "Ch 1, sc in same st, ch 3, skip 2 sts, *sc in next st, ch 3, skip 2 sts; repeat from * around. Join with sl st. (12 loops)",
        "Sl st into first ch-3 space. Ch 3, work 4 dc in same space, sc in next space, *5 dc in next space, sc in next space; repeat from * around. Join and fasten off.",
    )
    confidences = (0.96, 0.95, 0.93, 0.92, 0.91)
    return [
        PatternInstruction(id=f"row-{row}", row=row, text=text, confidence=confidences[row - 1])
        for row, text in enumerate(rows, start=1)
    ]
