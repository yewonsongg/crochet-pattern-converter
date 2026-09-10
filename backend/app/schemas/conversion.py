from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "crochet-pattern-converter"


class OrientedBoundingBox(BaseModel):
    centerX: float = Field(ge=0, le=1)
    centerY: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    rotation: float


class DetectedSymbol(BaseModel):
    id: str
    label: str
    confidence: float = Field(ge=0, le=1)
    boundingBox: OrientedBoundingBox
    pageNumber: int = Field(ge=1)


class PdfPageResult(BaseModel):
    pageNumber: int = Field(ge=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    detectedSymbols: list[DetectedSymbol]


class PatternInstruction(BaseModel):
    id: str
    row: int = Field(ge=1)
    text: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class ConversionResponse(BaseModel):
    status: Literal["completed", "failed"]
    mock: bool
    title: str
    instructions: list[PatternInstruction]
    pages: list[PdfPageResult]
    averageConfidence: float | None = Field(default=None, ge=0, le=1)
    message: str | None = None
