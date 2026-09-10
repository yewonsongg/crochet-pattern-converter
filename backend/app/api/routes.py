from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.conversion import ConversionResponse, HealthResponse


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
READ_CHUNK_SIZE = 1024 * 1024
PDF_SIGNATURE = b"%PDF-"


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.post("/convert", response_model=ConversionResponse)
async def convert_pattern(
    request: Request,
    file: Annotated[UploadFile | None, File(description="Crochet chart PDF")] = None,
) -> ConversionResponse:
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A PDF file is required in the 'file' form field.",
        )

    filename = file.filename or "upload.pdf"
    looks_like_pdf = (
        file.content_type == "application/pdf"
        or filename.lower().endswith(".pdf")
    )
    if not looks_like_pdf:
        await file.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    temp_path: Path | None = None
    try:
        temp_path = await _save_validated_pdf(
            file,
            max_size=request.app.state.settings.max_upload_size_bytes,
        )
        return request.app.state.conversion_service.convert(temp_path, filename)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected conversion failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The conversion service could not process the PDF. Please try again.",
        ) from None
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


async def _save_validated_pdf(file: UploadFile, *, max_size: int) -> Path:
    total_size = 0
    first_chunk = True
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="crochet-upload-",
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temp_path = Path(temporary_file.name)
            while chunk := await file.read(READ_CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > max_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"PDF exceeds the maximum upload size of {_format_bytes(max_size)}.",
                    )
                if first_chunk:
                    first_chunk = False
                    if not chunk.startswith(PDF_SIGNATURE):
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail="The uploaded file does not have a valid PDF signature.",
                        )
                temporary_file.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded PDF is empty.",
            )
        return temp_path
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def _format_bytes(size: int) -> str:
    if size % (1024 * 1024) == 0:
        return f"{size // (1024 * 1024)} MB"
    if size % 1024 == 0:
        return f"{size // 1024} KB"
    return f"{size} bytes"
