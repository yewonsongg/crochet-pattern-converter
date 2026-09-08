from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
DEFAULT_FRONTEND_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


@dataclass(frozen=True, slots=True)
class Settings:
    max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES
    frontend_origins: tuple[str, ...] = DEFAULT_FRONTEND_ORIGINS

    @classmethod
    def from_env(cls) -> "Settings":
        max_upload_size = _positive_int(
            os.getenv("MAX_UPLOAD_SIZE_BYTES"),
            default=DEFAULT_MAX_UPLOAD_SIZE_BYTES,
        )
        origins = tuple(
            origin.strip().rstrip("/")
            for origin in os.getenv("FRONTEND_ORIGINS", ",".join(DEFAULT_FRONTEND_ORIGINS)).split(",")
            if origin.strip()
        )
        return cls(
            max_upload_size_bytes=max_upload_size,
            frontend_origins=origins or DEFAULT_FRONTEND_ORIGINS,
        )


def _positive_int(value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("MAX_UPLOAD_SIZE_BYTES must be an integer.") from exc
    if parsed <= 0:
        raise ValueError("MAX_UPLOAD_SIZE_BYTES must be greater than zero.")
    return parsed
