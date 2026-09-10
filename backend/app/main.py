from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import Settings
from app.services.conversion_service import ConversionService
from app.services.detectors import MockSymbolDetector


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    application = FastAPI(
        title="Crochet Pattern Converter API",
        version="0.1.0",
        debug=False,
    )
    application.state.settings = resolved_settings
    application.state.conversion_service = ConversionService(MockSymbolDetector())

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved_settings.frontend_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )
    application.include_router(router)
    return application


app = create_app()
