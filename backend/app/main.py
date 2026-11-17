"""FastAPI application entry point: ``uvicorn app.main:app --reload`` from ``backend/``."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_finetune, routes_pipeline, routes_system, routes_worker
from app.config.settings import get_settings
from app.database.session import init_engine


@asynccontextmanager
async def lifespan(_: FastAPI):  # noqa: ANN201
    settings = get_settings()
    settings.ensure_dirs()
    init_engine()
    yield


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(routes_system.router)
    app.include_router(routes_pipeline.router)
    app.include_router(routes_worker.router)
    app.include_router(routes_finetune.router)
    return app


app = create_app()
