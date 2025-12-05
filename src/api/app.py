"""
FastAPI application factory for LexiMind.

Creates and configures the REST API application.

Author: Oliver Perrin
Date: December 2025
"""

from fastapi import FastAPI

from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="LexiMind")
    app.include_router(router)
    return app
