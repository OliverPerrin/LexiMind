"""FastAPI application entrypoint."""
from fastapi import FastAPI

from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="LexiMind")
    app.include_router(router)
    return app
