"""Logging setup."""
import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging. Call once during application setup."""

    logging.basicConfig(level=level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
