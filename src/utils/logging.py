"""
Logging utilities for LexiMind.

Provides centralized logging configuration and logger factory.

Author: Oliver Perrin
Date: December 2025
"""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging. Call once during application setup."""

    logging.basicConfig(level=level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
