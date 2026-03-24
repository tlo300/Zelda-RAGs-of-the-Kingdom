# pipeline/tests/test_smoke.py
# Smoke tests: verify imports and shared utilities work correctly.

import logging
from pipeline.config import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test.smoke")
    assert isinstance(logger, logging.Logger)


def test_get_logger_name_preserved():
    logger = get_logger("test.name")
    assert logger.name == "test.name"
