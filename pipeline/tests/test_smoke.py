# pipeline/tests/test_smoke.py
# Smoke tests: verify imports and shared utilities work correctly.

import logging
from pathlib import Path

import pipeline.scrape
import pipeline.process
import pipeline.build_db
from pipeline.config import DATA_DIR, RAW_DIR, PROCESSED_DIR, DB_PATH, get_logger


def test_data_dir_is_path():
    assert isinstance(DATA_DIR, Path)


def test_raw_dir_is_path():
    assert isinstance(RAW_DIR, Path)


def test_processed_dir_is_path():
    assert isinstance(PROCESSED_DIR, Path)


def test_db_path_is_path():
    assert isinstance(DB_PATH, Path)


def test_raw_dir_is_inside_data_dir():
    assert RAW_DIR.parent == DATA_DIR


def test_processed_dir_is_inside_data_dir():
    assert PROCESSED_DIR.parent == DATA_DIR


def test_db_path_is_inside_data_dir():
    assert DB_PATH.parent == DATA_DIR


def test_scrape_importable():
    # Passes if pipeline/scrape/__init__.py exists and is syntax-error free
    assert pipeline.scrape is not None


def test_process_importable():
    assert pipeline.process is not None


def test_build_db_importable():
    assert pipeline.build_db is not None


def test_get_logger_returns_logger():
    logger = get_logger("test.smoke")
    assert isinstance(logger, logging.Logger)


def test_get_logger_name_preserved():
    logger = get_logger("test.name")
    assert logger.name == "test.name"
