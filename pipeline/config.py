# config.py
# Shared path constants for the data pipeline. All paths are pathlib.Path objects.

import logging
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "knowledge_base.db"


def get_logger(name: str) -> logging.Logger:
    """Return a logger with timestamped console output.

    Uses basicConfig so the first call configures the root logger;
    subsequent calls from other modules are no-ops for configuration
    but still return the correct named logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
