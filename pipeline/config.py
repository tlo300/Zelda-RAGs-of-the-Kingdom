# config.py
# Shared path constants for the data pipeline. All paths are pathlib.Path objects.

from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
DATA_DIR = PIPELINE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "knowledge_base.db"
