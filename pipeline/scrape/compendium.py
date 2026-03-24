# pipeline/scrape/compendium.py
# Fetches all Tears of the Kingdom data from the Hyrule Compendium API.
# Saves one JSON file per category to pipeline/data/raw/compendium/.

import json
import time
from pathlib import Path

import requests

from pipeline.config import RAW_DIR, get_logger

API_BASE = "https://botw-compendium.herokuapp.com/api/v3"
CATEGORIES = ("creatures", "equipment", "materials", "monsters", "treasure")
OUT_DIR = RAW_DIR / "compendium"
RATE_DELAY = 1.5  # seconds between requests
MAX_RETRIES = 3
_USER_AGENT = "ZeldaTotKGuide/1.0 (personal iOS app; contact via GitHub)"


def fetch_category(session: requests.Session, category: str) -> list[dict]:
    """Fetch all TotK entries for a category with retry and exponential backoff."""
    logger = get_logger(__name__)
    url = f"{API_BASE}/compendium/category/{category}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params={"game": "totk"}, timeout=30)
            response.raise_for_status()
            return response.json().get("data", [])
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            logger.warning(
                "Attempt %d/%d failed for %s: %s. Retrying in %ds",
                attempt, MAX_RETRIES, category, exc, wait,
            )
            time.sleep(wait)

    return []  # unreachable


def run(out_dir: Path = OUT_DIR) -> None:
    """Fetch all TotK compendium categories and write one JSON file each."""
    logger = get_logger(__name__)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = _USER_AGENT

    for i, category in enumerate(CATEGORIES):
        if i > 0:
            time.sleep(RATE_DELAY)
        logger.info("Fetching category: %s", category)
        entries = fetch_category(session, category)
        out_path = out_dir / f"{category}.json"
        out_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("  -> %d entries saved to %s", len(entries), out_path)


if __name__ == "__main__":
    run()
