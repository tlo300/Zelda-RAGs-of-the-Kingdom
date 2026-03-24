# pipeline/scrape/zelda_dungeon.py
# Fetches TotK walkthrough content from Zelda Dungeon Wiki via the MediaWiki API.
# Saves one .txt file of raw wikitext per page to pipeline/data/raw/zelda_dungeon/.
# Resume-safe: skips pages that already have a saved file.

import re
import time
from pathlib import Path

import requests

from pipeline.config import RAW_DIR, get_logger

API_URL = "https://www.zeldadungeon.net/wiki/api.php"
OUTPUT_DIR = RAW_DIR / "zelda_dungeon"
HEADERS = {"User-Agent": "ZeldaTotKGuide/1.0 (personal iOS app; contact via GitHub)"}
DELAY = 1.5  # seconds between requests

# TotK categories to crawl. Extend this list to add more content areas.
TOTK_CATEGORIES = [
    "Tears_of_the_Kingdom_Walkthroughs",
    "Tears_of_the_Kingdom_Shrines",
    "Tears_of_the_Kingdom_Side_Quests",
    "Tears_of_the_Kingdom_Locations",
    "Tears_of_the_Kingdom",
]

logger = get_logger(__name__)


def _safe_filename(title: str) -> str:
    """Convert a wiki page title to a safe filename."""
    safe = re.sub(r'[\\/*?:"<>|]', "_", title)
    safe = safe.replace(" ", "_")
    return safe + ".txt"


def list_category_pages(
    session: requests.Session, category: str
) -> list[str]:
    """Return all page titles in a MediaWiki category, following continuation tokens."""
    titles: list[str] = []
    params: dict = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": "page",
        "cmlimit": "500",
        "format": "json",
    }
    while True:
        response = session.get(API_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        time.sleep(DELAY)
        cont = data.get("continue", {}).get("cmcontinue")
        if cont is None:
            break
        params["cmcontinue"] = cont
    return titles


def fetch_wikitext(session: requests.Session, title: str) -> str | None:
    """Fetch raw wikitext for a single page. Returns None on error."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "format": "json",
    }
    try:
        response = session.get(API_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.error("HTTP error fetching %r: %s", title, exc)
        return None
    time.sleep(DELAY)
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        slots = page.get("revisions", [{}])[0].get("slots", {})
        return slots.get("main", {}).get("*")
    return None


def scrape(output_dir: Path = OUTPUT_DIR) -> None:
    """Scrape all TotK pages from Zelda Dungeon Wiki and save wikitext to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    # Collect unique page titles across all categories.
    all_titles: set[str] = set()
    for category in TOTK_CATEGORIES:
        logger.info("Listing category: %s", category)
        try:
            titles = list_category_pages(session, category)
        except requests.RequestException as exc:
            logger.error("Failed to list category %r: %s", category, exc)
            continue
        logger.info("  Found %d pages in %s", len(titles), category)
        all_titles.update(titles)

    logger.info("Total unique pages to fetch: %d", len(all_titles))
    fetched = skipped = errors = 0

    for title in sorted(all_titles):
        dest = output_dir / _safe_filename(title)
        if dest.exists():
            skipped += 1
            continue
        wikitext = fetch_wikitext(session, title)
        if wikitext is None:
            errors += 1
            continue
        dest.write_text(wikitext, encoding="utf-8")
        fetched += 1
        logger.info("[%d fetched / %d skipped / %d errors] %s", fetched, skipped, errors, title)

    logger.info(
        "Done. fetched=%d skipped=%d errors=%d total=%d",
        fetched,
        skipped,
        errors,
        len(all_titles),
    )


if __name__ == "__main__":
    scrape()
