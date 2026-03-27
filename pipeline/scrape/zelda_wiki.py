# pipeline/scrape/zelda_wiki.py
# Fetches TotK lore, NPC, and mechanic pages from Zelda Wiki (zeldawiki.wiki)
# via the MediaWiki API. Saves one .txt file of raw wikitext per page to
# pipeline/data/raw/zelda_wiki/. Resume-safe: skips pages already saved.

import re
import time
from pathlib import Path

import requests

from pipeline.config import RAW_DIR, get_logger

API_URL = "https://zeldawiki.wiki/w/api.php"
OUTPUT_DIR = RAW_DIR / "zelda_wiki"
HEADERS = {"User-Agent": "ZeldaTotKGuide/1.0 (personal iOS app; contact via GitHub)"}
DELAY = 1.5  # seconds between requests

# TotK sub-categories to crawl. These are the actual category names on zeldawiki.wiki.
TOTK_CATEGORIES = [
    "Characters_in_Tears_of_the_Kingdom",
    "Locations_in_Tears_of_the_Kingdom",
    "Enemies_in_Tears_of_the_Kingdom",
    "Items_in_Tears_of_the_Kingdom",
    "Main_Quests_in_Tears_of_the_Kingdom",
    "Side_Quests_in_Tears_of_the_Kingdom",
    "Shrine_Quests_in_Tears_of_the_Kingdom",
    "Mechanics_in_Tears_of_the_Kingdom",
    "Bosses_in_Tears_of_the_Kingdom",
    # Additional categories not covered by Locations — dungeons, abilities, shrines
    "Dungeons_in_Tears_of_the_Kingdom",
    "Abilities_in_Tears_of_the_Kingdom",
    "Shrines_of_Light",
]

# Namespace prefixes to skip — user sandboxes, talk pages, etc.
_SKIP_PREFIXES = ("User:", "User talk:", "Talk:", "File:", "Template:", "Category:")

# TotK cooking effect prefixes — these produce hundreds of near-identical food
# variant pages (e.g. "Biting Meat Skewer") that add noise without useful RAG content.
# The base food pages (e.g. "Meat Skewer") are kept.
_COOKING_PREFIXES = (
    "Biting ", "Chilly ", "Electro ", "Enduring ", "Energizing ",
    "Fireproof ", "Hasty ", "Hearty ", "Mighty ", "Sneaky ", "Spicy ", "Tough ",
)

logger = get_logger(__name__)


def _safe_filename(title: str) -> str:
    """Convert a wiki page title to a safe filename."""
    safe = re.sub(r'[\\/*?:"<>|]', "_", title)
    safe = safe.replace(" ", "_")
    return safe + ".txt"


def list_category_pages(session: requests.Session, category: str) -> list[str]:
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
    """Scrape all TotK pages from Zelda Wiki and save wikitext to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    all_titles: set[str] = set()
    for category in TOTK_CATEGORIES:
        logger.info("Listing category: %s", category)
        try:
            titles = list_category_pages(session, category)
        except requests.RequestException as exc:
            logger.error("Failed to list category %r: %s", category, exc)
            continue
        clean = [
            t for t in titles
            if not t.startswith(_SKIP_PREFIXES) and not t.startswith(_COOKING_PREFIXES)
        ]
        skipped_ns = len(titles) - len(clean)
        if skipped_ns:
            logger.info("  Skipped %d non-article/cooking-variant pages in %s", skipped_ns, category)
        logger.info("  Found %d pages in %s", len(clean), category)
        all_titles.update(clean)

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
