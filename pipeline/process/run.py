# pipeline/process/run.py
# Orchestration script: read all raw sources, clean, chunk, write chunks.jsonl.
#
# Usage:
#   python -m pipeline.process.run
#
# Output: pipeline/data/processed/chunks.jsonl
# Each line is a JSON object: {source, page_title, chunk_index, text}

import json
from pathlib import Path

from pipeline.config import RAW_DIR, PROCESSED_DIR, get_logger
from pipeline.process.clean import clean_wiki_page, format_compendium_entry
from pipeline.process.chunk import chunk_text, MIN_TOKENS

log = get_logger(__name__)

_WIKI_SOURCES = {
    "zelda_wiki": RAW_DIR / "zelda_wiki",
    "zelda_dungeon": RAW_DIR / "zelda_dungeon",
}

_COMPENDIUM_DIR = RAW_DIR / "compendium"
_OUTPUT_PATH = PROCESSED_DIR / "chunks.jsonl"


def _process_wiki_dir(source: str, raw_dir: Path) -> list[dict]:
    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        log.warning("No .txt files found in %s", raw_dir)
        return []

    records: list[dict] = []
    for path in txt_files:
        page_title = path.stem.replace("_", " ")
        wikitext = path.read_text(encoding="utf-8")
        plain = clean_wiki_page(wikitext)

        chunks = chunk_text(plain, source=source, page_title=page_title)
        if not chunks:
            log.debug("Skipped stub: %s (%d tokens)", page_title, len(plain.split()))
        else:
            records.extend(c.to_dict() for c in chunks)

    log.info("%s: %d pages → %d chunks", source, len(txt_files), len(records))
    return records


def _process_compendium_dir(comp_dir: Path) -> list[dict]:
    json_files = sorted(comp_dir.glob("*.json"))
    if not json_files:
        log.warning("No .json files found in %s", comp_dir)
        return []

    records: list[dict] = []
    for path in json_files:
        category = path.stem  # e.g. "monsters"
        source = f"compendium/{category}"
        entries: list[dict] = json.loads(path.read_text(encoding="utf-8"))

        for entry in entries:
            plain = format_compendium_entry(entry, source_category=category)
            page_title = entry.get("name", "unknown").replace("_", " ")
            chunks = chunk_text(
                plain,
                source=source,
                page_title=page_title,
                min_tokens=1,  # compendium entries are intentionally short
            )
            records.extend(c.to_dict() for c in chunks)

        log.info("compendium/%s: %d entries → %d chunks", category, len(entries), len(records))

    return records


def run(output_path: Path = _OUTPUT_PATH) -> int:
    """Clean and chunk all raw sources. Returns total chunk count."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []

    for source, raw_dir in _WIKI_SOURCES.items():
        if raw_dir.exists():
            all_records.extend(_process_wiki_dir(source, raw_dir))
        else:
            log.warning("Raw directory not found, skipping: %s", raw_dir)

    if _COMPENDIUM_DIR.exists():
        all_records.extend(_process_compendium_dir(_COMPENDIUM_DIR))
    else:
        log.warning("Compendium directory not found, skipping: %s", _COMPENDIUM_DIR)

    with output_path.open("w", encoding="utf-8") as fh:
        for record in all_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info("Wrote %d chunks to %s", len(all_records), output_path)
    return len(all_records)


if __name__ == "__main__":
    run()
