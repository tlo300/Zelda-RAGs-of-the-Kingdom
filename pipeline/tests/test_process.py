# pipeline/tests/test_process.py
# Unit tests for the text cleaning and chunking pipeline.
# No live HTTP or filesystem I/O (uses tmp_path fixtures for run.py tests).

import json
from pathlib import Path

import pytest

from pipeline.process.clean import clean_wiki_page, format_compendium_entry
from pipeline.process.chunk import chunk_text, Chunk, CHUNK_SIZE, CHUNK_OVERLAP, MIN_TOKENS


# ---------------------------------------------------------------------------
# clean_wiki_page — markup removal
# ---------------------------------------------------------------------------

SIMPLE_WIKITEXT = """\
== Overview ==
[[Link]] is the protagonist of {{TotK}}.
He uses a [[sword]] to fight enemies.

== References ==
<ref>Some citation</ref>
"""

def test_clean_wiki_removes_links():
    result = clean_wiki_page(SIMPLE_WIKITEXT)
    assert "[[" not in result
    assert "]]" not in result


def test_clean_wiki_removes_templates():
    result = clean_wiki_page(SIMPLE_WIKITEXT)
    assert "{{" not in result
    assert "}}" not in result


def test_clean_wiki_removes_refs():
    result = clean_wiki_page(SIMPLE_WIKITEXT)
    assert "<ref>" not in result
    assert "Some citation" not in result


def test_clean_wiki_drops_references_section():
    result = clean_wiki_page(SIMPLE_WIKITEXT)
    assert "References" not in result


def test_clean_wiki_keeps_prose():
    result = clean_wiki_page(SIMPLE_WIKITEXT)
    assert "Link" in result
    assert "protagonist" in result


def test_clean_wiki_infobox_stripped():
    wikitext = """\
{{Infobox Quest
|game= TotK
|giver= [[Josha]]
}}
Link must find the temple.
"""
    result = clean_wiki_page(wikitext)
    assert "{{" not in result
    assert "Josha" not in result  # infobox content stripped
    assert "Link must find the temple" in result


def test_clean_wiki_empty_page_returns_empty():
    result = clean_wiki_page("{{Stub|TotK}}")
    # After removing stub template there's nothing left.
    assert result == ""


def test_clean_wiki_no_excess_newlines():
    wikitext = "Line one.\n\n\n\n\nLine two."
    result = clean_wiki_page(wikitext)
    assert "\n\n\n" not in result


# ---------------------------------------------------------------------------
# format_compendium_entry — prose formatting
# ---------------------------------------------------------------------------

MONSTER_ENTRY = {
    "id": 101,
    "name": "water octorok",
    "category": "monsters",
    "description": "They burst out when they sense someone.",
    "common_locations": ["Lanayru Great Spring", "West Necluda"],
    "drops": ["octorok tentacle", "octo balloon"],
    "dlc": False,
}

EQUIPMENT_ENTRY = {
    "id": 201,
    "name": "boko shield",
    "category": "equipment",
    "description": "A crude shield made by Bokoblins.",
    "common_locations": [],
    "drops": [],
    "attack": 0,
    "defense": 3,
}


def test_format_compendium_contains_name():
    result = format_compendium_entry(MONSTER_ENTRY, "monsters")
    assert "Water Octorok" in result


def test_format_compendium_contains_description():
    result = format_compendium_entry(MONSTER_ENTRY, "monsters")
    assert "They burst out when they sense someone" in result


def test_format_compendium_contains_locations():
    result = format_compendium_entry(MONSTER_ENTRY, "monsters")
    assert "Lanayru Great Spring" in result
    assert "West Necluda" in result


def test_format_compendium_contains_drops():
    result = format_compendium_entry(MONSTER_ENTRY, "monsters")
    assert "octorok tentacle" in result


def test_format_compendium_no_raw_json():
    result = format_compendium_entry(MONSTER_ENTRY, "monsters")
    assert "{" not in result
    assert "}" not in result
    assert '"' not in result


def test_format_compendium_equipment_stats():
    result = format_compendium_entry(EQUIPMENT_ENTRY, "equipment")
    assert "Defense power: 3" in result


def test_format_compendium_empty_locations_no_crash():
    entry = {**MONSTER_ENTRY, "common_locations": None}
    result = format_compendium_entry(entry, "monsters")
    assert "Water Octorok" in result


# ---------------------------------------------------------------------------
# chunk_text — chunking and overlap
# ---------------------------------------------------------------------------

def _make_text(n_words: int, word: str = "word") -> str:
    return " ".join([word] * n_words)


def test_chunk_stub_returns_empty():
    text = _make_text(MIN_TOKENS - 1)
    assert chunk_text(text, "zelda_wiki", "Short Page") == []


def test_chunk_exactly_min_tokens_produces_one_chunk():
    text = _make_text(MIN_TOKENS)
    chunks = chunk_text(text, "zelda_wiki", "Page")
    assert len(chunks) == 1


def test_chunk_metadata_fields():
    text = _make_text(600)
    chunks = chunk_text(text, "zelda_wiki", "Test Page")
    assert chunks[0].source == "zelda_wiki"
    assert chunks[0].page_title == "Test Page"
    assert chunks[0].chunk_index == 0


def test_chunk_indices_sequential():
    text = _make_text(1200)
    chunks = chunk_text(text, "zelda_wiki", "Page")
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_size_within_bounds():
    # First chunk is always exactly CHUNK_SIZE (or less for the tail).
    text = _make_text(CHUNK_SIZE * 3)
    chunks = chunk_text(text, "zelda_wiki", "Page")
    # All but last should have exactly CHUNK_SIZE words.
    for chunk in chunks[:-1]:
        assert len(chunk.text.split()) == CHUNK_SIZE


def test_chunk_overlap_is_correct():
    """The last CHUNK_OVERLAP words of chunk N must equal the first CHUNK_OVERLAP words of chunk N+1."""
    text = _make_text(CHUNK_SIZE + CHUNK_OVERLAP + 10, word="w")
    # Use distinct words so we can detect overlap precisely.
    words = [f"w{i}" for i in range(CHUNK_SIZE + CHUNK_OVERLAP + 10)]
    text = " ".join(words)
    chunks = chunk_text(text, "zelda_wiki", "Page")
    assert len(chunks) >= 2

    tail = chunks[0].text.split()[-CHUNK_OVERLAP:]
    head = chunks[1].text.split()[:CHUNK_OVERLAP]
    assert tail == head


def test_chunk_to_dict_keys():
    chunks = chunk_text(_make_text(200), "zelda_wiki", "Page")
    d = chunks[0].to_dict()
    assert set(d.keys()) == {"source", "page_title", "chunk_index", "text"}


# ---------------------------------------------------------------------------
# run.py — integration (uses tmp_path, no live I/O)
# ---------------------------------------------------------------------------

def test_run_produces_jsonl(tmp_path: Path):
    from pipeline.process.run import run

    # Set up minimal raw dirs.
    wiki_dir = tmp_path / "zelda_wiki"
    wiki_dir.mkdir()
    # Write a page that is long enough to survive stub filtering.
    long_text = "{{Infobox}}\n" + " ".join(["Word"] * 600) + "\n"
    (wiki_dir / "Test_Page.txt").write_text(long_text, encoding="utf-8")

    comp_dir = tmp_path / "compendium"
    comp_dir.mkdir()
    entry = {"id": 1, "name": "bokoblin", "description": "A basic enemy.", "common_locations": [], "drops": []}
    (comp_dir / "monsters.json").write_text(json.dumps([entry]), encoding="utf-8")

    output = tmp_path / "chunks.jsonl"

    # Monkeypatch the module-level constants so run() uses tmp_path.
    import pipeline.process.run as run_mod
    orig_wiki = run_mod._WIKI_SOURCES
    orig_comp = run_mod._COMPENDIUM_DIR

    run_mod._WIKI_SOURCES = {"zelda_wiki": wiki_dir}
    run_mod._COMPENDIUM_DIR = comp_dir

    try:
        count = run(output_path=output)
    finally:
        run_mod._WIKI_SOURCES = orig_wiki
        run_mod._COMPENDIUM_DIR = orig_comp

    assert output.exists()
    assert count > 0
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == count
    first = json.loads(lines[0])
    assert set(first.keys()) == {"source", "page_title", "chunk_index", "text"}


def test_run_skips_stub_pages(tmp_path: Path):
    from pipeline.process.run import run

    wiki_dir = tmp_path / "zelda_wiki"
    wiki_dir.mkdir()
    (wiki_dir / "Stub.txt").write_text("Short text.", encoding="utf-8")

    output = tmp_path / "chunks.jsonl"

    import pipeline.process.run as run_mod
    orig_wiki = run_mod._WIKI_SOURCES
    orig_comp = run_mod._COMPENDIUM_DIR

    run_mod._WIKI_SOURCES = {"zelda_wiki": wiki_dir}
    run_mod._COMPENDIUM_DIR = tmp_path / "no_compendium"  # non-existent

    try:
        count = run(output_path=output)
    finally:
        run_mod._WIKI_SOURCES = orig_wiki
        run_mod._COMPENDIUM_DIR = orig_comp

    assert count == 0
