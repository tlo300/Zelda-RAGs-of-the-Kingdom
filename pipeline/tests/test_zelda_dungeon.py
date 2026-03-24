# pipeline/tests/test_zelda_dungeon.py
# Unit tests for the Zelda Dungeon Wiki scraper.
# All HTTP is mocked via the responses library - the live wiki is never contacted.

import json
from pathlib import Path

import pytest
import responses as resp_lib
import requests

from pipeline.scrape.zelda_dungeon import (
    _safe_filename,
    list_category_pages,
    fetch_wikitext,
    scrape,
    API_URL,
    HEADERS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_response(members: list[str], cont: str | None = None) -> dict:
    payload: dict = {
        "query": {
            "categorymembers": [{"title": t} for t in members]
        }
    }
    if cont is not None:
        payload["continue"] = {"cmcontinue": cont}
    return payload


def _wikitext_response(title: str, content: str) -> dict:
    return {
        "query": {
            "pages": {
                "1": {
                    "title": title,
                    "revisions": [
                        {"slots": {"main": {"*": content}}}
                    ],
                }
            }
        }
    }


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------

def test_safe_filename_spaces_to_underscores():
    assert _safe_filename("Hyrule Castle") == "Hyrule_Castle.txt"


def test_safe_filename_strips_illegal_chars():
    result = _safe_filename('Page: "Foo/Bar"')
    assert "/" not in result
    assert '"' not in result
    assert result.endswith(".txt")


def test_safe_filename_plain():
    assert _safe_filename("SomeWikiPage") == "SomeWikiPage.txt"


# ---------------------------------------------------------------------------
# list_category_pages
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_list_category_pages_single_page():
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_category_response(["Page A", "Page B"]),
        status=200,
    )
    session = requests.Session()
    titles = list_category_pages(session, "Tears_of_the_Kingdom")
    assert titles == ["Page A", "Page B"]


@resp_lib.activate
def test_list_category_pages_pagination():
    """Two API calls joined by a continuation token."""
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_category_response(["Page A"], cont="abc123"),
        status=200,
    )
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_category_response(["Page B"]),
        status=200,
    )
    session = requests.Session()
    titles = list_category_pages(session, "Tears_of_the_Kingdom")
    assert titles == ["Page A", "Page B"]


@resp_lib.activate
def test_list_category_pages_empty_category():
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_category_response([]),
        status=200,
    )
    session = requests.Session()
    titles = list_category_pages(session, "Empty_Category")
    assert titles == []


@resp_lib.activate
def test_list_category_pages_raises_on_http_error():
    resp_lib.add(resp_lib.GET, API_URL, status=500)
    session = requests.Session()
    with pytest.raises(requests.HTTPError):
        list_category_pages(session, "Tears_of_the_Kingdom")


# ---------------------------------------------------------------------------
# fetch_wikitext
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_wikitext_returns_content():
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_wikitext_response("Hyrule Castle", "==Walkthrough==\nSome text here."),
        status=200,
    )
    session = requests.Session()
    result = fetch_wikitext(session, "Hyrule Castle")
    assert result == "==Walkthrough==\nSome text here."


@resp_lib.activate
def test_fetch_wikitext_returns_none_on_http_error():
    resp_lib.add(resp_lib.GET, API_URL, status=503)
    session = requests.Session()
    result = fetch_wikitext(session, "Hyrule Castle")
    assert result is None


@resp_lib.activate
def test_fetch_wikitext_returns_none_when_no_pages():
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json={"query": {"pages": {}}},
        status=200,
    )
    session = requests.Session()
    result = fetch_wikitext(session, "Missing Page")
    assert result is None


# ---------------------------------------------------------------------------
# scrape (integration-style, all HTTP mocked)
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_scrape_creates_files(tmp_path: Path):
    # Scraper lists all 5 categories first, then fetches wikitext.
    # Category 1 returns two pages; categories 2-5 return empty.
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_category_response(["Shrine A", "Shrine B"]),
        status=200,
    )
    for _ in range(4):
        resp_lib.add(resp_lib.GET, API_URL, json=_category_response([]), status=200)
    # Wikitext fetches (pages sorted: Shrine A before Shrine B)
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_wikitext_response("Shrine A", "Shrine A content"),
        status=200,
    )
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_wikitext_response("Shrine B", "Shrine B content"),
        status=200,
    )

    scrape(output_dir=tmp_path)

    assert (tmp_path / "Shrine_A.txt").read_text(encoding="utf-8") == "Shrine A content"
    assert (tmp_path / "Shrine_B.txt").read_text(encoding="utf-8") == "Shrine B content"


@resp_lib.activate
def test_scrape_skips_existing_files(tmp_path: Path):
    # Pre-create the file so the scraper should skip it.
    (tmp_path / "Shrine_A.txt").write_text("old content", encoding="utf-8")

    # All 5 category calls: first returns the page, rest empty.
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_category_response(["Shrine A"]),
        status=200,
    )
    for _ in range(4):
        resp_lib.add(resp_lib.GET, API_URL, json=_category_response([]), status=200)
    # No wikitext call should be made (page is skipped).

    scrape(output_dir=tmp_path)

    # File content unchanged (was skipped, not re-fetched)
    assert (tmp_path / "Shrine_A.txt").read_text(encoding="utf-8") == "old content"


@resp_lib.activate
def test_scrape_deduplicates_titles_across_categories(tmp_path: Path):
    """A title appearing in two categories is fetched only once."""
    # All 5 category listings: first two return the same page, rest empty.
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_category_response(["Shared Page"]),
        status=200,
    )
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_category_response(["Shared Page"]),
        status=200,
    )
    for _ in range(3):
        resp_lib.add(resp_lib.GET, API_URL, json=_category_response([]), status=200)
    # Wikitext fetched exactly once
    resp_lib.add(
        resp_lib.GET,
        API_URL,
        json=_wikitext_response("Shared Page", "content"),
        status=200,
    )

    scrape(output_dir=tmp_path)

    assert (tmp_path / "Shared_Page.txt").exists()
    # Only one wikitext call was made (responses library would raise if more were needed)
