# pipeline/tests/test_zelda_wiki.py
# Unit tests for the Zelda Wiki scraper.
# All HTTP is mocked via the responses library - the live wiki is never contacted.

from pathlib import Path

import pytest
import responses as resp_lib
import requests

from pipeline.scrape.zelda_wiki import (
    _safe_filename,
    list_category_pages,
    fetch_wikitext,
    scrape,
    API_URL,
    TOTK_CATEGORIES,
    _SKIP_PREFIXES,
    _COOKING_PREFIXES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_response(members: list[str], cont: str | None = None) -> dict:
    payload: dict = {
        "query": {"categorymembers": [{"title": t} for t in members]}
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
                    "revisions": [{"slots": {"main": {"*": content}}}],
                }
            }
        }
    }


def _empty_categories(n: int) -> None:
    """Register n empty category responses (for leftover category calls)."""
    for _ in range(n):
        resp_lib.add(resp_lib.GET, API_URL, json=_category_response([]), status=200)


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


# ---------------------------------------------------------------------------
# list_category_pages
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_list_category_pages_returns_titles():
    resp_lib.add(
        resp_lib.GET, API_URL,
        json=_category_response(["Characters in Tears of the Kingdom", "Enemies in Tears of the Kingdom"]),
        status=200,
    )
    session = requests.Session()
    titles = list_category_pages(session, "Tears_of_the_Kingdom")
    assert "Characters in Tears of the Kingdom" in titles
    assert "Enemies in Tears of the Kingdom" in titles


@resp_lib.activate
def test_list_category_pages_follows_continuation():
    resp_lib.add(resp_lib.GET, API_URL, json=_category_response(["Page A"], cont="tok1"), status=200)
    resp_lib.add(resp_lib.GET, API_URL, json=_category_response(["Page B"]), status=200)
    session = requests.Session()
    assert list_category_pages(session, "Tears_of_the_Kingdom") == ["Page A", "Page B"]


@resp_lib.activate
def test_list_category_pages_empty():
    resp_lib.add(resp_lib.GET, API_URL, json=_category_response([]), status=200)
    assert list_category_pages(requests.Session(), "Empty") == []


@resp_lib.activate
def test_list_category_pages_raises_on_http_error():
    resp_lib.add(resp_lib.GET, API_URL, status=403)
    with pytest.raises(requests.HTTPError):
        list_category_pages(requests.Session(), "Tears_of_the_Kingdom")


# ---------------------------------------------------------------------------
# fetch_wikitext
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_fetch_wikitext_returns_content():
    resp_lib.add(
        resp_lib.GET, API_URL,
        json=_wikitext_response("Link", "==Link==\nProtagonist of the series."),
        status=200,
    )
    result = fetch_wikitext(requests.Session(), "Link")
    assert result == "==Link==\nProtagonist of the series."


@resp_lib.activate
def test_fetch_wikitext_returns_none_on_http_error():
    resp_lib.add(resp_lib.GET, API_URL, status=500)
    assert fetch_wikitext(requests.Session(), "Link") is None


@resp_lib.activate
def test_fetch_wikitext_returns_none_when_no_pages():
    resp_lib.add(resp_lib.GET, API_URL, json={"query": {"pages": {}}}, status=200)
    assert fetch_wikitext(requests.Session(), "Missing") is None


# ---------------------------------------------------------------------------
# scrape
# ---------------------------------------------------------------------------

@resp_lib.activate
def test_scrape_creates_files(tmp_path: Path):
    # Category listings: first returns two pages, rest empty
    resp_lib.add(resp_lib.GET, API_URL, json=_category_response(["Link", "Zelda"]), status=200)
    _empty_categories(len(TOTK_CATEGORIES) - 1)
    # Wikitext fetches (sorted: Link before Zelda)
    resp_lib.add(resp_lib.GET, API_URL, json=_wikitext_response("Link", "Link content"), status=200)
    resp_lib.add(resp_lib.GET, API_URL, json=_wikitext_response("Zelda", "Zelda content"), status=200)

    scrape(output_dir=tmp_path)

    assert (tmp_path / "Link.txt").read_text(encoding="utf-8") == "Link content"
    assert (tmp_path / "Zelda.txt").read_text(encoding="utf-8") == "Zelda content"


@resp_lib.activate
def test_scrape_skips_existing_files(tmp_path: Path):
    (tmp_path / "Link.txt").write_text("old content", encoding="utf-8")
    resp_lib.add(resp_lib.GET, API_URL, json=_category_response(["Link"]), status=200)
    _empty_categories(len(TOTK_CATEGORIES) - 1)
    # No wikitext call expected

    scrape(output_dir=tmp_path)

    assert (tmp_path / "Link.txt").read_text(encoding="utf-8") == "old content"


@resp_lib.activate
def test_scrape_deduplicates_titles_across_categories(tmp_path: Path):
    resp_lib.add(resp_lib.GET, API_URL, json=_category_response(["Shared"]), status=200)
    resp_lib.add(resp_lib.GET, API_URL, json=_category_response(["Shared"]), status=200)
    _empty_categories(len(TOTK_CATEGORIES) - 2)
    resp_lib.add(resp_lib.GET, API_URL, json=_wikitext_response("Shared", "content"), status=200)

    scrape(output_dir=tmp_path)

    assert (tmp_path / "Shared.txt").exists()


@resp_lib.activate
def test_scrape_filters_user_pages(tmp_path: Path):
    """User: and other non-article pages are excluded and never fetched."""
    resp_lib.add(
        resp_lib.GET, API_URL,
        json=_category_response(["Link", "User:Tartine/Sandbox", "Talk:Link", "File:Link.png"]),
        status=200,
    )
    _empty_categories(len(TOTK_CATEGORIES) - 1)
    # Only "Link" should be fetched
    resp_lib.add(resp_lib.GET, API_URL, json=_wikitext_response("Link", "Link content"), status=200)

    scrape(output_dir=tmp_path)

    assert (tmp_path / "Link.txt").exists()
    assert not (tmp_path / "User_Tartine_Sandbox.txt").exists()
    assert not any(f.name.startswith("Talk_") for f in tmp_path.iterdir())
    assert not any(f.name.startswith("File_") for f in tmp_path.iterdir())


def test_skip_prefixes_covers_user_namespace():
    assert any(p == "User:" for p in _SKIP_PREFIXES)


@resp_lib.activate
def test_scrape_filters_cooking_variants(tmp_path: Path):
    """Cooking variant pages (e.g. 'Biting Meat Skewer') are excluded; base foods are kept."""
    resp_lib.add(
        resp_lib.GET, API_URL,
        json=_category_response(["Meat Skewer", "Biting Meat Skewer", "Spicy Meat Skewer", "Hasty Elixir"]),
        status=200,
    )
    _empty_categories(len(TOTK_CATEGORIES) - 1)
    # Only "Meat Skewer" should be fetched
    resp_lib.add(resp_lib.GET, API_URL, json=_wikitext_response("Meat Skewer", "content"), status=200)

    scrape(output_dir=tmp_path)

    assert (tmp_path / "Meat_Skewer.txt").exists()
    assert not (tmp_path / "Biting_Meat_Skewer.txt").exists()
    assert not (tmp_path / "Spicy_Meat_Skewer.txt").exists()
    assert not (tmp_path / "Hasty_Elixir.txt").exists()


def test_cooking_prefixes_includes_common_variants():
    for prefix in ("Biting ", "Spicy ", "Hearty ", "Hasty "):
        assert prefix in _COOKING_PREFIXES
