# pipeline/tests/test_compendium.py
# Unit tests for the Hyrule Compendium API scraper.
# All HTTP is mocked with the responses library - no live API calls.

import json
from unittest.mock import patch

import pytest
import requests
import responses as resp

from pipeline.scrape.compendium import API_BASE, CATEGORIES, fetch_category, run


def _payload(entries):
    return {"data": entries}


@resp.activate
def test_fetch_category_returns_entries():
    resp.add(
        resp.GET,
        f"{API_BASE}/compendium/category/materials",
        json=_payload([{"id": 1, "name": "apple"}]),
    )
    result = fetch_category(requests.Session(), "materials")
    assert result == [{"id": 1, "name": "apple"}]


@resp.activate
def test_fetch_category_sends_totk_game_param():
    resp.add(
        resp.GET,
        f"{API_BASE}/compendium/category/monsters",
        json=_payload([{"id": 2}]),
    )
    fetch_category(requests.Session(), "monsters")
    assert "game=totk" in resp.calls[0].request.url


@resp.activate
def test_fetch_category_retries_on_server_error():
    resp.add(resp.GET, f"{API_BASE}/compendium/category/equipment", status=500)
    resp.add(resp.GET, f"{API_BASE}/compendium/category/equipment", json=_payload([{"id": 3}]))
    with patch("time.sleep"):
        result = fetch_category(requests.Session(), "equipment")
    assert result == [{"id": 3}]
    assert len(resp.calls) == 2


@resp.activate
def test_fetch_category_raises_after_max_retries():
    for _ in range(3):
        resp.add(resp.GET, f"{API_BASE}/compendium/category/treasure", status=503)
    with patch("time.sleep"):
        with pytest.raises(requests.HTTPError):
            fetch_category(requests.Session(), "treasure")


@resp.activate
def test_run_saves_one_file_per_category(tmp_path):
    for category in CATEGORIES:
        resp.add(
            resp.GET,
            f"{API_BASE}/compendium/category/{category}",
            json=_payload([{"id": idx, "name": f"{category}_{idx}"} for idx in range(3)]),
        )
    with patch("time.sleep"):
        run(out_dir=tmp_path)
    for category in CATEGORIES:
        out_file = tmp_path / f"{category}.json"
        assert out_file.exists(), f"Missing output file for {category}"
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert len(data) == 3


@resp.activate
def test_run_logs_fetch_and_count(tmp_path, caplog):
    import logging

    for category in CATEGORIES:
        resp.add(
            resp.GET,
            f"{API_BASE}/compendium/category/{category}",
            json=_payload([{"id": 1}]),
        )
    with patch("time.sleep"):
        with caplog.at_level(logging.INFO):
            run(out_dir=tmp_path)

    messages = [r.message for r in caplog.records]
    assert any("Fetching category" in m for m in messages)
    assert any("entries saved" in m for m in messages)
