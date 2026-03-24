# pipeline/tests/test_validate_kb.py
# Tests for knowledge base validation script.
# Uses a mock SentenceTransformer and in-memory/tmp SQLite databases — no live model or network.

import json
import math
import sqlite3
import struct
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import sqlite_vec

from pipeline.build_db.embed import EMBEDDING_DIM, run as build_db
from pipeline.validate_kb import (
    check_chunk_count,
    check_embedding_dimensions,
    check_no_bad_embeddings,
    check_no_short_chunks,
    check_sources,
    copy_to_ios,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model(bad_indices: set[int] | None = None) -> MagicMock:
    """Return a mock SentenceTransformer producing deterministic valid vectors."""
    bad_indices = bad_indices or set()
    model = MagicMock()

    def encode(text_or_texts, **kwargs):
        if isinstance(text_or_texts, str):
            return np.full(EMBEDDING_DIM, 0.5, dtype=np.float32)
        vecs = []
        for i, _ in enumerate(text_or_texts):
            vec = np.full(EMBEDDING_DIM, 0.1 * (i + 1), dtype=np.float32)
            if i in bad_indices:
                vec[:] = 0.0
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)

    model.encode.side_effect = encode
    return model


def _write_chunks(path: Path, count: int, sources: list[str] | None = None) -> Path:
    all_sources = sources or ["compendium/monsters", "zelda_wiki"]
    records = [
        {
            "source": all_sources[i % len(all_sources)],
            "page_title": f"Page {i}",
            "chunk_index": 0,
            "text": f"This is a sufficiently long chunk of text for page {i}. " * 3,
        }
        for i in range(count)
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


@pytest.fixture
def valid_db(tmp_path: Path):
    """A well-formed DB with 10 chunks from both sources."""
    chunks_path = _write_chunks(tmp_path / "chunks.jsonl", 10)
    db_path = tmp_path / "knowledge_base.db"
    build_db(chunks_path=chunks_path, db_path=db_path, model=_make_model())
    return db_path


@pytest.fixture
def valid_db_large(tmp_path: Path):
    """A well-formed DB with 3000 chunks from both sources (passes count check)."""
    chunks_path = _write_chunks(tmp_path / "chunks.jsonl", 3000)
    db_path = tmp_path / "knowledge_base.db"
    build_db(chunks_path=chunks_path, db_path=db_path, model=_make_model())
    return db_path


# ---------------------------------------------------------------------------
# check_chunk_count
# ---------------------------------------------------------------------------


def test_chunk_count_pass(valid_db_large):
    conn = _open_db(valid_db_large)
    assert check_chunk_count(conn) is True
    conn.close()


def test_chunk_count_too_few(valid_db):
    conn = _open_db(valid_db)
    assert check_chunk_count(conn) is False  # 10 < 2000
    conn.close()


def test_chunk_count_too_many(tmp_path):
    chunks_path = _write_chunks(tmp_path / "chunks.jsonl", 60000)
    db_path = tmp_path / "kb.db"
    build_db(chunks_path=chunks_path, db_path=db_path, model=_make_model())
    conn = _open_db(db_path)
    assert check_chunk_count(conn) is False
    conn.close()


# ---------------------------------------------------------------------------
# check_sources
# ---------------------------------------------------------------------------


def test_sources_all_present(valid_db):
    conn = _open_db(valid_db)
    assert check_sources(conn) is True
    conn.close()


def test_sources_missing_compendium(tmp_path):
    chunks_path = _write_chunks(tmp_path / "chunks.jsonl", 10, sources=["zelda_wiki"])
    db_path = tmp_path / "kb.db"
    build_db(chunks_path=chunks_path, db_path=db_path, model=_make_model())
    conn = _open_db(db_path)
    assert check_sources(conn) is False
    conn.close()


def test_sources_missing_zelda_wiki(tmp_path):
    chunks_path = _write_chunks(tmp_path / "chunks.jsonl", 10, sources=["compendium"])
    db_path = tmp_path / "kb.db"
    build_db(chunks_path=chunks_path, db_path=db_path, model=_make_model())
    conn = _open_db(db_path)
    assert check_sources(conn) is False
    conn.close()


# ---------------------------------------------------------------------------
# check_no_short_chunks
# ---------------------------------------------------------------------------


def test_no_short_chunks_pass(valid_db):
    conn = _open_db(valid_db)
    assert check_no_short_chunks(conn) is True
    conn.close()


def test_short_chunk_detected(tmp_path):
    records = [
        {"source": "zelda_wiki", "page_title": "P0", "chunk_index": 0, "text": "Short."},
        {"source": "compendium", "page_title": "P1", "chunk_index": 0, "text": "Also short text here."},
    ]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    db_path = tmp_path / "kb.db"
    build_db(chunks_path=chunks_path, db_path=db_path, model=_make_model())
    conn = _open_db(db_path)
    assert check_no_short_chunks(conn) is False
    conn.close()


# ---------------------------------------------------------------------------
# check_embedding_dimensions
# ---------------------------------------------------------------------------


def test_embedding_dimensions_pass(valid_db):
    conn = _open_db(valid_db)
    assert check_embedding_dimensions(conn) is True
    conn.close()



# ---------------------------------------------------------------------------
# check_no_bad_embeddings
# ---------------------------------------------------------------------------


def test_no_bad_embeddings_pass(valid_db):
    conn = _open_db(valid_db)
    assert check_no_bad_embeddings(conn) is True
    conn.close()


def test_bad_embedding_zero_vector_detected(tmp_path):
    chunks_path = _write_chunks(tmp_path / "chunks.jsonl", 200)
    db_path = tmp_path / "kb.db"
    build_db(chunks_path=chunks_path, db_path=db_path, model=_make_model(bad_indices={0}))

    # embed.py skips bad embeddings, so manually inject one
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    zero_blob = struct.pack(f"{EMBEDDING_DIM}f", *([0.0] * EMBEDDING_DIM))
    max_rowid = conn.execute("SELECT MAX(rowid) FROM chunk_embeddings").fetchone()[0]
    max_chunk_id = conn.execute("SELECT MAX(id) FROM chunks").fetchone()[0]
    conn.execute(
        "INSERT INTO chunks (id, chunk_text, source, page_title, chunk_index) VALUES (?, ?, ?, ?, ?)",
        (max_chunk_id + 1, "x" * 60, "zelda_wiki", "Bad", 0),
    )
    conn.execute(
        "INSERT INTO chunk_embeddings(rowid, embedding) VALUES (?, ?)",
        (max_rowid + 1, zero_blob),
    )
    conn.commit()
    conn.close()

    conn2 = _open_db(db_path)
    assert check_no_bad_embeddings(conn2) is False
    conn2.close()


def test_bad_embedding_nan_detected(tmp_path):
    chunks_path = _write_chunks(tmp_path / "chunks.jsonl", 200)
    db_path = tmp_path / "kb.db"
    build_db(chunks_path=chunks_path, db_path=db_path, model=_make_model())

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    nan_vec = [0.5] * EMBEDDING_DIM
    nan_vec[7] = float("nan")
    nan_blob = struct.pack(f"{EMBEDDING_DIM}f", *nan_vec)
    max_rowid = conn.execute("SELECT MAX(rowid) FROM chunk_embeddings").fetchone()[0]
    max_chunk_id = conn.execute("SELECT MAX(id) FROM chunks").fetchone()[0]
    conn.execute(
        "INSERT INTO chunks (id, chunk_text, source, page_title, chunk_index) VALUES (?, ?, ?, ?, ?)",
        (max_chunk_id + 1, "x" * 60, "zelda_wiki", "NaN page", 0),
    )
    conn.execute(
        "INSERT INTO chunk_embeddings(rowid, embedding) VALUES (?, ?)",
        (max_rowid + 1, nan_blob),
    )
    conn.commit()
    conn.close()

    conn2 = _open_db(db_path)
    assert check_no_bad_embeddings(conn2) is False
    conn2.close()


# ---------------------------------------------------------------------------
# copy_to_ios
# ---------------------------------------------------------------------------


def test_copy_to_ios_creates_file(valid_db, tmp_path):
    ios_resources = tmp_path / "ios" / "Resources"
    result = copy_to_ios(valid_db, ios_resources)
    assert result is True
    assert (ios_resources / "knowledge_base.db").exists()


def test_copy_to_ios_creates_parent_dirs(valid_db, tmp_path):
    ios_resources = tmp_path / "deep" / "nested" / "dir"
    copy_to_ios(valid_db, ios_resources)
    assert (ios_resources / "knowledge_base.db").exists()


# ---------------------------------------------------------------------------
# run() — integration
# ---------------------------------------------------------------------------


def test_run_passes_with_large_valid_db(valid_db_large, tmp_path):
    ios_resources = tmp_path / "Resources"
    model = _make_model()
    result = run(db_path=valid_db_large, ios_resources=ios_resources, model=model)
    assert result is True
    assert (ios_resources / "knowledge_base.db").exists()


def test_run_fails_when_db_missing(tmp_path):
    result = run(
        db_path=tmp_path / "nonexistent.db",
        ios_resources=tmp_path / "Resources",
        model=_make_model(),
    )
    assert result is False


def test_run_fails_when_chunk_count_too_low(valid_db, tmp_path):
    ios_resources = tmp_path / "Resources"
    model = _make_model()
    result = run(db_path=valid_db, ios_resources=ios_resources, model=model)
    assert result is False  # 10 chunks < 2000


def test_run_exits_nonzero_on_failure(valid_db, tmp_path, monkeypatch):
    """run() returns False; __main__ block exits 1."""
    import pipeline.validate_kb as vkb

    ios_resources = tmp_path / "Resources"
    model = _make_model()
    ok = vkb.run(db_path=valid_db, ios_resources=ios_resources, model=model)
    assert ok is False
