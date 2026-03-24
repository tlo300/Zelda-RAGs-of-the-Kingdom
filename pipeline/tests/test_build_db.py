# pipeline/tests/test_build_db.py
# Tests for embedding generation and SQLite vector DB build.
# SentenceTransformer is mocked — no live model downloads in tests.

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import sqlite_vec

from pipeline.build_db.embed import (
    EMBEDDING_DIM,
    _is_bad_embedding,
    _open_db,
    _pack_embedding,
    run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model(bad_indices: set[int] | None = None) -> MagicMock:
    """Return a mock SentenceTransformer.

    Vectors at positions in bad_indices are zero-vectors; all others are valid.
    """
    bad_indices = bad_indices or set()
    model = MagicMock()

    def encode(texts, **kwargs):
        vecs = []
        for i, _ in enumerate(texts):
            vec = np.full(EMBEDDING_DIM, 0.1 * (i + 1), dtype=np.float32)
            if i in bad_indices:
                vec[:] = 0.0
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)

    model.encode.side_effect = encode
    return model


@pytest.fixture
def mock_model():
    return _make_model()


def _write_chunks(path: Path, count: int) -> Path:
    records = [
        {
            "source": "zelda_wiki",
            "page_title": f"Page {i}",
            "chunk_index": 0,
            "text": f"This is chunk text number {i}. " * 10,
        }
        for i in range(count)
    ]
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


@pytest.fixture
def sample_chunks_file(tmp_path: Path) -> Path:
    """10-chunk file — enough for happy-path tests."""
    return _write_chunks(tmp_path / "chunks.jsonl", 10)


@pytest.fixture
def large_chunks_file(tmp_path: Path) -> Path:
    """200-chunk file — 1 bad chunk stays well under 1% threshold."""
    return _write_chunks(tmp_path / "chunks_large.jsonl", 200)


# ---------------------------------------------------------------------------
# _is_bad_embedding
# ---------------------------------------------------------------------------


def test_is_bad_embedding_zero_vector():
    assert _is_bad_embedding([0.0] * EMBEDDING_DIM) is True


def test_is_bad_embedding_nan_in_vector():
    vec = [0.5] * EMBEDDING_DIM
    vec[10] = float("nan")
    assert _is_bad_embedding(vec) is True


def test_is_bad_embedding_valid_vector():
    assert _is_bad_embedding([0.1] * EMBEDDING_DIM) is False


def test_is_bad_embedding_single_nonzero_is_valid():
    vec = [0.0] * EMBEDDING_DIM
    vec[0] = 0.001
    assert _is_bad_embedding(vec) is False


# ---------------------------------------------------------------------------
# _pack_embedding
# ---------------------------------------------------------------------------


def test_pack_embedding_byte_length():
    packed = _pack_embedding([0.5] * EMBEDDING_DIM)
    assert len(packed) == EMBEDDING_DIM * 4  # 4 bytes per float32


def test_pack_embedding_roundtrips():
    import struct

    original = [float(i) / EMBEDDING_DIM for i in range(EMBEDDING_DIM)]
    packed = _pack_embedding(original)
    unpacked = list(struct.unpack(f"{EMBEDDING_DIM}f", packed))
    assert unpacked == pytest.approx(original, abs=1e-6)


# ---------------------------------------------------------------------------
# run() — happy path
# ---------------------------------------------------------------------------


def test_run_creates_db_file(tmp_path, mock_model, sample_chunks_file):
    db_path = tmp_path / "kb.db"
    run(chunks_path=sample_chunks_file, db_path=db_path, model=mock_model)
    assert db_path.exists()


def test_run_returns_stats(tmp_path, mock_model, sample_chunks_file):
    db_path = tmp_path / "kb.db"
    stats = run(chunks_path=sample_chunks_file, db_path=db_path, model=mock_model)
    assert stats["inserted"] == 10
    assert stats["skipped"] == 0


def test_chunks_table_row_count(tmp_path, mock_model, sample_chunks_file):
    db_path = tmp_path / "kb.db"
    run(chunks_path=sample_chunks_file, db_path=db_path, model=mock_model)
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    assert count == 10


def test_chunks_table_columns(tmp_path, mock_model, sample_chunks_file):
    db_path = tmp_path / "kb.db"
    run(chunks_path=sample_chunks_file, db_path=db_path, model=mock_model)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT id, chunk_text, source, page_title, chunk_index FROM chunks LIMIT 1"
    ).fetchone()
    conn.close()
    assert row is not None
    assert len(row) == 5


def test_chunks_metadata_stored_correctly(tmp_path, mock_model, sample_chunks_file):
    db_path = tmp_path / "kb.db"
    run(chunks_path=sample_chunks_file, db_path=db_path, model=mock_model)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT source, page_title, chunk_index FROM chunks WHERE id = 1"
    ).fetchone()
    conn.close()
    assert row == ("zelda_wiki", "Page 0", 0)


# ---------------------------------------------------------------------------
# run() — embeddings table
# ---------------------------------------------------------------------------


def test_embeddings_table_exists(tmp_path, mock_model, sample_chunks_file):
    db_path = tmp_path / "kb.db"
    run(chunks_path=sample_chunks_file, db_path=db_path, model=mock_model)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE name = 'chunk_embeddings'"
    ).fetchone()
    conn.close()
    assert result is not None


def test_embeddings_row_count_matches_chunks(tmp_path, mock_model, sample_chunks_file):
    db_path = tmp_path / "kb.db"
    run(chunks_path=sample_chunks_file, db_path=db_path, model=mock_model)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    count = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
    conn.close()
    assert count == 10


# ---------------------------------------------------------------------------
# run() — FTS5 table
# ---------------------------------------------------------------------------


def test_fts_table_exists(tmp_path, mock_model, sample_chunks_file):
    db_path = tmp_path / "kb.db"
    run(chunks_path=sample_chunks_file, db_path=db_path, model=mock_model)
    conn = sqlite3.connect(db_path)
    result = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'chunks_fts'"
    ).fetchone()
    conn.close()
    assert result is not None


def test_fts_search_returns_results(tmp_path, mock_model, sample_chunks_file):
    db_path = tmp_path / "kb.db"
    run(chunks_path=sample_chunks_file, db_path=db_path, model=mock_model)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunk_text MATCH 'chunk' LIMIT 5"
    ).fetchall()
    conn.close()
    assert len(rows) > 0


# ---------------------------------------------------------------------------
# run() — bad embedding filtering
# ---------------------------------------------------------------------------


def test_skips_zero_vector(tmp_path, large_chunks_file):
    db_path = tmp_path / "kb.db"
    model = _make_model(bad_indices={0})
    stats = run(chunks_path=large_chunks_file, db_path=db_path, model=model)
    assert stats["skipped"] == 1
    assert stats["inserted"] == 199


def test_skips_nan_vector(tmp_path, large_chunks_file):
    db_path = tmp_path / "kb.db"
    model = MagicMock()

    def encode(texts, **kwargs):
        vecs = []
        for i, _ in enumerate(texts):
            vec = np.full(EMBEDDING_DIM, 0.5, dtype=np.float32)
            if i == 3:
                vec[5] = float("nan")
            vecs.append(vec)
        return np.array(vecs, dtype=np.float32)

    model.encode.side_effect = encode
    stats = run(chunks_path=large_chunks_file, db_path=db_path, model=model)
    assert stats["skipped"] == 1
    assert stats["inserted"] == 199


def test_skipped_chunk_not_in_db(tmp_path, large_chunks_file):
    db_path = tmp_path / "kb.db"
    model = _make_model(bad_indices={0})  # "Page 0" is skipped
    run(chunks_path=large_chunks_file, db_path=db_path, model=model)
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT id FROM chunks WHERE page_title = 'Page 0'"
    ).fetchone()
    conn.close()
    assert row is None


# ---------------------------------------------------------------------------
# run() — 1% skip threshold
# ---------------------------------------------------------------------------


def test_exits_when_skip_rate_exceeds_threshold(tmp_path, sample_chunks_file):
    """2 bad out of 10 = 20% > 1% — should sys.exit(1)."""
    db_path = tmp_path / "kb.db"
    model = _make_model(bad_indices={0, 1})
    with pytest.raises(SystemExit) as exc_info:
        run(chunks_path=sample_chunks_file, db_path=db_path, model=model)
    assert exc_info.value.code == 1


def test_no_exit_when_skip_rate_at_threshold(tmp_path):
    """Exactly 1 bad out of 101 ≈ 0.99% < 1% — should not exit."""
    chunks_path = tmp_path / "chunks.jsonl"
    records = [
        {"source": "zelda_wiki", "page_title": f"P{i}", "chunk_index": 0, "text": f"text {i} " * 10}
        for i in range(101)
    ]
    chunks_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    model = _make_model(bad_indices={0})  # 1/101 ≈ 0.99%
    db_path = tmp_path / "kb.db"
    stats = run(chunks_path=chunks_path, db_path=db_path, model=model)
    assert stats["skipped"] == 1
    assert stats["inserted"] == 100
