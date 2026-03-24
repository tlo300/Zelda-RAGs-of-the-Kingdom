# pipeline/build_db/embed.py
# Embed all chunks with all-MiniLM-L6-v2 and store in SQLite with sqlite-vec.
#
# Usage:
#   python -m pipeline.build_db.embed
#
# Reads:  pipeline/data/processed/chunks.jsonl
# Writes: pipeline/data/knowledge_base.db
#
# Schema:
#   chunks              — id, chunk_text, source, page_title, chunk_index
#   chunk_embeddings    — vec0 virtual table; rowid matches chunks.id
#   chunks_fts          — fts5 virtual table on chunk_text (hybrid search fallback)

import json
import math
import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pipeline.config import PROCESSED_DIR, DB_PATH, get_logger

EMBEDDING_DIM = 384
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"

log = get_logger(__name__)


def _load_model() -> SentenceTransformer:
    log.info("Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    dim = len(model.encode("probe"))
    if dim != EMBEDDING_DIM:
        log.error(
            "Expected embedding dimension %d, got %d — wrong model loaded. Exiting.",
            EMBEDDING_DIM,
            dim,
        )
        sys.exit(1)
    log.info("Model loaded. Embedding dimension: %d", dim)
    return model


def _pack_embedding(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _is_bad_embedding(vec) -> bool:
    """Return True if the vector is all-zeros or contains any NaN."""
    return all(v == 0.0 for v in vec) or any(math.isnan(v) for v in vec)


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _setup_schema(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS chunk_embeddings")
    conn.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("""
        CREATE TABLE chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_text  TEXT    NOT NULL,
            source      TEXT    NOT NULL,
            page_title  TEXT    NOT NULL,
            chunk_index INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
            embedding float[384]
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_text,
            content='chunks',
            content_rowid='id'
        )
    """)
    conn.commit()


def run(
    chunks_path: Path = CHUNKS_PATH,
    db_path: Path = DB_PATH,
    model: SentenceTransformer | None = None,
) -> dict:
    """Embed all chunks and write knowledge_base.db. Returns {inserted, skipped}."""
    if model is None:
        model = _load_model()

    raw = chunks_path.read_text(encoding="utf-8").strip().splitlines()
    chunks = [json.loads(line) for line in raw if line.strip()]
    log.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _open_db(db_path)
    _setup_schema(conn)

    log.info("Encoding %d chunks...", len(chunks))
    embeddings = model.encode(
        [c["text"] for c in chunks],
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    skipped = 0
    inserted = 0

    for chunk, vec in tqdm(
        zip(chunks, embeddings),
        total=len(chunks),
        desc="Writing DB",
        unit="chunk",
    ):
        if _is_bad_embedding(vec):
            log.warning(
                "Skipped bad embedding: chunk_index=%d page_title=%r",
                chunk["chunk_index"],
                chunk["page_title"],
            )
            skipped += 1
            continue

        conn.execute(
            "INSERT INTO chunks (chunk_text, source, page_title, chunk_index) VALUES (?, ?, ?, ?)",
            (chunk["text"], chunk["source"], chunk["page_title"], chunk["chunk_index"]),
        )
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO chunk_embeddings(rowid, embedding) VALUES (?, ?)",
            (row_id, _pack_embedding(vec.tolist())),
        )
        conn.execute(
            "INSERT INTO chunks_fts(rowid, chunk_text) VALUES (?, ?)",
            (row_id, chunk["text"]),
        )
        inserted += 1

    conn.commit()
    conn.close()

    total = inserted + skipped
    log.info("Embedding complete — inserted: %d, skipped: %d", inserted, skipped)

    if total > 0 and skipped / total > 0.01:
        log.error(
            "Skipped %d / %d chunks (%.1f%%) exceeds 1%% threshold — CI failure.",
            skipped,
            total,
            100.0 * skipped / total,
        )
        sys.exit(1)

    return {"inserted": inserted, "skipped": skipped}


if __name__ == "__main__":
    run()
