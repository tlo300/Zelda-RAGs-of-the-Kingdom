# pipeline/validate_kb.py
# Validate knowledge_base.db quality before bundling in the iOS app.
#
# Usage:
#   python -m pipeline.validate_kb
#
# Checks:
#   - Total chunk count between 2000 and 50000
#   - All active sources present (compendium, zelda_wiki)
#   - No chunk_text shorter than 50 characters
#   - All embedding rows are 384-dimensional
#   - No NaN or all-zero embedding vectors
#   - 10 sample queries printed with top-3 results for manual review
#   - knowledge_base.db copied to ios/ZeldaGuide/Resources/
#
# Exits non-zero if any check fails.

import math
import shutil
import sqlite3
import struct
import sys
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer

from pipeline.config import DB_PATH, get_logger

EMBEDDING_DIM = 384
IOS_RESOURCES_DIR = Path(__file__).parent.parent / "ios" / "ZeldaGuide" / "Resources"

SAMPLE_QUERIES = [
    "How do I find the Hylian Shield?",
    "Where do Gloom Hands spawn?",
    "How to solve the Jiosin shrine puzzle?",
    "How do I complete the Find the Fifth Sage quest?",
    "Who is Purah and where is she?",
    "Where can I find Zonaite?",
    "How to defeat a Lynel?",
    "Where are the Dragon's Tears memories located?",
    "What does the Autobuild ability do?",
    "Where is Impa located in Hyrule?",
]

log = get_logger(__name__)


def _open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def check_chunk_count(conn: sqlite3.Connection) -> bool:
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if 2000 <= count <= 50000:
        log.info("PASS chunk count: %d (expected 2000–50000)", count)
        return True
    log.error("FAIL chunk count: %d is outside expected range 2000–50000", count)
    return False


def check_sources(conn: sqlite3.Connection) -> bool:
    sources = {row[0] for row in conn.execute("SELECT DISTINCT source FROM chunks").fetchall()}
    log.info("Sources present: %s", sources)
    # compendium sources are stored as compendium/<sub-category>
    has_compendium = any(s == "compendium" or s.startswith("compendium/") for s in sources)
    has_zelda_wiki = "zelda_wiki" in sources
    missing = []
    if not has_compendium:
        missing.append("compendium")
    if not has_zelda_wiki:
        missing.append("zelda_wiki")
    if not missing:
        log.info("PASS all required sources present")
        return True
    log.error("FAIL missing sources: %s", missing)
    return False


def check_no_short_chunks(conn: sqlite3.Connection) -> bool:
    rows = conn.execute(
        "SELECT id, chunk_text FROM chunks WHERE length(chunk_text) < 50"
    ).fetchall()
    if not rows:
        log.info("PASS no chunks under 50 characters")
        return True
    log.error("FAIL %d chunk(s) under 50 characters", len(rows))
    for row_id, text in rows[:5]:
        log.error("  id=%d text=%r", row_id, text[:80])
    return False


def check_embedding_dimensions(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("SELECT rowid, embedding FROM chunk_embeddings LIMIT 100").fetchall()
    bad = [rowid for rowid, blob in rows if len(blob) != EMBEDDING_DIM * 4]
    if not bad:
        total = conn.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
        log.info("PASS embedding dimensions OK (checked up to 100 / %d rows)", total)
        return True
    log.error("FAIL %d embedding(s) with wrong byte length (expected %d)", len(bad), EMBEDDING_DIM * 4)
    return False


def check_no_bad_embeddings(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("SELECT rowid, embedding FROM chunk_embeddings").fetchall()
    bad_count = 0
    for rowid, blob in rows:
        vec = list(struct.unpack(f"{EMBEDDING_DIM}f", blob))
        if all(v == 0.0 for v in vec) or any(math.isnan(v) for v in vec):
            bad_count += 1
            log.error("  Bad embedding (NaN or all-zero) at rowid=%d", rowid)
    if bad_count == 0:
        log.info("PASS no NaN or all-zero embeddings found")
        return True
    log.error("FAIL %d bad embedding(s) detected", bad_count)
    return False


def run_sample_queries(conn: sqlite3.Connection, model: SentenceTransformer) -> None:
    log.info("--- Sample query results (top-3 per query) ---")
    for query in SAMPLE_QUERIES:
        vec = model.encode(query).tolist()
        packed = struct.pack(f"{EMBEDDING_DIM}f", *vec)
        rows = conn.execute(
            """
            SELECT c.page_title, c.source, c.chunk_text,
                   vec_distance_cosine(ce.embedding, ?) AS dist
            FROM chunk_embeddings ce
            JOIN chunks c ON ce.rowid = c.id
            ORDER BY dist ASC
            LIMIT 3
            """,
            (packed,),
        ).fetchall()
        log.info("Query: %r", query)
        for i, (title, source, text, dist) in enumerate(rows, 1):
            snippet = text[:120].replace("\n", " ")
            log.info("  #%d [%s / %s  dist=%.4f] %s", i, source, title, dist, snippet)


def copy_to_ios(db_path: Path, ios_resources: Path) -> bool:
    ios_resources.mkdir(parents=True, exist_ok=True)
    dest = ios_resources / "knowledge_base.db"
    shutil.copy2(db_path, dest)
    log.info("PASS copied knowledge_base.db -> %s", dest)
    return True


def run(
    db_path: Path = DB_PATH,
    ios_resources: Path = IOS_RESOURCES_DIR,
    model: SentenceTransformer | None = None,
) -> bool:
    """Run all validation checks. Returns True if all pass, False otherwise."""
    if not db_path.exists():
        log.error("FAIL knowledge_base.db not found: %s", db_path)
        return False

    conn = _open_db(db_path)

    checks = [
        check_chunk_count(conn),
        check_sources(conn),
        check_no_short_chunks(conn),
        check_embedding_dimensions(conn),
        check_no_bad_embeddings(conn),
    ]

    if model is None:
        log.info("Loading embedding model for sample queries...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

    run_sample_queries(conn, model)
    conn.close()

    checks.append(copy_to_ios(db_path, ios_resources))

    if all(checks):
        log.info("All checks passed.")
        return True
    log.error("One or more checks FAILED.")
    return False


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
