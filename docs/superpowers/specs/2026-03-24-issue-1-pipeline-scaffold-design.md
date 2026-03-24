# Issue #1 — Python Pipeline Scaffold and Dev Environment

**Date:** 2026-03-24
**Issue:** #1
**Milestone:** 1 - Data pipeline and knowledge base

---

## Context

The pipeline directory already has `config.py` (Path constants), `requirements.txt`, and `__init__.py` files for all three sub-packages (`scrape`, `process`, `build_db`). What is missing is a working pytest setup and a shared logging helper.

---

## Design

### Four changes, nothing more

#### 1. `pytest.ini` (repo root)

```ini
[pytest]
testpaths = pipeline/tests
```

Allows both `pytest pipeline/tests/` and bare `pytest` to work from the repo root. Kept at the repo root so future milestones (iOS XCTest, model conversion tests) can each add their own testpath without restructuring.

#### 2. `pipeline/tests/__init__.py`

Empty file. Makes the directory a proper Python package so pytest can collect it and relative imports work.

#### 3. `pipeline/tests/test_smoke.py`

Verifies two things:

- `pipeline.config` exports `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `DB_PATH` as `pathlib.Path` instances.
- `pipeline.scrape`, `pipeline.process`, and `pipeline.build_db` are all importable (confirms `__init__.py` files are present and syntax-error free).

No fixtures, no mock HTTP — this is a pure import smoke test.

#### 4. `get_logger(name)` in `pipeline/config.py`

```python
import logging

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
```

Each pipeline script calls `logger = get_logger(__name__)` at module level. `basicConfig` is a no-op if the root logger is already configured, so calling this from multiple scripts in the same process does not duplicate handlers.

---

## What is explicitly out of scope

- `conftest.py` or shared fixtures — belong in the issues that need them (#2–#7).
- Logging to file — console output is sufficient for a one-time local pipeline run.
- `pyproject.toml` migration — `pytest.ini` is simpler and already familiar.

---

## Acceptance criteria mapping

| Criterion | Covered by |
|-----------|-----------|
| `pip install -r requirements.txt` completes on Windows | Already satisfied; no change needed |
| `pytest pipeline/tests/` runs and passes | `pytest.ini` + `tests/__init__.py` + `test_smoke.py` |
| Sub-packages have `__init__.py` | Already satisfied |
| `config.py` defines Path constants | Already satisfied |
| Timestamped logging in all pipeline scripts | `get_logger` in `config.py` |
