# Issue #1 — Python Pipeline Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Python pipeline scaffold so `pytest pipeline/tests/` passes and every future pipeline script has a logging helper available.

**Architecture:** Three new files are added (`pytest.ini`, `pipeline/tests/__init__.py`, `pipeline/tests/test_smoke.py`) and one existing file is modified (`pipeline/config.py`). No scraping, embedding, or database work is done — this issue is purely dev-environment setup.

**Tech Stack:** Python 3.11+, pytest 8.2.2, standard-library `logging`, `pathlib.Path`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `pytest.ini` | Tell pytest where to find tests when run from repo root |
| Create | `conftest.py` | Repo-root conftest — ensures the root is on sys.path for all pytest runs |
| Create | `pipeline/tests/__init__.py` | Make tests a proper Python package |
| Create | `pipeline/tests/test_smoke.py` | Import smoke tests for config constants and sub-packages |
| Modify | `pipeline/config.py` | Add `get_logger(name)` helper |

---

### Task 1: Add pytest.ini and root conftest.py

**Files:**
- Create: `pytest.ini` (repo root)
- Create: `conftest.py` (repo root)

- [ ] **Step 1: Create `pytest.ini`**

```ini
[pytest]
testpaths = pipeline/tests
```

- [ ] **Step 2: Create `conftest.py` at the repo root (empty)**

```python
# conftest.py
# Empty root conftest. Presence of this file ensures pytest adds the repo root
# to sys.path, making `import pipeline` work without pip install -e .
```

- [ ] **Step 3: Verify pytest discovers the (not-yet-existing) test directory without error**

Run from repo root:
```
pytest --collect-only
```
Expected output: warning or empty collection — no crash, no import error. (Tests don't exist yet, that's fine.)

- [ ] **Step 4: Commit**

```bash
git add pytest.ini conftest.py
git commit -m "chore: add pytest.ini and root conftest for test discovery"
```

---

### Task 2: Add get_logger to config.py

**Files:**
- Modify: `pipeline/config.py`

- [ ] **Step 1: Write the failing test first**

Create `pipeline/tests/test_smoke.py` with just this one test (more will be added in Task 3):

```python
# pipeline/tests/test_smoke.py
# Smoke tests: verify imports and shared utilities work correctly.

import logging
from pipeline.config import get_logger


def test_get_logger_returns_logger():
    logger = get_logger("test.smoke")
    assert isinstance(logger, logging.Logger)


def test_get_logger_name_preserved():
    logger = get_logger("test.name")
    assert logger.name == "test.name"
```

Also create `pipeline/tests/__init__.py` (empty):

```
# empty
```

- [ ] **Step 2: Run the test to confirm it fails**

Run from repo root:
```
pytest pipeline/tests/test_smoke.py -v
```
Expected: `FAILED` — `ImportError: cannot import name 'get_logger' from 'pipeline.config'`

- [ ] **Step 3: Add `get_logger` to `pipeline/config.py`**

Append to the bottom of the existing file:

```python
import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger with timestamped console output.

    Uses basicConfig so the first call configures the root logger;
    subsequent calls from other modules are no-ops for configuration
    but still return the correct named logger.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(name)
```

- [ ] **Step 4: Run the tests to confirm they pass**

```
pytest pipeline/tests/test_smoke.py -v
```
Expected:
```
PASSED pipeline/tests/test_smoke.py::test_get_logger_returns_logger
PASSED pipeline/tests/test_smoke.py::test_get_logger_name_preserved
```

- [ ] **Step 5: Commit**

```bash
git add pipeline/config.py pipeline/tests/__init__.py pipeline/tests/test_smoke.py
git commit -m "feat: add get_logger helper to config.py"
```

---

### Task 3: Add import smoke tests for Path constants and sub-packages

**Files:**
- Modify: `pipeline/tests/test_smoke.py`

- [ ] **Step 1: Add tests for Path constants and sub-package imports**

Replace the contents of `pipeline/tests/test_smoke.py` with the full smoke test suite:

```python
# pipeline/tests/test_smoke.py
# Smoke tests: verify imports and shared utilities work correctly.

import logging
from pathlib import Path

import pipeline.scrape
import pipeline.process
import pipeline.build_db
from pipeline.config import DATA_DIR, RAW_DIR, PROCESSED_DIR, DB_PATH, get_logger


def test_data_dir_is_path():
    assert isinstance(DATA_DIR, Path)


def test_raw_dir_is_path():
    assert isinstance(RAW_DIR, Path)


def test_processed_dir_is_path():
    assert isinstance(PROCESSED_DIR, Path)


def test_db_path_is_path():
    assert isinstance(DB_PATH, Path)


def test_raw_dir_is_inside_data_dir():
    assert RAW_DIR.parent == DATA_DIR


def test_processed_dir_is_inside_data_dir():
    assert PROCESSED_DIR.parent == DATA_DIR


def test_db_path_is_inside_data_dir():
    assert DB_PATH.parent == DATA_DIR


def test_scrape_importable():
    # Passes if pipeline/scrape/__init__.py exists and is syntax-error free
    assert pipeline.scrape is not None


def test_process_importable():
    assert pipeline.process is not None


def test_build_db_importable():
    assert pipeline.build_db is not None


def test_get_logger_returns_logger():
    logger = get_logger("test.smoke")
    assert isinstance(logger, logging.Logger)


def test_get_logger_name_preserved():
    logger = get_logger("test.name")
    assert logger.name == "test.name"
```

- [ ] **Step 2: Run all tests**

```
pytest pipeline/tests/ -v
```
Expected: all 12 tests PASS.

- [ ] **Step 3: Run via bare `pytest` to confirm pytest.ini testpaths works**

```
pytest -v
```
Expected: same 12 tests collected and passed (no other test directories discovered).

- [ ] **Step 4: Commit**

```bash
git add pipeline/tests/test_smoke.py
git commit -m "test: add smoke tests for config constants and sub-package imports"
```

---

### Task 4: GitHub workflow label and issue close

- [ ] **Step 1: Verify all acceptance criteria**

| Criterion | How to verify |
|-----------|---------------|
| `pip install -r requirements.txt` works on Windows | Already passes — no changes to requirements.txt |
| `pytest pipeline/tests/` runs and passes | Run `pytest pipeline/tests/ -v` — expect 12 PASSED |
| Sub-packages have `__init__.py` | `ls pipeline/scrape/__init__.py pipeline/process/__init__.py pipeline/build_db/__init__.py` |
| `config.py` defines Path constants | Covered by smoke tests `test_*_is_path` |
| Timestamped logging in all pipeline scripts | `get_logger` available in `config.py`; verified by `test_get_logger_*` tests |

- [ ] **Step 2: Push branch and open PR**

```bash
git push origin HEAD
gh pr create --title "Python pipeline scaffold and dev environment" \
  --body "Closes #1" \
  --repo tlo300/Zelda-RAGs-of-the-Kingdom
```

- [ ] **Step 3: Update CLAUDE.md project state**

In `CLAUDE.md`, update the "Current state" section:
- `Last completed` → `#1 Python pipeline scaffold and dev environment`
- Issue #1 row in the issue table → `Status: done`

```bash
git add CLAUDE.md
git commit -m "docs: update project state after #1"
```

- [ ] **Step 4: Remove in-progress label from issue**

```bash
gh issue edit 1 --remove-label in-progress --repo tlo300/Zelda-RAGs-of-the-Kingdom
```
