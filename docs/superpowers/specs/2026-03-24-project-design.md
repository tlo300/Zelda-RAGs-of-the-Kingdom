# Zelda TotK Guide — Project Design Document

**Date:** 2026-03-24
**Project:** Zelda-RAGs-of-the-Kingdom
**Repo:** https://github.com/tlo300/Zelda-RAGs-of-the-Kingdom

---

## 1. Project Context

### What it is
An offline-first iOS app that acts as an AI-powered guide for The Legend of Zelda: Tears of the Kingdom. Users ask natural language questions ("How do I get the Hylian Shield?", "Where are the Shrine locations near Akkala?") and receive grounded answers drawn from wiki and compendium data — fully offline, no internet required after the initial setup.

The core technique is RAG (Retrieval-Augmented Generation): a local vector database finds the most relevant knowledge chunks for a given query, and those chunks are fed to an on-device LLM that generates a contextual answer. Because the model only reasons over retrieved context, it cannot hallucinate invented game content.

### Who it's for
Personal single-user use. There are no accounts, no backend, no cloud sync. The app and all its data live entirely on the device.

### Key constraints

| Constraint | Impact |
|------------|--------|
| Fully offline at runtime | No API calls, no model downloads, no telemetry |
| Windows-only dev machine | Data pipeline and tooling must run on Windows; no local Xcode/macOS build |
| No paid Apple Developer account | Distribution via AltStore free-ID sideloading; app must stay within the 3-app limit |
| GitHub Actions macOS runners | All Core ML model conversion and .ipa builds happen in CI |
| AltStore 7-day signing expiry | AltStore Daemon auto-renews while phone is on same Wi-Fi as the PC |
| App size budget | 1B model: ~750 MB total; 3B model: ~2.5 GB (requires paid account for OTA) |

---

## 2. Architecture Overview

The project has three distinct execution environments that run in sequence:

```
[Windows PC]              [GitHub Actions macOS]       [iPhone]
─────────────             ──────────────────────       ────────
Data pipeline       →     Model conversion        →    iOS app
  scrape wikis              MiniLM → Core ML            VectorSearchService
  clean + chunk             Llama 1B/3B → Core ML       LLMService
  embed chunks              4-bit quantization          RAGEngine
  build SQLite DB                                       SwiftUI chat UI
  commit DB to repo         build unsigned .ipa
```

### End-to-end data flow for a user question

1. User types a question in the chat UI.
2. `VectorSearchService` embeds the question using `MiniLMEmbedder.mlpackage` (on-device, 384 dims).
3. The 384-dimensional query vector is compared against all stored embeddings in `knowledge_base.db` using cosine similarity via `sqlite-vec`.
4. The top 5 chunks by similarity score are retrieved.
5. `RAGEngine` assembles a structured prompt: system instruction + retrieved chunks + user question.
6. `LLMService` feeds the prompt to `LlamaModel.mlpackage` and streams tokens back.
7. The answer is displayed in the chat UI. Source attributions link back to the original wiki pages.

---

## 3. Component Design

### 3.1 Data Pipeline

**Purpose:** Transform raw wiki and API data into `knowledge_base.db`, a SQLite database with pre-computed 384-dimensional embeddings, ready to be bundled into the iOS app.

**Sub-steps:**

| Step | Script location | Input | Output |
|------|----------------|-------|--------|
| Scrape | `pipeline/scrape/` | Live APIs / wiki (one-time) | `pipeline/data/raw/` (git-ignored) |
| Clean + chunk | `pipeline/process/` | Raw files | `pipeline/data/processed/` |
| Embed + build DB | `pipeline/build_db/` | Processed chunks | `pipeline/data/knowledge_base.db` |

**Scraping rules:**
- Minimum 1.5-second delay between requests.
- `User-Agent: ZeldaTotKGuide/1.0 (personal iOS app; contact via GitHub)`
- Only fetch TotK-related pages.

**Data sources:**

| Source | Method | Content | Licence |
|--------|--------|---------|---------|
| Hyrule Compendium API | REST (no key) | Items, enemies, weapons, regions | Fan-compiled, free |
| Zelda Dungeon Wiki | MediaWiki API | Walkthroughs, shrines, quests | GFDL |
| Zelda Wiki (zeldawiki.wiki) | MediaWiki API | Lore, NPCs, mechanics | GFDL |

**Interface:** The pipeline is invoked as a series of Python scripts. `pipeline/config.py` provides `pathlib.Path` constants for all shared directories. Steps are independent and can be re-run individually.

**Dependencies:** Python 3.11+, `mwparserfromhell`, `requests`, `BeautifulSoup4`, `sentence-transformers` (all-MiniLM-L6-v2), `sqlite-vec`, `coremltools`.

**Testing:** `pytest` with a 10-page sample dataset in fixtures. Scrapers are tested against a local mock HTTP server using the `responses` library — live wikis are never contacted in tests.

---

### 3.2 Model Conversion

**Purpose:** Produce two Core ML artifacts that are bundled in the iOS app:
- `MiniLMEmbedder.mlpackage` — the sentence embedding model for on-device query embedding.
- `LlamaModel.mlpackage` — the quantized LLM for on-device answer generation.

**Execution environment:** GitHub Actions `macos-14` runners.

**MiniLM conversion (`model/convert_embeddings.py`):**
- Loads `all-MiniLM-L6-v2` from HuggingFace.
- Converts to Core ML using `coremltools`.
- Output: `MiniLMEmbedder.mlpackage` with a single `embed(text: String) → [Float]` interface.
- The output dimension is always 384 and must match what the pipeline stored in SQLite.

**Llama conversion (`model/convert_llm.py`):**
- Reads the `MODEL_VARIANT` environment variable (`"1B"` or `"3B"`, default `"1B"`).
- Downloads the corresponding Llama 3.2 Instruct checkpoint.
- Applies 4-bit quantization via `coremltools`.
- Output: `LlamaModel.mlpackage`.

**Switching models:** Change `MODEL_VARIANT` in the `convert-model` CI job and re-run it. No Swift code changes needed anywhere else — `ModelConfig.swift` handles all parameter differences.

**Dependencies:** `coremltools`, `torch`, `transformers` (via `model/requirements.txt`).

---

### 3.3 iOS Services

All services live under `ios/ZeldaGuide/Services/`.

#### ModelConfig.swift

The single source of truth for the active model variant. Defines:
- Model filename (`LlamaModel.mlpackage`)
- Context length, generation temperature, max tokens
- Any other model-specific generation parameters

No other Swift file may reference a model filename or hardcode generation parameters. This is a hard code-review rule enforced at PR time.

#### VectorSearchService

**What it does:** Embeds a query string using the on-device MiniLM Core ML model and performs a cosine-similarity vector search against `knowledge_base.db`.

**Interface:**
```swift
func search(query: String, topK: Int = 5) async throws -> [KnowledgeChunk]
```

**Dependencies:** `MiniLMEmbedder.mlpackage`, `knowledge_base.db` (bundled in app resources), `sqlite-vec` Swift package.

**Testing:** XCTest with a small fixture database. No network required.

#### LLMService

**What it does:** Loads `LlamaModel.mlpackage` and generates a streamed token response for a given prompt string.

**Interface:**
```swift
func generate(prompt: String) -> AsyncThrowingStream<String, Error>
```

**Dependencies:** `LlamaModel.mlpackage`, `ModelConfig.swift` (reads all generation parameters from here).

**Testing:** XCTest with a short prompt; asserts non-empty output stream and no crash.

#### RAGEngine

**What it does:** Orchestrates retrieval and generation. Calls `VectorSearchService`, assembles the prompt, calls `LLMService`, and returns a streaming answer with source metadata.

**Interface:**
```swift
func answer(question: String) -> AsyncThrowingStream<RAGResponse, Error>
// RAGResponse carries either a token chunk or the final list of KnowledgeChunk sources
```

**Dependencies:** `VectorSearchService`, `LLMService`.

**Testing:** XCTest with mock services; asserts that retrieved chunks appear in the assembled prompt, and that the response stream completes.

---

### 3.4 iOS UI

#### Chat View (`ios/ZeldaGuide/Views/`)

A single-screen SwiftUI view with:
- A scrollable message list showing the conversation history.
- A text input field at the bottom.
- Streaming token display as the LLM generates its response (tokens appended to the latest assistant message in real time).
- A "Sources" disclosure group below each answer that expands to show `SourceAttributionView`.

State is managed with `@StateObject` / `@ObservableObject`; no UIKit.

#### Source Attribution View

Displays the list of `KnowledgeChunk` sources that informed the answer:
- `page_title` of each chunk.
- The source wiki name.
- A short excerpt of `chunk_text`.

This satisfies the GFDL attribution requirement for wiki content.

---

### 3.5 CI/CD Workflows

All workflows live under `.github/workflows/`.

#### `pipeline.yml` — Data pipeline CI
- Runs on `ubuntu-latest`.
- Triggered on push/PR touching `pipeline/`.
- Installs Python deps, runs `pytest` with mock HTTP fixtures.
- Does not run the live scraper; that is a manual one-time step.

#### `convert-model.yml` — Core ML conversion
- Runs on `macos-14`.
- Accepts `MODEL_VARIANT` input (`"1B"` or `"3B"`).
- Runs `model/convert_embeddings.py` and `model/convert_llm.py`.
- Uploads `MiniLMEmbedder.mlpackage` and `LlamaModel.mlpackage` as build artifacts.

#### `build-ipa.yml` — Unsigned .ipa build
- Runs on `macos-14`.
- Downloads model artifacts from `convert-model.yml`.
- Copies `knowledge_base.db` and `.mlpackage` files into `ios/ZeldaGuide/Resources/`.
- Builds the Xcode project with `xcodebuild` and packages the unsigned `.ipa`.
- Uploads the `.ipa` as a release artifact for AltStore to consume.

---

## 4. Key Architecture Decisions

### ModelConfig.swift is the single source of truth for model variant

**Decision:** All model filenames, context lengths, and generation parameters live only in `ModelConfig.swift`. No other Swift file may reference them.

**Rationale:** Switching from 1B to 3B must be a one-line change with zero risk of missing a hardcoded reference. Centralisation is enforced structurally so that PRs that violate it are obviously wrong at review time.

### knowledge_base.db is built once and committed to the repo

**Decision:** The database is built offline on the dev PC, committed to the repo, and bundled into the app at build time. It is never updated at runtime.

**Rationale:** Runtime updates would require either network access (violating offline-first) or complex on-device pipeline logic. The corpus (a single game's wiki) changes slowly; periodic offline rebuilds are sufficient.

### all-MiniLM-L6-v2 at 384 dimensions for both pipeline and on-device

**Decision:** The same model family is used to generate embeddings during the pipeline and to embed queries on-device. The dimension is fixed at 384.

**Rationale:** Cosine similarity is only meaningful when the query vector and stored vectors come from identical embedding spaces. Using the same model guarantees this. 384 dimensions is a good balance between retrieval quality and storage / compute cost on mobile.

**Risk:** If the pipeline and the on-device model ever diverge (e.g., the pipeline is upgraded to a different model), vector search will silently return garbage results rather than erroring. This must be caught by integration tests that compare pipeline and on-device embeddings on known inputs.

### RAG retrieves top-5 chunks by cosine similarity

**Decision:** The RAG engine always retrieves exactly 5 chunks and includes all of them in the prompt context.

**Rationale:** Five chunks fits comfortably within the 1B model's context window while providing enough coverage for multi-part questions. The system prompt instructs the model to answer only from provided context and say "I don't know" if the answer is absent — this prevents hallucination about game content.

### Zero network requests at runtime

**Decision:** The iOS app makes no network requests at any point during normal operation.

**Rationale:** The app is designed for use while playing (often without reliable connectivity). All model inference, vector search, and knowledge retrieval are fully local. This also eliminates privacy concerns about query logging.

### AltStore distribution with unsigned .ipa

**Decision:** The app is distributed as an unsigned `.ipa` via AltStore with a free Apple ID. A paid Apple Developer account is only needed if the 3B model variant is chosen (due to app size).

**Rationale:** Paid Apple Developer accounts cost $99/year and are unnecessary for personal-use distribution. AltStore's free-ID path has two real constraints: the 3-app limit and the 7-day re-signing requirement. AltStore Daemon handles the 7-day renewal automatically over Wi-Fi. The 3-app limit is a user responsibility.

---

## 5. Data Schemas

### 5.1 SQLite Schema — knowledge_base.db

```sql
CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_text  TEXT    NOT NULL,
    embedding   BLOB    NOT NULL,  -- 384 x float32, stored as raw bytes
    source      TEXT    NOT NULL,  -- "compendium" | "zelda_dungeon" | "zelda_wiki"
    page_title  TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL   -- position of this chunk within the original page
);

-- Full-text search for optional keyword fallback
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_text,
    content='chunks',
    content_rowid='id'
);

-- sqlite-vec virtual table for vector search
CREATE VIRTUAL TABLE chunks_vec USING vec0(
    embedding float[384]
);
```

The `embedding` column in `chunks` stores the raw float32 bytes. The `chunks_vec` virtual table mirrors the embeddings and is queried via sqlite-vec's cosine similarity operator. `chunks_fts` is available as a fallback if vector search returns low-confidence results.

### 5.2 KnowledgeChunk Swift Struct

```swift
struct KnowledgeChunk: Identifiable {
    let id: Int64
    let chunkText: String
    let source: String      // "compendium" | "zelda_dungeon" | "zelda_wiki"
    let pageTitle: String
    let chunkIndex: Int
    let similarityScore: Float  // populated after retrieval, 0–1
}
```

### 5.3 RAG Prompt Assembly Format

```
<|system|>
You are a guide for The Legend of Zelda: Tears of the Kingdom.
Answer the user's question using only the context provided below.
If the answer is not contained in the context, say "I don't have information about that."
Do not invent locations, items, or mechanics.

Context:
--- Source 1: {chunk.pageTitle} ({chunk.source}) ---
{chunk.chunkText}

--- Source 2: {chunk.pageTitle} ({chunk.source}) ---
{chunk.chunkText}

[... up to 5 sources ...]
<|end|>
<|user|>
{userQuestion}
<|end|>
<|assistant|>
```

The Llama 3.2 Instruct chat template uses `<|system|>`, `<|user|>`, `<|assistant|>` turn markers. `ModelConfig.swift` owns the exact template string so it can be adjusted for model variants.

---

## 6. Non-Obvious Constraints and Risks

### AltStore 3-app limit and 7-day signing expiry
A free Apple ID can only have 3 apps sideloaded simultaneously. If the user already has 2 other AltStore apps, installing this guide will consume the last slot. The 7-day expiry is handled automatically by AltStore Daemon when the phone is on the same Wi-Fi network as the PC where AltStore is running. If the phone is away from home Wi-Fi for more than 7 days, the app will stop launching until it is re-signed manually.

### Silent embedding dimension mismatch
If the pipeline is ever rebuilt with a different embedding model (different dimensions or normalization), the vector search will not error — it will simply return incorrect results with high confidence. Prevention: integration tests that embed identical strings in both the pipeline and iOS app environments and assert cosine similarity > 0.99. The SQLite schema should store the dimension and model name as metadata that the iOS app validates at startup.

### 1B vs 3B affects distribution method
The 3B model (~2.5 GB total app size) exceeds what can be practically distributed via AltStore's free-ID path (which requires the `.ipa` to be downloaded and signed locally). At 3B, a paid Apple Developer account ($99/year) is needed for TestFlight or App Store distribution. This is a design constraint, not just a performance tradeoff.

### GFDL attribution for wiki content
Content from Zelda Dungeon Wiki and Zelda Wiki is licensed under the GNU Free Documentation Licence. The app must display attribution. This is fulfilled by the Source Attribution View shown under each answer. The attribution wording must credit the specific wiki by name and link (or display the URL as text). Failure to attribute is a licence violation.

### Model warm-up latency
Core ML models have a cold-start cost. The first inference after app launch will be noticeably slower than subsequent ones. Issue #18 (performance tuning) covers pre-warming the model during the app's loading screen so the first user question is not affected.

### Windows-only dev machine
There is no local Xcode available. All Swift compilation, Core ML conversion, and `.ipa` packaging happen exclusively in GitHub Actions. This means build iteration cycles are longer than on a native Mac. The pipeline and test logic (Python) can be iterated locally. iOS code changes must be pushed to a branch and validated through CI.

---

## 7. Build and Distribution Flow

### Step 1: Build the knowledge base (Windows PC, one-time per refresh)

```
cd pipeline
pip install -r requirements.txt
python scrape/scrape_compendium.py
python scrape/scrape_zelda_dungeon.py
python scrape/scrape_zelda_wiki.py
python process/clean_and_chunk.py
python build_db/build_knowledge_base.py
```

Output: `pipeline/data/knowledge_base.db`

Commit the database to the repo:
```
git add pipeline/data/knowledge_base.db
git commit -m "data: rebuild knowledge base"
git push origin main
```

### Step 2: Convert models (GitHub Actions, triggered manually or on push)

Trigger the `convert-model` workflow via the GitHub Actions UI or `gh workflow run`:
```
gh workflow run convert-model.yml -f MODEL_VARIANT=1B \
  --repo tlo300/Zelda-RAGs-of-the-Kingdom
```

This produces `MiniLMEmbedder.mlpackage` and `LlamaModel.mlpackage` as workflow artifacts. These are automatically consumed by the next step.

### Step 3: Build the unsigned .ipa (GitHub Actions, triggered after model conversion)

Trigger the `build-ipa` workflow:
```
gh workflow run build-ipa.yml \
  --repo tlo300/Zelda-RAGs-of-the-Kingdom
```

The workflow:
1. Downloads model artifacts from the latest `convert-model` run.
2. Copies `knowledge_base.db` and `.mlpackage` files into `ios/ZeldaGuide/Resources/`.
3. Builds the Xcode project unsigned.
4. Uploads `ZeldaGuide.ipa` as a release artifact.

### Step 4: Sideload onto iPhone via AltStore

Prerequisites:
- AltStore installed on the iPhone (via the AltStore installer on Windows).
- AltServer running on the Windows PC.
- Phone and PC on the same Wi-Fi network.

Procedure:
1. Download `ZeldaGuide.ipa` from the GitHub Actions artifact.
2. Open AltStore on the iPhone → My Apps → (+) → select the `.ipa` file (via AltServer on PC, or AltStore's file import).
3. AltStore signs the `.ipa` with the free Apple ID and installs it.
4. The app is now available and fully offline.

### Step 5: Automatic re-signing (ongoing)

AltStore Daemon running on the Windows PC will re-sign the app every 7 days when the phone is on the same Wi-Fi. No manual action required under normal use.

### Upgrading from 1B to 3B model

1. Edit `ModelConfig.swift`: change the model variant constant from `"1B"` to `"3B"`.
2. Re-run `convert-model.yml` with `MODEL_VARIANT=3B`.
3. Re-run `build-ipa.yml`.
4. Install the new `.ipa` via AltStore (requires a paid Apple Developer account due to app size).

Full details: `docs/upgrade-to-3b.md`.
