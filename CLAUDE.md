# Zelda TotK Guide - project context

## What this is
An offline-first iOS app that acts as an AI-powered guide for The Legend of Zelda: Tears of the Kingdom.
Users ask natural language questions ("How do I get the Hylian Shield?", "Where are the Shrine locations near Akkala?")
and get grounded answers drawn from wiki and compendium data - fully offline, no internet required after setup.

Built using RAG (Retrieval-Augmented Generation): a local vector database finds relevant knowledge chunks,
which are fed to an on-device LLM to generate a contextual answer. No hallucination about made-up locations.

## Intended user
Personal use - single user, no accounts, no backend, no cloud. The app and its data live entirely on the device.
Distributed via SideStore sideloading - no Apple Developer account required.

---

## Model configuration

The LLM is fully configurable via a single file (ModelConfig.swift). Default is 1B for free AltStore distribution.

| Setting | 1B (default) | 3B (upgrade path) |
|---------|-------------|-------------------|
| Model | Qwen2.5 1.5B Instruct | Qwen2.5 3B Instruct |
| Quantized size | ~750MB | ~1.5GB |
| App total size | ~900MB | ~1.8GB |
| Min device | Any iPhone (iOS 18) | Any iPhone (iOS 18) |
| Speed on A15 | ~15 tok/sec | ~8 tok/sec |
| Speed on A17 Pro | ~25 tok/sec | ~20 tok/sec |
| Answer quality | Good for factual Q&A | Better for complex questions |
| Distribution | SideStore free Apple ID | SideStore free Apple ID |

Switching from 1B to 3B requires only:
1. Change one constant in ModelConfig.swift
2. Re-run the convert-model CI job with MODEL_VARIANT=3B
3. Replace the .mlpackage in the app bundle
No other code changes needed anywhere.

---

## Tech stack

### Data pipeline (runs on Windows PC, one-time)
- Python 3.11+
- mwparserfromhell - parse and clean MediaWiki wikitext
- requests + BeautifulSoup4 - HTTP fetching with polite rate limiting
- sentence-transformers (all-MiniLM-L6-v2) - generate 384-dim embeddings
- sqlite-vec - store embeddings in SQLite for the knowledge base
- coremltools - convert embedding model to Core ML format for on-device use

### Model conversion (runs via GitHub Actions on macos-14 runner)
- coremltools - convert Llama 3.2 1B or 3B to Core ML .mlpackage
- 4-bit quantization
- Controlled by MODEL_VARIANT env var: "1B" (default) or "3B"

### iOS app
- SwiftUI + Swift concurrency
- Core ML - run quantized Llama model on-device
- sqlite-vec Swift package - on-device vector search
- Apple NaturalLanguage framework - query embeddings on-device
- knowledge_base.db bundled in the app at build time

### Distribution
- SideStore sideloading via free Apple ID
- GitHub Actions builds an unsigned .ipa artifact
- SideStore signs it with your free Apple ID
- SideStore auto-renews the signature every 7 days over the internet (no home Wi-Fi needed)

---

## Repo
- GitHub: https://github.com/tlo300/Zelda-RAGs-of-the-Kingdom
- Project board: https://github.com/users/tlo300/projects/3
- Default branch: main (protected - no direct pushes)

---

## Data sources

| Source | Method | Content | Licence |
|--------|--------|---------|---------|
| Hyrule Compendium API | REST API (no key needed) | All items, enemies, weapons, regions | Fan-compiled, free to use |
| Zelda Dungeon Wiki | MediaWiki API | Walkthroughs, shrines, quests, collectibles | GFDL - attribute in app |
| Zelda Wiki (zeldawiki.wiki) | MediaWiki API | Lore, NPCs, timeline, game mechanics | GFDL - attribute in app |

Scraping rules:
- Minimum 1.5 second delay between requests
- User-Agent: ZeldaTotKGuide/1.0 (personal iOS app; contact via GitHub)
- Only fetch TotK-related pages
- Store raw content in data/raw/ before processing

---

## Project layout
Zelda-RAGs-of-the-Kingdom/
  pipeline/
    scrape/                   # Data fetching scripts
    process/                  # Cleaning, chunking
    build_db/                 # Embedding and SQLite assembly
    data/
      raw/                    # Git-ignored - large raw scrape output
      processed/              # Cleaned text chunks
      knowledge_base.db       # Committed to repo - bundled in app
    tests/                    # pytest - always uses mock HTTP
    config.py                 # Shared paths (pathlib.Path constants)
    requirements.txt
  model/
    convert_embeddings.py     # all-MiniLM-L6-v2 -> Core ML
    convert_llm.py            # Qwen2.5 1.5B or 3B -> Core ML (MODEL_VARIANT)
    requirements.txt
  ios/
    ZeldaGuide/
      Views/                  # SwiftUI views
      Models/                 # Swift data models
      Services/
        ModelConfig.swift     # SINGLE SOURCE OF TRUTH for model variant
        VectorSearchService.swift
        LLMService.swift
        RAGEngine.swift
      Resources/              # knowledge_base.db, MiniLMEmbedder.mlpackage, LlamaModel.mlpackage
  docs/
    decisions/                # Architecture Decision Records
    build.md                  # AltStore sideload instructions
    upgrade-to-3b.md          # Steps to switch to 3B when ready
  .github/
    workflows/
      pipeline.yml            # Python pipeline CI (ubuntu-latest)
      convert-model.yml       # Core ML conversion (macos-14, MODEL_VARIANT)
      build-ipa.yml           # Unsigned .ipa build (macos-14)
  CLAUDE.md
  session-start.md

---

## Key architecture decisions
Do not change these without an ADR in docs/decisions/.

- MODEL SWITCHING is entirely contained in ModelConfig.swift.
  No other Swift file may reference a model filename or hardcode generation parameters.
  This is enforced in code review - any PR that puts model names elsewhere is rejected.

- knowledge_base.db is built once on PC and committed to the repo.
  It does not update at runtime. Re-run the pipeline to refresh content.

- Embeddings use all-MiniLM-L6-v2 (384 dimensions).
  The same model is converted to Core ML and bundled in the app for query embedding.
  Both pipeline and iOS app must use identical dimensions or vector search breaks silently.

- RAG retrieves top 5 chunks by cosine similarity, assembles a prompt with a system message
  that instructs the model to answer only from provided context and say it does not know
  if the answer is absent. Prevents hallucination about game content.

- The iOS app makes zero network requests at runtime. All inference is local.

- SideStore distribution uses an unsigned .ipa. SideStore signs it with the user's free Apple ID.
  The 3-app limit and 7-day expiry are the only real constraints of a free Apple ID.
  SideStore auto-renews over the internet via its built-in WireGuard VPN — no PC or home Wi-Fi needed.

---

## GitHub workflow

### Starting an issue
1. git pull origin main
2. gh issue view {number} --repo tlo300/Zelda-RAGs-of-the-Kingdom
3. git checkout -b {number}-short-description
4. gh issue edit {number} --add-label in-progress --repo tlo300/Zelda-RAGs-of-the-Kingdom

### Finishing an issue
1. All acceptance criteria checkboxes met
2. Relevant tests pass
3. git push origin {branch}
4. gh pr create --title "{issue title}" --body "Closes #{number}" --repo tlo300/Zelda-RAGs-of-the-Kingdom
5. gh issue edit {number} --remove-label in-progress --repo tlo300/Zelda-RAGs-of-the-Kingdom
6. Update Current state section below
7. git add CLAUDE.md && git commit -m "docs: update project state after #{number}"

### Writing ADRs
Create docs/decisions/NNN-short-title.md with:
- Context, Decision, Consequences

---

## Testing rules
- Python: pytest with 10-page sample dataset fixtures
- Scrapers tested against local mock HTTP server (responses library) - never live wikis in tests
- iOS: XCTest for VectorSearchService, LLMService, RAGEngine
- UI tests not required for v1

---

## Current state
Active milestone : 4 - Polish and distribution
Last completed  : #116 fix HyruleSage maxContextTokens overflow (merged 2026-03-28)
In progress     : build-ipa-llama run 23687306531 (HyruleSage fix IPA, awaiting user test)
Blocked         : nothing
Last session    : 2026-03-28 — Fixed HyruleSage blank responses (#116, maxContextTokens 1500→380); IPA building (run 23687306531)

Notes:
- HyruleSage: IPA from run 23687306531 is the one to test — fixes blank/empty responses when asking questions
- HyruleSage blank response root cause: maxContextTokens=1500 caused ~1500-token prefill, overflowing 512-slot KV cache; same bug as ZeldaGuide #111, fixed in #116 (maxContextTokens=380)
- HyruleSage model artifacts: convert-model-llama run 23683351896 (LlamaModel-1B.mlpackage + tokenizer-llama.json)
- ZeldaGuide (Qwen): R36 is the current IPA — has context-overflow fix (#111) + RAG echo fix (chunk format)
- ZeldaGuide R36 fix: changed chunk headers from [N] (source: ...) to <<source | title>>; added Answer: cue; fixes Qwen 1.5B echoing context instead of answering
- ZeldaGuide R33 and earlier: do not use for generation testing
- Fixed-size KV cache design (merged in #109):
  - convert_llm.py: past_kv [L,2,H,512,D] fixed; new_kv [L,2,H,1,D] per token; CPU_AND_GPU compile
  - LLMService.swift: CoreMLPredictor circular buffer + write pointer; .cpuAndGPU load
  - ModelConfig.swift: llmMaxKVLen = 512
  - Palettization: per_grouped_channel with group_size=32
  - ct.convert() with CPU_AND_GPU completes in ~1.5h (CPU_AND_NE hit 6h GitHub limit)
- Context fix (merged in #111): maxContextTokens = 380; charsPerChunk = 380*4/5 = 304; fits in 512-token KV cache
- NE support: future improvement once GPU baseline is confirmed working
- Do NOT sideload R26-R31 — all crash.
- FTS5 fallback triggers on similarity < 0.3; os.Logger debug logging added
- GitHub Releases: .ipa downloadable directly from Safari on iPhone via releases/latest
- KB now has 3480 chunks (was 2502); includes all 153 shrines, Wind/Lightning Temple, sage abilities
- Zelda Dungeon wiki is permanently Cloudflare-blocked (JS challenge) — not a viable source

---

## Issue status

| Issue | Title | Milestone | Status |
|-------|-------|-----------|--------|
| #1    | Python pipeline scaffold and dev environment | 1 | merged |
| #2    | Hyrule Compendium API scraper | 1 | merged |
| #3    | Zelda Dungeon Wiki MediaWiki scraper | 1 | closed (Cloudflare JS challenge — unbypassable) |
| #4    | Zelda Wiki MediaWiki scraper | 1 | merged |
| #23   | Re-run cleaning and chunking pipeline (full dataset) | 1 | PR #26 |
| #5    | Text cleaning and chunking pipeline | 1 | merged |
| #6    | Embedding generation and SQLite vector DB build | 1 | merged |
| #7    | Knowledge base validation and quality check | 1 | PR #27 |
| #8    | Embedding model Core ML conversion (CI) | 2 | PR #28 |
| #9    | LLM Core ML conversion and quantization (CI) | 2 | merged |
| #10   | GitHub Actions CI workflows | 2 | merged |
| #11   | iOS project scaffold and ModelConfig | 3 | merged |
| #12   | On-device vector search service | 3 | merged |
| #13   | On-device LLM inference service | 3 | merged |
| #14   | RAG engine (retrieval, prompt assembly, generation) | 3 | merged |
| #15   | Chat UI - question input and streaming answer | 3 | merged |
| #16   | Source attribution view | 3 | merged |
| #17   | AltStore sideload build and distribution docs | 4 | merged |
| #107  | Publish .ipa as GitHub Release asset | 4 | merged |
| #66   | Fix bundle real Core ML models in .ipa | 4 | merged |
| #70   | Fix embedding input shape (1x40 vs 1x128) | 4 | merged |
| #72   | Simulator CI + embedding shape unit tests | 4 | merged |
| #75   | Fix embedding output key mismatch | 4 | merged |
| #77   | Fix llm.load() never called + EOS token + race condition | 4 | merged |
| #78   | Replace BundledTokenizer with real Qwen2.5 BPE tokenizer | 4 | merged |
| #89   | Enhance debug-simulator for full RAG pipeline + bump 2048-context limits | 4 | merged |
| #103  | Fix OOM crash (logits slice to last token) + im_start stop token | 4 | merged |
| #18   | App performance tuning and model warm-up | 4 | merged |
| #111  | Fix RAG prompt overflow (no answer generated) | 4 | merged |
| #115  | Add HyruleSage app — Llama 3.2 powered Zelda TotK guide | 4 | merged |
| #116  | Fix HyruleSage blank responses (maxContextTokens overflow) | 4 | merged |
| #69   | Add in-app log viewer for on-device debugging | 4 | backlog |

---

## Starting a new session
VS Code Claude Code chat panel:
  Read CLAUDE.md then tell me the current state and suggest the next issue.

Terminal:
  Get-Content session-start.md -Raw | claude
