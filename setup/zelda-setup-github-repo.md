# Zelda TotK Guide - GitHub repo setup

You are setting up a GitHub repository for an offline iOS AI guide for The Legend of Zelda: Tears of the Kingdom.
Use the GitHub CLI (gh) and PowerShell throughout - never bash syntax.
Work through every step in order, confirm each succeeded, and fix failures before moving on.
Do not stop to ask questions.

---

## Before you start

```powershell
$OWNER = gh api user --jq .login
Write-Host "Owner: $OWNER"
```

---

## Step 1 - Create the repository

```powershell
gh repo create zelda-totk-guide --private --description "Offline iOS AI guide for Zelda: Tears of the Kingdom - on-device RAG with Llama 3.2"
```

---

## Step 2 - Clone and scaffold

```powershell
gh repo clone "$OWNER/zelda-totk-guide"
Set-Location zelda-totk-guide
```

Create folder structure:

```powershell
New-Item -ItemType Directory -Force `
  pipeline/scrape, `
  pipeline/process, `
  pipeline/build_db, `
  pipeline/data/raw, `
  pipeline/data/processed, `
  pipeline/tests, `
  model, `
  ios/ZeldaGuide/Views, `
  ios/ZeldaGuide/Models, `
  ios/ZeldaGuide/Services, `
  ios/ZeldaGuide/Resources, `
  docs/decisions, `
  .github/workflows
```

Create `.gitignore`:

```powershell
@"
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
dist/
build/

# Large data - raw scrape output not committed
pipeline/data/raw/

# Environment
.env
.env.local

# Xcode
ios/*.xcodeproj/xcuserdata/
ios/DerivedData/
ios/.build/
*.ipa

# Models - too large for git, downloaded by CI or manually
model/*.mlpackage
model/*.gguf
model/*.bin
ios/ZeldaGuide/Resources/*.mlpackage

# OS
.DS_Store
Thumbs.db

# IDEs
.idea/
.vscode/
*.swp
"@ | Set-Content .gitignore
```

Create `README.md`:

```powershell
@"
# Zelda: Tears of the Kingdom - Offline AI Guide

An iOS app that answers your TotK questions entirely on-device. No internet required after install.
Distributed free via AltStore - no Apple Developer account needed.

## How it works

1. A Python pipeline scrapes the Hyrule Compendium API, Zelda Dungeon Wiki, and Zelda Wiki
2. Content is chunked, embedded, and stored in a SQLite vector database
3. The database and a quantized Llama 3.2 model are bundled in the iOS app
4. On your phone, questions trigger a RAG pipeline: vector search finds relevant chunks,
   the on-device LLM generates a grounded answer

## Model variants

Default: Llama 3.2 1B (~750MB app, works on any iPhone, free AltStore distribution)
Upgrade: Llama 3.2 3B (~2.5GB app, better quality, requires paid Apple Developer account)

To switch models: change one constant in ModelConfig.swift and re-run the CI conversion job.

## Requirements

- iPhone running iOS 17 or later
- ~800MB free storage (1B model)
- AltStore installed on your phone and PC

## Install

1. Build the .ipa via GitHub Actions (workflow_dispatch on build-ipa.yml)
2. Download the .ipa artifact
3. Open AltStore on your phone and sideload the .ipa

See docs/build.md for full instructions.

## Data sources

- Hyrule Compendium API (https://gadhagod.github.io/Hyrule-Compendium-API)
- Zelda Dungeon Wiki (https://www.zeldadungeon.net/wiki) - GFDL licence
- Zelda Wiki (https://zeldawiki.wiki) - GFDL licence
"@ | Set-Content README.md
```

Create `.env.example`:

```powershell
@"
# MODEL_VARIANT controls which Llama model to convert and bundle
# "1B" = Llama 3.2 1B Instruct (default, free AltStore distribution)
# "3B" = Llama 3.2 3B Instruct (better quality, requires paid Apple Developer account)
MODEL_VARIANT=1B

# Hugging Face token - required to download Llama model weights
# Get yours at https://huggingface.co/settings/tokens
# Accept the Llama 3.2 licence at https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
HF_TOKEN=
"@ | Set-Content .env.example
```

Create pipeline requirements:

```powershell
@"
requests==2.31.0
beautifulsoup4==4.12.3
mwparserfromhell==0.6.6
sentence-transformers==2.7.0
sqlite-vec==0.1.6
tqdm==4.66.4
pytest==8.2.2
pytest-mock==3.14.0
responses==0.25.3
"@ | Set-Content pipeline/requirements.txt
```

Create model conversion requirements:

```powershell
@"
coremltools==8.0
transformers==4.44.0
torch==2.4.0
"@ | Set-Content model/requirements.txt
```

Create ModelConfig.swift - the single file that controls model switching:

```powershell
@"
// ModelConfig.swift
// THE single source of truth for the LLM variant.
// To switch from 1B to 3B: change modelVariant to "3B", re-run the convert-model CI job,
// and replace LlamaModel.mlpackage in Resources/. No other files need to change.

import Foundation

enum ModelVariant: String {
    case llama1B = "1B"
    case llama3B = "3B"
}

struct ModelConfig {

    // MARK: - Change this to switch model variants
    static let activeVariant: ModelVariant = .llama1B

    // MARK: - Derived config (do not edit below this line)

    static var modelFilename: String {
        switch activeVariant {
        case .llama1B: return "LlamaModel-1B.mlpackage"
        case .llama3B: return "LlamaModel-3B.mlpackage"
        }
    }

    static var maxContextTokens: Int {
        switch activeVariant {
        case .llama1B: return 4096
        case .llama3B: return 8192
        }
    }

    static var maxOutputTokens: Int { 512 }
    static var ragTopK: Int { 5 }
    static var embeddingDimensions: Int { 384 }
    static var embeddingModelFilename: String { "MiniLMEmbedder.mlpackage" }
    static var knowledgeBaseFilename: String { "knowledge_base.db" }

    static var systemPrompt: String {
        return """
        You are a helpful guide for The Legend of Zelda: Tears of the Kingdom. \
        Answer questions using only the context provided below. \
        If the answer is not in the context, say you do not have that information. \
        Be concise and specific.
        """
    }
}
"@ | Set-Content ios/ZeldaGuide/Services/ModelConfig.swift
```

Create AltStore build docs placeholder:

```powershell
@"
# Building and installing with AltStore

## Prerequisites
- AltStore installed on your iPhone and PC (https://altstore.io)
- AltStore Daemon running on your PC
- GitHub account with this repo

## Building the .ipa

1. Go to the GitHub Actions tab in this repo
2. Select the 'Build iOS .ipa' workflow
3. Click 'Run workflow'
4. Wait ~15 minutes for the build to complete
5. Download the .ipa artifact from the completed run

## Installing with AltStore

1. Connect your iPhone to your PC via USB (first time only)
2. Open AltStore on your iPhone
3. Tap the + button and select the downloaded .ipa file
4. The app will install and sign with your free Apple ID

## Auto-renewal

AltStore Daemon on your PC will automatically re-sign the app every 7 days
when your iPhone is on the same Wi-Fi network as your PC.
You do not need to manually reinstall unless you delete the app.

## Switching to the 3B model

See docs/upgrade-to-3b.md when you have a paid Apple Developer account.
"@ | Set-Content docs/build.md
```

Create upgrade docs placeholder:

```powershell
@"
# Upgrading from 1B to 3B model

When you have a paid Apple Developer account and want better answer quality:

## Steps

1. Edit ios/ZeldaGuide/Services/ModelConfig.swift
   Change: static let activeVariant: ModelVariant = .llama1B
   To:     static let activeVariant: ModelVariant = .llama3B

2. Go to GitHub Actions > Convert Core ML model > Run workflow
   Set MODEL_VARIANT input to: 3B

3. Wait for the conversion job to complete (~45 minutes)

4. Download the LlamaModel-3B.mlpackage artifact

5. Replace ios/ZeldaGuide/Resources/LlamaModel-1B.mlpackage with LlamaModel-3B.mlpackage

6. Rebuild the .ipa via GitHub Actions > Build iOS .ipa

7. Sideload the new .ipa (or use Xcode direct install with your Developer account)

## Notes

- The 3B model requires ~1.8GB of storage vs ~600MB for 1B
- All other code stays the same - ModelConfig.swift handles all differences
- You will need to set up Apple Developer signing in build-ipa.yml (see comments in that file)
"@ | Set-Content docs/upgrade-to-3b.md
```

Create CI workflow placeholders:

```powershell
@"
# .github/workflows/pipeline.yml
# Runs the Python data pipeline and validates the knowledge base
# Implemented in issue #10
name: Data pipeline
on:
  workflow_dispatch:
jobs:
  placeholder:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
"@ | Set-Content .github/workflows/pipeline.yml

@"
# .github/workflows/convert-model.yml
# Converts Llama 3.2 (1B or 3B) to Core ML on a macOS runner
# MODEL_VARIANT input: "1B" (default) or "3B"
# Implemented in issue #10
name: Convert Core ML model
on:
  workflow_dispatch:
    inputs:
      model_variant:
        description: 'Model variant to convert (1B or 3B)'
        required: true
        default: '1B'
        type: choice
        options: ['1B', '3B']
jobs:
  placeholder:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
"@ | Set-Content .github/workflows/convert-model.yml

@"
# .github/workflows/build-ipa.yml
# Builds unsigned .ipa for AltStore sideloading
# Implemented in issue #10
# Note: for paid Developer account distribution, see docs/upgrade-to-3b.md
name: Build iOS .ipa
on:
  workflow_dispatch:
jobs:
  placeholder:
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
"@ | Set-Content .github/workflows/build-ipa.yml
```

Create pipeline README:

```powershell
@"
# Data pipeline

Builds the knowledge base SQLite file from three sources. Runs on Windows (PowerShell).

## Setup

``````powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
``````

## Run all steps

``````powershell
python scrape/hyrule_compendium.py    # Fetch all item/enemy/weapon data
python scrape/zelda_dungeon.py        # Fetch walkthrough content
python scrape/zelda_wiki.py           # Fetch lore and NPC content
python process/clean_and_chunk.py    # Clean wikitext, chunk to ~500 tokens
python build_db/embed_and_store.py   # Generate embeddings, build SQLite DB
python build_db/validate.py          # Quality checks - run before committing DB
``````

## Output

data/knowledge_base.db - committed to repo, bundled in iOS app

## Run tests

``````powershell
pytest tests/ -v
``````
"@ | Set-Content pipeline/README.md
```

Create CLAUDE.md (fill {owner} with actual value):

```powershell
$claudeContent = Get-Content zelda-CLAUDE.md -Raw
$claudeContent = $claudeContent -replace '\{owner\}', $OWNER
Set-Content CLAUDE.md $claudeContent
```

Create session-start.md (fill {owner} with actual value):

```powershell
$sessionContent = Get-Content zelda-session-start.md -Raw
$sessionContent = $sessionContent -replace '\{owner\}', $OWNER
Set-Content session-start.md $sessionContent
```

Commit and push:

```powershell
git add .
git commit -m "chore: initial project scaffold"
git push origin main
```

---

## Step 3 - Branch protection

```powershell
$protection = @{
  required_status_checks = @{ strict = $true; contexts = @() }
  enforce_admins = $false
  required_pull_request_reviews = @{
    required_approving_review_count = 0
    dismiss_stale_reviews = $true
  }
  restrictions = $null
  allow_force_pushes = $false
  allow_deletions = $false
} | ConvertTo-Json -Depth 5

$protection | gh api "repos/$OWNER/zelda-totk-guide/branches/main/protection" --method PUT --input -
```

---

## Step 4 - Labels

```powershell
$defaultLabels = @("bug","documentation","duplicate","enhancement","good first issue","help wanted","invalid","question","wontfix")
foreach ($label in $defaultLabels) {
    gh label delete $label --repo "$OWNER/zelda-totk-guide" --yes 2>$null
}

gh label create "story"       --color "0075ca" --repo "$OWNER/zelda-totk-guide"
gh label create "bug"         --color "d73a4a" --repo "$OWNER/zelda-totk-guide"
gh label create "chore"       --color "e4e669" --repo "$OWNER/zelda-totk-guide"
gh label create "pipeline"    --color "1D9E75" --repo "$OWNER/zelda-totk-guide"
gh label create "ios"         --color "7F77DD" --repo "$OWNER/zelda-totk-guide"
gh label create "ci"          --color "BA7517" --repo "$OWNER/zelda-totk-guide"
gh label create "model"       --color "D85A30" --repo "$OWNER/zelda-totk-guide"
gh label create "data"        --color "378ADD" --repo "$OWNER/zelda-totk-guide"
gh label create "S"           --color "c2e0c6" --repo "$OWNER/zelda-totk-guide"
gh label create "M"           --color "fef2c0" --repo "$OWNER/zelda-totk-guide"
gh label create "L"           --color "f9d0c4" --repo "$OWNER/zelda-totk-guide"
gh label create "XL"          --color "e99695" --repo "$OWNER/zelda-totk-guide"
gh label create "in-progress" --color "0052cc" --repo "$OWNER/zelda-totk-guide"
gh label create "blocked"     --color "b60205" --repo "$OWNER/zelda-totk-guide"
```

---

## Step 5 - Milestones

```powershell
gh api "repos/$OWNER/zelda-totk-guide/milestones" --method POST --field title="1 - Data pipeline and knowledge base"
gh api "repos/$OWNER/zelda-totk-guide/milestones" --method POST --field title="2 - Model conversion and CI"
gh api "repos/$OWNER/zelda-totk-guide/milestones" --method POST --field title="3 - iOS app core"
gh api "repos/$OWNER/zelda-totk-guide/milestones" --method POST --field title="4 - Polish and distribution"
```

Verify and store milestone numbers:

```powershell
gh api "repos/$OWNER/zelda-totk-guide/milestones" --jq '.[] | {number, title}'

$M1 = 1   # 1 - Data pipeline and knowledge base
$M2 = 2   # 2 - Model conversion and CI
$M3 = 3   # 3 - iOS app core
$M4 = 4   # 4 - Polish and distribution
```

Update variables if the API returned different numbers.

---

## Step 6 - Issues

### Milestone 1 - Data pipeline and knowledge base

**ISSUE-001**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Python pipeline scaffold and dev environment" `
  --milestone $M1 `
  --label "story,pipeline,S" `
  --body "Set up the Python project structure for the data pipeline. Virtual environment, requirements, pytest config, and a smoke test confirming all modules import correctly.

## Acceptance criteria
- [ ] python -m venv .venv and pip install -r requirements.txt completes without error on Windows
- [ ] pytest pipeline/tests/ runs and passes (even with placeholder tests)
- [ ] pipeline/scrape/, pipeline/process/, pipeline/build_db/ all have __init__.py
- [ ] pipeline/config.py defines DATA_DIR, RAW_DIR, PROCESSED_DIR, DB_PATH as pathlib.Path constants
- [ ] Logging configured to write timestamped progress to console in all pipeline scripts"
```

**ISSUE-002**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Hyrule Compendium API scraper" `
  --milestone $M1 `
  --label "story,pipeline,data,S" `
  --body "Fetch all Tears of the Kingdom item, enemy, weapon, and region data from the Hyrule Compendium API.

API base: https://botw-compendium.herokuapp.com/api/v3

## Acceptance criteria
- [ ] Fetches all categories: creatures, equipment, materials, monsters, treasure
- [ ] Filters to TotK entries only
- [ ] Saves one JSON file per category to pipeline/data/raw/compendium/
- [ ] Handles API errors with retry: 3 attempts, exponential backoff
- [ ] Logs count of entries fetched per category
- [ ] Unit tests mock HTTP responses and verify filtering and file output"
```

**ISSUE-003**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Zelda Dungeon Wiki MediaWiki scraper" `
  --milestone $M1 `
  --label "story,pipeline,data,M" `
  --body "Fetch TotK walkthrough content from Zelda Dungeon Wiki using the MediaWiki API.

Target pages: main quest walkthroughs, shrine guides, side quest guides, collectible locations.
API: https://www.zeldadungeon.net/wiki/api.php

## Acceptance criteria
- [ ] Uses MediaWiki API action=query to list all pages in TotK categories
- [ ] Fetches raw wikitext for each page
- [ ] Minimum 1.5 second delay between requests
- [ ] User-Agent: ZeldaTotKGuide/1.0 (personal iOS app; contact via GitHub)
- [ ] Saves one .txt file per page to pipeline/data/raw/zelda_dungeon/
- [ ] Skips already-downloaded pages (resume-safe)
- [ ] Logs page count and any HTTP errors
- [ ] Unit tests use a local mock server - never hit the live wiki in tests"
```

**ISSUE-004**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Zelda Wiki MediaWiki scraper" `
  --milestone $M1 `
  --label "story,pipeline,data,M" `
  --body "Fetch TotK lore, NPC, and mechanic pages from Zelda Wiki (zeldawiki.wiki).

API: https://zeldawiki.wiki/api.php

## Acceptance criteria
- [ ] Uses MediaWiki API to list and fetch all pages in TotK categories
- [ ] Minimum 1.5 second delay between requests
- [ ] Same User-Agent header as issue #3
- [ ] Saves raw wikitext to pipeline/data/raw/zelda_wiki/
- [ ] Skips already-downloaded pages (resume-safe)
- [ ] Unit tests use a local mock server"
```

**ISSUE-005**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Text cleaning and chunking pipeline" `
  --milestone $M1 `
  --label "story,pipeline,data,M" `
  --body "Clean raw wikitext and compendium JSON into plain text chunks ready for embedding.

Wiki pages: strip MediaWiki markup with mwparserfromhell, remove navigation templates, infoboxes, categories, external links. Split into ~500 token chunks with 50-token overlap.
Compendium entries: format as readable text paragraphs (name, category, description, locations, properties).

## Acceptance criteria
- [ ] Wiki pages cleaned with mwparserfromhell - no raw markup in output
- [ ] Chunks are 400-600 tokens (whitespace tokenizer approximation)
- [ ] Each chunk includes metadata: source, page_title, chunk_index
- [ ] Compendium entries formatted as readable paragraphs, not raw JSON
- [ ] Stub pages under 100 tokens after cleaning are discarded
- [ ] Output is pipeline/data/processed/chunks.jsonl
- [ ] Unit tests verify markup removal and correct chunk overlap"
```

**ISSUE-006**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Embedding generation and SQLite vector DB build" `
  --milestone $M1 `
  --label "story,pipeline,data,L" `
  --body "Generate 384-dimension embeddings for all chunks using all-MiniLM-L6-v2 and store in a SQLite database with sqlite-vec. The resulting knowledge_base.db is bundled in the iOS app.

## Acceptance criteria
- [ ] all-MiniLM-L6-v2 generates float32 embeddings (384 dimensions) for each chunk
- [ ] Embeddings stored in SQLite using sqlite-vec virtual table
- [ ] Each row: id, chunk_text, embedding, source, page_title, chunk_index
- [ ] FTS5 full-text search table also created on chunk_text (hybrid search fallback)
- [ ] Final DB written to pipeline/data/knowledge_base.db
- [ ] DB file size under 150MB
- [ ] Query test: 'Where is the Hylian Shield' returns relevant results in top 5
- [ ] Progress bar (tqdm) shows chunks/sec during embedding"
```

**ISSUE-007**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Knowledge base validation and quality check" `
  --milestone $M1 `
  --label "story,pipeline,data,S" `
  --body "Run quality checks on knowledge_base.db before bundling in the iOS app.

## Acceptance criteria
- [ ] Total chunk count is between 5000 and 50000
- [ ] All three sources represented (compendium, zelda_dungeon, zelda_wiki)
- [ ] No empty or under-50-character chunk_text values
- [ ] Vector dimensions consistently 384 across all rows
- [ ] 10 sample queries printed with top-3 results for manual review
- [ ] Sample queries cover: item lookup, enemy location, shrine solution, main quest step, NPC name
- [ ] knowledge_base.db copied to ios/ZeldaGuide/Resources/"
```

---

### Milestone 2 - Model conversion and CI

**ISSUE-008**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Embedding model Core ML conversion" `
  --milestone $M2 `
  --label "story,model,ci,M" `
  --body "Convert all-MiniLM-L6-v2 to Core ML so the iOS app can generate query embeddings on-device.
Runs via GitHub Actions on a macos-14 runner.

## Acceptance criteria
- [ ] model/convert_embeddings.py converts all-MiniLM-L6-v2 to Core ML using coremltools
- [ ] Output: model/MiniLMEmbedder.mlpackage
- [ ] Model input: string text, output: float32 array of 384 dimensions
- [ ] Verified: Core ML embedding of 'Hylian Shield' matches Python output within float tolerance
- [ ] .mlpackage copied to ios/ZeldaGuide/Resources/
- [ ] GitHub Actions runs this script on macos-14 on workflow_dispatch"
```

**ISSUE-009**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "LLM Core ML conversion and quantization" `
  --milestone $M2 `
  --label "story,model,ci,XL" `
  --body "Download Llama 3.2 1B (default) or 3B (optional upgrade) from Hugging Face, convert to Core ML, apply 4-bit quantization. Controlled by MODEL_VARIANT workflow input.

Runs via GitHub Actions on a macos-14 runner. Output is uploaded as an artifact (not committed to git).

## Acceptance criteria
- [ ] model/convert_llm.py reads MODEL_VARIANT env var ('1B' default, '3B' optional)
- [ ] Downloads the correct model from HuggingFace using HF_TOKEN secret
- [ ] Converts to Core ML .mlpackage using coremltools stateful conversion
- [ ] Applies 4-bit palettization quantization
- [ ] 1B output is under 700MB, 3B output is under 2GB
- [ ] Smoke test generates 20 tokens from a short prompt and confirms coherent output
- [ ] .mlpackage uploaded as a GitHub Actions artifact
- [ ] README in model/ explains how to download the artifact and place it in ios/ZeldaGuide/Resources/
- [ ] ModelConfig.swift filename constants match the output filenames (LlamaModel-1B.mlpackage or LlamaModel-3B.mlpackage)"
```

**ISSUE-010**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "GitHub Actions CI workflows" `
  --milestone $M2 `
  --label "story,ci,M" `
  --body "Implement the three GitHub Actions workflows: data pipeline, model conversion, and iOS .ipa build.

## Acceptance criteria
- [ ] pipeline.yml: ubuntu-latest, Python setup, runs pipeline on sample data, runs pytest
- [ ] convert-model.yml: macos-14, MODEL_VARIANT input ('1B'/'3B'), converts model, uploads .mlpackage artifact
- [ ] build-ipa.yml: macos-14, builds unsigned .ipa with xcodebuild archive, uploads .ipa artifact
- [ ] All workflows trigger on workflow_dispatch only (not every push)
- [ ] HF_TOKEN configured as a GitHub Actions secret (documented in README)
- [ ] build-ipa.yml includes a comment block explaining how to add Apple signing for the 3B upgrade path
- [ ] pip dependencies cached between runs
- [ ] Workflow run summary shows model variant used and output file size"
```

---

### Milestone 3 - iOS app core

**ISSUE-011**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "iOS project scaffold and ModelConfig" `
  --milestone $M3 `
  --label "story,ios,S" `
  --body "Create the Xcode project, add Swift package dependencies, and confirm ModelConfig.swift correctly drives all model-related constants.

## Acceptance criteria
- [ ] Xcode project at ios/ZeldaGuide.xcodeproj, deployment target iOS 17.0
- [ ] Bundle identifier: com.{owner}.ZeldaGuide
- [ ] Swift packages added: sqlite-vec Swift package
- [ ] knowledge_base.db and MiniLMEmbedder.mlpackage added to app bundle resources
- [ ] ModelConfig.swift is the only file that references model filenames or generation parameters
- [ ] App builds and launches on iPhone 15 simulator showing a placeholder screen
- [ ] xcodebuild build succeeds from command line"
```

**ISSUE-012**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "On-device vector search service" `
  --milestone $M3 `
  --label "story,ios,M" `
  --body "Swift service that opens knowledge_base.db from the app bundle and performs cosine similarity search.

## Acceptance criteria
- [ ] VectorSearchService opens knowledge_base.db from Bundle.main
- [ ] search(query: [Float], topK: Int) -> [KnowledgeChunk] returns top-K chunks by cosine similarity
- [ ] KnowledgeChunk: id, chunkText, source, pageTitle, similarityScore
- [ ] Uses ModelConfig.embeddingDimensions (not hardcoded 384)
- [ ] Search completes under 200ms for topK=5 on iPhone 15 simulator
- [ ] Falls back to FTS5 keyword search if vector search returns 0 results
- [ ] XCTest unit tests verify correct ranking using a small test DB fixture"
```

**ISSUE-013**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "On-device LLM inference service" `
  --milestone $M3 `
  --label "story,ios,L" `
  --body "Swift service that loads the Core ML Llama model and generates streaming text output.

## Acceptance criteria
- [ ] LLMService loads the model from ModelConfig.modelFilename (not a hardcoded name)
- [ ] Model loaded asynchronously at app startup
- [ ] generate(prompt: String) -> AsyncStream<String> streams tokens as generated
- [ ] Generation can be cancelled mid-stream
- [ ] Max output tokens respects ModelConfig.maxOutputTokens
- [ ] Memory warning handler pauses generation and logs a warning
- [ ] XCTest unit test verifies the stream produces non-empty output for a short prompt"
```

**ISSUE-014**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "RAG engine - retrieval, prompt assembly, generation" `
  --milestone $M3 `
  --label "story,ios,M" `
  --body "Wire together vector search and LLM into a complete RAG pipeline.

## Acceptance criteria
- [ ] RAGEngine.answer(question: String) -> AsyncStream<String> is the single public entry point
- [ ] Generates query embedding on-device using Core ML MiniLM model
- [ ] Retrieves top ModelConfig.ragTopK chunks using VectorSearchService
- [ ] Assembles prompt using ModelConfig.systemPrompt + retrieved chunks
- [ ] Streams LLM output token by token
- [ ] Exposes source chunks alongside the answer stream for attribution UI
- [ ] End-to-end test: 'Where is the Hylian Shield?' produces an answer mentioning Hyrule Castle"
```

**ISSUE-015**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Chat UI - question input and streaming answer" `
  --milestone $M3 `
  --label "story,ios,L" `
  --body "Main chat interface in SwiftUI. Clean and readable with a subtle Zelda aesthetic.

## Acceptance criteria
- [ ] Scrollable message list with text input bar at the bottom
- [ ] User types a question and taps send or return
- [ ] Loading indicator shows while RAG engine retrieves chunks
- [ ] Answer streams in word by word as the LLM generates
- [ ] Stop button cancels generation mid-stream
- [ ] Previous Q&A shown in the session (not persisted between app launches)
- [ ] Dark mode fully supported
- [ ] Input bar disabled while response is generating"
```

**ISSUE-016**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "Source attribution view" `
  --milestone $M3 `
  --label "story,ios,S" `
  --body "Show the source chunks used to generate each answer. Required for GFDL attribution compliance.

## Acceptance criteria
- [ ] Each answer has a 'Sources' disclosure group below it
- [ ] Expanded view lists page titles and source wiki for top-3 retrieved chunks
- [ ] Tapping a source opens the wiki URL in SFSafariViewController
- [ ] Attribution text always visible: 'Content from Zelda Dungeon Wiki and Zelda Wiki, licenced under GFDL'"
```

---

### Milestone 4 - Polish and distribution

**ISSUE-017**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "AltStore sideload build and distribution docs" `
  --milestone $M4 `
  --label "story,ci,ios,M" `
  --body "Finalise the unsigned .ipa build workflow and write clear AltStore installation instructions.

## Acceptance criteria
- [ ] build-ipa.yml produces a valid unsigned .ipa that AltStore can sign and install
- [ ] docs/build.md covers: AltStore setup, downloading the .ipa artifact, sideloading, auto-renewal
- [ ] docs/upgrade-to-3b.md documents the exact steps to switch to the 3B model with a paid account
- [ ] README has a concise install section linking to docs/build.md
- [ ] The workflow run summary shows app size and which model variant was bundled
- [ ] Tested: .ipa installs successfully on a physical iPhone via AltStore"
```

**ISSUE-018**

```powershell
gh issue create `
  --repo "$OWNER/zelda-totk-guide" `
  --title "App performance tuning and model warm-up" `
  --milestone $M4 `
  --label "story,ios,M" `
  --body "Make the app feel fast and responsive. Model warm-up on launch, smooth streaming, graceful memory handling.

## Acceptance criteria
- [ ] LLM model loads in background immediately on app launch
- [ ] Subtle loading indicator in toolbar while model warms up
- [ ] First token appears within 5 seconds of submitting a question on iPhone 12
- [ ] Generation speed is at least 10 tokens/sec on iPhone 12 (A15)
- [ ] App does not crash or hang on memory warning during generation
- [ ] Vector search completes under 150ms on the full knowledge base
- [ ] App launch to ready state under 3 seconds on iPhone 12"
```

---

## Step 7 - GitHub Project board

```powershell
gh project create --owner $OWNER --title "Zelda TotK Guide"
```

Note the project number returned, then add all issues:

```powershell
$projectNumber = 1  # update if different
$issues = gh issue list --repo "$OWNER/zelda-totk-guide" --limit 25 --json number | ConvertFrom-Json
foreach ($issue in $issues) {
    gh project item-add $projectNumber --owner $OWNER --url "https://github.com/$OWNER/zelda-totk-guide/issues/$($issue.number)"
}
```

---

## Step 8 - Final verification

```powershell
Write-Host "=== Repo ==="
gh repo view "$OWNER/zelda-totk-guide" --json name,isPrivate

Write-Host "=== Labels (expect 14) ==="
gh label list --repo "$OWNER/zelda-totk-guide" --limit 20

Write-Host "=== Milestones (expect 4) ==="
gh api "repos/$OWNER/zelda-totk-guide/milestones" --jq '.[] | {number, title}'

Write-Host "=== Issues (expect 18) ==="
gh issue list --repo "$OWNER/zelda-totk-guide" --limit 25 --json number,title | ConvertFrom-Json | Format-Table

Write-Host "=== Branch protection ==="
gh api "repos/$OWNER/zelda-totk-guide/branches/main/protection" --jq '{allow_force_pushes, allow_deletions}'

Write-Host "=== Key files present ==="
Test-Path CLAUDE.md
Test-Path session-start.md
Test-Path ios/ZeldaGuide/Services/ModelConfig.swift
Test-Path docs/build.md
Test-Path docs/upgrade-to-3b.md
```

Expected: repo private, 14 labels, 4 milestones, 18 issues, branch protection active, all 5 files present.

Fix any failures before finishing. End with a summary of everything created.
