# Zelda: Tears of the Kingdom - Offline AI Guide

An iOS app that answers your TotK questions entirely on-device. No internet required after install.
Distributed free via AltStore - no Apple Developer account needed.

## How it works

1. A Python pipeline scrapes the Hyrule Compendium API, Zelda Dungeon Wiki, and Zelda Wiki
2. Content is chunked, embedded, and stored in a SQLite vector database
3. The database and a quantized Qwen2.5 model are bundled in the iOS app
4. On your phone, questions trigger a RAG pipeline: vector search finds relevant chunks,
   the on-device LLM generates a grounded answer

## Model variants

Default: Qwen2.5 1B (~750 MB app, works on any iPhone, free SideStore distribution)
Upgrade: Qwen2.5 3B (~1.8 GB app, better answer quality, also free via SideStore)

To switch models: change one constant in ModelConfig.swift and re-run the CI conversion job.
See [docs/upgrade-to-3b.md](docs/upgrade-to-3b.md) for the exact steps.

## Requirements

- iPhone running iOS 18 or later
- ~800 MB free storage (1B model)
- SideStore installed on your iPhone (free, no PC required after setup)

## Install

1. Install SideStore on your iPhone — see [docs/build.md](docs/build.md) for setup.
2. Run **Actions → Build iOS .ipa** in this repo and download the `ZeldaGuide.ipa` artifact.
3. Transfer the `.ipa` to your iPhone (iCloud Drive, AirDrop, etc.).
4. Open SideStore, tap **+**, select the `.ipa`, and install.

See [docs/build.md](docs/build.md) for detailed instructions including troubleshooting.

## GitHub Actions secrets

The following secret must be configured in your repository settings before running CI workflows
(Settings → Secrets and variables → Actions → New repository secret):

| Secret | Used by | Purpose |
|--------|---------|---------|
| `HF_TOKEN` | `convert-model.yml` | Authenticate with Hugging Face to download the Qwen2.5 model weights |

## Data sources

- Hyrule Compendium API (https://gadhagod.github.io/Hyrule-Compendium-API)
- Zelda Dungeon Wiki (https://www.zeldadungeon.net/wiki) - GFDL licence
- Zelda Wiki (https://zeldawiki.wiki) - GFDL licence
