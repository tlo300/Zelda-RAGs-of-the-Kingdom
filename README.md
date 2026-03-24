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
