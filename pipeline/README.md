# Data pipeline

Builds the knowledge base SQLite file from three sources. Runs on Windows (PowerShell).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run all steps

```powershell
python scrape/hyrule_compendium.py    # Fetch all item/enemy/weapon data
python scrape/zelda_dungeon.py        # Fetch walkthrough content
python scrape/zelda_wiki.py           # Fetch lore and NPC content
python process/clean_and_chunk.py    # Clean wikitext, chunk to ~500 tokens
python build_db/embed_and_store.py   # Generate embeddings, build SQLite DB
python build_db/validate.py          # Quality checks - run before committing DB
```

## Output

data/knowledge_base.db - committed to repo, bundled in iOS app

## Run tests

```powershell
pytest tests/ -v
```
