# pipeline/generate_walkthrough_content.py
# Reads chunks.jsonl (read-only) and populates walkthrough section content in
# WalkthroughData.swift using keyword scoring.
#
# Usage:
#   python -m pipeline.generate_walkthrough_content
#
# Reads:  pipeline/data/processed/chunks.jsonl  (read-only, not modified)
# Writes: ios/ZeldaWalkthrough/ZeldaWalkthrough/Models/WalkthroughData.swift

import json
import re
import textwrap
from pathlib import Path

from pipeline.config import PROCESSED_DIR, get_logger

log = get_logger(__name__)

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
SWIFT_OUT = (
    Path(__file__).parent.parent
    / "ios/ZeldaWalkthrough/ZeldaWalkthrough/Models/WalkthroughData.swift"
)

MAX_CONTENT_CHARS = 1200
MAX_CHUNK_CHARS = 420
CHUNK_LIMIT = 6

# ── Search queries ────────────────────────────────────────────────────────────
# Maps section id → list of keyword strings. A chunk scores 1 point per keyword
# found (case-insensitive). The first keyword in each list is the primary term
# and scores 3 points if found in page_title.
SECTION_QUERIES: dict[str, list[str]] = {
    # Getting Started
    "gs-controls":  ["controls", "Purah Pad", "abilities", "map", "inventory", "interface"],
    "gs-abilities": ["Ultrahand", "Fuse", "Ascend", "Recall", "abilities"],
    "gs-tips":      ["hearts", "stamina", "weapons", "armor", "food", "cooking", "tips"],
    # Great Sky Island
    "gsi-wake":      ["Great Sky Island", "awakening", "Rauru", "Sonia", "Zonai"],
    "gsi-ukouh":     ["Ukouh Shrine", "Ultrahand", "construct", "hook"],
    "gsi-inisa":     ["In-isa Shrine", "Fuse", "weapon", "flint"],
    "gsi-gutanbac":  ["Gutanbac Shrine", "Ascend", "ceiling"],
    "gsi-nachoyah":  ["Nachoyah Shrine", "Recall", "rewind", "time"],
    "gsi-temple":    ["Temple of Time", "Rauru", "Construct", "sky island"],
    "gsi-descent":   ["descend", "fall", "dive", "Hyrule", "paraglider"],
    # Arriving in Hyrule
    "aih-lookout":   ["Lookout Landing", "Purah", "tower", "survey"],
    "aih-shelter":   ["Emergency Shelter", "underground", "cave", "Hyrule Castle"],
    "aih-camera":    ["Camera Work", "Depths", "photograph", "Gloom"],
    "aih-castle":    ["Hyrule Castle", "Gloom", "Phantom Ganon", "moat"],
    "aih-phenomena": ["Regional Phenomena", "temple", "sage", "four regions"],
    # Wind Temple
    "wt-rito":   ["Rito Village", "Hebra", "snowstorm", "blizzard"],
    "wt-tulin":  ["Tulin", "Rito", "companion", "wind gust"],
    "wt-storm":  ["blizzard", "storm", "snowfield", "Hebra Mountains"],
    "wt-temple": ["Wind Temple", "propeller", "fan", "puzzle"],
    "wt-boss":   ["Colgera", "Wind Temple", "ice", "boss"],
    # Water Temple
    "wat-zora":    ["Zora's Domain", "sludge", "rain", "Zora"],
    "wat-sidon":   ["Sidon", "Zora", "water", "sage"],
    "wat-sludge":  ["sludge", "Zora's Domain", "clear", "Gloom"],
    "wat-temple":  ["Water Temple", "water flow", "valve", "puzzle"],
    "wat-boss":    ["Mucktorok", "Water Temple", "sludge", "boss"],
    # Fire Temple
    "ft-goron":   ["Goron City", "Yunobo", "marbled rock", "stone"],
    "ft-yunobo":  ["Yunobo", "Goron", "sage", "fire", "roll"],
    "ft-mine":    ["YunoboCo", "mine", "coal", "cart", "Goron"],
    "ft-temple":  ["Fire Temple", "minecart", "gong", "fire", "puzzle"],
    "ft-boss":    ["Marbled Gohma", "Fire Temple", "eye", "boss"],
    # Lightning Temple
    "lt-gerudo":   ["Gerudo Town", "voe", "disguise", "Gerudo"],
    "lt-riju":     ["Riju", "Gerudo", "lightning", "sage"],
    "lt-yiga":     ["Yiga Clan", "Kohga", "hideout", "banana"],
    "lt-temple":   ["Lightning Temple", "mirror", "light", "puzzle"],
    "lt-boss":     ["Queen Gibdo", "Lightning Temple", "Gibdo", "boss"],
    # Master Sword
    "ms-tears":    ["Dragon's Tears", "memories", "Zelda", "geoglyph"],
    "ms-rings":    ["Ring Ruins", "Dragonhead Island", "Mineru", "ancient"],
    "ms-dragon":   ["Light Dragon", "Master Sword", "sky", "dragon"],
    "ms-obtain":   ["Master Sword", "obtain", "sword", "pull"],
    # Hyrule Castle
    "hc-crisis":    ["Hyrule Castle", "Gloom", "Phantom Ganon", "crisis"],
    "hc-guidance":  ["Rauru", "sages", "spirit", "guidance", "ages past"],
    "hc-ascent":    ["Hyrule Castle", "ascend", "throne room", "Ganondorf"],
    # Final Battle
    "fb-gloom":    ["Gloom", "Chasm", "Depths", "secret stone"],
    "fb-army":     ["Demon King", "army", "Ganondorf", "Gloom Hands"],
    "fb-king":     ["Ganondorf", "Demon King", "boss", "final"],
    "fb-dragons":  ["dragon", "showdown", "Zelda", "Light Dragon", "Ganondorf"],
    # Shrines
    "shr-sky":      ["shrine", "Zonai", "sky", "Ukouh", "Gutanbac", "Nachoyah"],
    "shr-eldin":    ["Eldin", "Akkala", "shrine", "volcano"],
    "shr-lanayru":  ["Lanayru", "Necluda", "shrine"],
    "shr-faron":    ["Faron", "shrine", "jungle", "south"],
    "shr-central":  ["Central Hyrule", "shrine", "Lookout Landing"],
    "shr-gerudo":   ["Gerudo", "Rito", "shrine", "west"],
    "shr-depths":   ["Depths", "shrine", "Lightroot", "underground"],
    # Side Quests
    "sq-stables":   ["stable", "quest", "horse", "taming"],
    "sq-addison":   ["Addison", "Hudson", "sign", "prop"],
    "sq-zelda":     ["Potential Princess Sightings", "Zelda", "sighting"],
    "sq-lurelin":   ["Lurelin Village", "restoration", "quest"],
    "sq-phantom":   ["Phantom Ganon", "armor", "set", "Gloom"],
    # Side Adventures
    "sa-sages":     ["Sage's Will", "crystal", "light", "upgrade"],
    "sa-dispelling": ["Gloom", "dispel", "underground", "Depths"],
    "sa-construct": ["Steward Construct", "quest", "reward"],
    # Collectibles
    "col-korok":    ["Korok", "seed", "Hestu", "maracas"],
    "col-bubbul":   ["Bubbulfrog", "gem", "Koltin", "cave"],
    "col-bargainer": ["Bargainer Statue", "poe", "soul", "Depths"],
    "col-bosses":   ["Hinox", "Talus", "Molduga", "overworld boss"],
    "col-dragons":  ["dragon", "parts", "scales", "claws", "horn", "farming"],
    # Tips & Tricks
    "tt-combat":    ["combat", "flurry rush", "parry", "dodge", "battle"],
    "tt-building":  ["Ultrahand", "vehicle", "build", "attach", "construct"],
    "tt-rupees":    ["rupees", "money", "farming", "ore", "sell"],
    "tt-cooking":   ["cooking", "recipe", "food", "elixir", "effect"],
    "tt-armor":     ["armor", "upgrade", "Great Fairy", "defense", "set bonus"],
    # Equipment
    "eq-weapons":   ["weapon", "fusion", "fuse", "material", "durability"],
    "eq-shields":   ["shield", "surf", "bash", "fuse", "parry"],
    "eq-bows":      ["bow", "arrow", "multishot", "spread", "snipe"],
    "eq-special":   ["special weapon", "unique", "unbreakable", "rare", "legendary"],
}


# ── Keyword scoring ───────────────────────────────────────────────────────────

def _score_chunk(chunk: dict, keywords: list[str]) -> float:
    text_lower = chunk["text"].lower()
    title_lower = chunk["page_title"].lower()
    score = 0.0
    primary = keywords[0].lower()
    if primary in title_lower:
        score += 3.0
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in text_lower:
            score += 1.0
        if kw_lower in title_lower:
            score += 1.0
    return score


def _search(chunks: list[dict], keywords: list[str], limit: int = CHUNK_LIMIT) -> list[dict]:
    scored = [(c, _score_chunk(c, keywords)) for c in chunks]
    scored = [(c, s) for c, s in scored if s > 0]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored[:limit]]


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_chunk(text: str) -> str:
    text = re.sub(r'\[\[(File|Image):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'={2,}[^=]+=+', '', text)
    text = re.sub(r"'{2,3}", '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'^\s*[|!{][|!}].*$', '', text, flags=re.MULTILINE)
    # Strip empty quoted title patterns like: "" is a Main Quest in .
    text = re.sub(r'^""\s+is\s+a\s+\S[^.]*\.', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _trim_to_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_end = max(
        truncated.rfind('. '),
        truncated.rfind('.\n'),
        truncated.rfind('! '),
        truncated.rfind('? '),
    )
    if last_end > max_chars // 2:
        return truncated[:last_end + 1].rstrip()
    return truncated.rstrip()


def _assemble_content(chunks: list[dict]) -> str:
    seen_titles: set[str] = set()
    parts: list[str] = []
    total = 0

    for chunk in chunks:
        cleaned = _clean_chunk(chunk["text"])
        if not cleaned or len(cleaned) < 60:
            continue
        if chunk["page_title"] in seen_titles and len(parts) >= 2:
            continue
        seen_titles.add(chunk["page_title"])

        trimmed = _trim_to_sentence(cleaned, MAX_CHUNK_CHARS)
        remaining = MAX_CONTENT_CHARS - total
        if len(trimmed) > remaining:
            trimmed = _trim_to_sentence(trimmed, remaining)
            if len(trimmed) < 60:
                break
        parts.append(trimmed)
        total += len(trimmed)
        if total >= MAX_CONTENT_CHARS:
            break

    return '\n\n'.join(parts)


# ── Swift generation ──────────────────────────────────────────────────────────

def _swift_str(text: str) -> str:
    text = text.replace('\\', '\\\\')
    text = text.replace('"', '\\"')
    text = text.replace('\n', '\\n')
    return text


def _generate_swift(section_content: dict[str, str]) -> str:
    def section_line(sid: str, title: str) -> str:
        content = _swift_str(section_content.get(sid, ""))
        return f'            WalkthroughSection(id: "{sid}", title: "{title}", content: "{content}"),'

    def chapter_block(cid: str, title: str, category: str, icon: str, sections: list[tuple[str, str]]) -> str:
        sec_lines = '\n'.join(section_line(sid, stitle) for sid, stitle in sections)
        return (
            f'        WalkthroughChapter(\n'
            f'            id: "{cid}",\n'
            f'            title: "{title}",\n'
            f'            category: .{category},\n'
            f'            icon: "{icon}",\n'
            f'            sections: [\n'
            f'{sec_lines}\n'
            f'            ]\n'
            f'        ),'
        )

    chapters = '\n\n'.join([
        chapter_block("getting-started", "Getting Started", "mainQuest", "star.circle", [
            ("gs-controls",  "Controls & UI Overview"),
            ("gs-abilities", "Link's Core Abilities"),
            ("gs-tips",      "Early Game Tips"),
        ]),
        chapter_block("great-sky-island", "The Great Sky Island", "mainQuest", "cloud", [
            ("gsi-wake",     "Waking Up"),
            ("gsi-ukouh",    "Ukouh Shrine — Ultrahand"),
            ("gsi-inisa",    "In-isa Shrine — Fuse"),
            ("gsi-gutanbac", "Gutanbac Shrine — Ascend"),
            ("gsi-nachoyah", "Nachoyah Shrine — Recall"),
            ("gsi-temple",   "Reach the Temple of Time"),
            ("gsi-descent",  "Descend to Hyrule"),
        ]),
        chapter_block("arriving-in-hyrule", "Arriving in Hyrule", "mainQuest", "mappin.circle", [
            ("aih-lookout",   "Lookout Landing"),
            ("aih-shelter",   "Emergency Shelter"),
            ("aih-camera",    "Camera Work in the Depths"),
            ("aih-castle",    "Infiltrating Hyrule Castle"),
            ("aih-phenomena", "The Regional Phenomena"),
        ]),
        chapter_block("wind-temple", "Rito Village & the Wind Temple", "mainQuest", "wind", [
            ("wt-rito",   "Reach Rito Village"),
            ("wt-tulin",  "Tulin of Rito Village"),
            ("wt-storm",  "Approach the Storm"),
            ("wt-temple", "Wind Temple — Puzzles"),
            ("wt-boss",   "Boss: Colgera"),
        ]),
        chapter_block("water-temple", "Zora's Domain & the Water Temple", "mainQuest", "drop", [
            ("wat-zora",   "Reach Zora's Domain"),
            ("wat-sidon",  "Sidon of the Zora"),
            ("wat-sludge", "Clear the Sludge"),
            ("wat-temple", "Water Temple — Puzzles"),
            ("wat-boss",   "Boss: Mucktorok"),
        ]),
        chapter_block("fire-temple", "Goron City & the Fire Temple", "mainQuest", "flame", [
            ("ft-goron",  "Reach Goron City"),
            ("ft-yunobo", "Yunobo of Goron City"),
            ("ft-mine",   "YunoboCo HQ & the Mine"),
            ("ft-temple", "Fire Temple — Puzzles"),
            ("ft-boss",   "Boss: Marbled Gohma"),
        ]),
        chapter_block("lightning-temple", "Gerudo Town & the Lightning Temple", "mainQuest", "bolt", [
            ("lt-gerudo", "Reach Gerudo Town"),
            ("lt-riju",   "Riju of Gerudo Town"),
            ("lt-yiga",   "Infiltrate the Yiga Clan Hideout"),
            ("lt-temple", "Lightning Temple — Puzzles"),
            ("lt-boss",   "Boss: Queen Gibdo"),
        ]),
        chapter_block("master-sword", "The Master Sword", "mainQuest", "sparkles", [
            ("ms-tears",  "The Dragon's Tears — All Memories"),
            ("ms-rings",  "Secret of the Ring Ruins"),
            ("ms-dragon", "Reach the Light Dragon"),
            ("ms-obtain", "Obtain the Master Sword"),
        ]),
        chapter_block("hyrule-castle", "Hyrule Castle", "mainQuest", "building.columns", [
            ("hc-crisis",   "Crisis at Hyrule Castle"),
            ("hc-guidance", "Guidance from Ages Past"),
            ("hc-ascent",   "Ascend the Castle"),
        ]),
        chapter_block("final-battle", "The Final Battle", "mainQuest", "crown", [
            ("fb-gloom",   "Gloom's Origin"),
            ("fb-army",    "The Demon King's Army"),
            ("fb-king",    "Boss: The Demon King"),
            ("fb-dragons", "The Dragon Showdown"),
        ]),
        chapter_block("shrines", "Shrines", "sideContent", "diamond", [
            ("shr-sky",     "Sky Islands Shrines"),
            ("shr-eldin",   "Eldin & Akkala Shrines"),
            ("shr-lanayru", "Lanayru & Necluda Shrines"),
            ("shr-faron",   "Faron Shrines"),
            ("shr-central", "Central Hyrule Shrines"),
            ("shr-gerudo",  "Gerudo & Rito Shrines"),
            ("shr-depths",  "Depths Shrines"),
        ]),
        chapter_block("side-quests", "Side Quests", "sideContent", "list.bullet.clipboard", [
            ("sq-stables", "Stable Quests"),
            ("sq-addison", "Addison's Hudson Signs"),
            ("sq-zelda",   "Potential Princess Sightings"),
            ("sq-lurelin", "Lurelin Village Restoration"),
            ("sq-phantom", "The Phantom Ganon Armor"),
        ]),
        chapter_block("side-adventures", "Side Adventures", "sideContent", "figure.walk", [
            ("sa-sages",      "Sage's Will Locations"),
            ("sa-dispelling", "Dispelling Gloom"),
            ("sa-construct",  "Steward Construct Quests"),
        ]),
        chapter_block("collectibles", "Collectibles & Completion", "sideContent", "star.circle.fill", [
            ("col-korok",     "Korok Seeds (1000)"),
            ("col-bubbul",    "Bubbulfrogs"),
            ("col-bargainer", "Bargainer Statues"),
            ("col-bosses",    "Hinoxes, Taluses & Moldugas"),
            ("col-dragons",   "Dragon Parts Farming"),
        ]),
        chapter_block("tips-and-tricks", "Tips & Tricks", "reference", "lightbulb", [
            ("tt-combat",   "Combat Tips"),
            ("tt-building", "Ultrahand & Building"),
            ("tt-rupees",   "Farming Rupees"),
            ("tt-cooking",  "Cooking & Elixirs"),
            ("tt-armor",    "Best Armor Sets"),
        ]),
        chapter_block("equipment", "Weapons & Equipment", "reference", "shield", [
            ("eq-weapons", "Best Weapon Fusions"),
            ("eq-shields", "Shield Surfing & Fusions"),
            ("eq-bows",    "Bow & Arrow Types"),
            ("eq-special", "Unique & Unbreakable Weapons"),
        ]),
    ])

    return (
        '// WalkthroughData.swift\n'
        '// Static chapter and section structure for the TotK walkthrough.\n'
        '// Content populated from chunks.jsonl via keyword search.\n'
        '// Re-generate with: python -m pipeline.generate_walkthrough_content\n'
        '\n'
        'import Foundation\n'
        '\n'
        'enum ChapterCategory: String, CaseIterable {\n'
        '    case mainQuest = "Main Quest"\n'
        '    case sideContent = "Side Content"\n'
        '    case reference = "Reference"\n'
        '}\n'
        '\n'
        'struct WalkthroughSection: Identifiable, Hashable {\n'
        '    let id: String\n'
        '    let title: String\n'
        '    var content: String = ""\n'
        '}\n'
        '\n'
        'struct WalkthroughChapter: Identifiable {\n'
        '    let id: String\n'
        '    let title: String\n'
        '    let category: ChapterCategory\n'
        '    let icon: String\n'
        '    let sections: [WalkthroughSection]\n'
        '}\n'
        '\n'
        'enum WalkthroughData {\n'
        '    static let chapters: [WalkthroughChapter] = [\n'
        '\n'
        '        // MARK: - Main Quest\n'
        '\n'
        f'{chapters}\n'
        '\n'
        '    ]\n'
        '\n'
        '    static func chapters(for category: ChapterCategory) -> [WalkthroughChapter] {\n'
        '        chapters.filter { $0.category == category }\n'
        '    }\n'
        '}\n'
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def run(chunks_path: Path = CHUNKS_PATH, out_path: Path = SWIFT_OUT) -> None:
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at {chunks_path}")

    log.info("Loading chunks from %s", chunks_path)
    with open(chunks_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d chunks", len(chunks))

    section_content: dict[str, str] = {}
    populated = 0

    for section_id, keywords in SECTION_QUERIES.items():
        hits = _search(chunks, keywords)
        content = _assemble_content(hits)
        section_content[section_id] = content
        if content:
            populated += 1
            log.info("  %-20s  %d chars  (%d hits)", section_id, len(content), len(hits))
        else:
            log.warning("  %-20s  no content found", section_id)

    conn_note = f"Populated {populated} / {len(SECTION_QUERIES)} sections"
    log.info(conn_note)

    swift_source = _generate_swift(section_content)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(swift_source, encoding="utf-8")
    log.info("Wrote %s (%d bytes)", out_path, len(swift_source))


if __name__ == "__main__":
    run()
