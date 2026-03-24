# pipeline/process/clean.py
# Clean raw wikitext and compendium JSON into plain text ready for chunking.
#
# Wiki pages:  strip MediaWiki markup via mwparserfromhell, collapse whitespace.
# Compendium:  format each JSON entry as a readable prose paragraph.

import re
from typing import Any

import mwparserfromhell

# Wikitext templates whose sole purpose is navigation / boilerplate.
# After stripping, these leave behind empty or near-empty sections that add
# no value to a RAG corpus.
_BOILERPLATE_TEMPLATES = re.compile(
    r"^\s*(Stub|Navbox|Navigation|Navbar|"
    r"Spoiler|EndSpoiler|Clear|TOC|TOC right|"
    r"Languages|DEFAULTSORT)\s*$",
    re.IGNORECASE,
)

# Sections that are pure meta-content and not useful for answering questions.
_SKIP_SECTIONS = re.compile(
    r"^(References?|External links?|See also|Nomenclature|Gallery|"
    r"Trivia|Notes?|Navigation|In other languages?)$",
    re.IGNORECASE,
)

# Collapse runs of 3+ newlines into exactly two (one blank line).
_EXCESS_NEWLINES = re.compile(r"\n{3,}")

# Strip leading/trailing whitespace from each line, then drop blank runs.
_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)


def _remove_skip_sections(wikicode: Any) -> None:
    """Remove entire sections whose heading matches _SKIP_SECTIONS in-place."""
    for section in wikicode.get_sections(levels=[2, 3], include_headings=True):
        headings = section.filter_headings()
        if headings and _SKIP_SECTIONS.match(headings[0].title.strip()):
            wikicode.remove(section)


def clean_wiki_page(wikitext: str) -> str:
    """Return plain text from a MediaWiki page, or '' if the page is empty."""
    wikicode = mwparserfromhell.parse(wikitext)

    # Drop boilerplate templates (Stub, Navbox, …) by name.
    for template in wikicode.filter_templates():
        name = template.name.strip()
        if _BOILERPLATE_TEMPLATES.match(name):
            try:
                wikicode.remove(template)
            except ValueError:
                pass  # already removed as part of a parent node

    # Drop entire meta-sections (References, Gallery, …).
    _remove_skip_sections(wikicode)

    # strip_code() removes all remaining markup: templates, links, tags, refs.
    text = wikicode.strip_code(
        normalize=True,
        collapse=True,
        keep_template_params=False,
    )

    # Clean up whitespace artefacts.
    text = _TRAILING_SPACES.sub("", text)
    text = _EXCESS_NEWLINES.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Compendium formatting
# ---------------------------------------------------------------------------

_CATEGORY_LABELS: dict[str, str] = {
    "creatures": "Creature",
    "equipment": "Equipment",
    "materials": "Material",
    "monsters": "Monster",
    "treasure": "Treasure",
}


def _title_case(name: str) -> str:
    return name.replace("_", " ").title()


def format_compendium_entry(entry: dict[str, Any], source_category: str) -> str:
    """Format a single Hyrule Compendium entry as a readable prose paragraph."""
    kind = _CATEGORY_LABELS.get(source_category, source_category.capitalize())
    name = _title_case(entry.get("name", "Unknown"))
    description = (entry.get("description") or "").strip()

    parts: list[str] = [f"{name} is a {kind} in Tears of the Kingdom."]

    if description:
        parts.append(description)

    locations: list[str] = entry.get("common_locations") or []
    if locations:
        loc_str = ", ".join(locations)
        parts.append(f"It can be found in: {loc_str}.")

    drops: list[str] = entry.get("drops") or []
    if drops:
        drop_str = ", ".join(drops)
        parts.append(f"It drops: {drop_str}.")

    # Equipment-specific fields
    attack = entry.get("attack")
    defense = entry.get("defense")
    if attack is not None:
        parts.append(f"Attack power: {attack}.")
    if defense is not None:
        parts.append(f"Defense power: {defense}.")

    return " ".join(parts)
