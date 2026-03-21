"""Stream-parse Wikiquote XML dump and extract clean quotes."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import click
import mwparserfromhell

from wikiquotes_tagger import db

# MediaWiki XML namespace
MW_NS = "http://www.mediawiki.org/xml/export-0.11/"

# Section titles that contain valid quotes (lowercase for matching)
QUOTE_SECTIONS = frozenset({
    "quotes",
    "sourced",
    "sourced quotes",
    "attributed",
    "attributed quotes",
    "unsorted",
})

# Section titles to skip entirely (lowercase for matching)
SKIP_SECTIONS = frozenset({
    "misattributed",
    "misattributed quotes",
    "disputed",
    "disputed quotes",
    "external links",
    "see also",
    "references",
    "notes",
    "bibliography",
    "further reading",
    "works",
    "publications",
    "filmography",
    "discography",
    "cast",
    "dialogue",
    "tagline",
    "taglines",
    "lyrics",
    "cast and characters",
})

# Prefixes that indicate "about" sections (not quotes BY the person)
ABOUT_PREFIXES = ("quotes about", "about ")

# Sections that indicate a media/non-person page
MEDIA_INDICATOR_SECTIONS = frozenset({
    "cast",
    "dialogue",
    "tagline",
    "taglines",
    "lyrics",
    "cast and characters",
    "episodes",
    "season ",
    "track listing",
})

MIN_QUOTE_LENGTH = 20
MAX_QUOTE_LENGTH = 500

# Regex patterns for cleanup
RE_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL)
RE_REF_SELF_CLOSE = re.compile(r"<ref[^/]*/\s*>")
RE_HTML_TAG = re.compile(r"<[^>]+>")
RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RE_MULTI_SPACE = re.compile(r"\s+")
RE_TEMPLATE = re.compile(r"\{\{[^}]*\}\}")

# --- Literary work detection ---

# Regex patterns for structural section headings in literary works
RE_ACT = re.compile(r"^act\s+[ivxlc\d]+", re.IGNORECASE)
RE_PART = re.compile(
    r"^part\s+(?:[ivxlc\d]+|one|two|three|four|five|six|seven|eight|nine|ten)",
    re.IGNORECASE,
)
RE_BOOK = re.compile(r"^book\s+[ivxlc\d]+", re.IGNORECASE)
RE_CHAPTER = re.compile(r"^chapters?\s+\d", re.IGNORECASE)

LITERARY_EXACT_SECTIONS = frozenset({
    "prologue", "epilogue", "appendix", "introduction", "preface",
})

# Extract author from literary work intro: "is a [genre] by [[Author]]"
RE_INTRO_AUTHOR = re.compile(
    r"\bis\s+(?:a|an)\b.{0,80}?"
    r"\b(?:novel|play|tragedy|comedy|poem|book|epic|novella|satire|fable|"
    r"allegory|romance|story|stories|collection|memoir|treatise|essay|dialogue|parable)"
    r"\b.{0,50}?"
    r"\bby\b.{0,80}?"
    r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]",
    re.IGNORECASE | re.DOTALL,
)

# Detect film/TV intro to avoid misclassifying as literary work
RE_INTRO_FILM = re.compile(
    r"\bis\s+(?:a|an)\b.{0,50}?\b(?:film|television|TV\s+series)\b",
    re.IGNORECASE,
)

# Theme page detection: alphabetical single-letter section headings
RE_ALPHA_HEADING = re.compile(r"^==\s*[A-Z]\s*==$", re.MULTILINE)


@dataclass
class RawQuote:
    text: str
    author: str
    source_work: str | None
    source_confidence: str


def parse_dump(xml_path: Path) -> Iterator[RawQuote]:
    """Stream-parse the XML dump, yielding RawQuote for each valid quote.

    Uses iterparse with start+end events on <page> elements.
    Clears elements after processing to keep memory constant.
    Filters to namespace 0 (main articles) only.
    """
    ns = ""
    root = None

    def tag(name: str) -> str:
        return f"{ns}{name}"

    for event, elem in ET.iterparse(str(xml_path), events=("start", "end")):
        # Capture root element and detect namespace from the first start event
        if root is None:
            root = elem
            if "}" in elem.tag:
                ns = elem.tag.split("}")[0] + "}"
            continue

        # Only process end events for <page> elements
        if event != "end" or elem.tag != tag("page"):
            continue

        # Extract namespace — only process main namespace (0)
        ns_elem = elem.find(tag("ns"))
        if ns_elem is None or ns_elem.text != "0":
            elem.clear()
            root.clear()
            continue

        # Extract title
        title_elem = elem.find(tag("title"))
        title = title_elem.text if title_elem is not None else None
        if not title:
            elem.clear()
            root.clear()
            continue

        # Extract wikitext from the latest revision
        revision = elem.find(tag("revision"))
        if revision is None:
            elem.clear()
            root.clear()
            continue

        text_elem = revision.find(tag("text"))
        wikitext = text_elem.text if text_elem is not None else None
        if not wikitext:
            elem.clear()
            root.clear()
            continue

        # Extract quotes from this page
        yield from _extract_quotes_from_page(title, wikitext)

        # Clear processed elements to free memory
        elem.clear()
        root.clear()


def _is_theme_page(wikitext: str) -> bool:
    """Detect theme/concept pages organized alphabetically (e.g., Love, War)."""
    lower = wikitext.lower()
    if "{{tocalpha}}" in lower or "{{tocright}}" in lower:
        return True
    alpha_headings = RE_ALPHA_HEADING.findall(wikitext)
    return len(alpha_headings) >= 5


def _is_literary_section(section_name: str) -> bool:
    """Check if a section heading indicates literary work structure."""
    if section_name in LITERARY_EXACT_SECTIONS:
        return True
    return any(p.match(section_name) for p in (RE_ACT, RE_PART, RE_BOOK, RE_CHAPTER))


def _extract_intro_author(wikitext: str) -> str | None:
    """Extract author from a literary work page's intro text.

    Literary work intros follow the pattern:
        '''''Title''''' is a novel by [[Author]]
    Returns the author name if found, None otherwise.
    """
    intro = wikitext[:1000]
    # Reject films/TV — they also use {{italic title}} and similar intros
    if RE_INTRO_FILM.search(intro):
        return None
    match = RE_INTRO_AUTHOR.search(intro)
    if match:
        return match.group(1).strip()
    return None


def _extract_quotes_from_page(title: str, wikitext: str) -> list[RawQuote]:
    """Parse wikitext for a single page and extract quotes.

    Handles person pages (author = page title) and literary work pages
    (author extracted from intro, source_work = page title).
    Skips media pages (films/TV) and theme pages (Love, War).
    """
    # Quick pre-filter: skip redirects
    if wikitext.strip().upper().startswith("#REDIRECT"):
        return []

    # Skip pages that look like media (films, TV, music)
    lower_text = wikitext.lower()
    for indicator in MEDIA_INDICATOR_SECTIONS:
        if f"== {indicator}" in lower_text or f"=={indicator}" in lower_text:
            return []

    # Skip theme pages (alphabetical organization)
    if _is_theme_page(wikitext):
        return []

    # Detect literary work pages and extract author from intro
    intro_author = _extract_intro_author(wikitext)
    is_literary_work = intro_author is not None

    if is_literary_work:
        author = intro_author
    else:
        author = title

    try:
        parsed = mwparserfromhell.parse(wikitext)
    except Exception:
        return []

    sections = parsed.get_sections(levels=[2], include_lead=False)
    quotes: list[RawQuote] = []
    has_valid_section = False

    for section in sections:
        headings = section.filter_headings()
        if not headings:
            continue

        section_name = headings[0].title.strip_code().strip().lower()

        # Skip non-quote sections
        if _should_skip_section(section_name):
            continue

        # Determine if this section contains quotes
        if is_literary_work:
            # Accept standard quote sections AND structural sections (Act, Part, etc.)
            if section_name not in QUOTE_SECTIONS and not _is_literary_section(section_name):
                continue
        else:
            # Person pages: only accept known quote sections
            if section_name not in QUOTE_SECTIONS:
                continue

        has_valid_section = True
        confidence = _section_confidence(section_name)

        extracted = _extract_from_section(author, str(section), confidence)
        if is_literary_work:
            for q in extracted:
                q.source_work = title
        quotes.extend(extracted)

    if not has_valid_section:
        return []

    return quotes


def _extract_from_section(
    author: str,
    section_text: str,
    confidence: str,
) -> list[RawQuote]:
    """Extract individual quotes from a wiki section's text.

    Lines starting with '* ' are quote text.
    Lines starting with '** ' are attribution/source_work.
    """
    quotes: list[RawQuote] = []
    current_quote: str | None = None
    current_source: str | None = None

    for line in section_text.split("\n"):
        stripped = line.strip()

        # Attribution/source line (must check before quote line since ** starts with *)
        if stripped.startswith("** ") or stripped.startswith("**\t"):
            if current_quote is not None and current_source is None:
                source_text = stripped.lstrip("*").strip()
                current_source = _clean_wikitext(source_text)
                if len(current_source) < 3:
                    current_source = None
            continue

        # Quote line
        if stripped.startswith("* ") or stripped.startswith("*\t"):
            # Save previous quote if any
            if current_quote is not None:
                _maybe_add_quote(quotes, current_quote, author, current_source, confidence)

            raw_text = stripped.lstrip("*").strip()
            cleaned = _clean_wikitext(raw_text)
            if cleaned:
                current_quote = cleaned
                current_source = None
            else:
                current_quote = None
                current_source = None
            continue

        # Lines starting with '#' are also used as quote markers on some pages
        if stripped.startswith("# ") and not stripped.startswith("##"):
            if current_quote is not None:
                _maybe_add_quote(quotes, current_quote, author, current_source, confidence)

            raw_text = stripped.lstrip("#").strip()
            cleaned = _clean_wikitext(raw_text)
            if cleaned:
                current_quote = cleaned
                current_source = None
            else:
                current_quote = None
                current_source = None
            continue

    # Don't forget the last quote
    if current_quote is not None:
        _maybe_add_quote(quotes, current_quote, author, current_source, confidence)

    return quotes


def _maybe_add_quote(
    quotes: list[RawQuote],
    text: str,
    author: str,
    source_work: str | None,
    confidence: str,
) -> None:
    """Add a quote to the list if it passes length filters."""
    if len(text) < MIN_QUOTE_LENGTH or len(text) > MAX_QUOTE_LENGTH:
        return
    quotes.append(RawQuote(
        text=text,
        author=author,
        source_work=source_work,
        source_confidence=confidence,
    ))


def _clean_wikitext(text: str) -> str:
    """Strip wiki markup to produce clean plain text."""
    # Remove HTML comments first
    text = RE_HTML_COMMENT.sub("", text)

    # Remove <ref> tags and their content before mwparserfromhell sees them
    text = RE_REF.sub("", text)
    text = RE_REF_SELF_CLOSE.sub("", text)

    # Use mwparserfromhell for structured cleanup
    try:
        parsed = mwparserfromhell.parse(text)

        # Remove templates ({{cite}}, {{lang}}, etc.) — do this before strip_code
        for template in parsed.filter_templates():
            try:
                parsed.remove(template)
            except ValueError:
                pass

        # strip_code handles bold, italic, wikilinks, and remaining HTML
        text = parsed.strip_code(normalize=True, collapse=True)
    except Exception:
        # Fallback: regex-based cleanup
        text = RE_TEMPLATE.sub("", text)
        text = RE_HTML_TAG.sub("", text)
        text = text.replace("'''", "").replace("''", "")
        text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)

    # Final cleanup
    text = RE_MULTI_SPACE.sub(" ", text).strip()

    # Remove leading/trailing quotes if the entire string is wrapped
    if len(text) >= 2:
        if (text[0] == '"' and text[-1] == '"') or (text[0] == "\u201c" and text[-1] == "\u201d"):
            text = text[1:-1].strip()

    return text


def _should_skip_section(section_name: str) -> bool:
    """Check if a section title indicates non-quote content."""
    if section_name in SKIP_SECTIONS:
        return True
    for prefix in ABOUT_PREFIXES:
        if section_name.startswith(prefix):
            return True
    return False


def _section_confidence(section_name: str) -> str:
    """Map section name to source_confidence value."""
    if section_name in ("attributed", "attributed quotes"):
        return "attributed"
    return "sourced"


def parse_and_store(xml_path: Path, db_path: Path) -> int:
    """Parse the XML dump and insert all quotes into SQLite.

    Returns the number of quotes inserted (excluding duplicates).
    Commits every 1000 quotes for crash safety.
    """
    conn = db.get_connection(db_path)
    inserted = 0
    duplicates = 0
    total_seen = 0

    click.echo(f"Parsing {xml_path.name}...")

    try:
        for quote in parse_dump(xml_path):
            total_seen += 1
            if db.insert_quote(
                conn,
                text=quote.text,
                author=quote.author,
                source_work=quote.source_work,
                source_confidence=quote.source_confidence,
            ):
                inserted += 1
            else:
                duplicates += 1

            if total_seen % 1000 == 0:
                conn.commit()
                click.echo(
                    f"  ... {inserted:,} quotes inserted, {duplicates:,} duplicates skipped"
                )

        conn.commit()
    finally:
        conn.close()

    click.echo(f"Parse complete: {inserted:,} quotes inserted, {duplicates:,} duplicates skipped")
    return inserted
