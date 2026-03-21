"""Tests for the wikitext parser and quote extraction."""

from wikiquotes_tagger.parser import (
    _clean_wikitext,
    _extract_intro_author,
    _extract_quotes_from_page,
    _is_literary_section,
    _is_theme_page,
    _should_skip_section,
    _section_confidence,
)


class TestCleanWikitext:
    def test_strips_bold_and_italic(self):
        assert _clean_wikitext("'''bold''' and ''italic''") == "bold and italic"

    def test_strips_wiki_links_keeps_display(self):
        assert _clean_wikitext("a [[Philosophy|philosophical]] idea") == "a philosophical idea"

    def test_strips_plain_wiki_links(self):
        assert _clean_wikitext("the [[universe]] is vast") == "the universe is vast"

    def test_strips_ref_tags(self):
        result = _clean_wikitext('Some text<ref name="foo">citation</ref> more text')
        assert "citation" not in result
        assert "Some text" in result
        assert "more text" in result

    def test_strips_templates(self):
        result = _clean_wikitext("Hello {{cite book|title=Foo}} world")
        assert "cite" not in result
        assert "Hello" in result

    def test_removes_wrapping_quotes(self):
        assert _clean_wikitext('"A famous quote"') == "A famous quote"
        assert _clean_wikitext('\u201cA famous quote\u201d') == "A famous quote"

    def test_collapses_whitespace(self):
        assert _clean_wikitext("too   many    spaces") == "too many spaces"

    def test_empty_string(self):
        assert _clean_wikitext("") == ""


class TestShouldSkipSection:
    def test_skips_misattributed(self):
        assert _should_skip_section("misattributed") is True

    def test_skips_external_links(self):
        assert _should_skip_section("external links") is True

    def test_skips_quotes_about(self):
        assert _should_skip_section("quotes about einstein") is True

    def test_does_not_skip_sourced(self):
        assert _should_skip_section("sourced") is False

    def test_does_not_skip_quotes(self):
        assert _should_skip_section("quotes") is False


class TestSectionConfidence:
    def test_sourced(self):
        assert _section_confidence("sourced") == "sourced"
        assert _section_confidence("quotes") == "sourced"

    def test_attributed(self):
        assert _section_confidence("attributed") == "attributed"
        assert _section_confidence("attributed quotes") == "attributed"


class TestExtractQuotesFromPage:
    def test_basic_person_page(self):
        wikitext = """
== Sourced ==
* The unexamined life is not worth living.
** ''[[Apology (Plato)|Apology]]'', 38a5-6

* I know that I know nothing.
** ''Apology''

== Misattributed ==
* This was not said by Socrates.
"""
        quotes = _extract_quotes_from_page("Socrates", wikitext)
        assert len(quotes) == 2
        assert quotes[0].author == "Socrates"
        assert "unexamined life" in quotes[0].text
        assert quotes[0].source_confidence == "sourced"
        assert quotes[0].source_work is not None
        assert "Apology" in quotes[0].source_work

    def test_attributed_section(self):
        wikitext = """
== Attributed ==
* Be kind, for everyone you meet is fighting a hard battle.
"""
        quotes = _extract_quotes_from_page("Plato", wikitext)
        assert len(quotes) == 1
        assert quotes[0].source_confidence == "attributed"

    def test_skips_media_pages(self):
        wikitext = """
== Cast ==
* '''Character Name''': Some quote here.

== Dialogue ==
* Line one
* Line two
"""
        quotes = _extract_quotes_from_page("Star Wars", wikitext)
        assert len(quotes) == 0

    def test_skips_redirects(self):
        quotes = _extract_quotes_from_page("Foo", "#REDIRECT [[Bar]]")
        assert len(quotes) == 0

    def test_skips_pages_without_quote_sections(self):
        wikitext = """
This is a simple page about Love with no structured sections.
"""
        quotes = _extract_quotes_from_page("Love", wikitext)
        assert len(quotes) == 0

    def test_filters_short_quotes(self):
        wikitext = """
== Quotes ==
* Too short.
* This is a sufficiently long quote that should pass the minimum length filter easily.
"""
        quotes = _extract_quotes_from_page("Test Author", wikitext)
        assert len(quotes) == 1
        assert "sufficiently long" in quotes[0].text


class TestIsThemePage:
    def test_detects_alphabetical_headings(self):
        wikitext = """
== A ==
* Quote by someone starting with A.
== B ==
* Quote by someone starting with B.
== C ==
* Quote
== D ==
* Quote
== E ==
* Quote
"""
        assert _is_theme_page(wikitext) is True

    def test_detects_tocalpha_template(self):
        assert _is_theme_page("Some intro text.\n{{TOCalpha}}\n== A ==\n* Quote") is True

    def test_rejects_normal_page(self):
        wikitext = """
== Sourced ==
* A quote here.
== Attributed ==
* Another quote.
"""
        assert _is_theme_page(wikitext) is False


class TestIsLiterarySection:
    def test_act_headings(self):
        assert _is_literary_section("act i") is True
        assert _is_literary_section("act iv") is True
        assert _is_literary_section("act 3") is True

    def test_part_headings(self):
        assert _is_literary_section("part one") is True
        assert _is_literary_section("part two") is True
        assert _is_literary_section("part iii") is True
        assert _is_literary_section("part 2") is True

    def test_book_headings(self):
        assert _is_literary_section("book i") is True
        assert _is_literary_section("book xii") is True
        assert _is_literary_section("book 3") is True

    def test_chapter_headings(self):
        assert _is_literary_section("chapter 1") is True
        assert _is_literary_section("chapters 1-10") is True

    def test_exact_sections(self):
        assert _is_literary_section("prologue") is True
        assert _is_literary_section("epilogue") is True
        assert _is_literary_section("appendix") is True

    def test_rejects_non_literary(self):
        assert _is_literary_section("sourced") is False
        assert _is_literary_section("external links") is False
        assert _is_literary_section("biography") is False


class TestExtractIntroAuthor:
    def test_simple_novel(self):
        intro = "'''''Moby-Dick''''' (1851) is a novel by [[Herman Melville]]."
        assert _extract_intro_author(intro) == "Herman Melville"

    def test_play_with_adjective(self):
        intro = "'''''Hamlet''''' is a revenge tragedy by [[William Shakespeare]]."
        assert _extract_intro_author(intro) == "William Shakespeare"

    def test_novel_with_wikilink_genre(self):
        intro = "'''Pride and Prejudice''' (1813) is a [[w:novel of manners|novel of manners]] by [[Jane Austen]]."
        assert _extract_intro_author(intro) == "Jane Austen"

    def test_novel_with_descriptor_before_author(self):
        intro = "'''''Nineteen Eighty-Four''''' is a dystopian novel by the English writer [[George Orwell]]."
        assert _extract_intro_author(intro) == "George Orwell"

    def test_epic_poem_with_long_descriptor(self):
        intro = "'''''Paradise Lost''''' (1667) is an epic poem by the 17th century English poet [[John Milton]]."
        assert _extract_intro_author(intro) == "John Milton"

    def test_piped_wikilink_author(self):
        intro = "'''''The Little Prince''''' (1943) is a novel by [[w:Antoine de Saint-Exupéry|Antoine de Saint-Exupéry]]."
        assert _extract_intro_author(intro) == "Antoine de Saint-Exupéry"

    def test_rejects_film(self):
        intro = "'''''The Godfather''''' is a 1972 film directed by [[Francis Ford Coppola]]."
        assert _extract_intro_author(intro) is None

    def test_rejects_tv_series(self):
        intro = "'''''Breaking Bad''''' is an American TV series created by [[Vince Gilligan]]."
        assert _extract_intro_author(intro) is None

    def test_rejects_person_page(self):
        intro = "'''Albert Einstein''' (1879-1955) was a theoretical physicist."
        assert _extract_intro_author(intro) is None


class TestExtractLiteraryWorkQuotes:
    def test_literary_works_skipped(self):
        """Literary work pages are currently skipped (person pages only)."""
        wikitext = """'''''Pride and Prejudice''''' (1813) is a novel by [[Jane Austen]].

== Quotes ==
* It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.
** Ch. 1
"""
        quotes = _extract_quotes_from_page("Pride and Prejudice", wikitext)
        assert len(quotes) == 0

    def test_plays_skipped(self):
        """Play pages are currently skipped (person pages only)."""
        wikitext = """'''''Hamlet''''' is a revenge tragedy by [[William Shakespeare]].

== Act I ==
* Though yet of Hamlet our dear brother's death the memory be green.
"""
        quotes = _extract_quotes_from_page("Hamlet", wikitext)
        assert len(quotes) == 0

    def test_skips_theme_pages(self):
        wikitext = """'''Love''' is a complex emotion.
{{TOCalpha}}
== A ==
* Love is composed of a single soul inhabiting two bodies.
** [[Aristotle]]
== B ==
* The giving of love is an education in itself.
** [[Eleanor Roosevelt]]
"""
        quotes = _extract_quotes_from_page("Love", wikitext)
        assert len(quotes) == 0

    def test_person_page_still_works(self):
        """Ensure person pages are unaffected by literary work changes."""
        wikitext = """
== Sourced ==
* The only thing we have to fear is fear itself.
** First inaugural address (1933)
"""
        quotes = _extract_quotes_from_page("Franklin D. Roosevelt", wikitext)
        assert len(quotes) == 1
        assert quotes[0].author == "Franklin D. Roosevelt"
        assert quotes[0].source_work is not None
        assert quotes[0].source_confidence == "sourced"

    def test_film_page_still_excluded(self):
        wikitext = """'''''The Godfather''''' is a 1972 film directed by [[Francis Ford Coppola]].

== Cast ==
* '''Marlon Brando''' as Vito Corleone

== Dialogue ==
* I'm gonna make him an offer he can't refuse.
"""
        quotes = _extract_quotes_from_page("The Godfather", wikitext)
        assert len(quotes) == 0
