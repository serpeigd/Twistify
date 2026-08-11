"""Tests for build_green_corpus (Milestone 1, D3 in docs/DESIGN.md).

The whole point of this module is a safety property -- plot text must
NEVER reach a generator -- so the tests are built around trying to make
that fail, not just checking the happy path. No network: monkeypatches
preshow.wikipedia's two entry points instead of hitting the real API.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preshow import retrieval, wikipedia  # noqa: E402
from preshow.schemas import SpoilerTier  # noqa: E402

FAKE_EXTRACT = """This is the lead paragraph, introducing the premise safely.

== Plot ==
This section reveals that the butler did it and everyone dies at the end.

== Production ==
Filmed in 1974 on a budget of $2 million with a crew of 40.

== Critical response ==
Critics praised the film but noted the twist where the narrator is unreliable.

== Accolades ==
Won Best Picture at the 1975 Academy Awards.
"""


def _patch_wikipedia(monkeypatch, extract=FAKE_EXTRACT):
    monkeypatch.setattr(wikipedia, "find_page_title", lambda title, year: "Fake Movie (1974 film)")
    monkeypatch.setattr(
        wikipedia,
        "fetch_article",
        lambda page_title: {"title": page_title, "url": "https://en.wikipedia.org/wiki/Fake_Movie", "extract": extract},
    )


def test_plot_section_never_becomes_a_source_doc(monkeypatch):
    """The core safety property: no SourceDoc in the returned corpus may
    contain the plot section's text, under any tier."""
    _patch_wikipedia(monkeypatch)
    corpus = retrieval.build_green_corpus("Fake Movie", 1974)
    for doc in corpus:
        assert "butler did it" not in doc.text
        assert "everyone dies" not in doc.text


def test_reception_section_excluded_even_though_it_mentions_a_twist(monkeypatch):
    """AMBER treated as RED by default (D3) -- "Critical response" is
    never GREEN here, unlike webapp/research_assist.py's separate,
    human-reviewed path."""
    _patch_wikipedia(monkeypatch)
    corpus = retrieval.build_green_corpus("Fake Movie", 1974)
    for doc in corpus:
        assert "unreliable" not in doc.text
        assert doc.section != "reception"


def test_green_sections_are_included_and_tagged(monkeypatch):
    _patch_wikipedia(monkeypatch)
    corpus = retrieval.build_green_corpus("Fake Movie", 1974)
    sections = {doc.section for doc in corpus}
    assert sections == {"overview", "production", "accolades"}
    for doc in corpus:
        assert doc.tier == SpoilerTier.GREEN
        assert doc.origin == "wikipedia"
        assert doc.url == "https://en.wikipedia.org/wiki/Fake_Movie"

    by_section = {doc.section: doc.text for doc in corpus}
    assert "lead paragraph" in by_section["overview"]
    assert "$2 million" in by_section["production"]
    assert "Best Picture" in by_section["accolades"]


def test_missing_article_returns_empty_corpus_not_an_error(monkeypatch):
    monkeypatch.setattr(wikipedia, "find_page_title", lambda title, year: None)
    assert retrieval.build_green_corpus("Totally Obscure Film", 1901) == []


def test_missing_sections_are_simply_omitted(monkeypatch):
    """A stub with no Production/Accolades headings shouldn't crash or
    fabricate a SourceDoc for a section that isn't there."""
    _patch_wikipedia(monkeypatch, extract="Just a lead paragraph, nothing else.")
    corpus = retrieval.build_green_corpus("Fake Movie", 1974)
    sections = {doc.section for doc in corpus}
    assert sections == {"overview"}
