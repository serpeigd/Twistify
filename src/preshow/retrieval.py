"""Milestone 1: builds the GREEN-only corpus a pre-show generator is
allowed to see (D3 in docs/DESIGN.md — "the traffic light tags the
corpus, not the output"). This is the piece that was designed
(`SpoilerLabel`/`SourceDoc.tier` in schemas.py) but never wired up while
Milestone 0 ran with `corpus=[]`.

The safety property this exists for: it's not the generator's job to
avoid repeating plot details it was given. It's this module's job to
never GIVE it plot details in the first place. A prompt instruction
("don't spoil anything") is something the model can ignore or misjudge --
exactly what this whole project's judge-calibration saga (D12-D15) spent
weeks proving is unreliable to check after the fact. A context the plot
was never in can't leak it, full stop.

Tagging, using `preshow.wikipedia.relevant_sections()`'s existing
plot/production/reception/accolades split:
  - GREEN (passed to the generator): "overview" (the lead paragraph --
    genuinely pre-viewing safe in practice: premise/setup, not the
    ending), "production" (cast/crew/budget/shoot facts), "accolades"
    (awards, festival wins).
  - RED, dropped entirely, never even tagged into a SourceDoc: "plot".
    This is THE spoiler corpus. It is never constructed as a SourceDoc,
    never passed anywhere near a generator prompt -- there's no
    intermediate step where it exists in a form a bug could accidentally
    forward.
  - AMBER, dropped by the same default D3 states ("AMBER is treated as
    RED by default"): "reception"/critical response sections. Real
    reviews routinely discuss plot points in their analysis (that's
    what a review IS), so unlike "overview", this can't be trusted
    pre-viewing-safe just because its heading isn't "Plot".

Caveat, stated plainly: heading-based section splitting is a heuristic,
not a guarantee. A "Production" section that happens to mention a plot
twist in passing (rare, but possible) would slip through GREEN. This is
a real, known limitation -- not fixed here, worth flagging if Milestone
1's own leakage_rate comes back non-zero and the evidence points at a
GREEN-tagged source rather than the generator inventing something.
"""

from __future__ import annotations

from . import wikipedia
from .schemas import SourceDoc, SpoilerTier

GREEN_SECTIONS = ("overview", "production", "accolades")
# "plot" is never included -- see module docstring. "reception" is
# excluded too (AMBER treated as RED by D3's stated default), even
# though relevant_sections() extracts it -- that extraction exists for
# webapp/research_assist.py's post-viewing content, a different context
# with a human review gate before publishing (D14), which this
# generator-facing path doesn't have.


def build_green_corpus(title: str, year: int) -> list[SourceDoc]:
    """Retrieves the title's Wikipedia article and returns ONLY the
    GREEN-tagged chunks as SourceDocs -- this is the exact list a
    Milestone 1 generator's `corpus` argument should receive. Returns []
    (not an error) if the article can't be found or fetched, matching
    Milestone 0's `richness`-not-`leakage`-or-`grounding` philosophy:
    "no sources available" should show up as a low richness/grounded
    score, not crash the run over one title.
    """
    page_title = wikipedia.find_page_title(title, year)
    if not page_title:
        return []
    article = wikipedia.fetch_article(page_title)
    if not article:
        return []

    sections = wikipedia.relevant_sections(article["extract"])
    corpus: list[SourceDoc] = []
    for i, name in enumerate(GREEN_SECTIONS):
        text = sections.get(name)
        if not text:
            continue
        corpus.append(
            SourceDoc(
                source_id=f"wikipedia:{name}",
                origin="wikipedia",
                text=text,
                tier=SpoilerTier.GREEN,
                url=article["url"],
                section=name,
            )
        )
    return corpus
