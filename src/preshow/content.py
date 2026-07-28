"""Data contract for RESEARCHED CONTENT (what the app shows).

Distinct from `schemas.py`: that one models what a generator PRODUCES and
the harness MEASURES. This one models what a human/research effort has
researched and the app SERVES. Kept separate on purpose -- mixing them would
make it impossible to tell "a model wrote this" apart from "this came from
a source."

Principle inherited from D2 (see docs/DESIGN.md): the schema ALLOWS
incomplete states. A `SourcedText` with no `source_id` is valid and renders
as "no source" in the UI. If the schema rejected it, the gap would be an
exception instead of a visible data point -- and the visible gap IS the
product: it shows where the content is grounded and where it isn't.

PHASE PARTITION (D3): `PRE_SHOW_FIELDS` and `POST_SHOW_FIELDS` aren't
documentation, they're the boundary the API uses to decide what gets sent
to the browser before the user declares they've seen the work. A
post-viewing field isn't filtered with CSS: it never leaves the server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# The spoiler boundary. The API never sends POST_SHOW_FIELDS unless the
# client explicitly declares having seen the work.
PRE_SHOW_FIELDS = (
    "story",
    "context_bullets",
    "before_watching",
    "author_voice",
    "emotional_temperature",
    "why_now",
)

POST_SHOW_FIELDS = (
    "metaphors",
    "intertextual_refs",
    "production_trivia",
    "scene_analysis",
    "critical_consensus",
    "strengths",
    "weaknesses",
    "verdict",
    "useless_fact",
    "fun_facts",
)

# Neither pre nor post: questions/debate, they don't reveal plot by themselves.
NEUTRAL_FIELDS = ("questions", "debate_prompts", "cta", "sources", "director", "themes")


class SourcedText(BaseModel):
    """Text with an optional source. Missing `source_id` => unsupported,
    and the UI marks it as such instead of hiding it."""

    text: str
    source_id: str | None = None
    kind: Literal["fact", "interpretation"] | None = None

    @property
    def is_grounded(self) -> bool:
        return self.kind == "interpretation" or bool(self.source_id)


class SceneNote(BaseModel):
    scene: str
    text: str
    source_id: str | None = None


class FactBullet(BaseModel):
    """A fact with a bold headline + explanation. Used in 'before
    watching' (spoiler-free) and 'fun facts' (post-viewing)."""

    lead: str
    text: str
    source_id: str | None = None


class Score(BaseModel):
    source: str
    value: str
    url: str | None = None


class CriticalConsensus(BaseModel):
    summary: str | None = None
    scores: list[Score] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)


class ContentPack(BaseModel):
    """Full package for a work. Everything is optional except the id: a
    half-researched entry is a legitimate state and the UI must be able to
    show it."""

    title_id: str

    # --- Phase 1: pre-viewing (spoiler-free) ---
    story: str | None = None
    context_bullets: list[str] = Field(default_factory=list)
    before_watching: list[FactBullet] = Field(default_factory=list)
    author_voice: list[SourcedText] = Field(default_factory=list)
    emotional_temperature: str | None = None
    why_now: str | None = None

    # --- Phase 2: post-viewing (spoilers are the product here) ---
    metaphors: list[SourcedText] = Field(default_factory=list)
    intertextual_refs: list[SourcedText] = Field(default_factory=list)
    production_trivia: list[SourcedText] = Field(default_factory=list)
    scene_analysis: list[SceneNote] = Field(default_factory=list)

    # --- Phase 3: critical reception ---
    critical_consensus: CriticalConsensus = Field(default_factory=CriticalConsensus)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    verdict: str | None = None
    useless_fact: str | None = None
    fun_facts: list[FactBullet] = Field(default_factory=list)

    # --- Phase 4: engagement ---
    questions: list[str] = Field(default_factory=list)
    debate_prompts: list[str] = Field(default_factory=list)
    cta: str | None = None

    sources: list[str] = Field(default_factory=list)

    # --- Catalogue metadata (for filtering/sorting with no spoilers) ---
    director: str | None = None
    themes: list[str] = Field(default_factory=list)

    # ---- Completeness / grounding metrics ----

    def all_sourced(self) -> list[SourcedText]:
        as_st = lambda b: SourcedText(text=b.text, source_id=b.source_id, kind="fact")
        return [
            *self.author_voice,
            *self.metaphors,
            *self.intertextual_refs,
            *self.production_trivia,
            *[as_st(b) for b in self.before_watching],
            *[as_st(b) for b in self.fun_facts],
        ]

    def grounding(self) -> tuple[int, int]:
        """(claims that need a source, how many have one).

        Interpretations don't count: demanding a source for a personal
        reading of the work makes no sense (same rule as evals/metrics.py).
        """
        needs = [s for s in self.all_sourced() if s.kind != "interpretation"]
        return len(needs), sum(1 for s in needs if s.source_id)

    def completeness(self) -> dict:
        """Which sections have content. Gaps are shown, not disguised: an
        empty field means 'we found no source', which is useful
        information, not a failure to hide."""
        sections = {
            "story": bool(self.story),
            "before_watching": bool(self.before_watching),
            "author_voice": bool(self.author_voice),
            "fun_facts": bool(self.fun_facts),
            "metaphors": bool(self.metaphors),
            "intertextual_refs": bool(self.intertextual_refs),
            "production_trivia": bool(self.production_trivia),
            "scene_analysis": bool(self.scene_analysis),
            "critical_consensus": bool(self.critical_consensus.scores),
            "strengths": bool(self.strengths),
            "weaknesses": bool(self.weaknesses),
            "questions": bool(self.questions),
        }
        filled = sum(sections.values())
        return {
            "sections": sections,
            "filled": filled,
            "total": len(sections),
            "pct": round(100 * filled / len(sections)),
        }

    def pre_show_text(self) -> list[tuple[str, str]]:
        """Pre-viewing surface as (location, text), to run through the leak
        detector. Same criterion as evals/metrics.brief_surface: if the
        user can read it before seeing the work, it's surface."""
        out: list[tuple[str, str]] = []
        if self.story:
            out.append(("story", self.story))
        for i, b in enumerate(self.context_bullets):
            out.append((f"context_bullets[{i}]", b))
        for i, b in enumerate(self.before_watching):
            out.append((f"before_watching[{i}].lead", b.lead))
            out.append((f"before_watching[{i}]", b.text))
        for i, a in enumerate(self.author_voice):
            out.append((f"author_voice[{i}]", a.text))
        if self.emotional_temperature:
            out.append(("emotional_temperature", self.emotional_temperature))
        if self.why_now:
            out.append(("why_now", self.why_now))
        return [(loc, t) for loc, t in out if t.strip()]

    def public_dump(self, seen: bool) -> dict:
        """Serialize for the client. If `seen` is False, post-viewing
        fields are NOT included -- they're omitted server-side, not hidden
        in the browser. That's the difference between a guarantee and a
        suggestion (see D3 in docs/DESIGN.md)."""
        return strip_post_show(self.model_dump(), seen)


def strip_post_show(data: dict, seen: bool) -> dict:
    """Same partition `ContentPack.public_dump` applies, but usable on a
    plain dict -- so the machine-translated dump (webapp/app.py) enforces
    the identical spoiler boundary instead of re-deriving it."""
    if seen:
        return data
    data = dict(data)
    for field in POST_SHOW_FIELDS:
        data.pop(field, None)
    return data


def load_pack(path: Path) -> ContentPack:
    return ContentPack(**json.loads(path.read_text(encoding="utf-8")))


def load_all(directory: Path) -> dict[str, ContentPack]:
    packs = {}
    for p in sorted(directory.glob("*.json")):
        pack = load_pack(p)
        packs[pack.title_id] = pack
    return packs
