"""System data contract.

CENTRAL DESIGN DECISION
-----------------------
A `Claim` can exist without a source. This looks like a bug: why not force
`source_id: str` and have Pydantic reject unsupported claims?

Because you couldn't measure it. If the parser rejects the invalid state,
the failure becomes an exception instead of a metric, and all you learn is
"it blew up." You need the model to be ABLE to produce a claim with no
source so you can count how often it happens, on which kind of titles, and
whether your intervention reduces it.

Validation doesn't live in the schema. It lives in the verifier, which is
an explicit pipeline stage with its own metric.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SpoilerTier(str, Enum):
    """Traffic light, but assigned to the CORPUS, not the output.

    In the original prompt the model self-labeled. Here the tier is decided
    on the retrieved documents, before generation. The pre-experience
    generator only receives GREEN. Security is a property of context, not
    an instruction the model can ignore.
    """

    GREEN = "green"  # safe pre-experience
    AMBER = "amber"  # ambiguous -> treated as RED by default
    RED = "red"      # post-experience only


Origin = Literal["tmdb", "omdb", "wikipedia", "openlibrary", "model_memory"]


class SourceDoc(BaseModel):
    """A retrieved chunk. `model_memory` exists for the baseline (Milestone
    0): it lets you represent the case 'this came from the weights, not a
    source'."""

    source_id: str
    origin: Origin
    text: str
    tier: SpoilerTier
    url: str | None = None
    section: str | None = None


class Claim(BaseModel):
    """Atomic unit of content.

    `kind` separates the verifiable from the opinionated. An interpretation
    with no source is legitimate (it's your reading of the work). A fact
    with no source is garbage. Mixing them in the same field, like the
    original prompt did, makes it impossible to apply different rules to
    each.
    """

    text: str
    kind: Literal["fact", "interpretation"]
    source_id: str | None = None
    quote: str | None = Field(
        default=None,
        description="Literal excerpt from the source that backs the claim.",
    )

    @property
    def is_grounded(self) -> bool:
        return self.kind == "interpretation" or self.source_id is not None


class ScriptBlock(BaseModel):
    """A timed block of the script.

    Data, not markdown. If you ever chain this into TTS + automated
    editing, you need `start_s`/`end_s` as numbers, not a table cell.
    """

    start_s: int
    end_s: int
    on_screen_text: str
    voiceover: str
    visual_direction: str


class PreShowBrief(BaseModel):
    """PHASE 1. Generated with GREEN context only."""

    title_id: str
    context_bullets: list[Claim] = Field(max_length=3)
    author_voice: list[Claim] = Field(max_length=3)
    emotional_temperature: str  # sensory metaphor: always an interpretation
    why_now: str
    script: list[ScriptBlock]

    @property
    def all_claims(self) -> list[Claim]:
        return [*self.context_bullets, *self.author_voice]


class DeepDive(BaseModel):
    """PHASES 2+3. Full context. Spoilers are the product here."""

    title_id: str
    metaphors: list[Claim]
    intertextual_refs: list[Claim]
    production_trivia: list[Claim]
    critical_consensus: list[Claim]
    strengths: list[Claim]
    weaknesses: list[Claim]
    verdict: str

    @property
    def all_claims(self) -> list[Claim]:
        return [
            *self.metaphors,
            *self.intertextual_refs,
            *self.production_trivia,
            *self.critical_consensus,
            *self.strengths,
            *self.weaknesses,
        ]


class SpoilerLabel(BaseModel):
    """Hand-labeled ground truth. A fact that must NEVER show up in a
    PreShowBrief.

    `paraphrases` exists because a real leak is almost never literal. The
    model doesn't write "the protagonist is dead"; it writes "a final
    revelation reframes everything you've seen." If your detector only
    looks for substrings, your measured leak rate will be zero, and it will
    be a lie.
    """

    id: str
    canonical: str
    paraphrases: list[str] = Field(default_factory=list)
    severity: Literal["core", "major", "minor"]


class TitleCase(BaseModel):
    """A case from the evaluation set."""

    title_id: str
    title: str
    year: int
    kind: Literal["film", "book"]
    stratum: Literal["mainstream", "longtail"]
    tmdb_id: int | None = None
    notes: str | None = None
