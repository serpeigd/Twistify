"""Generation interface.

PRINCIPLE: the evaluation harness is built and tested against a FAKE
generator before touching any API.

Why: if you write the harness and the real generator at the same time, when
a metric looks wrong you won't know whether the generator is bad or the
metric is broken. A deterministic fake with *planted* leaks gives you a
test with a known answer: you know there are exactly 2 leaks, so if your
metric doesn't say 2, the metric is broken. This is what lets you trust the
number in the README.
"""

from __future__ import annotations

from typing import Protocol

from .schemas import Claim, DeepDive, PreShowBrief, ScriptBlock, SourceDoc, TitleCase


class Generator(Protocol):
    """Injectable. The evals runner doesn't know whether there's an LLM or
    a dict behind it."""

    name: str

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief: ...

    def deep_dive(self, case: TitleCase, corpus: list[SourceDoc]) -> DeepDive: ...


class ScriptedFakeGenerator:
    """Returns fixed outputs per title. Zero network, zero cost,
    deterministic.

    Used in tests/test_metrics.py to verify that the metrics count what
    they claim to count.
    """

    name = "fake"

    def __init__(self, canned: dict[str, PreShowBrief]):
        self._canned = canned

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief:
        return self._canned[case.title_id]

    def deep_dive(self, case: TitleCase, corpus: list[SourceDoc]) -> DeepDive:
        raise NotImplementedError


def make_brief(
    title_id: str,
    facts: list[tuple[str, str | None]],
    voiceover: str = "",
) -> PreShowBrief:
    """Helper to build synthetic briefs in tests.

    `facts` are (text, source_id) tuples. source_id=None => unsupported fact.
    """
    return PreShowBrief(
        title_id=title_id,
        context_bullets=[
            Claim(text=t, kind="fact", source_id=s) for t, s in facts[:3]
        ],
        author_voice=[],
        emotional_temperature="like opening a fridge in the dark",
        why_now="",
        script=[
            ScriptBlock(
                start_s=0,
                end_s=15,
                on_screen_text="",
                voiceover=voiceover,
                visual_direction="",
            )
        ],
    )
