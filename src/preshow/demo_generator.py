"""DEMO generator. Doesn't call any LLM, costs nothing, isn't the real
baseline (that's `baseline.py`, which does need ANTHROPIC_API_KEY and
spends credit).

Builds a PreShowBrief from fixed templates and only the safe fields we
already have for free in the dataset itself (title, year, stratum, notes
from `titles.yaml`). Never touches `evals/dataset/spoilers/*.yaml`: that
file is exactly what a pre-viewing brief must not leak.

Exists so there's something to see and verify in a UI without spending
real credit. Don't confuse this with a measurement of the system -- the
`leakage_rate` here is 0.0 by construction (there's no real retrieval
behind it), not because the system "works."
"""

from __future__ import annotations

from .schemas import Claim, DeepDive, PreShowBrief, ScriptBlock, SourceDoc, TitleCase

_STRATUM_VOICE = {
    "mainstream": "one of those movies everyone has an opinion on, even if they haven't seen it",
    "longtail": "one of those movies almost nobody has seen, but everyone who has never forgets it",
}


class DemoGenerator:
    """Demo Generator (Protocol). Deterministic: same title -> same brief."""

    name = "demo"

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief:
        voice = _STRATUM_VOICE.get(case.stratum, "a movie worth watching without knowing anything beforehand")

        context_bullets = [
            Claim(
                text=f"{case.title} was released in {case.year}.",
                kind="fact",
                source_id=None,
            ),
        ]
        if case.notes:
            context_bullets.append(
                Claim(text=case.notes, kind="interpretation", source_id=None)
            )

        return PreShowBrief(
            title_id=case.title_id,
            context_bullets=context_bullets[:3],
            author_voice=[
                Claim(text=voice, kind="interpretation", source_id=None),
            ],
            emotional_temperature="like opening a door without knowing what's behind it",
            why_now=f"{case.title} ({case.year}) — one of those you should watch without reading anything first.",
            script=[
                ScriptBlock(
                    start_s=0,
                    end_s=15,
                    on_screen_text=case.title,
                    voiceover=voice,
                    visual_direction="static shot, no music until second 10",
                )
            ],
        )

    def deep_dive(self, case: TitleCase, corpus: list[SourceDoc]) -> DeepDive:
        raise NotImplementedError("The demo only covers pre_show; deep_dive doesn't exist yet")
