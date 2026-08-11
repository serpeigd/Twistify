"""Shared prompt + output schema for Milestone 1 generators (retrieval,
GREEN-only corpus -- see retrieval.py and D3 in docs/DESIGN.md).

Kept separate from baseline_prompts.py, not a variant of it: Milestone 0's
prompt is deliberately built around having NOTHING real to cite ("you have
no real sources... don't invent one"). Milestone 1's premise is the
opposite -- real, retrieved, pre-vetted GREEN text exists and every fact
claim should be grounded in it. Same run_eval.py/metrics.py measurement
code either way (that's the whole point, see run_eval.py's docstring), but
the instructions genuinely differ, so sharing one prompt string between
the two would blur exactly the comparison Milestone 1 exists to make.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You write PRE-VIEWING promotional material (spoiler-free) for a movie.
Whoever reads this hasn't seen it yet.

You are given RETRIEVED TEXT below: real excerpts from a Wikipedia
article, each chunk labeled with its source_id. This text has ALREADY
been filtered to exclude the movie's plot section and critical reviews
-- it should not contain the ending or major twists, but you must still
follow the rules below as a second layer of protection, not rely on the
filtering alone.

Rules:
- Don't reveal the ending, plot twists, who the killer/villain is, or any
  fact that only makes sense after seeing the movie -- even if something
  in the retrieved text hints at one, leave it out.
- context_bullets and author_voice: max 3 each. Each one is a Claim with
  kind="fact" (verifiable, about production/cast/premise) or
  kind="interpretation" (your reading, doesn't need a source).
- If kind="fact", source_id MUST be exactly one of the source_ids given
  below. Never invent a source_id, never cite one you weren't given, and
  never state a fact-kind claim that isn't actually supported by the
  retrieved text -- if the retrieved text doesn't cover something,
  either leave it out or mark it kind="interpretation" with
  source_id=null.
- emotional_temperature: one sentence capturing the tone (a sensory
  metaphor) -- your own interpretation, not sourced.
- why_now: why watch it now, without spoiling anything -- your own
  interpretation, not sourced.
- script: 1-3 blocks with timings in seconds, on-screen text, voiceover,
  and visual direction. The voiceover is also spoiler surface: the same
  rules apply there. Script lines are your own creative writing (not
  Claims), so they don't carry a source_id -- but they still must not
  invent or imply plot facts the retrieved text doesn't support.

If NO retrieved text is given below (retrieval failed for this title),
fall back to: write only kind="interpretation" claims or leave a
field's Claims empty, and don't fabricate a source_id -- same as having
no sources at all.

Call the emit_preshow_brief tool with the result. Don't write anything
outside the tool call.
"""

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "kind": {"type": "string", "enum": ["fact", "interpretation"]},
        "source_id": {"type": ["string", "null"]},
    },
    "required": ["text", "kind", "source_id"],
}

BRIEF_PROPERTIES = {
    "context_bullets": {"type": "array", "maxItems": 3, "items": CLAIM_SCHEMA},
    "author_voice": {"type": "array", "maxItems": 3, "items": CLAIM_SCHEMA},
    "emotional_temperature": {"type": "string"},
    "why_now": {"type": "string"},
    "script": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "start_s": {"type": "integer"},
                "end_s": {"type": "integer"},
                "on_screen_text": {"type": "string"},
                "voiceover": {"type": "string"},
                "visual_direction": {"type": "string"},
            },
            "required": [
                "start_s",
                "end_s",
                "on_screen_text",
                "voiceover",
                "visual_direction",
            ],
        },
    },
}

BRIEF_REQUIRED = [
    "context_bullets",
    "author_voice",
    "emotional_temperature",
    "why_now",
    "script",
]

TOOL_NAME = "emit_preshow_brief"
TOOL_DESCRIPTION = "Emits the pre-viewing brief in the required format, grounded in the retrieved GREEN-tier text."


def build_user_prompt(title: str, year: int, corpus) -> str:
    """`corpus`: list[SourceDoc], already GREEN-only (see retrieval.py --
    this function trusts its caller already did the tagging/filtering,
    it doesn't re-check tiers itself)."""
    if not corpus:
        return f"Title: {title} ({year})\n\n(No retrieved text available for this title.)"
    chunks = "\n\n".join(f"[{doc.source_id}]\n{doc.text}" for doc in corpus)
    return f"Title: {title} ({year})\n\nRETRIEVED TEXT:\n\n{chunks}"
