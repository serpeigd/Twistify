"""Shared prompt + output schema for Milestone 0 baseline generators.

Kept in one place so every provider (Anthropic, Groq, ...) gets literally
the same instructions and the same output schema. If the wording drifted
between providers, a difference in leakage_rate between them would be
confounded by prompt differences instead of measuring the model itself --
exactly the kind of harness inconsistency run_eval.py's own docstring
warns against.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You write PRE-VIEWING promotional material (spoiler-free) for a movie.
Whoever reads this hasn't seen it yet.

Rules:
- Don't reveal the ending, plot twists, who the killer/villain is, or any
  fact that only makes sense after seeing the movie.
- context_bullets and author_voice: max 3 each. Each one is a Claim with
  kind="fact" (verifiable, about production/cast/premise) or
  kind="interpretation" (your reading, doesn't need a source). If it's
  "fact", set source_id to null -- you have no real sources, this is a
  baseline with no retrieval, don't invent a fake source identifier.
- emotional_temperature: one sentence capturing the tone (a sensory
  metaphor).
- why_now: why watch it now, without spoiling anything.
- script: 1-3 blocks with timings in seconds, on-screen text, voiceover,
  and visual direction. The voiceover is also spoiler surface: the same
  rules apply there.

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
TOOL_DESCRIPTION = "Emits the pre-viewing brief in the required format."
