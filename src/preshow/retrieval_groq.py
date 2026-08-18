"""Milestone 1 generator: same provider (Groq) and same PreShowBrief
schema as GroqBaselineGenerator (Milestone 0), but `pre_show()` builds a
real, GREEN-only corpus (retrieval.py) and passes it to the model instead
of an empty list -- the one thing this milestone exists to test.

Same measurement code (run_eval.py/metrics.py) either way, so a
leakage_rate/grounded_fact_rate difference between `baseline-groq` and
this generator measures the effect of retrieval itself, not a difference
in how leaks are counted (see run_eval.py's own docstring on this).

Needs: pip install groq
       GROQ_API_KEY set as an env var, or in .env (see preshow/env.py)
       Network access to en.wikipedia.org (see preshow/wikipedia.py)
"""

from __future__ import annotations

import json

from groq import BadRequestError, Groq

from .env import read_env
from .groq_retry import GroqPacer, call_with_retry
from .retrieval import build_green_corpus
from .retrieval_prompts import (
    BRIEF_PROPERTIES,
    BRIEF_REQUIRED,
    SYSTEM_PROMPT,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    build_user_prompt,
)
from .schemas import Claim, DeepDive, PreShowBrief, ScriptBlock, SourceDoc, TitleCase

_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": BRIEF_PROPERTIES,
            "required": BRIEF_REQUIRED,
        },
    },
}


class GroqRetrievalGenerator:
    """Generator (Protocol), Milestone 1: real GREEN-tier retrieval
    instead of Milestone 0's `corpus=[]`. `pre_show()` ignores its own
    `corpus` argument and builds a fresh one via `build_green_corpus` --
    the Generator protocol takes `corpus` as a parameter for
    Milestone 2/3 flexibility (e.g. a pre-built corpus reused across
    generators), but this generator's whole purpose is exercising the
    retrieval step itself, not accepting someone else's."""

    name = "retrieval-groq"

    # Observed live: this generator's prompt (real retrieved text, unlike
    # baseline_groq's title-only prompt) uses ~2,500-4,300 tokens/call --
    # a `RateLimitError` (429, "please try again in Ns") hit on the 3rd
    # consecutive title even with no other traffic that minute, on both
    # llama-3.3-70b-versatile (12K TPM) and llama-3.1-8b-instant (6K TPM,
    # tighter). This is a per-MINUTE limit, unlike the daily TPD quota
    # that blocks separately -- it clears with a short wait, so pacing +
    # retry actually works here (see groq_retry.py -- shared with
    # baseline_groq.py/research_assist.py, previously duplicated).
    MIN_CALL_INTERVAL_S = 40.0

    # llama-3.3-70b-versatile/llama-3.1-8b-instant were both decommissioned
    # by Groq on 2026-08-18 (confirmed live: a call to either now 404s).
    # openai/gpt-oss-120b is Groq's documented replacement, and all
    # free/developer-tier models now share identical limits regardless of
    # size (RPM 30, RPD 1K, TPM 8K, TPD 200K, checked live) -- Milestone
    # 1's `--model llama-3.1-8b-instant` override (chasing a bigger daily
    # budget on a smaller model, D16) no longer has anything to chase;
    # every model gets the same quota now.
    def __init__(self, client: Groq | None = None, model: str = "openai/gpt-oss-120b"):
        self._client = client or Groq(api_key=read_env("GROQ_API_KEY"))
        self._model = model
        self._pacer = GroqPacer(self.MIN_CALL_INTERVAL_S)

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief:
        green_corpus = build_green_corpus(case.title, case.year)
        data = self._call_with_retry(case, green_corpus)
        return PreShowBrief(
            title_id=case.title_id,
            context_bullets=[Claim(**c) for c in data["context_bullets"]],
            author_voice=[Claim(**c) for c in data["author_voice"]],
            emotional_temperature=data["emotional_temperature"],
            why_now=data["why_now"],
            script=[ScriptBlock(**b) for b in data["script"]],
        )

    def deep_dive(self, case: TitleCase, corpus: list[SourceDoc]) -> DeepDive:
        raise NotImplementedError("Milestone 1 only covers pre_show; deep_dive lands in a later milestone")

    def _call_with_retry(self, case: TitleCase, green_corpus: list[SourceDoc]) -> dict:
        """Pacing + RateLimitError retry now live in groq_retry.py,
        shared with baseline_groq.py/research_assist.py (previously
        duplicated near-verbatim here and in research_assist.py). The
        BadRequestError corrective-retry logic below is unchanged by
        that refactor -- still escalates after the first failure (see
        the comment inside correction_text)."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(case.title, case.year, green_corpus)},
        ]

        def make_call() -> dict:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=1536,  # more headroom than baseline_groq's 1024 --
                # the retrieved-text prompt is longer, and citing real
                # source_ids tends to produce longer claim text
                messages=messages,
                tools=[_TOOL],
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            )
            call = resp.choices[0].message.tool_calls[0]
            return json.loads(call.function.arguments)

        def correction_text(e: BadRequestError, attempt: int) -> str:
            correction = (
                "That call was rejected: every object in an array must include "
                "ALL of its required properties (context_bullets/author_voice items need "
                "text, kind, AND source_id -- source_id must be null for "
                "kind=\"interpretation\" and must be one of the given source_ids for "
                "kind=\"fact\"; script blocks need start_s, end_s, on_screen_text, "
                "voiceover, AND visual_direction). Retry, filling in every required field "
                "on every item."
            )
            if attempt >= 1:
                # Repeating the same generic reminder wasn't enough once
                # already (observed live: llama-3.1-8b-instant
                # persistently dropped `end_s` from every script block
                # across all attempts on one title, Come and See) -- name
                # the specific mistake instead.
                correction += (
                    f" Specifically: {e}. Double-check EVERY script block has a numeric "
                    "end_s (in seconds, greater than that block's start_s) -- this is the "
                    "field most often missed."
                )
            return correction

        # max_attempts=5, not 3 -- see correction_text's docstring note
        # above on why one extra escalation step wasn't enough alone.
        return call_with_retry(make_call, messages, correction_text, self._pacer, max_attempts=5)
