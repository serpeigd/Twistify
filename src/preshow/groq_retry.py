"""Shared Groq call pacing + retry helper.

Two independent free-tier failure modes recur across every Groq-backed
generator/drafter in this project (baseline_groq.py, retrieval_groq.py,
webapp/research_assist.py) -- this factors out the fix for the one this
code CAN do anything about, previously duplicated near-verbatim in
retrieval_groq.py and research_assist.py:

- RateLimitError on tokens-per-minute (TPM): recoverable with a short
  wait. GroqPacer paces calls MIN_CALL_INTERVAL_S apart up front, and
  call_with_retry backs off + retries the same interval on a 429.
- RateLimitError on tokens-per-day (TPD) / APIStatusError 413 "Request
  too large": NOT recoverable by waiting seconds or retrying the same
  request -- call_with_retry still raises once max_attempts is used up,
  same as before this refactor. Fixing these needs a different model, a
  smaller request, or literally waiting out the daily quota -- see each
  caller's own comments (D16 in docs/DESIGN.md has the full history).

Does NOT own the BadRequestError corrective-retry text or attempt count
-- those differ per caller (retrieval_groq.py's escalates after one
failure, baseline_groq.py's/research_assist.py's don't; baseline_groq.py
also keeps its own lower max_attempts) and are exactly what
`correction_text` and `max_attempts` are for. Behavior there is
unchanged by this refactor -- only the pacing/RateLimitError plumbing
around it is now shared.
"""

from __future__ import annotations

import time
from typing import Callable

from groq import BadRequestError, RateLimitError


class GroqPacer:
    """Owns MIN_CALL_INTERVAL_S pacing state (the monotonic timestamp of
    the last call) across repeated Groq calls from one generator/drafter
    -- e.g. research_assist.py's draft_best_of() making 3 calls for one
    title, or one generator instance's calls across a whole run_eval.py
    run. One instance per generator/drafter (research_assist.py, which
    has no class to hang this off, keeps a single module-level instance
    -- equivalent to its previous module-level `_last_call` global)."""

    def __init__(self, min_interval_s: float):
        self.min_interval_s = min_interval_s
        self._last_call = 0.0

    def wait(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def mark_call(self) -> None:
        self._last_call = time.monotonic()


def call_with_retry(
    make_call: Callable[[], dict],
    messages: list[dict],
    correction_text: Callable[[BadRequestError, int], str],
    pacer: GroqPacer,
    max_attempts: int,
) -> dict:
    """Shared retry loop used by every Groq-backed caller in this
    project.

    make_call: zero-arg callable that performs ONE Groq API call against
    the CURRENT contents of `messages` and returns the parsed tool-call
    arguments as a dict -- must raise the real
    groq.BadRequestError/RateLimitError on failure, not swallow them
    (typically a closure over the caller's own client/model/tools).

    messages: the caller's own conversation list (mutated in place --
    this function appends one corrective user message to it on a
    BadRequestError, before the next attempt).

    correction_text(exception, attempt): returns whatever corrective
    text that caller wants appended before retrying. May print its own
    diagnostics as a side effect (kept per-caller: baseline_groq.py
    stays silent, research_assist.py logs, retrieval_groq.py's message
    escalates by attempt) -- this refactor doesn't touch that behavior,
    only where the surrounding loop lives.

    max_attempts: total attempts including the first, per caller (not
    unified -- baseline_groq.py deliberately kept its own lower count).
    """
    for attempt in range(max_attempts):
        pacer.wait()
        try:
            result = make_call()
            pacer.mark_call()
            return result
        except BadRequestError as e:
            pacer.mark_call()
            if attempt == max_attempts - 1:
                raise
            messages.append({"role": "user", "content": correction_text(e, attempt)})
        except RateLimitError as e:
            pacer.mark_call()
            if attempt == max_attempts - 1:
                raise
            print(f"  rate limited, backing off {pacer.min_interval_s:.0f}s: {e}")
            time.sleep(pacer.min_interval_s)
    raise RuntimeError("unreachable")
