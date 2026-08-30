"""Tests for preshow/groq_retry.py -- the pacing + retry loop shared by
baseline_groq.py, retrieval_groq.py, and webapp/research_assist.py
(previously duplicated near-verbatim across the last two). Uses real
groq.BadRequestError/RateLimitError instances (constructible without a
network call -- see their __init__ signature) so this exercises the
actual except clauses, not stand-ins for them. No network, no sleeping:
GroqPacer.wait() is a no-op when min_interval_s=0.
"""

import sys
from pathlib import Path

import pytest

httpx = pytest.importorskip("httpx")
groq = pytest.importorskip("groq")
BadRequestError = groq.BadRequestError
RateLimitError = groq.RateLimitError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preshow.groq_retry import GroqPacer, call_with_retry  # noqa: E402


def _bad_request(message: str = "bad") -> BadRequestError:
    resp = httpx.Response(status_code=400, request=httpx.Request("POST", "https://example.com"))
    return BadRequestError(message, response=resp, body=None)


def _rate_limited(message: str = "rate limited") -> RateLimitError:
    resp = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
    return RateLimitError(message, response=resp, body=None)


def test_succeeds_on_first_attempt_without_any_correction():
    calls = []

    def make_call():
        calls.append(1)
        return {"ok": True}

    messages = [{"role": "system", "content": "sys"}]
    result = call_with_retry(make_call, messages, lambda e, a: "unused", GroqPacer(0.0), max_attempts=3)

    assert result == {"ok": True}
    assert len(calls) == 1
    assert messages == [{"role": "system", "content": "sys"}]  # nothing appended


def test_bad_request_appends_correction_and_retries():
    attempts = []

    def make_call():
        attempts.append(1)
        if len(attempts) < 3:
            raise _bad_request(f"missing field on attempt {len(attempts)}")
        return {"ok": True}

    seen_corrections = []

    def correction_text(e, attempt):
        seen_corrections.append((attempt, str(e)))
        return f"correction for attempt {attempt}"

    messages = [{"role": "system", "content": "sys"}]
    result = call_with_retry(make_call, messages, correction_text, GroqPacer(0.0), max_attempts=5)

    assert result == {"ok": True}
    assert len(attempts) == 3
    # one corrective message appended per failed attempt (2 failures)
    assert [m["content"] for m in messages[1:]] == ["correction for attempt 0", "correction for attempt 1"]
    assert [a for a, _ in seen_corrections] == [0, 1]


def test_bad_request_raises_after_max_attempts_exhausted():
    def make_call():
        raise _bad_request("always rejected")

    messages = []
    with pytest.raises(BadRequestError):
        call_with_retry(make_call, messages, lambda e, a: "correction", GroqPacer(0.0), max_attempts=3)


def test_rate_limit_retries_without_touching_messages():
    attempts = []

    def make_call():
        attempts.append(1)
        if len(attempts) < 2:
            raise _rate_limited()
        return {"ok": True}

    messages = [{"role": "system", "content": "sys"}]
    result = call_with_retry(make_call, messages, lambda e, a: "unused", GroqPacer(0.0), max_attempts=3)

    assert result == {"ok": True}
    assert len(attempts) == 2
    assert messages == [{"role": "system", "content": "sys"}]  # RateLimitError never touches messages


def test_rate_limit_raises_after_max_attempts_exhausted():
    def make_call():
        raise _rate_limited()

    with pytest.raises(RateLimitError):
        call_with_retry(make_call, [], lambda e, a: "unused", GroqPacer(0.0), max_attempts=2)


def test_pacer_marks_call_after_every_attempt_including_failures():
    pacer = GroqPacer(0.0)
    assert pacer._last_call == 0.0

    def make_call():
        raise _bad_request()

    with pytest.raises(BadRequestError):
        call_with_retry(make_call, [], lambda e, a: "c", pacer, max_attempts=1)

    assert pacer._last_call > 0.0  # marked even though every attempt failed
