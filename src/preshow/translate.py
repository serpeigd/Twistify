"""Best-effort machine translation of researched content into Spanish.

NOT the same thing as the hand-researched English content (see docs/DESIGN.md
"Pending" for why the two shouldn't be conflated): this is automatic,
free-tier translation for the UI toggle, not cited research. Uses MyMemory's
free public API (https://mymemory.translated.net) -- no API key, stdlib only,
so it doesn't add a dependency or a paid-key requirement to the project.

Every string is translated independently and falls back to the original
English on any failure (timeout, quota, network) -- a missed translation
shows English text, never an error.
"""

from __future__ import annotations

import copy
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, wait

_MAX_CHARS = 480  # MyMemory works best comfortably under ~500 chars/query
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# MyMemory's free anonymous tier throttles bursts hard (measured: 16-way
# concurrency triggers 429s within a couple of calls, and this specific
# host's shared egress IP looks close to its daily cap during dev/testing).
# Low concurrency + retry/backoff on 429 is the well-behaved way to call it;
# the overall time budget in translate_pack_dump is the actual safety net --
# if MyMemory is degraded when this runs for real, the page must still load
# in English within a bounded time, not hang.
_MAX_WORKERS = 4
_MAX_RETRIES = 2
_BACKOFF_BASE_S = 1.5
_TOTAL_BUDGET_S = 25


def _translate_chunk(text: str) -> str:
    if not text or not text.strip():
        return text
    url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode(
        {"q": text, "langpair": "en|es"}
    )
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            translated = data.get("responseData", {}).get("translatedText")
            if translated and data.get("responseStatus") == 200:
                return translated
            return text  # a real (non-429) API error -- fall back, don't retry
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < _MAX_RETRIES - 1:
                time.sleep(_BACKOFF_BASE_S * (attempt + 1))
                continue
            return text
        except Exception:
            return text
    return text


def translate_text(text: str | None) -> str | None:
    """Translates arbitrary-length text by chunking on paragraph/sentence
    boundaries so no single request exceeds MyMemory's practical limit."""
    if not text or not text.strip():
        return text

    translated_paragraphs = []
    for para in text.split("\n\n"):
        if len(para) <= _MAX_CHARS:
            translated_paragraphs.append(_translate_chunk(para))
            continue
        chunks: list[str] = []
        buf = ""
        for sentence in _SENTENCE_SPLIT.split(para):
            candidate = f"{buf} {sentence}".strip()
            if len(candidate) > _MAX_CHARS and buf:
                chunks.append(buf)
                buf = sentence
            else:
                buf = candidate
        if buf:
            chunks.append(buf)
        translated_paragraphs.append(" ".join(_translate_chunk(c) for c in chunks))
    return "\n\n".join(translated_paragraphs)


def translate_pack_dump(data: dict) -> tuple[dict, bool]:
    """Translates the prose fields of a `ContentPack.model_dump()`-shaped
    dict. Deliberately left untranslated: `title_id`, `director`, `themes`
    (canonical filter vocabulary, matched against the frontend's THEME_META),
    `sources` (URLs), and every `source_id`/`url`/`kind` -- none of that is
    prose, translating it would break matching or cite the wrong thing.

    Returns `(translated_dict, all_translated)`. `all_translated` is True
    only if every single job succeeded -- see the all-or-nothing note
    further down for why a weaker bar (e.g. "at least one job succeeded")
    isn't safe to cache on.

    A full pack is 60-90 short strings. Collecting them into `jobs` and
    running the HTTP calls through a thread pool (network-bound, GIL isn't
    the bottleneck) is what keeps the first, uncached view of a title from
    taking a minute or more -- translating them one at a time was the
    actual bug behind an early timeout during testing."""
    out = copy.deepcopy(data)
    jobs: list[tuple[dict, str]] = []  # (container, key) -- container[key] gets overwritten

    def register(container, key):
        if container.get(key):
            jobs.append((container, key))

    def register_list(lst):
        for i in range(len(lst)):
            jobs.append((lst, i))

    register(out, "story")
    register_list(out.get("context_bullets") or [])

    for b in out.get("before_watching") or []:
        register(b, "lead")
        register(b, "text")
    for a in out.get("author_voice") or []:
        register(a, "text")

    register(out, "emotional_temperature")
    register(out, "why_now")

    for m in out.get("metaphors") or []:
        register(m, "text")
    for r in out.get("intertextual_refs") or []:
        register(r, "text")
    for p in out.get("production_trivia") or []:
        register(p, "text")
    for s in out.get("scene_analysis") or []:
        register(s, "scene")
        register(s, "text")

    cc = out.get("critical_consensus")
    if cc:
        register(cc, "summary")
        register_list(cc.get("awards") or [])

    register_list(out.get("strengths") or [])
    register_list(out.get("weaknesses") or [])
    register(out, "verdict")
    register(out, "useless_fact")
    for f in out.get("fun_facts") or []:
        register(f, "lead")
        register(f, "text")

    register_list(out.get("questions") or [])
    register_list(out.get("debate_prompts") or [])
    register(out, "cta")

    translated_count = 0
    pool = ThreadPoolExecutor(max_workers=_MAX_WORKERS)
    try:
        futures = {
            pool.submit(translate_text, container[key]): (container, key)
            for container, key in jobs
        }
        done, not_done = wait(futures, timeout=_TOTAL_BUDGET_S)
        for future in done:
            container, key = futures[future]
            try:
                result = future.result()
            except Exception:
                continue  # container[key] already holds the pre-translation English text
            if result != container[key]:
                translated_count += 1
            container[key] = result
        # `not_done` (budget ran out): leave the English text already in place.
        # Don't block returning on threads still mid-request -- shutdown(wait=False)
        # lets them finish or die in the background instead of hanging the request.
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    # All-or-nothing on purpose: a partial run (some fields translated, some
    # rate-limited back to English) is worse than a full failure, because it
    # LOOKS complete -- cached and marked auto_translated, quietly showing a
    # patchwork of Spanish and English forever. Anything less than "every
    # job succeeded" doesn't get cached, so the next attempt (e.g. after
    # switching IP/VPN) starts clean instead of only mopping up stragglers
    # from a half-cached file. This was found the hard way: an earlier
    # "translated something" bar cached 3 of 7 titles half-translated.
    all_translated = bool(jobs) and translated_count == len(jobs)
    return out, all_translated
