"""Research assistant: drafts a ContentPack (content/researched/*.json
shape) for a movie by retrieving REAL text (Wikipedia + TMDB) and asking
an LLM to draft from that text only -- never from its own memory.

This does NOT relax D6/D7 (ground truth must be cited, real sources).
What it automates is the LABOR of finding and reading sources, not the
citation requirement itself: every factual claim's source_id must be a
URL this script actually fetched, and the prompt explicitly forbids
grounding a claim in anything else. What it can't automate is the final
check that the draft is actually accurate -- an LLM can still misread
retrieved text correctly-cited. That's why this writes to
content/_drafts/ (gitignored), not straight into content/researched/:
a human review pass before publishing is still the gate, same as the
existing 8 entries, just reviewing instead of writing from scratch.

Sources used: Wikipedia (CC BY-SA, via src/preshow/wikipedia.py) for
plot/production/reception/accolades text, TMDB (src/preshow/tmdb.py) for
director/year/genres. Deliberately NOT scraping Rotten Tomatoes or
Metacritic directly (no simple free API for either) -- where Wikipedia's
own "Critical response" prose states a score, it's cited to the
Wikipedia URL actually fetched, not to rottentomatoes.com/metacritic.com,
because this script never visited those pages itself. That's a real gap
against the 8 hand-researched entries (which do cite RT/Metacritic
directly) -- said plainly, not hidden.

Usage:
    python webapp/research_assist.py "Citizen Kane" 1941 [n_candidates=3]

Generates `n_candidates` independent drafts from the same retrieved text
and keeps the one with the most grounded claims (see draft_best_of) --
added after testing showed a single call from this free model is
inconsistent run to run (D14 in docs/DESIGN.md).

Needs a free GROQ_API_KEY in .env (same as evals/run_eval.py --generator
baseline-groq).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from groq import BadRequestError, Groq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preshow import tmdb, wikipedia  # noqa: E402
from preshow.env import read_env  # noqa: E402
from preshow.groq_retry import GroqPacer, call_with_retry, safe_print  # noqa: E402

DRAFTS_DIR = ROOT / "content" / "_drafts"
# llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-18
# (confirmed live: a call to it now 404s) -- openai/gpt-oss-120b is
# Groq's documented replacement, and the only one of the three
# replacement models with a comparable context/capability profile for
# this script's richer 20-field schema. All free/developer-tier models
# now share one flat TPM cap (8,000, checked live), tighter than the old
# 70b model's 12K -- a full 5-chunk request at the old truncation asked
# for 10,805 tokens on Dune (413 "Request too large"). Explicit call
# with the user: cut MAX_CHARS_PER_CHUNK and max_tokens below to fit
# (real content-richness tradeoff, not free), rather than wait out a
# bigger model that no longer exists.
MODEL = "openai/gpt-oss-120b"

# Same lesson as evals/run_eval.py's --model flag / retrieval_groq.py's
# pacing (see D16): draft_best_of() alone makes 2n calls/title (pre-show
# + deep-dive per candidate, see PRE_SHOW_PROPERTIES' comment) against
# this model's 8K TPM / 200K TPD limits, well before any multi-title
# batch. Module-level (not per-instance -- this file has no class to
# hang it off) because draft_best_of calls _call_pre_show/_call_deep_dive
# several times in a row for the same title. Pacing/RateLimitError retry
# now live in preshow/groq_retry.py, shared with
# baseline_groq.py/retrieval_groq.py (previously duplicated near-verbatim
# here and in retrieval_groq.py).
MIN_CALL_INTERVAL_S = 40.0
_pacer = GroqPacer(MIN_CALL_INTERVAL_S)

# Must match webapp/index.html's THEME_META -- the app only filters by
# this closed vocabulary, so a theme outside it would silently never
# match any filter chip.
THEME_VOCAB = [
    "Identity",
    "Grief and Loss",
    "Obsession",
    "Class and Power",
    "Morality and Justice",
    "Perception",
]

SYSTEM_PROMPT = f"""You are drafting a movie research entry for Twistify, a spoiler-safety \
project whose entire premise is that factual claims must be traceable to a real source, \
never invented. You will be given RETRIEVED TEXT below, each chunk labeled with its exact \
URL. This retrieved text is the ONLY source of facts you may use.

Rules, none negotiable:
1. Every claim you mark kind="fact" (or every before_watching/production_trivia/fun_facts/
   critical_consensus item) must be directly supported by the retrieved text, and its
   source_id (or, for critical_consensus_scores, its "url") must be EXACTLY one of the URLs
   given below -- never a URL you weren't given, never invented, never your own general
   knowledge of the film.
2. You were NOT given a Rotten Tomatoes or Metacritic page directly -- only Wikipedia. If
   Wikipedia's own text reports a score from one of those sites, you may report the number
   in critical_consensus_scores, but its "url" must be null (you did not visit that page
   yourself) -- never fabricate a rottentomatoes.com or metacritic.com URL.
3. If the retrieved text doesn't support a field (e.g. no accolades are mentioned), leave
   that field empty. An empty list is correct and expected; a fabricated entry is not.
4. Interpretive claims (your own reading of the film -- metaphors, intertextual_refs,
   scene_analysis) get kind="interpretation" and source_id=null. These don't need retrieved
   support since they're your analysis, not a fact -- but say so honestly via "kind".
5. "themes" must be chosen ONLY from this exact list (pick 1-2, whichever genuinely fit):
   {", ".join(THEME_VOCAB)}
6. Write in the same voice as a film-literate critic, not marketing copy. Spoiler-free
   fields (story, context_bullets, before_watching, author_voice, emotional_temperature,
   why_now) must not reveal plot twists or the ending. On the pre-show call, you are
   simply NOT GIVEN the plot/reception text (see RETRIEVED TEXT below) -- that's the
   real guarantee, this instruction is reinforcement, not the mechanism. Even so, don't
   infer or guess at a twist from what production/overview text implies.
7. "critical_consensus_scores" is ONLY for a specific, named, numeric or ranked score from
   a real aggregator or organization found in the retrieved text (e.g. "88% on Rotten
   Tomatoes, 374 reviews", "#4 on WGA's list of 101 Greatest Screenplays"). A vague
   reputational claim ("widely considered one of the greatest films ever") is NOT a score
   -- it belongs in critical_consensus_summary as prose, never invented as a fake score
   entry just to fill the field.
8. "questions" and "debate_prompts" are NOT the same thing and must never repeat each
   other. "questions" (multiple, open-ended, no side to pick): things the film leaves you
   wondering about. "debate_prompts" (exactly ONE item): a genuine two-sided disagreement
   about the film itself -- something a real critic or the retrieved text disagrees about,
   framed so a reader can take one side or the other and argue it, e.g. "Is this film's
   [X] a real achievement, or does its [Y] undercut it?" If the retrieved text doesn't
   surface a real point of critical disagreement, write a debate prompt about form or
   reception (e.g. box office vs. critical reputation) rather than duplicating a question.
9. NO REPETITION ACROSS FIELDS. Each field must say something the others don't. If you
   catch yourself writing "corrupting influence of power" or "innovative cinematography"
   (or any other single idea) in more than one field, replace the repeat with a different,
   more specific observation -- pull another fact from the retrieved text instead of
   restating the same one three different ways.
10. MINE THE RETRIEVED TEXT FOR SPECIFICS. Prefer a real number, name, date, or direct quote
   found in the retrieved text over a generic critical-consensus sentence. "The film cost
   $839,727 to make" beats "the film was made on a modest budget"; an exhibitor's actual
   quoted reaction beats "critics praised the performances." If the retrieved text contains
   a quote, a dollar figure, a date, or a named person's stated opinion, use it verbatim or
   near-verbatim (with its source_id) instead of paraphrasing it into a vague generality.
   Generic film-criticism platitudes with no specific detail behind them are the failure
   mode this rule exists to prevent.
11. "author_voice" is a QUOTE, not your opinion. Each item must be something an actual
   person who made the film (director, screenwriter, cinematographer, composer, a lead
   actor) is reported as having SAID or WRITTEN about it, attributed by name in the
   retrieved text -- e.g. "Welles later said...", "In a 1941 interview, [name] explained...".
   Never write your own critical assessment here ("this film's cinematography is
   groundbreaking") even if it's true and even if it cites a source_id -- that belongs in
   critical_consensus_summary or scene_analysis instead. If the retrieved text contains no
   attributed quote from someone who made the film, leave author_voice EMPTY -- an empty
   list is correct and expected here (see rule 3), a critic-voice paraphrase is not.
12. DEPTH IS NOT OPTIONAL. metaphors, intertextual_refs, production_trivia,
   scene_analysis, strengths, weaknesses, fun_facts, and questions have no fixed max --
   aim for 3-5 items in EACH, not 1-2, whenever the retrieved text has that much real
   material (it usually does: a full Wikipedia article has many distinct facts, only a
   few of which any one item can cover). Stopping at one item per field after finding
   the FIRST usable fact is the single most common failure mode -- keep reading the
   retrieved text for a second and third distinct angle before moving to the next field.
   Rule 3 (leave a field empty/short if the text doesn't support more) still applies --
   this rule is about not stopping early when it DOES, not about padding with filler.

EXAMPLE OF THE TARGET DEPTH (illustrative only -- about a different, unnamed film, not the
one you're drafting -- match this level of specificity, not this content):

  BAD  (generic, would be rejected): "before_watching": [{{"lead": "A troubled production",
        "text": "The film had a difficult shoot with several challenges.", "source_id": "..."}}]

  GOOD (specific, grounded): "before_watching": [{{"lead": "The lead actress was a late
        replacement", "text": "The studio's first choice dropped out three weeks before
        filming after a scheduling conflict with another project; the director had worked
        with her eventual replacement once before, on a film that lost money.",
        "source_id": "..."}}]

  BAD  (generic): "production_trivia": [{{"text": "The film was made on a modest budget.",
        "source_id": "..."}}]

  GOOD (specific): "production_trivia": [{{"text": "Budgeted at $6.2 million, it went
        $1.4 million over after a hurricane delayed exterior shooting by eleven days.",
        "source_id": "..."}}]

  BAD  (this is YOUR opinion wearing a citation, not a quote -- rejected by rule 11):
        "author_voice": [{{"text": "As a film-literate critic, I approach this film with a
        deep appreciation for its groundbreaking cinematography.", "source_id": "..."}}]

  GOOD (an attributed quote from someone who made the film): "author_voice": [{{"text":
        "The director later said the ending was rewritten twice because test audiences
        found the original too bleak.", "source_id": "..."}}]

Every item you write should look like the GOOD examples: a real number, name, date, or
quote from the retrieved text -- not a paraphrase that could apply to almost any film.
"""

_SOURCED = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "source_id": {"type": ["string", "null"]},
    },
    "required": ["text", "source_id"],
}
_SOURCED_KIND = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "kind": {"type": "string", "enum": ["fact", "interpretation"]},
        "source_id": {"type": ["string", "null"]},
    },
    "required": ["text", "kind", "source_id"],
}
_FACT_BULLET = {
    "type": "object",
    "properties": {
        "lead": {"type": "string"},
        "text": {"type": "string"},
        "source_id": {"type": ["string", "null"]},
    },
    "required": ["lead", "text", "source_id"],
}

CONTENT_PROPERTIES = {
    "story": {"type": "string", "description": "Spoiler-free narrative hook, 3-5 short paragraphs."},
    "context_bullets": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    "before_watching": {"type": "array", "items": _FACT_BULLET, "maxItems": 3},
    "author_voice": {
        "type": "array",
        "items": _SOURCED,
        "maxItems": 3,
        "description": "Attributed quotes from someone who made the film (director, "
        "writer, cast), never the drafter's own critical opinion. Empty list if the "
        "retrieved text has no such quote -- see rule 11.",
    },
    "emotional_temperature": {"type": "string"},
    "why_now": {"type": "string"},
    "metaphors": {
        "type": "array", "items": _SOURCED_KIND,
        "description": "3-5 distinct readings when the film supports that many -- see rule 12. "
        "One item per field is under-depth, not caution.",
    },
    "intertextual_refs": {
        "type": "array", "items": _SOURCED_KIND,
        "description": "3-5 distinct references/influences/comparisons when the retrieved text "
        "supports that many -- see rule 12.",
    },
    "production_trivia": {
        "type": "array", "items": _SOURCED,
        "description": "3-5 distinct, specific facts (budget, schedule, casting, technical "
        "choices, behind-the-scenes) when the retrieved text supports that many -- see rule 12.",
    },
    "scene_analysis": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "scene": {"type": "string"},
                "text": {"type": "string"},
                "source_id": {"type": ["string", "null"]},
            },
            "required": ["scene", "text", "source_id"],
        },
        "description": "2-4 distinct scenes/moments worth analyzing when the retrieved text "
        "supports that many -- see rule 12.",
    },
    "critical_consensus_summary": {"type": ["string", "null"]},
    "critical_consensus_scores": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "value": {"type": "string"},
                "url": {"type": ["string", "null"]},
            },
            "required": ["source", "value", "url"],
        },
    },
    "critical_consensus_awards": {"type": "array", "items": {"type": "string"}},
    "strengths": {
        "type": "array", "items": {"type": "string"},
        "description": "3-5 distinct strengths when the retrieved text supports that many -- see rule 12.",
    },
    "weaknesses": {
        "type": "array", "items": {"type": "string"},
        "description": "2-4 distinct weaknesses/criticisms when the retrieved text supports that many -- see rule 12.",
    },
    "verdict": {"type": "string"},
    "useless_fact": {"type": ["string", "null"]},
    "fun_facts": {
        "type": "array", "items": _FACT_BULLET, "maxItems": 4,
        "description": "Up to 4 -- use all 4 when the retrieved text supports that many, see rule 12.",
    },
    "questions": {
        "type": "array", "items": {"type": "string"},
        "description": "3-4 distinct open-ended questions -- see rule 12.",
    },
    "debate_prompts": {"type": "array", "items": {"type": "string"}, "maxItems": 1},
    "cta": {"type": "string"},
    "themes": {"type": "array", "items": {"type": "string", "enum": THEME_VOCAB}, "maxItems": 2},
}
CONTENT_REQUIRED = [
    "story", "context_bullets", "before_watching", "author_voice",
    "emotional_temperature", "why_now", "metaphors", "intertextual_refs",
    "production_trivia", "scene_analysis", "critical_consensus_summary",
    "critical_consensus_scores", "critical_consensus_awards", "strengths",
    "weaknesses", "verdict", "useless_fact", "fun_facts", "questions",
    "debate_prompts", "cta", "themes",
]

# --- Two-call structural partition (2026-08-18, replacing a single call
# over the full schema) -----------------------------------------------
#
# D3/D16's whole thesis is that spoiler safety must be a context
# PARTITION (the generator never sees the spoiler text), not an
# instruction the model could ignore -- exactly the property
# src/preshow/retrieval.py's build_green_corpus() already gives the
# measurement track. This script didn't have that: one call got ALL
# retrieved text (including "plot") and rule 6 above just ASKED it to
# keep plot/reception out of the pre-viewing fields. Live proof this
# doesn't hold: promoting the first batch of drafts and diffing them
# against evals/dataset/spoilers/ found 4/13 titles stating a
# core/major documented spoiler directly in "story" (Tetsuo, Come and
# See, Coherence, Cronocrímenes -- see git history same day).
#
# Fix: PRE_SHOW_PROPERTIES/PRE_SHOW_REQUIRED (the same six fields rule 6
# already named) are generated from a call that is only ever given
# overview/production/accolades chunks -- plot and reception are
# filtered out before the request is built, not just discouraged
# in-prompt (see _call_pre_show/PRE_SHOW_SECTIONS below). Everything
# else (DEEP_DIVE_PROPERTIES/DEEP_DIVE_REQUIRED) is a separate call that
# DOES get the full retrieved text, same as ContentPack's own Phase
# 2+3 grouping (content.py) -- "spoilers are the product" there, by
# design.
#
# IMPORTANT CAVEAT, not fully solved by this: D16 also found the leak
# for Tetsuo/Hard to Be a God specifically traces to Wikipedia's
# `overview` section ITSELF stating the twist, not to `plot` leaking
# through -- a section this fix still includes in the pre-show call
# (it has to; overview is legitimately needed for a real synopsis).
# This fix removes the "plot/reception leaks into pre-show" failure
# class (Coherence/Cronocrímenes' likely cause) but does NOT guarantee
# every title's `overview` prose is itself spoiler-free. Human review
# before content/researched/ stays the real gate, not this alone.
PRE_SHOW_PROPERTIES = {
    k: CONTENT_PROPERTIES[k]
    for k in ("story", "context_bullets", "before_watching", "author_voice", "emotional_temperature", "why_now")
}
PRE_SHOW_REQUIRED = list(PRE_SHOW_PROPERTIES)

DEEP_DIVE_PROPERTIES = {k: v for k, v in CONTENT_PROPERTIES.items() if k not in PRE_SHOW_PROPERTIES}
DEEP_DIVE_REQUIRED = [f for f in CONTENT_REQUIRED if f not in PRE_SHOW_PROPERTIES]

PRE_SHOW_TOOL_NAME = "emit_pre_show_fields"
DEEP_DIVE_TOOL_NAME = "emit_deep_dive_fields"

_PRE_SHOW_TOOL = {
    "type": "function",
    "function": {
        "name": PRE_SHOW_TOOL_NAME,
        "description": "Emit the spoiler-free, pre-viewing fields of a Twistify content draft. "
        "You have NOT been given plot or reception text for this call -- work only from what "
        "you were actually given (overview/production/accolades, where available).",
        "parameters": {
            "type": "object",
            "properties": PRE_SHOW_PROPERTIES,
            "required": PRE_SHOW_REQUIRED,
        },
    },
}
_DEEP_DIVE_TOOL = {
    "type": "function",
    "function": {
        "name": DEEP_DIVE_TOOL_NAME,
        "description": "Emit the post-viewing, critical-reception, and engagement fields of a "
        "Twistify content draft -- spoilers are expected and fine here.",
        "parameters": {
            "type": "object",
            "properties": DEEP_DIVE_PROPERTIES,
            "required": DEEP_DIVE_REQUIRED,
        },
    },
}

# GREEN tier, same as src/preshow/retrieval.py's GREEN_SECTIONS (D3/D16)
# -- the pre-show call is only ever built from chunks in this set.
PRE_SHOW_SECTIONS = {"wikipedia:overview", "wikipedia:production", "wikipedia:accolades"}


def sanitize_grounding(pack: dict, allowed_urls: set[str]) -> int:
    """Hard enforcement of "no fabricated citation", in code rather than
    only via prompt instruction (same lesson as D3: don't trust the
    generating model to police itself). Walks every source_id/url in the
    pack and nulls out anything that isn't exactly one of the URLs this
    script actually retrieved. Returns how many were stripped -- a
    non-zero count means the model tried to cite something it wasn't
    given, worth surfacing to whoever reviews the draft."""
    stripped = 0

    def clean_field(container: dict, key: str) -> None:
        nonlocal stripped
        val = container.get(key)
        if val and val not in allowed_urls:
            container[key] = None
            stripped += 1

    for field in ("before_watching", "author_voice", "metaphors", "intertextual_refs",
                  "production_trivia", "fun_facts"):
        for item in pack.get(field, []):
            clean_field(item, "source_id")
    for item in pack.get("scene_analysis", []):
        clean_field(item, "source_id")
    for item in pack["critical_consensus"].get("scores", []):
        clean_field(item, "url")
    return stripped


def slugify(title: str, year: int) -> str:
    base = "".join(c if c.isalnum() else "_" for c in title.lower()).strip("_")
    while "__" in base:
        base = base.replace("__", "_")
    return f"{base}_{year}"


# Was 6000, then 3000 -- cut twice after Groq's 2026-08-18 model
# decommission (see MODEL above) left every free/developer-tier model at
# a flat 8K TPM, tighter than the old 70b model's 12K. 6000 chars/chunk
# asked for 10,805 tokens on Dune (413 "Request too large"). 3000
# chars/chunk fit under TPM but then max_tokens=3000 wasn't enough
# completion budget for this rich schema (a DIFFERENT failure -- "Failed
# to parse tool call arguments as JSON", the model ran out of room
# mid-response, not a 413 -- see _call's make_call comment). 2000
# chars/chunk x up to 5 chunks (worst case ~2500 tokens) + the
# ~1,672-token SYSTEM_PROMPT + max_tokens=3500 leaves real margin under
# 8000 even in the worst case -- not reverified against every title, so
# a 413 is still possible on an unusually long article; call_with_retry
# doesn't turn that into an infinite loop, it just raises (D16).
MAX_CHARS_PER_CHUNK = 2000


def retrieve(title: str, year: int) -> tuple[list[dict], dict]:
    """Fetches everything this script is allowed to cite from. Returns
    (source_chunks, tmdb_meta). Each chunk: {label, url, text}."""
    chunks: list[dict] = []

    page_title = wikipedia.find_page_title(title, year)
    if page_title:
        article = wikipedia.fetch_article(page_title)
        if article:
            sections = wikipedia.relevant_sections(article["extract"])
            for label, text in sections.items():
                if text:
                    chunks.append(
                        {"label": f"wikipedia:{label}", "url": article["url"], "text": text[:MAX_CHARS_PER_CHUNK]}
                    )

    tmdb_movie = tmdb.best_match(title, year)
    director = tmdb.get_director(tmdb_movie["tmdb_id"]) if tmdb_movie else None
    meta = {
        "tmdb_id": tmdb_movie["tmdb_id"] if tmdb_movie else None,
        "director": director,
        "genres": tmdb_movie.get("genres") if tmdb_movie else [],
        "poster_url": tmdb_movie.get("poster_url") if tmdb_movie else None,
    }
    return chunks, meta


def build_user_prompt(title: str, year: int, chunks: list[dict], meta: dict, phase: str) -> str:
    parts = [f"MOVIE: {title} ({year})", f"PHASE: {phase}"]
    if meta.get("director"):
        parts.append(f"Director (from TMDB, already confirmed, don't re-derive): {meta['director']}")
    parts.append("\nRETRIEVED TEXT (only source of facts allowed):\n")
    for c in chunks:
        parts.append(f"--- [{c['label']}] URL: {c['url']} ---\n{c['text']}\n")
    if not chunks:
        parts.append("(No retrieved text found -- leave all fact fields empty; only interpretation-kind fields may be filled.)")
    return "\n".join(parts)


def _call(
    client: Groq,
    messages: list[dict],
    tool: dict,
    tool_name: str,
    max_tokens: int,
    required_fields_hint: str,
) -> dict:
    """Shared low-level call + schema-validation retry for both the
    pre-show and deep-dive calls below. Pacing/backoff on RateLimitError
    lives in preshow/groq_retry.py, shared with
    baseline_groq.py/retrieval_groq.py. Returns the raw tool-call
    arguments (not yet assembled into a pack)."""
    max_attempts = 5

    def make_call() -> dict:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            temperature=0.8,
            messages=messages,
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": tool_name}},
        )
        call = resp.choices[0].message.tool_calls[0]
        return json.loads(call.function.arguments)

    def correction_text(e: BadRequestError, attempt: int) -> str:
        safe_print(f"  schema validation failed (attempt {attempt + 1}/{max_attempts}), retrying: {e}")
        if "Failed to parse tool call arguments as JSON" in str(e):
            # Different failure mode from a missing-field validation
            # error -- the completion got cut off by max_tokens before
            # finishing valid JSON (observed live on the old single-call
            # schema; still possible on the deep-dive call's larger
            # schema). The generic "fill in every required field"
            # correction below doesn't address THIS cause and risks
            # truncating again in a different place; ask for brevity
            # instead so the full schema actually fits this time.
            return (
                "That call was cut off before it finished producing valid JSON -- you ran "
                "out of space. Be noticeably more concise in prose fields so the ENTIRE "
                "response, every field, fits and is valid JSON. A shorter complete answer "
                "is better than a longer incomplete one."
            )
        return (
            f"That call was rejected: every object in an array must include ALL of its "
            f"required properties ({required_fields_hint}). Retry, filling in every "
            "required field on every item."
        )

    return call_with_retry(make_call, messages, correction_text, _pacer, max_attempts=max_attempts)


def _call_pre_show(client: Groq, title: str, year: int, chunks: list[dict], meta: dict) -> dict:
    """The structural half of the fix (see PRE_SHOW_PROPERTIES' comment
    above): filters to GREEN-tier chunks BEFORE building the prompt, so
    plot/reception text is never in this call's context at all -- not
    just discouraged by instruction."""
    pre_show_chunks = [c for c in chunks if c["label"] in PRE_SHOW_SECTIONS]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(title, year, pre_show_chunks, meta, phase="pre-show (spoiler-free)")},
    ]
    return _call(
        client, messages, _PRE_SHOW_TOOL, PRE_SHOW_TOOL_NAME, max_tokens=1800,
        required_fields_hint="before_watching items need 'lead', 'text', AND 'source_id'; "
        "author_voice items need 'text' AND 'source_id'",
    )


def _call_deep_dive(client: Groq, title: str, year: int, chunks: list[dict], meta: dict) -> dict:
    """Gets the FULL retrieved text (plot/reception included) -- spoilers
    are expected and fine in these fields, same as ContentPack's own
    Phase 2+3 grouping (content.py)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(title, year, chunks, meta, phase="deep-dive (spoilers expected)")},
    ]
    return _call(
        client, messages, _DEEP_DIVE_TOOL, DEEP_DIVE_TOOL_NAME, max_tokens=3400,
        # 3200 -> 3400 alongside rule 12's richness push (2026-08-19) --
        # worst-case prompt (~1,934-token SYSTEM_PROMPT + 5 chunks at
        # MAX_CHARS_PER_CHUNK + user overhead) leaves ~3,400 tokens of
        # margin under the flat 8K TPM cap, matched here rather than left
        # unused.
        required_fields_hint="e.g. before_watching/fun_facts items need both 'lead' and 'text'; "
        "scene_analysis items need both 'scene' and 'text'",
    )


def _assemble_pack(
    title: str, year: int, pre_show_data: dict, deep_dive_data: dict, chunks: list[dict], meta: dict
) -> dict:
    title_id = slugify(title, year)
    return {
        "title_id": title_id,
        "story": pre_show_data["story"],
        "context_bullets": pre_show_data["context_bullets"],
        "before_watching": pre_show_data["before_watching"],
        "author_voice": pre_show_data["author_voice"],
        "emotional_temperature": pre_show_data["emotional_temperature"],
        "why_now": pre_show_data["why_now"],
        "metaphors": deep_dive_data["metaphors"],
        "intertextual_refs": deep_dive_data["intertextual_refs"],
        "production_trivia": deep_dive_data["production_trivia"],
        "scene_analysis": deep_dive_data["scene_analysis"],
        "critical_consensus": {
            "summary": deep_dive_data["critical_consensus_summary"],
            "scores": deep_dive_data["critical_consensus_scores"],
            "awards": deep_dive_data["critical_consensus_awards"],
        },
        "strengths": deep_dive_data["strengths"],
        "weaknesses": deep_dive_data["weaknesses"],
        "verdict": deep_dive_data["verdict"],
        "useless_fact": deep_dive_data["useless_fact"],
        "fun_facts": deep_dive_data["fun_facts"],
        "questions": deep_dive_data["questions"],
        "debate_prompts": deep_dive_data["debate_prompts"],
        "cta": deep_dive_data["cta"],
        "sources": sorted({c["url"] for c in chunks}),
        "director": meta.get("director"),
        "themes": deep_dive_data["themes"],
        # Only load-bearing for a title outside evals/dataset/titles.yaml's
        # 20-title measurement set (e.g. one added on request, not part of
        # the stratified sample) -- see ContentPack's docstring on these
        # fields. Harmless to always include: for a measurement-track
        # title, webapp/app.py's CASES (titles.yaml) is authoritative and
        # these are ignored.
        "title": title,
        "year": year,
        "tmdb_id": meta.get("tmdb_id"),
    }


def grounded_count(pack: dict) -> tuple[int, int]:
    """(claims that need a source, how many have one) -- same accounting
    as ContentPack.grounding(). Used both to report to a human and, in
    draft_best_of, as the score to rank candidates by: more real grounded
    claims is a reasonable, cheap proxy for "richer, more useful draft",
    same spirit as the eval harness's own richness metric."""
    needs = grounded = 0
    for field in ("author_voice", "metaphors", "intertextual_refs", "production_trivia"):
        for item in pack.get(field, []):
            if item.get("kind") == "interpretation":
                continue
            needs += 1
            if item.get("source_id"):
                grounded += 1
    for field in ("before_watching", "fun_facts"):
        for item in pack.get(field, []):
            needs += 1
            if item.get("source_id"):
                grounded += 1
    return needs, grounded


def grounding_summary(pack: dict) -> str:
    needs, grounded = grounded_count(pack)
    pct = round(100 * grounded / needs) if needs else 0
    return f"{grounded}/{needs} sourced claims grounded ({pct}%)"


def draft(title: str, year: int) -> dict:
    """Single-candidate draft (kept for scripts/tests that want exactly
    one call each). draft_best_of below is what main() actually uses.
    Two calls, not one -- see PRE_SHOW_PROPERTIES' comment for why."""
    chunks, meta = retrieve(title, year)
    client = Groq(api_key=read_env("GROQ_API_KEY"))
    pre_show_data = _call_pre_show(client, title, year, chunks, meta)
    deep_dive_data = _call_deep_dive(client, title, year, chunks, meta)
    pack = _assemble_pack(title, year, pre_show_data, deep_dive_data, chunks, meta)
    allowed_urls = {c["url"] for c in chunks}
    stripped = sanitize_grounding(pack, allowed_urls)
    if stripped:
        print(f"WARNING: stripped {stripped} citation(s) the model invented "
              f"(not among the {len(allowed_urls)} URL(s) actually retrieved)")
    return pack


def draft_best_of(title: str, year: int, n: int = 3, save_incrementally: bool = True) -> dict:
    """Generates `n` independent candidates from the SAME retrieved text
    and picks the one with the most grounded claims (after sanitizing
    each candidate's citations first, so a candidate can't win by
    fabricating extra ones). Exists because a single call from this free
    model is inconsistent run to run -- see D14 in docs/DESIGN.md: the
    same prompt against the same text produced anywhere from 4 to 15
    grounded claims across manual test runs. Retrieval happens once and
    is shared across all n candidates -- only the generation call varies.

    save_incrementally writes the best candidate seen so far to
    DRAFTS_DIR after every completed candidate, not just at the end --
    added after a multi-title batch lost a genuinely good candidate
    (13/13 grounded) to a mid-run TPD exhaustion, with nothing on disk
    to show for it (D16-adjacent lesson, same as run_eval.py's
    partial-results aggregation).

    Each candidate is now 2 calls (pre-show + deep-dive, see
    PRE_SHOW_PROPERTIES' comment), not 1 -- n=3 costs 6 calls/title, not
    3. Budget accordingly against Groq's flat 200K TPD (all
    free/developer models share it now, see MODEL's comment)."""
    chunks, meta = retrieve(title, year)
    print(f"retrieved {len(chunks)} source chunk(s): {[c['label'] for c in chunks]}")
    pre_show_chunks = [c for c in chunks if c["label"] in PRE_SHOW_SECTIONS]
    print(f"  of which {len(pre_show_chunks)} are GREEN-tier (pre-show call only sees these)")
    if meta.get("director"):
        print(f"director (TMDB): {meta['director']}")

    client = Groq(api_key=read_env("GROQ_API_KEY"))
    allowed_urls = {c["url"] for c in chunks}

    candidates = []
    for i in range(n):
        print(f"generating candidate {i + 1}/{n}...")
        pre_show_data = _call_pre_show(client, title, year, chunks, meta)
        deep_dive_data = _call_deep_dive(client, title, year, chunks, meta)
        pack = _assemble_pack(title, year, pre_show_data, deep_dive_data, chunks, meta)
        stripped = sanitize_grounding(pack, allowed_urls)
        needs, grounded = grounded_count(pack)
        print(f"  candidate {i + 1}: {grounded}/{needs} grounded claims"
              + (f", {stripped} fabricated citation(s) stripped" if stripped else ""))
        candidates.append((grounded, stripped, pack))

        if save_incrementally:
            _, _, best_so_far = max(candidates, key=lambda c: (c[0], -c[1]))
            DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = DRAFTS_DIR / f"{best_so_far['title_id']}.json"
            out_path.write_text(json.dumps(best_so_far, indent=2, ensure_ascii=False), encoding="utf-8")

    # Rank by most grounded claims; break ties by fewest citations that
    # had to be stripped (a candidate that tried to cite something it
    # wasn't given is less trustworthy than one that didn't, even at
    # equal grounded-claim counts).
    best_grounded, best_stripped, best_pack = max(candidates, key=lambda c: (c[0], -c[1]))
    print(f"selected the candidate with {best_grounded} grounded claims"
          + (f" ({best_stripped} stripped citations)" if best_stripped else ""))
    return best_pack


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: python webapp/research_assist.py \"<title>\" <year> [n_candidates=3]", file=sys.stderr)
        return 1
    title, year = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    pack = draft_best_of(title, year, n=n)
    print(grounding_summary(pack))

    DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DRAFTS_DIR / f"{pack['title_id']}.json"
    out_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
