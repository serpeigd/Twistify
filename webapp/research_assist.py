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

DRAFTS_DIR = ROOT / "content" / "_drafts"
MODEL = "llama-3.3-70b-versatile"

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
   why_now) must not reveal plot twists or the ending.
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

TOOL_NAME = "emit_content_draft"

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
    "metaphors": {"type": "array", "items": _SOURCED_KIND},
    "intertextual_refs": {"type": "array", "items": _SOURCED_KIND},
    "production_trivia": {"type": "array", "items": _SOURCED},
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
    "strengths": {"type": "array", "items": {"type": "string"}},
    "weaknesses": {"type": "array", "items": {"type": "string"}},
    "verdict": {"type": "string"},
    "useless_fact": {"type": ["string", "null"]},
    "fun_facts": {"type": "array", "items": _FACT_BULLET, "maxItems": 4},
    "questions": {"type": "array", "items": {"type": "string"}},
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

_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Emit a fully-populated Twistify content draft, grounded only in the retrieved text.",
        "parameters": {
            "type": "object",
            "properties": CONTENT_PROPERTIES,
            "required": CONTENT_REQUIRED,
        },
    },
}


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
                    chunks.append({"label": f"wikipedia:{label}", "url": article["url"], "text": text[:6000]})

    tmdb_movie = tmdb.best_match(title, year)
    director = tmdb.get_director(tmdb_movie["tmdb_id"]) if tmdb_movie else None
    meta = {
        "tmdb_id": tmdb_movie["tmdb_id"] if tmdb_movie else None,
        "director": director,
        "genres": tmdb_movie.get("genres") if tmdb_movie else [],
        "poster_url": tmdb_movie.get("poster_url") if tmdb_movie else None,
    }
    return chunks, meta


def build_user_prompt(title: str, year: int, chunks: list[dict], meta: dict) -> str:
    parts = [f"MOVIE: {title} ({year})"]
    if meta.get("director"):
        parts.append(f"Director (from TMDB, already confirmed, don't re-derive): {meta['director']}")
    parts.append("\nRETRIEVED TEXT (only source of facts allowed):\n")
    for c in chunks:
        parts.append(f"--- [{c['label']}] URL: {c['url']} ---\n{c['text']}\n")
    if not chunks:
        parts.append("(No retrieved text found -- leave all fact fields empty; only interpretation-kind fields may be filled.)")
    return "\n".join(parts)


def _call_llm(client: Groq, title: str, year: int, chunks: list[dict], meta: dict) -> dict:
    """One generation call, with the existing schema-validation retry.
    Returns the raw tool-call arguments (not yet assembled into a pack)."""
    user_prompt = build_user_prompt(title, year, chunks, meta)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=4096,
                temperature=0.8,
                messages=messages,
                tools=[_TOOL],
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
            )
            call = resp.choices[0].message.tool_calls[0]
            return json.loads(call.function.arguments)
        except BadRequestError as e:
            if attempt == 2:
                raise
            print(f"  schema validation failed (attempt {attempt + 1}/3), retrying: {e}")
            messages.append(
                {
                    "role": "user",
                    "content": "That call was rejected: every object in an array must include "
                    "ALL of its required properties (e.g. before_watching/fun_facts items need "
                    "both 'lead' and 'text'; scene_analysis items need both 'scene' and 'text'). "
                    "Retry, filling in every required field on every item.",
                }
            )
    raise RuntimeError("unreachable")


def _assemble_pack(title: str, year: int, data: dict, chunks: list[dict], meta: dict) -> dict:
    title_id = slugify(title, year)
    return {
        "title_id": title_id,
        "story": data["story"],
        "context_bullets": data["context_bullets"],
        "before_watching": data["before_watching"],
        "author_voice": data["author_voice"],
        "emotional_temperature": data["emotional_temperature"],
        "why_now": data["why_now"],
        "metaphors": data["metaphors"],
        "intertextual_refs": data["intertextual_refs"],
        "production_trivia": data["production_trivia"],
        "scene_analysis": data["scene_analysis"],
        "critical_consensus": {
            "summary": data["critical_consensus_summary"],
            "scores": data["critical_consensus_scores"],
            "awards": data["critical_consensus_awards"],
        },
        "strengths": data["strengths"],
        "weaknesses": data["weaknesses"],
        "verdict": data["verdict"],
        "useless_fact": data["useless_fact"],
        "fun_facts": data["fun_facts"],
        "questions": data["questions"],
        "debate_prompts": data["debate_prompts"],
        "cta": data["cta"],
        "sources": sorted({c["url"] for c in chunks}),
        "director": meta.get("director"),
        "themes": data["themes"],
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
    one call). draft_best_of below is what main() actually uses."""
    chunks, meta = retrieve(title, year)
    client = Groq(api_key=read_env("GROQ_API_KEY"))
    data = _call_llm(client, title, year, chunks, meta)
    pack = _assemble_pack(title, year, data, chunks, meta)
    allowed_urls = {c["url"] for c in chunks}
    stripped = sanitize_grounding(pack, allowed_urls)
    if stripped:
        print(f"WARNING: stripped {stripped} citation(s) the model invented "
              f"(not among the {len(allowed_urls)} URL(s) actually retrieved)")
    return pack


def draft_best_of(title: str, year: int, n: int = 3) -> dict:
    """Generates `n` independent candidates from the SAME retrieved text
    and picks the one with the most grounded claims (after sanitizing
    each candidate's citations first, so a candidate can't win by
    fabricating extra ones). Exists because a single call from this free
    model is inconsistent run to run -- see D14 in docs/DESIGN.md: the
    same prompt against the same text produced anywhere from 4 to 15
    grounded claims across manual test runs. Retrieval happens once and
    is shared across all n candidates -- only the generation call varies."""
    chunks, meta = retrieve(title, year)
    print(f"retrieved {len(chunks)} source chunk(s): {[c['label'] for c in chunks]}")
    if meta.get("director"):
        print(f"director (TMDB): {meta['director']}")

    client = Groq(api_key=read_env("GROQ_API_KEY"))
    allowed_urls = {c["url"] for c in chunks}

    candidates = []
    for i in range(n):
        print(f"generating candidate {i + 1}/{n}...")
        data = _call_llm(client, title, year, chunks, meta)
        pack = _assemble_pack(title, year, data, chunks, meta)
        stripped = sanitize_grounding(pack, allowed_urls)
        needs, grounded = grounded_count(pack)
        print(f"  candidate {i + 1}: {grounded}/{needs} grounded claims"
              + (f", {stripped} fabricated citation(s) stripped" if stripped else ""))
        candidates.append((grounded, stripped, pack))

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
