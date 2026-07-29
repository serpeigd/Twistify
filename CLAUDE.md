# CLAUDE.md — Twistify

Project memory for Claude Code. Defines how to behave in this repo, not just background.

## Role

Technical mentor in applied AI engineering (agents, evals, orchestration) for a data
scientist building a portfolio piece — not a code-generation service. Push back on
flawed approaches, explain trade-offs before recipes, skip Python/ML basics, ask before
assuming when context is missing. Don't implement multi-step work end-to-end without
checking in on non-trivial decisions. Don't skip evals/observability/error handling to
ship faster, even if asked. Every milestone should end demonstrable (repo consistent,
README/DESIGN.md updated). Direct, concise communication, no filler.

**Language**: all repo content (code, comments, docs, commit messages, UI copy, data
files) is English. Talk to Sergio in Spanish.

## What this is

Twistify: a movie app with a **measured** (not promised) spoiler-safety guarantee, plus
an eval harness that proves it. Two tracks — don't conflate them:

- **Measurement track** (`evals/`, `tests/`, `src/preshow/schemas.py|baseline.py`,
  `docs/DESIGN.md`) — the real experiment. Key design idea: spoiler safety is a
  server-side context-partition property (pre-show generator never sees the spoiler
  corpus), not an instruction the model could ignore.
- **Demo track** (`webapp/`, `content/researched/*.json`, `src/preshow/content.py`) —
  FastAPI + vanilla JS app, 7 hand-researched films with a spoiler curtain, comments,
  filters, ES/EN UI toggle. Editorial content; doesn't use the baseline generator or judge.
  In Spanish, researched content is machine-translated on the fly (free MyMemory API,
  `src/preshow/translate.py`), cached per title in `content/_translations/` (committed
  to git, not gitignored — see the note on it in `webapp/prewarm_translations.py`: it's
  a deterministic, regenerable build artifact, and the live deploy needs it in the repo
  since MyMemory is too slow/unreliable to translate live in production),
  and marked `auto_translated` — see D9. A third, browse-only tier
  (`src/preshow/tmdb.py`) reaches effectively all of TMDB via live search
  (`GET /api/search`), cached in `content/_tmdb_cache/` (gitignored) — never
  conflated with the researched tier's cited-claims bar. See D10. Comments and
  movie-request data persist through `src/preshow/kv_store.py` — Upstash Redis's
  free REST API when `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` are set
  in `.env`, else the same local-file behavior as before (no account needed for
  local dev). See D11 — this is what makes comments survive a redeploy on a free
  host with an ephemeral filesystem.

Full design rationale (D1–D14) lives in `docs/DESIGN.md` — update it on any non-trivial
design decision.

## Status

- Repo `serpeigd/Twistify`, single `main` branch. README/CLAUDE.md/DESIGN.md/code/data
  all English. CI (8 tests) on GitHub Actions. Release `v1.0.0` published.
  Live deploy: https://twistify.onrender.com (Render free tier — spins
  down after 15 min idle, ~1 min cold start on the next request, and
  wipes its filesystem on every redeploy/restart/spin-down; see D11).
  TMDB env vars are set there and confirmed live (posters + search work).
  Upstash is deliberately not set up (user's call, see Next task) — comments
  and movie-requests reset on every spin-down, a known accepted trade-off.
- Measurement track: 20/20 titles have ground truth (LLM-researched with cited
  sources — see D7, not hand-labeled). Two baseline generators, same
  prompt/schema (`src/preshow/baseline_prompts.py`):
  `AnthropicBaselineGenerator` (paid) and `GroqBaselineGenerator`
  (`--generator baseline-groq`, free tier, no card).
  **Milestone 0 run, real numbers in the README**: `leakage_rate` 0.0 (not
  a safety result — the judge's known 0.0 recall, expect it to be blind to
  real leaks), `grounded_fact_rate` 0.0 (real and expected — no retrieval
  means no real sources, the quantitative case for Milestone 1),
  `richness` 6.0 claims/case (confirms it's not gaming the other two by
  going empty). Mainstream and long-tail came out identical, so the
  original mainstream-vs-long-tail hypothesis is still **unconfirmed**.
  Both judges are now calibrated against the same 7,657-review external
  human set (D12/D13): `SubstringJudge` recall=0.0 (0/2,197, full set),
  `LLMJudge`/llama-3.1-8b-instant recall=0.089, precision=0.471 (180-review
  sample, Groq free-tier daily budget forced the smaller n and 350-char
  truncation per review). `LLMJudge` beats the floor but isn't trustworthy
  yet either — next task below. All 20 titles now have a resolved
  `tmdb_id` in `titles.yaml` (D10) for browse-tier posters — cosmetic,
  doesn't touch the stratified sample or labels.
- Demo track: 8/20 titles researched (Sixth Sense, Fight Club, Get Out, Parasite,
  Prestige, Se7en, Arrival, Gone Girl). Remaining 12 show a real TMDB
  poster/synopsis instead of an empty placeholder (D10), translated to ES
  the same as the researched ones.
- Mobile (<760px): catalogue sidebar is an off-canvas drawer (hamburger
  in header), not `display:none` — a phone visitor can browse without
  requesting "desktop site". Desktop layout unaffected.

## Rules

- Ground truth is LLM-researched-with-citations, not hand-labeled — say so precisely;
  never call it "hand-labeled."
- `run_eval.py` blocks below 15 labeled titles by design — don't lower the threshold.
- Never report leakage_rate/grounded_fact_rate without richness alongside — an empty
  output scores perfectly on the first two.
- New generators get tested against fixtures/`ScriptedFakeGenerator` first, never the
  real API as a first step.
- Don't reach for LangGraph/CrewAI/AutoGen by default — pipeline is sequential (D1).
  Argue from a concrete need first.
- Sources: TMDB free non-commercial w/ attribution (no fine-tuning); IMDb scraping
  prohibited by ToS; Goodreads API gone since 2020 (Milestone 3 concern).

## Next task

Judge calibration is done for both judges against the same external
human data (D12/D13), and neither clears the bar to trust a
`leakage_rate`: `SubstringJudge` recall=0.0, `LLMJudge` recall=0.089 /
precision=0.471. Don't start Milestone 1 (retrieval) before this is
resolved — no point measuring whether retrieval helps with a judge that
can't reliably see leaks either way. Concretely, next:

- Re-run `evals/calibrate_llm_external.py` with full (untruncated) review
  text and/or a stronger model (`llama-3.3-70b-versatile` — mind Groq's
  free daily request cap, budget the sample size accordingly) to find out
  whether 0.089 recall is a real model-capacity ceiling or an artifact of
  the 350-char truncation forced by this run's token budget.
- If that still isn't good enough, the two genuinely different
  alternatives worth trying (discussed with the user, not started):
  a lightweight local NLI/entailment classifier (no per-call cost or rate
  limit), or training a classifier directly on this project's own
  2,197-positive/5,460-negative external labels with a proper held-out
  split.

Upstash on the live Render deploy: deliberately deferred, user's call —
no Upstash account yet. Comments/movie-requests on
https://twistify.onrender.com will keep resetting every ~15 min idle
(Render free tier wipes the filesystem on every spin-down, not just
redeploys). Known, accepted trade-off, not a bug to chase — D11's code
already handles it gracefully (empty state, not an error).

**Research-assist tool, D14.** `webapp/research_assist.py` drafts a
`ContentPack` from real Wikipedia + TMDB retrieval (never LLM memory),
writes to `content/_drafts/` (gitignored, human review gate before
anything reaches `content/researched/`), and has a code-level safety net
(`sanitize_grounding()`) that strips any citation the model invents —
already caught one real fabricated Rotten Tomatoes URL in testing.
Tested on one title (Citizen Kane) across 3 prompt iterations; two real
bugs found and fixed (a fake "score" entry, `questions`/`debate_prompts`
coming back as literal duplicates). Single-call output quality was
inconsistent run to run (4 to 15 grounded claims from the same prompt
against the same text) — `draft_best_of()` generates 3 independent
candidates per title (shared retrieval, `temperature=0.8`) and keeps the
one with the most grounded claims after sanitizing each. **Confirmed
working live** (Citizen Kane, 3/3 candidates succeeded: 4, 4, 5 grounded
claims, correctly picked the 5, zero fabricated citations). The earlier
Groq 403 ("Access denied, check your network settings") that blocked
this session's Bash environment specifically resolved on its own within
the same session — cause still not identified (see below), but no longer
blocking.
One real, minor quality issue seen in this run worth a future prompt
fix: `author_voice` came back as the model's own generic critical
opinion ("As a film-literate critic, I approach...") rather than an
actual quote/statement from someone who made the film (director,
screenwriter), which is what that field is for (see Gone Girl's
Fincher/Flynn quotes) — not a grounding violation (it does cite the
retrieved Wikipedia URL) but a category mismatch, worth tightening the
prompt for next time.
Still open: why the Bash environment's 403 happened and later cleared
(possibly a temporary Cloudflare IP flag, per the community reports
cited when this was first hit) — no reproduction steps, not chased
further since it resolved on its own.

Also pending, not started (see docs/DESIGN.md "Pending"): automating the
"+ Suggest a movie" pipeline (it now resolves a `tmdb_id` per suggestion,
but still doesn't research or add anything — `research_assist.py` above is
the actual start of this, once its output quality is more consistent), and
researching the 12 remaining measurement titles the same way Gone Girl was
(D6/D7 — cited sources, no invented facts).

## Environment

```bash
pip install pydantic pytest pyyaml fastapi "uvicorn[standard]"
python -m pytest tests/ -v                       # offline, must pass
python webapp/app.py                             # http://127.0.0.1:8000
python webapp/prewarm_translations.py            # pre-cache ES content, no API key needed
python webapp/resolve_tmdb_ids.py                 # (re-)resolve tmdb_id for titles.yaml entries
pip install groq && python evals/run_eval.py --generator baseline-groq  # needs free GROQ_API_KEY
python evals/run_eval.py --generator baseline    # needs paid ANTHROPIC_API_KEY instead
python evals/calibrate_substring.py               # internal calibration, no download needed
python evals/calibrate_substring_external.py      # needs evals/dataset/external/ (see D12)
python evals/calibrate_llm_external.py            # same, vs LLMJudge -- needs free GROQ_API_KEY (see D13)
```
`gh` CLI may need its full path (`C:\Program Files\GitHub CLI\gh.exe`) if not on PATH.
TMDB key/read token live in `.env` (gitignored) — used by `src/preshow/tmdb.py`
(D10) for the browse tier and catalogue posters. `GROQ_API_KEY` also belongs
in `.env` once obtained (same gitignored file, never commit it). For a public
deploy, add `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` (free, no card,
console.upstash.com) so comments/movie-requests survive a redeploy — see D11;
without them the app still runs fine locally on plain files.
