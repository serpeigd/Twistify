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
  `src/preshow/translate.py`), cached per title in `content/_translations/` (gitignored),
  and marked `auto_translated` — see D9. A third, browse-only tier
  (`src/preshow/tmdb.py`) reaches effectively all of TMDB via live search
  (`GET /api/search`), cached in `content/_tmdb_cache/` (gitignored) — never
  conflated with the researched tier's cited-claims bar. See D10. Comments and
  movie-request data persist through `src/preshow/kv_store.py` — Upstash Redis's
  free REST API when `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` are set
  in `.env`, else the same local-file behavior as before (no account needed for
  local dev). See D11 — this is what makes comments survive a redeploy on a free
  host with an ephemeral filesystem.

Full design rationale (D1–D12) lives in `docs/DESIGN.md` — update it on any non-trivial
design decision.

## Status

- Repo `serpeigd/Twistify`, single `main` branch. README/CLAUDE.md/DESIGN.md/code/data
  all English. CI (8 tests) on GitHub Actions. Release `v1.0.0` published.
  Live deploy: https://twistify.onrender.com (Render free tier — spins
  down after 15 min idle, ~1 min cold start on the next request, and
  wipes its filesystem on every redeploy/restart/spin-down; see D11 and
  confirm Upstash env vars are actually set there, since a working demo
  right now looks identical whether it's backed by Upstash or by the
  ephemeral file until the next spin-down proves it one way or the other).
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
  `SubstringJudge` is now calibrated twice — offline against its own
  paraphrases (recall=0.0) AND against 2,197 real IMDb spoiler reviews
  restricted to our 9 covered titles (recall=0.0 again, 0/2,197 caught —
  see D12). The open question is no longer "find a benchmark" (done); it's
  that `SubstringJudge` is proven unfit and needs replacing with the
  already-stubbed `LLMJudge` (`evals/judge.py`) before the mainstream vs.
  long-tail comparison means anything. All 20 titles now have a resolved
  `tmdb_id` in `titles.yaml` (D10) for browse-tier posters — cosmetic,
  doesn't touch the stratified sample or labels.
- Demo track: 8/20 titles researched (Sixth Sense, Fight Club, Get Out, Parasite,
  Prestige, Se7en, Arrival, Gone Girl). Remaining 12 show a real TMDB
  poster/synopsis instead of an empty placeholder (D10).

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

Judge calibration is done, twice (D12) — and both results say the same
thing: `SubstringJudge` is not fit to report a trustworthy `leakage_rate`.
The next task is wiring up `LLMJudge` (already written, unused, in
`evals/judge.py`) as the real judge, then re-running calibration against
the same 2,197-review external set before drawing any mainstream vs.
long-tail conclusion. Don't start Milestone 1 (retrieval) before that —
there's no point measuring whether retrieval helps with a judge that's
blind either way.

Upstash on the live Render deploy: deliberately deferred, user's call —
no Upstash account yet. Comments/movie-requests on
https://twistify.onrender.com will keep resetting every ~15 min idle
(Render free tier wipes the filesystem on every spin-down, not just
redeploys). Known, accepted trade-off, not a bug to chase — D11's code
already handles it gracefully (empty state, not an error).

**Confirmed bug on Render, separate from Upstash**: `poster_url` is
`null` for every title (`GET /api/catalogue`) and `GET /api/search`
returns `[]` on the live deploy — `TMDB_API_KEY`/`TMDB_READ_ACCESS_TOKEN`
are only in the local `.env` (gitignored, never deployed), so
`tmdb._get()` (`src/preshow/tmdb.py`) silently returns `None` on Render,
same "reads degrade quietly" shape as kv_store but for a feature that's
supposed to always work. Fix: add both as environment variables in
Render's dashboard (Environment tab) with the same values already in
local `.env`, then redeploy. No new account needed — same TMDB keys
already in use locally.

Also pending, not started (see docs/DESIGN.md "Pending"): automating the
"+ Suggest a movie" pipeline (it now resolves a `tmdb_id` per suggestion,
but still doesn't research or add anything), and researching the 12
remaining measurement titles the same way Gone Girl was (D6/D7 — cited
sources, no invented facts).

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
```
`gh` CLI may need its full path (`C:\Program Files\GitHub CLI\gh.exe`) if not on PATH.
TMDB key/read token live in `.env` (gitignored) — used by `src/preshow/tmdb.py`
(D10) for the browse tier and catalogue posters. `GROQ_API_KEY` also belongs
in `.env` once obtained (same gitignored file, never commit it). For a public
deploy, add `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` (free, no card,
console.upstash.com) so comments/movie-requests survive a redeploy — see D11;
without them the app still runs fine locally on plain files.
