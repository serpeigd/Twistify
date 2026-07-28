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
  conflated with the researched tier's cited-claims bar. See D10.

Full design rationale (D1–D10) lives in `docs/DESIGN.md` — update it on any non-trivial
design decision.

## Status

- Repo `serpeigd/Twistify`, single `main` branch. README/CLAUDE.md/DESIGN.md/code/data
  all English. CI (8 tests) on GitHub Actions. Release `v1.0.0` published.
- Measurement track: 20/20 titles have ground truth (LLM-researched with cited
  sources — see D7, not hand-labeled). `SubstringJudge` calibrated offline
  (recall=0.0 by design). Two baseline generators, same prompt/schema
  (`src/preshow/baseline_prompts.py`): `AnthropicBaselineGenerator` (paid) and
  `GroqBaselineGenerator` (`--generator baseline-groq`, free tier, no card).
  Neither has been run yet. **Blocker**: needs a `GROQ_API_KEY` (or
  `ANTHROPIC_API_KEY`) to actually run `evals/run_eval.py` and put real
  leakage/grounding/richness numbers in the README. External judge calibration
  (TV Tropes/IMDB Spoiler Dataset) still blocked — no direct public download.
  All 20 titles now have a resolved `tmdb_id` in `titles.yaml` (D10) for
  browse-tier posters — cosmetic, doesn't touch the stratified sample or labels.
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

Get a free `GROQ_API_KEY` (console.groq.com/keys, no card), run
`python evals/run_eval.py --generator baseline-groq` over the 20 labeled
titles, paste real leakage/grounding/richness numbers into the README, then
calibrate the judge externally before trusting them. Don't start Milestone 1
(retrieval) before Milestone 0 has real numbers.

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
```
`gh` CLI may need its full path (`C:\Program Files\GitHub CLI\gh.exe`) if not on PATH.
TMDB key/read token live in `.env` (gitignored) — used by `src/preshow/tmdb.py`
(D10) for the browse tier and catalogue posters. `GROQ_API_KEY` also belongs
in `.env` once obtained (same gitignored file, never commit it).
