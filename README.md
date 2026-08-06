# 🎬 Twistify

**Spoiler-free before. Every twist after.**

[![tests](https://github.com/serpeigd/Twistify/actions/workflows/tests.yml/badge.svg)](https://github.com/serpeigd/Twistify/actions/workflows/tests.yml)
[![release](https://img.shields.io/github/v/release/serpeigd/Twistify)](https://github.com/serpeigd/Twistify/releases/tag/v1.0.0)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-D97757?logo=claude&logoColor=white)

Twistify is a movie app with a rule that's actually enforced, not just
promised: the plot never leaves the server until you say you've already
seen it. Underneath, an evaluation harness measures whether that promise
holds — with numbers, not a self-awarded green badge.

<p align="center">
  <img src="docs/screenshots/twistify-safe-mode.png" width="49%" alt="Spoiler-free mode">
  <img src="docs/screenshots/twistify-spoiler-mode.png" width="49%" alt="Spoiler mode unlocked">
</p>

---

## Try it in 2 minutes

```bash
git clone https://github.com/serpeigd/Twistify.git
cd Twistify
pip install -r requirements.txt
python webapp/app.py
# open http://127.0.0.1:8000
```

Pick a researched movie (Sixth Sense, Fight Club, Get Out, Parasite, The
Prestige, Se7en, Arrival, or Gone Girl), read the spoiler-free entry, and
when you're ready, open the curtain. Anything else you search for comes
from TMDB's browse tier (poster + synopsis, no spoiler curtain — see
[Architecture](#architecture)).

### Live demo

[twistify.onrender.com](https://twistify.onrender.com) — same code as
`main`, no setup needed. Two things to know before judging it as "broken":

- **Render's free tier spins down after ~15 min idle.** The first request
  after that wakes it back up and takes about a minute; it's cold-start
  latency, not a hang.
- **Comments and "suggest a movie" entries reset on every spin-down**, not
  just on redeploy. The free tier wipes the filesystem, and this deploy
  deliberately hasn't been wired to Upstash yet (see
  [Configuration](#configuration)) — a known,
  accepted trade-off, not a bug. TMDB posters/search work on the live
  deploy; the researched entries and the eval harness are unaffected
  either way (they don't depend on persisted state).

## What makes this different from "another movie CRUD"

- **The spoiler partition is a server-side property, not a UI promise.**
  Post-viewing content isn't sent to the browser until the client declares
  `seen=true` — opening devtools reveals nothing. It's not CSS hiding a
  `<div>`.
- **Every factual claim says whether it has a source or not.** No faking
  `source_id`s when there's no real retrieval behind it. The gap is shown,
  not disguised.
- **The spoiler-leak detector itself is measured, not assumed.** There's an
  evals harness (`evals/`) that calibrates the judge against planted leaks
  and reports its real recall — including the uncomfortable case where the
  cheap judge fails (see [Results](#under-the-hood-the-evaluation-harness)).
- **Filters that actually mean something.** Themes (identity, obsession,
  class and power…) that group several movies for real, not a one-off tag
  per title.
- **Usable on a phone, not just a laptop demo.** Below 760px the catalogue
  sidebar becomes an off-canvas drawer (hamburger toggle in the header)
  instead of disappearing — a mobile visitor can actually browse without
  requesting the desktop site. Desktop layout is unaffected.

## Stack

**Backend:** Python 3.12 · FastAPI · Pydantic v2 (typed data contracts, not
loose dicts) · pytest (evals harness, runs with no network, no API key)

**Frontend:** vanilla HTML/CSS/JS — no framework, on purpose: the app is
small enough that a framework would be cost without benefit, not "doesn't
know how to use one."

**AI / evaluation:** Anthropic Claude (tool use / structured output for the
baseline generator, no markdown parsing) · a custom evals harness design
(leakage / grounding / richness) with a calibrated judge, verified against
planted leaks.

**CI:** GitHub Actions runs the 8 tests on every push (see badge above).

## Architecture

Two tracks that share data contracts but never share responsibilities —
worth keeping straight, because the numbers below only mean something if
you know which track produced them:

- **Measurement track** — the actual experiment. A no-retrieval baseline
  generator writes a pre-show brief for each of 20 stratified titles; a
  judge checks it for spoiler leaks and source grounding; `evals/`
  reports `leakage_rate` / `grounded_fact_rate` / `richness` per stratum.
  Nothing here is hand-authored editorial content.
- **Demo track** — the app you can click through. 8 titles get a real,
  cited-source researched entry with a spoiler curtain; everything else
  in the catalogue falls back to a TMDB poster + synopsis (no curtain,
  no citations — it's a browse convenience, not a claim). This track
  doesn't call the baseline generator or the judge at all.

| Piece | What it is |
|---|---|
| `webapp/` | FastAPI + vanilla JS. Serves the catalogue, runs the spoiler gate, comments (edit/delete with no accounts, anonymous per-browser token), the TMDB browse tier, and `research_assist.py` (drafts new researched entries from real retrieval — human-reviewed before publishing, see D14 in `docs/DESIGN.md`). |
| `content/researched/*.json` | 8 hand-researched entries (Sixth Sense, Fight Club, Get Out, Parasite, The Prestige, Se7en, Arrival, Gone Girl — cited sources: Wikipedia, Hollywood Reporter, No Film School…), not generated by an unverified LLM. |
| `src/preshow/` | Data contracts (Pydantic) for both the researched content and the measurement harness, plus the TMDB/translation/key-value-store integrations. |
| `evals/` | The real experiment: leakage/grounding/richness metrics, calibrated judges (`SubstringJudge`, `LLMJudge`), the 20-title stratified dataset, and the external-calibration scripts. |
| `tests/` | Offline pytest suite (8 tests) — the harness tested against a fake generator with planted leaks, no network, no API key. |
| `docs/DESIGN.md` | Every non-trivial design decision (D1–D14) with its trade-off, written as it was made. |

### Folder structure

```
Twistify/
├── webapp/                  # FastAPI app + vanilla JS frontend, browse tier, research-assist tool
├── content/
│   ├── researched/*.json    # 8 hand-researched, cited-source entries (demo track)
│   ├── _drafts/             # research_assist.py output, gitignored, human review gate (D14)
│   ├── _translations/       # cached ES machine translations, committed (D9)
│   └── _tmdb_cache/         # cached TMDB responses, gitignored (D10)
├── src/preshow/             # Pydantic schemas, baseline generators, TMDB/translate/kv_store clients
├── evals/
│   ├── dataset/titles.yaml  # 20-title stratified measurement set (mainstream vs. long-tail)
│   ├── dataset/external/    # gitignored IMDB Spoiler Dataset download (D12), not ours to redistribute
│   ├── run_eval.py          # runs the baseline generator + judge over the 20 titles
│   ├── judge.py             # SubstringJudge, LLMJudge
│   ├── metrics.py           # leakage_rate / grounded_fact_rate / richness
│   └── calibrate_*.py       # internal + external judge calibration scripts
├── tests/test_metrics.py    # offline pytest suite (8 tests), no network, no API key
├── docs/
│   ├── DESIGN.md            # D1–D14 design decisions with trade-offs
│   └── screenshots/         # README images
└── requirements.txt
```

## Configuration

No environment variables are required to run the app or the offline test
suite locally — everything below is optional, and each feature degrades
gracefully (never crashes) when its variable is unset. Put them in a
gitignored `.env` file at the repo root (`src/preshow/env.py` reads it,
falling back to real environment variables — no `python-dotenv`
dependency).

| Variable | Used by | Required for | If unset |
|---|---|---|---|
| `TMDB_READ_ACCESS_TOKEN` | `src/preshow/tmdb.py` | Browse-tier posters, live search, `resolve_tmdb_ids.py`, `research_assist.py`'s director lookup | Browse tier returns empty results; researched entries and the eval harness are unaffected |
| `GROQ_API_KEY` | `src/preshow/baseline_groq.py`, `evals/calibrate_llm_external.py`, `webapp/research_assist.py` | Free-tier baseline generator (`--generator baseline-groq`), `LLMJudge` calibration, drafting new researched entries | Those specific commands fail with a clear error; the rest of the app is unaffected |
| `ANTHROPIC_API_KEY` | `src/preshow/baseline.py` (read implicitly by the `anthropic` SDK) | Paid baseline generator (`--generator baseline`) | That command fails; use `baseline-groq` instead (free) |
| `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` | `src/preshow/kv_store.py` | Comments/movie-requests surviving a redeploy on a free host (D11) | Falls back to local JSON files — fine for local dev, resets on redeploy for a public host |
| `PORT` | `webapp/app.py` | Hosting platforms that assign a port (e.g. Render) | Defaults to `8000` |

No variable here is required for a paid service by default — the only one
that costs money is `ANTHROPIC_API_KEY`, and it's optional (see the
free-only `baseline-groq` alternative).

## Status

| What | Status |
|---|---|
| Twistify app (catalogue, spoiler gate, filters, comments) | ✅ 8/20 entries researched |
| Browse catalogue (TMDB posters, live search, ES/EN) | ✅ 20/20 have posters, search reaches all of TMDB |
| Offline evals harness | ✅ 8 tests passing |
| Spoiler ground truth (20 titles) | ✅ 20/20, LLM-researched with cited sources (never hand-labeled — see [Ground truth, precisely](#ground-truth-precisely) below) |
| Baseline generator (no retrieval) | ✅ two providers — Anthropic (paid) and Groq (free tier, no card) |
| Judge calibration (offline + real external spoiler reviews) | ✅ both judges calibrated against the same 7,657-review external human set — `SubstringJudge` recall=0.0, `LLMJudge` recall=0.089/precision=0.471. Neither clears the bar to trust a `leakage_rate` yet |
| Measure the baseline over the 20 titles | ✅ done — see numbers and caveats below |
| Research-assist tool (drafts new researched entries from real retrieval) | ✅ working, human-review gated (D14) |
| Retrieval (TMDB/OMDb/Wikipedia) + verifier | ⬜ next milestone — blocked on judge calibration, see [Roadmap](#roadmap) |

## Ground truth, precisely

The 20-title measurement set's spoiler labels are **LLM-researched with
cited sources** — an LLM reads real sources (Wikipedia, reviews, cast
interviews) and drafts what counts as a spoiler and how severe it is, with
every claim traceable to a citation. That is a different, weaker claim
than **"hand-labeled"**, and this README (like `docs/DESIGN.md`'s D7)
deliberately never uses the second phrase for the first thing: conflating
them would quietly reintroduce the same self-coherence risk the project's
judge-calibration work exists to catch. See D6/D7 in
[`docs/DESIGN.md`](docs/DESIGN.md) for the full reasoning and the
trade-off that was made explicitly, not by default.

## Under the hood: the evaluation harness

The part you don't see in the screenshots is what backs the app's promise:
a system that **measures**, instead of promising, three things per entry:

1. **`leakage_rate`** — did any spoiler slip into the pre-viewing content?
2. **`grounded_fact_rate`** — how many claims carry a real source?
3. **`richness`** — how much does it actually say? (an empty output scores
   perfectly on the first two — that's why it's never reported without
   this one)

```bash
python -m pytest tests/ -q                            # 8/8, no network, no API key
python evals/run_eval.py --generator baseline-groq    # free tier, no card
python evals/run_eval.py --generator baseline         # or the paid Anthropic version
```

### Milestone 0 results (no-retrieval baseline, Groq/Llama-3.3-70B, 20 titles)

| Metric | Mainstream | Long-tail | Overall |
|---|---|---|---|
| `leakage_rate` | 0.0 | 0.0 | 0.0 |
| `grounded_fact_rate` | 0.0 | 0.0 | 0.0 |
| `richness` (claims/case) | 6.0 | 6.0 | 6.0 |

**Read this table with its caveats, not instead of them:**

- **`leakage_rate = 0.0` is not a safety result — it's the judge's blind spot.**
  `SubstringJudge` was calibrated offline at **recall = 0.0**: it only catches
  verbatim spoiler phrases, never a paraphrase. A 0.0 leakage rate here most
  likely means the judge failed to see leaks that are actually there, not
  that the baseline is safe. Trusting this number without the calibration
  note next to it is exactly the mistake this project exists to avoid.
- **`grounded_fact_rate = 0.0` is a real, expected finding.** The baseline is
  given no retrieval corpus (`corpus=[]`) and is explicitly instructed never
  to invent a source id. Zero real sources in, zero real sources out — this
  is the quantitative baseline Milestone 1 (retrieval) needs to beat, not a
  bug.
- **`richness = 6.0`** confirms the generator isn't gaming the first two
  metrics by returning an empty brief.
- **Mainstream and long-tail are identical here**, which means this run
  *cannot yet confirm or deny* the project's original hypothesis (that a
  no-retrieval baseline degrades on long-tail titles) — a judge with 0.0
  recall can't see a gap that might exist. See below: this is no longer a
  missing-dataset problem, it's a judge problem.

### External judge calibration (IMDB Spoiler Dataset, real user reviews)

The calibration above uses the project's own LLM-written paraphrases —
useful, but it's an LLM checking an LLM. `evals/calibrate_substring_external.py`
re-runs it against real IMDb user reviews (Misra's IMDB Spoiler Dataset,
Kaggle, free), restricted to the 9 of our 20 titles the dataset covers
(mostly mainstream — long-tail titles here barely have review coverage at
all, a small real echo of the project's own mainstream/long-tail split):

| | Value |
|---|---|
| Reviews evaluated | 2,197 real spoiler-tagged + 5,460 real non-spoiler-tagged |
| **Recall** | **0.0** — caught 0 of 2,197 |
| Precision | undefined (0 positive predictions made — not "wrong every time") |

**Same conclusion, now independently confirmed**: `SubstringJudge` doesn't
just fail on paraphrases it's never seen from itself — it fails on plain
human language. See D12 in `docs/DESIGN.md` for the one caveat this
comparison carries (a review can be a real spoiler for a plot point we
didn't document, which the method above counts as a miss even though it
isn't the judge's fault).

### `LLMJudge`, calibrated the same way (Groq free tier)

`evals/calibrate_llm_external.py` re-runs the exact same method against
`LLMJudge` (`llama-3.1-8b-instant`, Groq's free tier) instead of
`SubstringJudge`. Free-tier limits (1,000 requests/day, and every review
costs `len(that movie's labels)` calls) meant a smaller, seeded, stratified
sample — 180 reviews (20/title, balanced spoiler/not) instead of the full
7,657 — and review text truncated to 350 characters/call to stay under
the tokens/min cap:

| Judge | n | Recall | Precision |
|---|---|---|---|
| `SubstringJudge` | 7,657 | 0.0 | undefined |
| `LLMJudge` (llama-3.1-8b-instant) | 180 | **0.089** | 0.471 |

A real improvement over the free floor — it sees paraphrases the substring
judge structurally cannot — but not yet trustworthy: it misses ~91 of every
100 real spoiler reveals in this sample, and fewer than half its positive
calls are right. Two things this run can't separate (see D13 in
`docs/DESIGN.md`): whether that ceiling is the small model or the 350-char
truncation forced by the token budget. Neither judge currently clears the
bar to report a trustworthy `leakage_rate`.

The decisions behind this design (why there's no LangGraph, why the schema
allows invalid states on purpose, why the same model being measured can't
generate its own ground truth) are documented in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Available commands

```bash
# Setup
pip install -r requirements.txt                     # app + evals harness (no paid keys)
pip install groq                                     # only needed for the free baseline / LLMJudge / research-assist

# Run the app
python webapp/app.py                                  # http://127.0.0.1:8000
python webapp/prewarm_translations.py                 # pre-cache ES machine translations (D9), no API key
python webapp/resolve_tmdb_ids.py                      # (re-)resolve tmdb_id for titles.yaml entries, needs TMDB_READ_ACCESS_TOKEN
python webapp/research_assist.py                       # draft a new researched entry from real retrieval (D14), needs GROQ_API_KEY

# Tests
python -m pytest tests/ -v                             # offline, 8 tests, no network, no API key

# Evals (measurement track)
python evals/run_eval.py --generator baseline-groq     # free tier, needs GROQ_API_KEY
python evals/run_eval.py --generator baseline           # paid, needs ANTHROPIC_API_KEY
python evals/calibrate_substring.py                     # internal SubstringJudge calibration, no download needed
python evals/calibrate_substring_external.py            # vs. real IMDb reviews, needs evals/dataset/external/ (see D12)
python evals/calibrate_llm_external.py                  # same, vs. LLMJudge, needs GROQ_API_KEY (see D13)
```

## Roadmap

- **Judge recalibration** (blocking Milestone 1 — see
  [Status](#status) and D13 in `docs/DESIGN.md`): re-run
  `calibrate_llm_external.py` with full, untruncated review text and/or a
  stronger model (`llama-3.3-70b-versatile`) to find out whether 0.089
  recall is a real capability ceiling or a 350-char-truncation artifact.
  If that's still not enough, the two alternatives on the table are a
  local NLI/entailment classifier (no per-call cost or rate limit) or a
  classifier trained directly on this project's own 2,197/5,460 external
  labels with a proper held-out split.
- **Milestone 1 — retrieval + verifier** (TMDB/OMDb/Wikipedia): explicitly
  not started until the judge is trustworthy enough to measure whether it
  helped.
- **Research the remaining 12 measurement titles** the same way Gone Girl
  was (D6/D7 — cited sources, no invented facts), and grow the demo
  track's researched catalogue past 8 (the actual motivation behind D14's
  research-assist tool).
- **Wire Upstash on the live Render deploy** so comments/movie-requests
  stop resetting on spin-down — deliberately deferred, not blocked on
  anything technical (D11 already supports it).

## Limitations

- **`leakage_rate` is not currently a trustworthy safety number.** Both
  calibrated judges (`SubstringJudge` recall=0.0, `LLMJudge`
  recall=0.089/precision=0.471) miss the large majority of real spoiler
  leaks in an external human-labeled sample — see
  [Ground truth, precisely](#ground-truth-precisely) and the calibration
  tables above. Read every `leakage_rate`/`grounded_fact_rate` this
  project reports next to `richness` and next to this caveat, not on its
  own.
- **The baseline generator has no retrieval.** `grounded_fact_rate = 0.0`
  is expected, not a bug — there is nothing to ground claims in yet.
- **Demo-track researched entries: 8 of 20 measurement titles.** The other
  12 (plus anything else searched) fall back to a TMDB poster + synopsis,
  not a curtain-gated, cited-source entry.
- **The mainstream-vs-long-tail hypothesis is unconfirmed.** Milestone 0's
  two strata scored identically — that could mean "no gap exists" or
  "the judge can't see a gap that exists"; the data can't distinguish
  those yet.
- **Live deploy state doesn't persist across a spin-down.** See
  [Live demo](#live-demo) above — Upstash isn't configured on
  twistify.onrender.com yet, so comments/movie-request suggestions reset
  after ~15 min of inactivity. Nothing else (researched content, eval
  results) is affected.
- **No license defined yet.** Ask before reusing anything (see
  [License](#license)).

## FAQ

**Is the 20-title spoiler ground truth hand-labeled?**
No — it's LLM-researched with cited sources, a deliberately weaker claim.
See [Ground truth, precisely](#ground-truth-precisely).

**Why isn't the leakage_rate reported as a safety guarantee?**
Because the judges that produce it are calibrated at recall=0.0
(`SubstringJudge`) and recall=0.089 (`LLMJudge`) against real external
spoiler reviews — see the calibration tables above. A safety number is
only as trustworthy as the detector behind it.

**Why no LangGraph/CrewAI/AutoGen?**
The pipeline is sequential end to end (generate → judge → aggregate). D1
in `docs/DESIGN.md` scopes an orchestration framework to exactly one
future case (retrieval reformulation for obscure titles) and only if a
plain `if/elif` stops being legible when that's built — not by default.

**Does this cost money to run?**
No. `pip install -r requirements.txt` plus the app and the offline test
suite need zero API keys. The only paid option is
`ANTHROPIC_API_KEY` for the paid baseline generator variant, and it's
optional — `--generator baseline-groq` is the free equivalent.

**Why do the measurement track and the demo track use different title
counts (20 vs. 8)?**
They're deliberately different scopes — see
[Architecture](#architecture). 20 is the fixed, stratified experiment set;
8 is how many of those (plus any future additions) have a full
hand-researched, curtain-gated demo entry so far.

## Sources and legal restrictions

- **TMDB** — free for non-commercial use, requires attribution (shown in the
  app wherever TMDB data appears). Powers the browse tier (`src/preshow/tmdb.py`
  — live search, catalogue posters), separate from the hand-researched,
  cited-source tier (see D10 in `docs/DESIGN.md`). Its terms restrict using
  the content to *train* AI systems; inference with attribution is the
  usual reading, but review it before scaling this up further.
- **OMDb** — a path to Rotten Tomatoes/Metascore scores, free tier is
  limited.
- **Wikipedia** — CC BY-SA, already in use for researched entries.
- **Scraping IMDb** — forbidden by ToS, not done under any excuse.

## License

No license defined yet — personal portfolio repo. If you want to reuse
something, ask first.
