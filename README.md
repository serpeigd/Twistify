# 🎬 Twistify

**Spoiler-free before. Every twist after.**

[![tests](https://github.com/serpeigd/Twistify/actions/workflows/tests.yml/badge.svg)](https://github.com/serpeigd/Twistify/actions/workflows/tests.yml)
[![release](https://img.shields.io/github/v/release/serpeigd/Twistify)](https://github.com/serpeigd/Twistify/releases/tag/v1.0.0)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-D97757?logo=claude&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?logo=groq&logoColor=white)
![TMDB](https://img.shields.io/badge/TMDB-01B4E4?logo=themoviedatabase&logoColor=white)

Twistify is a movie app with a rule that's actually enforced, not just
promised: the plot never leaves the server until you say you've already
seen it. Underneath, an evaluation harness measures whether that promise
holds — with numbers, not a self-awarded green badge.

**Live demo:** [twistify.onrender.com](https://twistify.onrender.com)
(Render free tier — the instance spins down after 15 min idle, so the
first request after a while takes ~1 min to cold-start; that's expected,
not broken. It also wipes its filesystem on every redeploy/spin-down, so
comments and movie suggestions reset — see [Limitations](#limitations).)

<p align="center">
  <img src="docs/screenshots/twistify-safe-mode.png" width="49%" alt="Spoiler-free mode">
  <img src="docs/screenshots/twistify-spoiler-mode.png" width="49%" alt="Spoiler mode unlocked">
</p>

---

## Try it in 2 minutes

```bash
git clone https://github.com/serpeigd/Twistify.git
cd Twistify
pip install fastapi "uvicorn[standard]" pydantic pyyaml
python webapp/app.py
# open http://127.0.0.1:8000
```

Pick a researched movie (Sixth Sense, Fight Club, Get Out, Parasite, The
Prestige, Se7en, Arrival, or Gone Girl), read the spoiler-free entry, and
when you're ready, open the curtain. The other 12 titles in the catalogue
show a live TMDB poster/synopsis instead (see [Browse tier](#browse-tier-tmdb)) — no
API key needed for any of this, `TMDB_READ_ACCESS_TOKEN` just upgrades
those placeholders to real posters.

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
- **Actually usable on a phone.** Below 760px the catalogue sidebar
  (search, filters, film list) is a hamburger-triggered off-canvas
  drawer — not the old `display:none` that forced "desktop site" mode to
  browse at all.
- **Scaling the researched catalogue automates the labor, not the
  citation bar.** `webapp/research_assist.py` (see
  [Research-assist tool](#research-assist-tool-scaling-the-researched-catalogue))
  drafts a new researched entry from real Wikipedia + TMDB retrieval, but
  every claim still needs a real source URL — a code-level sanitizer strips
  any citation the model invents, and a human review gate still sits
  between a draft and `content/researched/`.

## Stack

**Backend:** Python 3.12 · FastAPI · Pydantic v2 (typed data contracts, not
loose dicts) · pytest (evals harness, runs with no network, no API key)

**Frontend:** vanilla HTML/CSS/JS — no framework, on purpose: the app is
small enough that a framework would be cost without benefit, not "doesn't
know how to use one."

**AI / evaluation:** Anthropic Claude (tool use / structured output for the
paid baseline generator) · Groq/Llama (free-tier alternative for the same
baseline generator, the `LLMJudge`, and the only model `research_assist.py`
uses — no Anthropic path exists there, so nothing in this project *requires*
a paid key) · a custom evals harness design (leakage / grounding / richness)
with a calibrated judge, verified against planted leaks and real human
spoiler reviews.

**Data sources:** TMDB (free, attributed — browse-tier posters/search,
research-assist metadata) · Wikipedia (CC BY-SA — research-assist
retrieval) · MyMemory (free — on-the-fly Spanish translation, cached) ·
Upstash Redis free REST API (optional — durable comments/movie-requests
on a redeploy; falls back to local files without it).

**CI:** GitHub Actions runs the 8 tests on every push (see badge above).

## How it's built

| Piece | What it is |
|---|---|
| `webapp/app.py` | FastAPI + vanilla JS. Serves the catalogue, runs the spoiler gate, comments (edit/delete with no accounts, anonymous per-browser token). `GET /api/stats` exposes the same grounding/leakage numbers for auditing — deliberately *not* rendered on the page itself (an earlier "no leaks detected" badge was removed for the same reason `SubstringJudge`'s 0.0 recall is disclosed everywhere else: a green seal the measurement can't back up). |
| `webapp/research_assist.py` | Drafts a new researched entry from real Wikipedia + TMDB retrieval (never LLM memory) — see [Research-assist tool](#research-assist-tool-scaling-the-researched-catalogue). |
| `webapp/prewarm_translations.py` | One-time build step: caches Spanish translations of researched + browse-tier content so the live deploy never calls the translation API on a visitor's request. |
| `webapp/resolve_tmdb_ids.py` | One-time helper: resolves a `tmdb_id` for every title in `evals/dataset/titles.yaml` so the browse tier can show a poster for all 20. |
| `content/researched/*.json` | 8 hand-researched entries (with cited sources: Wikipedia, Hollywood Reporter, No Film School…), not generated by an unverified LLM. |
| `src/preshow/` | Data contracts (Pydantic) for both the researched content and the measurement harness, plus the TMDB/Wikipedia/translation/KV-store clients. |
| `evals/` | The real experiment: leakage/grounding/richness metrics, calibrated judge, 20-title stratified dataset, external calibration scripts. |
| `docs/DESIGN.md` | Every non-trivial design decision (D1–D14) with its trade-off, written as it was made. |

## Status

| What | Status |
|---|---|
| Twistify app (catalogue, spoiler gate, filters, comments) | ✅ 8/20 entries researched |
| Browse catalogue (TMDB posters, live search, ES/EN) | ✅ 20/20 have posters, search reaches all of TMDB |
| Offline evals harness | ✅ 8 tests passing |
| Spoiler ground truth (20 titles) | ✅ 20/20, LLM-researched with cited sources (never hand-labeled — see [Ground truth, precisely](#ground-truth-precisely) below) |
| Baseline generator (no retrieval) | ✅ two providers — Anthropic (paid) and Groq (free tier, no card) |
| Judge calibration (offline + real spoiler reviews) | ✅ both judges calibrated against the same external human data — `SubstringJudge` recall=0.0, `LLMJudge` recall=0.089/precision=0.471 — **neither clears the bar to trust a `leakage_rate` yet** (see [Limitations](#limitations)) |
| Measure the baseline over the 20 titles | ✅ done — see numbers and caveats below |
| Research-assist tool (D14) | ✅ drafts a researched entry from Wikipedia + TMDB; tested end-to-end on one title (Citizen Kane) |
| Retrieval (Wikipedia, GREEN-only corpus) + `--generator retrieval-groq` | ✅ `grounded_fact_rate` 0.0→1.0 confirmed live (19/20, D16) |

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
| **Recall** | **0.0** — caught 0 of 2,197 (95% CI: 0.0–0.002) |
| Precision | undefined (0 positive predictions made — not "wrong every time") |

**Same conclusion, now independently confirmed**: `SubstringJudge` doesn't
just fail on paraphrases it's never seen from itself — it fails on plain
human language, and with n=2,197 the 95% confidence interval (Wilson score,
`evals/stats.py`) is tight enough (0.0–0.002) that this isn't sample noise —
it's a structural miss. See D12 in `docs/DESIGN.md` for the one caveat this
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

| Judge | n | Recall (95% CI) | Precision (95% CI) |
|---|---|---|---|
| `SubstringJudge` | 7,657 | 0.0 (0.0–0.002) | undefined |
| `LLMJudge` (llama-3.1-8b-instant) | 180 | **0.089** (0.046–0.166) | 0.471 (0.262–0.69) |

A real improvement over the free floor — it sees paraphrases the substring
judge structurally cannot — but not yet trustworthy: it misses ~91 of every
100 real spoiler reveals in this sample, and fewer than half its positive
calls are right. The 95% confidence intervals (Wilson score interval,
`evals/stats.py`) matter here more than the point estimates: with only
n=180 (forced by Groq's free daily request cap), true recall could
plausibly be anywhere from 0.046 to 0.166 and true precision from 0.262 to
0.69 — a much bigger error bar than `SubstringJudge`'s, which had 42x more
data. Two things this run couldn't separate: whether that ceiling was the
small model or the 350-char truncation forced by the token budget.

**Follow-up: it was mostly the truncation.** Same model, same 180-review
sample, truncation raised from 350 to 1,500 characters:

| Truncation | n | Recall (95% CI) | Precision (95% CI) |
|---|---|---|---|
| 350 chars | 180 | 0.089 (0.046–0.166) | 0.471 (0.262–0.69) |
| 1,500 chars | 180 | **0.356** (0.264–0.459) | **0.64** (0.501–0.759) |

Non-overlapping confidence intervals — a real, large effect. Truncating
reviews to 350 characters was hiding most of the model's actual recall.
Still not good enough to trust a `leakage_rate` on (0.356 recall still
misses ~64 of every 100 real spoilers), but meaningfully better than the
first run suggested.

**The remaining confound — does a stronger model help — is resolved: no,
barely at all.** After three free-tier failures (a daily token quota that
turned out to be a rolling window, and a transient Groq outage — see
D13's follow-up in `docs/DESIGN.md` for the full account), a complete run
of `llama-3.3-70b-versatile` at the same 350-char truncation as the
original baseline finished cleanly:

| Model | n | Recall (95% CI) | Precision (95% CI) |
|---|---|---|---|
| llama-3.1-8b-instant | 180 | 0.089 (0.046–0.166) | 0.471 (0.262–0.69) |
| llama-3.3-70b-versatile | 108 | **0.093** (0.04–0.199) | 0.714 (0.359–0.918) |

Recall is essentially flat — a model roughly 9x larger doesn't catch
meaningfully more real spoiler reveals. Precision looks better but the
CIs still overlap substantially. **Combined with the truncation result
above, this is a clean answer**: giving the judge more text mattered a
lot; paying for a bigger model didn't. `llama-3.1-8b-instant` is the
better default going forward — no documented daily token cap, and it's
what survived four attempts of testing without a single free-tier
failure. `LLMJudge`'s ceiling at reasonable free-tier settings looks to
be around recall≈0.35–0.4.

### A trained classifier beats every judge above

Two more approaches were tried (D15 in `docs/DESIGN.md`). A local NLI
(entailment) classifier failed outright — impractically slow on this
project's dev hardware, and even where it ran, its recall was ~0
(formal NLI entailment is a stricter bar than "could a viewer infer
this", a real task mismatch, not a bug). But training a small classifier
directly on this project's own external labels worked, and it's the
best judge calibrated so far:

| Threshold | Recall (95% CI) | Precision (95% CI) |
|---|---|---|
| 0.3 | **0.889** (0.876–0.902) | 0.357 (0.345–0.37) |
| 0.4 | 0.685 (0.665–0.704) | 0.45 (0.433–0.467) |
| 0.5 | 0.445 (0.424–0.466) | 0.571 (0.547–0.594) |

TF-IDF + Logistic Regression, trained on the **full** 7,657-review
external set (no free-tier sample-size limit), evaluated with grouped
k-fold by title (leave-one-title-out) so every reported prediction comes
from a model that never saw its own title during training — a genuine
held-out estimate, not a fit-and-report number. It beats `LLMJudge`'s
best result at every threshold from 0.4 down, with a far tighter
confidence interval. Per-title recall (0.285–0.668 across the 9 titles)
confirms it isn't just memorizing Fight Club, which is 32% of the
dataset. Caveats: trained only on the 9 mostly-mainstream titles with
review coverage, and it answers a different question than the other
three judges ("is this text spoiler-revealing" vs. "does it entail this
specific documented spoiler").

**Wired in**: `python evals/train_spoiler_classifier.py` persists the
trained model, and `python evals/run_eval.py --generator baseline-groq
--judge trained-classifier` uses it (default judge stays `substring`,
which needs nothing but the stdlib — this one needs the persisted model
plus `scikit-learn`). See D15 in `docs/DESIGN.md`.

**Validated in-domain, not just on IMDb reviews it was trained on**:
`evals/calibrate_trained_classifier_internal.py` tests it against real
spoiler-free pre-viewing text from the 8 hand-researched titles plus
this project's own 277 documented spoiler sentences — recall=0.838
(95% CI 0.79–0.876), precision=0.928 (95% CI 0.889–0.954) at the
shipped threshold, actually better than the external number.

**But it does NOT work on this project's actual generator output —
confirmed, not a theoretical risk.** The first live run
(`--generator baseline-groq --judge trained-classifier --show-leaks`)
scored **leakage_rate=0.95**, flagging things like a stage direction
("Black screen"), a cast credit ("The film stars Daniel Kaluuya..."),
and an award fact ("...won the Palme d'Or") as spoilers. The baseline
generators write short, fragmentary `on_screen_text`/`voiceover` lines —
a third register this judge was never trained or validated on (its
training data and validation set are both full sentences). Scoring the
flagged phrases directly showed every one landed in a narrow 0.30–0.47
band regardless of content — not a threshold problem, the model has no
real signal on fragments this short. `run_eval.py` now prints a warning
when `--judge trained-classifier` is selected; **don't report a
`leakage_rate` from it** until this is fixed. Full account in D15.

**Fix attempted, reduced but didn't solve it**: `--judge hybrid` routes
short text (< 15 words) to `SubstringJudge`, longer text to the trained
classifier. Live result: leakage_rate=0.6 (down from 0.95), but every
remaining flagged sentence was still a false positive — mood/theme/
production text, no real plot reveal — confirmed by scoring a
hand-invented neutral control sentence, which landed in the same noise
band. This classifier has no real signal on this generator's writing
register at any length tested.

**`--judge llm` tried too — same pattern.** A complete 20-title run
scored leakage_rate=0.8: release-year facts, director credits, and
marketing hooks ("Are you ready to have your mind blown?") all flagged
as core leaks. One catch looked genuinely plausible
(`sixth_sense_1999`), but it was a small signal in a lot of noise.

**Decision: stop iterating on judges.** `SubstringJudge` (recall=0.0, an
honest floor) stays the default; the others stay available for
comparison, none fit to report a real `leakage_rate`. Working hypothesis
instead: a `corpus=[]` generator has nothing real to write from, so it
defaults to vague "hints at something" prose that fools any
recall-favoring judge — not a judge problem, a grounding problem.
Milestone 1 (below) tests that directly.

### Milestone 1: real retrieval, GREEN-tier only (D16)

`src/preshow/retrieval.py` wires up the GREEN/AMBER/RED corpus-tagging
mechanism this project's schema was built for (`SourceDoc.tier`,
`PreShowBrief`'s own docstring: "Generated with GREEN context only") but
Milestone 0 never used. Fetches Wikipedia, returns only
`overview`/`production`/`accolades` as GREEN — the `plot` section is
never even constructed as a source, and `reception`/critical-response is
dropped too (real reviews discuss plot points). `--generator
retrieval-groq` uses it.

**First live result**: `grounded_fact_rate` went from 0.0 (Milestone 0)
to **1.0** (19/20 titles, hit a daily API quota on the 20th) — a
judge-independent, purely structural result (every fact-kind claim now
cites a real source_id). `leakage_rate` measured 0.0 under
`--judge substring`, but that alone doesn't prove zero leaks (substring
can't see a leak pulled from the model's own memory rather than the
corpus) — what IS proven independent of any judge is that the corpus
itself never contains plot spoilers to begin with. `run_eval.py
--save-briefs` (new) writes the full generated text so a human can check
the remaining question directly. Full account in D16, `docs/DESIGN.md`.

The decisions behind this design (why there's no LangGraph, why the schema
allows invalid states on purpose, why the same model being measured can't
generate its own ground truth) are documented in
[`docs/DESIGN.md`](docs/DESIGN.md).

## Research-assist tool: scaling the researched catalogue

Hand-researching an entry (cited sources, no invented facts) takes ~15
minutes each — fine for 8 titles, not for the "most important films in
cinema history" the catalogue is meant to grow into. `webapp/research_assist.py`
automates the *labor* of that process, not the citation bar itself:

```bash
python webapp/research_assist.py "Citizen Kane" 1941
```

1. **Real retrieval, never LLM memory.** Fetches the film's Wikipedia
   article (`src/preshow/wikipedia.py`) and its TMDB metadata/director
   (`src/preshow/tmdb.py`), and gives the model *only* that retrieved text
   to draft from — the prompt explicitly forbids citing anything else.
2. **Best-of-3 drafting.** A single free-model call was inconsistent run
   to run (4–15 grounded claims from the same prompt against the same
   text — see D14 in `docs/DESIGN.md`), so `draft_best_of()` generates 3
   independent candidates from the same retrieved text and keeps the one
   with the most grounded, unfabricated claims.
3. **A code-level safety net, not just a prompt instruction.**
   `sanitize_grounding()` walks every citation in the draft and nulls out
   anything that isn't exactly a URL this run actually retrieved — it
   already caught the model inventing a plausible-looking Rotten Tomatoes
   URL in testing.
4. **Output never bypasses human review.** Drafts are written to
   `content/_drafts/` (gitignored), never straight to
   `content/researched/` — a human still has to read and promote a draft
   before it's published, same bar as the existing 8 entries.

**Status:** tested end-to-end on one title (Citizen Kane, 3/3 candidates
succeeded: 4, 4, and 5 grounded claims, correctly picked the 5, zero
fabricated citations reached the output). Needs `GROQ_API_KEY` (free).
See [Roadmap](#roadmap) for what's next.

## Browse tier: TMDB

The 12 titles in the 20-title measurement set that aren't hand-researched
yet still show a real poster and synopsis instead of an empty
placeholder, and `/api/search` reaches effectively all of TMDB — this is
a deliberately separate, lower tier: it never claims to be spoiler-safe
or cited the way `content/researched/*.json` is (see D10 in
`docs/DESIGN.md`). Needs `TMDB_READ_ACCESS_TOKEN`; without it, the app
still runs, it just shows the "not researched yet" placeholder instead of
a poster.

## Configuration

Everything in the [quickstart](#try-it-in-2-minutes) works with **zero
keys** — the demo app, its comments, and the offline evals harness
(`pytest tests/`) need nothing. Every key below is an optional upgrade,
never a requirement; all are read from a gitignored `.env` in the repo
root (`src/preshow/env.py`, falls back to the real environment too — no
`python-dotenv` dependency).

| Variable | Unlocks | Cost |
|---|---|---|
| `TMDB_READ_ACCESS_TOKEN` | Browse-tier posters/search for the 12 not-yet-researched titles (D10) | Free, [themoviedb.org](https://www.themoviedb.org/settings/api) |
| `GROQ_API_KEY` | `--generator baseline-groq`, `LLMJudge` external calibration, `research_assist.py` | Free tier, no card, [console.groq.com/keys](https://console.groq.com/keys) |
| `ANTHROPIC_API_KEY` | `--generator baseline` (Claude instead of Groq for the same baseline generator) | Paid — the *only* piece of this project that costs money, and it's opt-in |
| `UPSTASH_REDIS_REST_URL` + `UPSTASH_REDIS_REST_TOKEN` | Comments/movie-requests survive a redeploy on a free host with an ephemeral filesystem (D11) | Free, no card, [console.upstash.com](https://console.upstash.com) |

Full install, including the optional pieces:

```bash
pip install fastapi "uvicorn[standard]" pydantic pyyaml pytest   # core app + harness + tests
pip install groq                                          # optional: baseline-groq, LLMJudge calibration, research_assist.py
pip install anthropic                                     # optional: paid baseline generator only
```

## Project layout

```
webapp/                 FastAPI app, one-time build helpers, research-assist tool
  app.py                 serves the catalogue, spoiler gate, comments, search
  index.html              vanilla HTML/CSS/JS frontend
  research_assist.py     drafts a researched entry from Wikipedia + TMDB (D14)
  prewarm_translations.py  build step: caches ES translations
  resolve_tmdb_ids.py     build step: resolves tmdb_id for every title

src/preshow/             shared library: schemas, clients, generators
  schemas.py               Pydantic data contracts (Claim, PreShowBrief, ContentPack…)
  baseline.py / baseline_groq.py / baseline_prompts.py   no-retrieval baseline generator (2 providers, 1 prompt)
  content.py                loads/serves content/researched/*.json
  tmdb.py / wikipedia.py    stdlib-only clients for the browse tier and research-assist retrieval
  translate.py               free MyMemory API client, disk-cached
  kv_store.py                Upstash Redis client with a local-file fallback
  env.py                     shared .env reader

evals/                   the real experiment (measurement track)
  run_eval.py               runs the baseline over the 20-title dataset
  judge.py                   SubstringJudge + LLMJudge
  metrics.py                  leakage_rate / grounded_fact_rate / richness
  calibrate_substring.py               internal calibration (LLM paraphrases)
  calibrate_substring_external.py      SubstringJudge vs. real IMDb reviews (D12)
  calibrate_llm_external.py             LLMJudge vs. the same real reviews (D13)
  dataset/titles.yaml                    20-title stratified spoiler ground truth

content/
  researched/*.json        8 hand-researched, cited entries
  _translations/            cached ES translations (committed — see D9)
  _drafts/                    research_assist.py output (gitignored, pre-review)
  _tmdb_cache/                 (gitignored)
  _wikipedia_cache/            research_assist.py's fetched articles (gitignored)

tests/                   offline pytest suite (8 tests), no network, no API key
docs/DESIGN.md            every design decision (D1–D14) with its trade-off
docs/screenshots/          the two screenshots at the top of this README
.github/workflows/tests.yml   CI: installs core deps, runs pytest on every push/PR
```

## Roadmap

Documented as open, not started, in `docs/DESIGN.md`'s "Open questions"
and `CLAUDE.md`'s "Next task":

- **Resolve the judge-trust gap before Milestone 1 (retrieval).** Neither
  judge currently clears the bar to report a trustworthy `leakage_rate`
  (see [Limitations](#limitations)). Next concrete step: re-run
  `evals/calibrate_llm_external.py` with full, untruncated review text
  and/or a stronger model (`llama-3.3-70b-versatile`) to find out whether
  today's 0.089 recall is a real model-capacity ceiling or an artifact of
  the 350-char truncation forced by this run's token budget. If that's
  still not good enough, two genuinely different alternatives are on the
  table but not started: a lightweight local NLI/entailment classifier
  (no per-call cost or rate limit), or a classifier trained directly on
  this project's own 2,197-positive/5,460-negative external labels with a
  proper held-out split.
- **Research the remaining 12 measurement titles** the same way Gone Girl
  was (cited sources, no invented facts) — `research_assist.py` is the
  start of automating this, once its output quality is more consistent
  than a single Citizen Kane run has proven so far.
- **Automate the "+ Suggest a movie" pipeline.** It already resolves a
  `tmdb_id` per suggestion via TMDB autocomplete, but doesn't research or
  add anything automatically yet — that's exactly what
  `research_assist.py` is the first step toward.
- **Upstash on the live Render deploy** — deliberately deferred (the
  project owner's call, not a blocker): comments/movie-requests on the
  live demo reset on every idle spin-down until an Upstash account is
  wired in. The code path (D11) already handles this gracefully (empty
  state, not an error).

## Limitations

Stated plainly, not hidden behind a green badge — this is the project's
own stated goal applied to itself:

- **Neither spoiler judge is trustworthy yet**, and no `leakage_rate` in
  this README should be read as a real safety measurement until this
  changes. `SubstringJudge` catches 0 of 2,197 real human spoiler reviews
  (recall = 0.0 — it only matches verbatim planted phrases). `LLMJudge`
  (Groq, `llama-3.1-8b-instant`) does better but still misses ~91 of every
  100 real spoiler reveals in its calibration sample (recall = 0.089,
  precision = 0.471, n=180 — see [`LLMJudge` calibration](#llmjudge-calibrated-the-same-way-groq-free-tier)).
  This is *why* Milestone 1 (retrieval) hasn't started — there's no point
  measuring whether retrieval helps against a judge that can't reliably
  see leaks either way.
- **`grounded_fact_rate = 0.0` on the current baseline is expected, not a
  bug** — the no-retrieval baseline is given no corpus on purpose; it's
  the number Milestone 1 needs to beat.
- **The mainstream vs. long-tail hypothesis is unconfirmed.** Both strata
  scored identically in the Milestone 0 run — a judge with ~0 real recall
  can't reveal a gap that might genuinely exist.
- **`research_assist.py` is tested on one title.** Best-of-3 generation
  fixed the run-to-run inconsistency seen on that one title (Citizen
  Kane); it hasn't been run across a larger, varied sample yet, so
  "consistent output quality" is a hypothesis, not yet a measured claim.
  It also can't cite Rotten Tomatoes/Metacritic directly (no simple free
  API for either) — a real, disclosed gap against the 8 hand-researched
  entries, which do cite those sites directly.
- **The live demo's comments/movie-requests reset on idle spin-down**
  (Render free tier wipes the filesystem; Upstash isn't wired in yet —
  see [Roadmap](#roadmap)). Known and accepted, not a bug to chase.

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
