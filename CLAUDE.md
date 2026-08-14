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
  FastAPI + vanilla JS app, 8 hand-researched films with a spoiler curtain, comments,
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
  human set (D12/D13): `SubstringJudge` recall=0.0 (0/2,197, full set,
  95% CI 0.0-0.002 -- tight, not sample noise). `LLMJudge`/llama-3.1-8b-instant
  started at recall=0.089/precision=0.471 (350-char truncation, forced by
  the original token budget), but a follow-up isolating that confound
  (same model, same 180-review sample, truncation raised to 1,500 chars)
  found recall=**0.356** (95% CI 0.264-0.459), precision=**0.64** (95% CI
  0.501-0.759) -- non-overlapping CIs vs. the 350-char run, so truncation
  was genuinely suppressing most of the model's real recall, not just
  adding noise. Still not trustworthy enough to report a `leakage_rate`
  (misses ~64/100 real spoilers). The other confound -- does a stronger
  model close more of the gap -- is now **resolved: barely at all**.
  After 3 free-tier failures (a `llama-3.3-70b-versatile` daily token
  quota that turned out to be a rolling window, plus a transient Groq
  503) a 4th attempt completed cleanly at the original 350-char
  truncation: recall=0.093 (95% CI 0.04-0.199, n=108) vs. the 8B model's
  0.089 (95% CI 0.046-0.166, n=180) -- essentially identical, heavily
  overlapping CIs. A ~9x larger model does not meaningfully improve
  recall. Precision looked better (0.714 vs 0.471) but both CIs are wide
  and overlap. Combined conclusion: truncation was the real lever, model
  size wasn't -- `llama-3.1-8b-instant` is the better default going
  forward (no daily token cap, survived all 4 attempts without a single
  free-tier failure). `LLMJudge`'s ceiling at reasonable settings looks
  to be recall~=0.35-0.4 (see `evals/stats.py` for the Wilson score
  interval helper all these runs use).
  **D15: two more approaches tried, one failed, one is now the best
  judge calibrated.** A local NLI classifier (`NLIJudge`) failed on two
  fronts -- impractically slow on this dev machine (antivirus-related,
  not fixed by an exclusion) and, where it did run, recall ~0 (formal
  NLI entailment is stricter than "could a viewer infer this", a real
  task mismatch). But `evals/train_spoiler_classifier.py` (TF-IDF +
  Logistic Regression, trained directly on this project's own
  2,197-positive/5,460-negative external labels, evaluated with grouped
  k-fold by title so every prediction is out-of-fold) beats every prior
  judge: recall=0.889 at threshold 0.3 (95% CI 0.876-0.902), or
  recall=0.445/precision=0.571 at threshold 0.5 -- on the FULL
  7,657-review set, no sample-size limit, confirmed not to be a
  single-title artifact (per-title recall 0.285-0.668 across all 9
  titles, Fight Club -- 32% of the data -- has the *lowest* per-title
  recall). **Now wired in**: `TrainedClassifierJudge` in `evals/judge.py`
  loads a persisted model (`joblib`, built by
  `python evals/train_spoiler_classifier.py`) and implements the same
  `.entails(text, label)` interface as the other judges (ignores `label`
  -- see its docstring for why); `run_eval.py --judge trained-classifier
  [--judge-threshold N]` opts into it (default judge stays `substring`,
  since that needs nothing but the stdlib and always works offline).
  Threshold defaults to 0.3 (recall-favoring, per D15). Sanity-checked
  live on 3 example texts (spoiler-y text -> 0.576/0.857 spoiler_prob,
  generic praise -> 0.14) -- confirmed working correctly, not yet run
  through an actual `--generator` end to end (needs live Groq/Anthropic
  access this session's Bash environment doesn't reliably have).
  **In-domain validation done** (`evals/calibrate_trained_classifier_internal.py`,
  since D15's calibration was on IMDb reviews, not the brief-style text
  this judge actually scores): first attempt reused the wrong dataset
  (calibrate_substring.py's cross-label negatives are real spoilers for
  a DIFFERENT movie -- valid negatives for a per-label judge, not for
  this one) and made the judge look worse than it is. Corrected version
  (277 real spoiler sentences vs. 62 real spoiler-free sentences from
  the 8 researched titles) shows it's actually BETTER in-domain:
  recall=0.838 (95% CI 0.79-0.876), precision=0.928 (95% CI
  0.889-0.954) at threshold 0.3 -- on COMPLETE sentences.
  **CONFIRMED BROKEN on the actual generator output, the same day it was
  wired in.** The first live `run_eval.py --generator baseline-groq
  --judge trained-classifier --show-leaks` run scored
  **leakage_rate=0.95**, flagging a stage direction ("Black screen"), a
  cast credit, and an award fact as spoilers. Root cause: baseline
  generators write SHORT FRAGMENTS (`on_screen_text`/`voiceover` lines,
  often 3-8 words) -- a third register, different from both this judge's
  IMDb-review training data and its own complete-sentence validation
  set. Scored the exact flagged phrases directly: all landed in a
  narrow 0.30-0.47 band with no separation between real fragments and
  benign ones -- not a threshold problem, the model has no signal left
  on text this short. `run_eval.py` now prints a warning when
  `--judge trained-classifier` is selected.
  **Fix attempt: `HybridJudge`** (`evals/judge.py`) routes by word count
  (< 15 words -- the cutoff sits just above the highest confirmed false
  positive, 14 words -- goes to `SubstringJudge`; >= 15 goes to
  `TrainedClassifierJudge`), not by field name (rejected: false
  positives hit `author_voice`/`context_bullets` as often as `script[]`
  lines, this generator writes short text everywhere). Regression-tested
  offline (`tests/test_hybrid_judge.py`, no scikit-learn needed, runs in
  CI): all 18 confirmed false positives route away from the classifier,
  a real literal leak is still caught.
  **Live run: reduced but did not fix it.** leakage_rate=0.6 (down from
  0.95), but `--show-leaks` showed every remaining flagged sentence is
  STILL a false positive (mood/theme/production text, no actual plot
  reveal) -- a hand-invented, genuinely neutral control sentence scored
  0.252, in the same noise band as the real false positives. This
  classifier has no real signal on this generator's writing style at any
  length; `hybrid` only changes how often the noise fires. `run_eval.py`
  now warns about `--judge hybrid` too.
  **`--judge llm` wired into `run_eval.py`** for the first time --
  `LLMJudge` was calibrated (D13) but never tested against this
  project's actual generator output before. First live attempt got 3/20
  cases in with a genuinely mixed signal (not the blatant false
  positives `TrainedClassifierJudge` gave -- but `parasite_2019` flagged
  "In a world where social classes collide", pure generic intrigue the
  judge's own prompt says shouldn't count), then hit a SEPARATE
  pre-existing bug: `GroqBaselineGenerator` had no retry on a malformed
  tool call (same `tool_use_failed` error hit once before with
  `prestige_2006`), losing the rest of the run. **Fixed**: added the
  same corrective-retry pattern `research_assist.py` already uses
  (`baseline_groq.py`'s new `_call_with_retry`, verified with a fake
  client that fails twice then succeeds) -- doesn't touch the shared
  SYSTEM_PROMPT/schema, retry behavior only. `run_eval.py`'s main loop
  also now catches any mid-run failure and aggregates whatever completed
  (`"partial"`/`"n_cases_completed"`) instead of losing the run.
  **Complete 20-title `--judge llm` run: leakage_rate=0.8, same pattern
  as every prior judge** -- release-year facts, director credits, and
  pure marketing hooks ("Are you ready to have your mind blown?") all
  flagged "core", though one catch (`sixth_sense_1999`'s "living are
  unaware of the dead") looked genuinely plausible. **Decision made**
  (D15's conclusion): stop iterating on judges. `SubstringJudge` stays
  the default (honest recall=0.0 floor); `trained-classifier`/`hybrid`/
  `llm` all stay available for comparison, none fit to report a real
  `leakage_rate`. Working hypothesis instead: Milestone 0's `corpus=[]`
  generator has nothing real to write from, so it defaults to vague
  "hints at something" prose that fools any recall-favoring judge --
  not a judge problem, a grounding problem.
  **D16: Milestone 1 built to test that hypothesis.**
  `src/preshow/retrieval.py`'s `build_green_corpus()` wires up the
  GREEN/AMBER/RED mechanism D3 designed but Milestone 0 never used
  (`PreShowBrief`'s own docstring already said "Generated with GREEN
  context only") -- fetches Wikipedia, returns only
  overview/production/accolades as GREEN `SourceDoc`s, never even
  constructs a `SourceDoc` for the Plot section, drops
  reception/critical-response too (AMBER treated as RED, D3's default).
  Verified live against real Wikipedia (Parasite) and with tests
  specifically trying to leak plot text through
  (`tests/test_retrieval.py`). New generator
  `GroqRetrievalGenerator`/`--generator retrieval-groq`
  (`retrieval_groq.py` + `retrieval_prompts.py`, a genuinely different
  prompt from Milestone 0's, not a shared variant -- see D16) uses it.
  **First live run: real signal.** 19/20 cases (hit the daily token
  quota on the last one -- larger, grounded prompt costs more/call than
  Milestone 0's) -- `grounded_fact_rate` **0.0 -> 1.0** vs. Milestone 0,
  a judge-independent structural result (just checks `source_id` is
  populated). `leakage_rate` stayed 0.0 under `--judge substring`, but
  that's *not* proof of zero leaks -- substring's recall=0.0 (D12) can't
  see a leak pulled from the model's own memory rather than the corpus.
  What IS proven independent of any judge: the `plot` section is never
  even constructed as a source (retrieval.py), so nothing the model
  draws FROM THE CORPUS can be a plot spoiler -- that's an input fact,
  not a judge verdict. Added `run_eval.py --save-briefs PATH` so a human
  can read the actual generated text directly instead of trusting a
  judge either way.
  **Human spot-check done (13/20 titles, second run hit the same
  quota): mostly excellent, one real leak found.** Sixth Sense, Fight
  Club, Se7en, The Prestige, Gone Girl -- the 5 highest-risk
  famous-twist titles -- all came back clean on direct human reading (no
  judge involved). But Los cronocrímenes' generated script said "One man
  must stop his other selves" -- this project's own documented ground
  truth lists "there are two, even three, versions of the same man
  coexisting on the same day" as a **core**-severity paraphrase of that
  film's core spoiler. The GREEN corpus for that title never mentions
  multiple selves, so this reveal came from the model's own memory, not
  the corpus -- confirms the exact risk D16 flagged as still open, with
  a concrete example instead of just a caveat. `SubstringJudge` didn't
  catch it either (not a literal string match) -- consistent with its
  known recall=0.0, not a new problem. All 20 titles now have a resolved
  `tmdb_id` in `titles.yaml` (D10) for browse-tier posters — cosmetic,
  doesn't touch the stratified sample or labels.
  **Milestone 1: all 20/20 titles now complete and hand-read.** Three
  infrastructure fixes (`llama-3.1-8b-instant` for its 500K TPD budget,
  `DEFAULT_MAX_CHARS_PER_SECTION` truncation to fix a 413 "request too
  large", `MIN_CALL_INTERVAL_S` pacing + retry for TPM 429s — see D16)
  cleared the remaining 7 titles. Full aggregate: `grounded_fact_rate`
  0.0 -> **1.0**, `leakage_rate` (substring) 0.0, `richness` 5.2/4.9
  mainstream/longtail (still not confirming the original hypothesis).
  **Hand-reading the final 5 found 2 more real leaks — Tetsuo and Hard
  to Be a God — both traced to Wikipedia's own `overview` section
  containing the film's core/major spoiler verbatim** (unlike
  Cronocrímenes, this wasn't the model's memory — the "safe" GREEN
  corpus itself wasn't safe for these titles). A third, Come and See,
  has the same risk latent in its `production` section though it didn't
  surface this run. **Explicit decision with the user: document as a
  known limit of M1 and close it (same reasoning as closing judge
  iteration — a content filter would carry the same false-negative risk
  every judge attempt already showed).** Milestone 1's real result
  stands (grounding genuinely improved, all 5 highest-risk titles
  clean) but is not a leak-proof guarantee — two independent leak
  mechanisms are now confirmed. Full account in D16's final section,
  docs/DESIGN.md.
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
- **Keep it short (2026-08-13, explicit request in chat).** Too much text, too much
  explaining. Lead with the answer or the change; give reasoning only where it would
  change a decision. Don't recap work already visible in the diff, don't restate the
  question before answering it, and don't close with a summing-up line. Applies to
  chat, commit messages and PR bodies. Reference docs (README, this file) can be
  longer, but only where the length earns it.
- Never report leakage_rate/grounded_fact_rate without richness alongside — an empty
  output scores perfectly on the first two.
- New generators get tested against fixtures/`ScriptedFakeGenerator` first, never the
  real API as a first step.
- Don't reach for LangGraph/CrewAI/AutoGen by default — pipeline is sequential (D1).
  Argue from a concrete need first.
- Sources: TMDB free non-commercial w/ attribution (no fine-tuning); IMDb scraping
  prohibited by ToS; Goodreads API gone since 2020 (Milestone 3 concern).
- **Scheduled documentation-sync runs (added 2026-08-07, explicit decision in
  chat): standing authorization to merge doc-only PRs from that recurring task
  yourself, without waiting for approval, once CI (`tests.yml`) is green —
  same bar as any other merge, just no confirmation step for this specific,
  narrow case (README/`docs/` changes only, never product/measurement-track
  code).** That scheduled run lands on a fresh randomly-named branch every
  time, so an unmerged PR from a previous run is never reused automatically.
  Before opening a new one, check for another open PR titled starting
  "docs: sync" — if found, fold any still-valid unique content from it into
  the new one, merge the more complete/accurate PR once CI is green, and
  close the other with a comment linking to the merged one. Don't leave two
  open at once.

## Next task

**Judge search is closed for good now (D15 first concluded this, D16
briefly reopened it for one more attempt, now closed again for real --
see the bullet list below for the final state).** Six judges tried
against this project's actual generator output across D15/D16:
`SubstringJudge` (recall=0.0, an honest floor -- the one that stuck),
`TrainedClassifierJudge` (leakage_rate 0.95 -- cast credits and stage
directions flagged as spoilers), `HybridJudge` (0.6, still all false
positives once inspected), `LLMJudge` (0.8, same pattern), and
`SimilarityJudge` (calibrated well offline, but a live false positive
outscored the one real leak it was built to catch). None fit to report a
real `leakage_rate`. `SubstringJudge` stays `run_eval.py`'s default; the
other five stay available for comparison, all clearly marked not for
reporting.

**Working hypothesis instead of another judge attempt**: Milestone 0's
`corpus=[]` generator has nothing real to write from, so it defaults to
vague, evocative "hints at something" prose -- and `JUDGE_PROMPT`
deliberately asks judges to flag hints (this project's own
recall-favoring principle), so ungrounded text fools any judge built
that way, more or less regardless of which judge. If true, no more
judge-tuning fixes this; giving the generator something real to write
from might.

**D16: Milestone 1 built to test that hypothesis -- confirmed live,
see Status above (grounded_fact_rate 0.0->1.0, one real leak found by
human review, sixth judge attempt also failed).**
`src/preshow/retrieval.py`'s `build_green_corpus(title, year)` wires up
the GREEN/AMBER/RED corpus-tagging mechanism D3 designed back in this
project's early decisions but Milestone 0 never used
(`schemas.py`'s `PreShowBrief` docstring already said "Generated with
GREEN context only" -- the schema was ready, the code wasn't). Fetches
Wikipedia (`preshow/wikipedia.py`, reused from `research_assist.py`),
returns ONLY `overview`/`production`/`accolades` sections as GREEN
`SourceDoc`s -- the `plot` section is never even constructed as a
`SourceDoc` (not filtered after the fact -- never built), and
`reception`/critical-response is dropped too (AMBER treated as RED, D3's
stated default, since real reviews discuss plot points). Verified live
against real Wikipedia (Parasite: premise/cast/awards, nothing
resembling the actual plot turns) and offline with tests specifically
trying to leak plot/reception text through a fabricated article
(`tests/test_retrieval.py`). New generator `GroqRetrievalGenerator`
(`retrieval_groq.py` + `retrieval_prompts.py` -- a genuinely different
prompt from Milestone 0's `baseline_prompts.py`, not a shared variant,
since M0's premise is "you have nothing to cite" and M1's is the
opposite) wired into `run_eval.py --generator retrieval-groq`.

Concretely, next:

- **Confirmed live: `grounded_fact_rate` 0.0 -> 1.0**, and **confirmed by
  direct human reading** (not a judge) that retrieval gets the highest-risk
  cases right: Sixth Sense, Fight Club, Se7en, The Prestige, Gone Girl all
  came back clean. **But a real leak was found too**: Los cronocrímenes'
  script revealed "his other selves," matching this project's own
  documented core spoiler paraphrase almost exactly -- and it came from
  the model's memory, not the (clean) corpus, since the retrieved GREEN
  text never mentions multiple selves. See Status above and D16's second
  update in docs/DESIGN.md for the full account. Milestone 1 measurably
  improved things, it didn't solve them -- don't report a Milestone 1
  `leakage_rate` as a safety claim yet either.
- **`SimilarityJudge` tested live -- also not good enough, and judge
  iteration is now CLOSED (explicit decision with the user).** Calibrated
  well offline (recall=0.87/precision=0.856) and caught the one confirmed
  leak it was built for, but a live run immediately found a false
  positive ("The film stars Bruce Willis as a child psychologist...",
  the film's own public premise) scoring 0.561 -- HIGHER than the
  confirmed real leak's 0.525. No threshold separates those two numbers.
  This is the SIXTH judge in this project (Substring, LLM-external, NLI,
  TrainedClassifier, Hybrid, Similarity) to either fail outright or
  calibrate well and then fail live -- six independent pieces of
  evidence, not one repeating bug. **Decision: stop building judges.**
  `SubstringJudge` (recall=0.0) stays the only judge this project trusts
  for `leakage_rate` reporting -- not because it's good, but because its
  failure mode (misses things, bounded and known) is safer than every
  alternative's (confidently flags non-leaks, sometimes MORE confidently
  than real leaks). The other five stay in the codebase for comparison
  only, all clearly warn-on-use. The real safety practice going forward
  is `--save-briefs` + a human reading the text directly -- which is what
  actually caught the one confirmed real leak in this project so far, not
  any automated judge. Full account in D16's final correction,
  docs/DESIGN.md.
- **Milestone 1 is now complete: all 20/20 titles run and hand-read.**
  The remaining 7 titles finished after three infrastructure fixes
  (`--model llama-3.1-8b-instant`, GREEN corpus truncation, call pacing
  + retry -- D16's final section). Hand-reading the last 5 found 2 more
  real leaks (Tetsuo, Hard to Be a God) plus one latent risk (Come and
  See) -- all three traced to Wikipedia's `overview`/`production`
  sections containing the spoiler verbatim, a different mechanism from
  Los cronocrímenes' model-memory leak. **Explicit decision with the
  user: document this as a known limit of M1 and close it (same
  reasoning as closing judge iteration -- see D16).** Milestone 1's next
  task is now whatever Sergio wants to tackle next, not more leak-fixing
  on this milestone.
- Groq's Dev Tier (paid, would remove the daily-quota bottleneck this
  project has repeatedly hit) is temporarily not accepting upgrades
  ("high demand", per Groq's own billing page, checked live). Cheaper/
  alternative options if this keeps blocking progress: wait and retry:
  OpenRouter (pay-as-you-go, no minimum spend, ~5.5% top-up fee, 25+
  free models with a 50 req/day cap) is the lowest-friction alternative
  for this project's tiny volume; Together AI and Fireworks AI both work
  too but require a real minimum prepay (~$5) for volume this project
  doesn't need. Switching provider is a real code change, not just an
  env var -- these use an OpenAI-compatible API shape, not Groq's own
  SDK, so `baseline_groq.py`/`retrieval_groq.py` would need a new client
  construction, not a drop-in swap.
- Also still open, low priority, only relevant if a judge is ever
  revisited (not planned): the long-form `why_now`-paragraph
  false-positive pattern found during `TrainedClassifierJudge`'s
  in-domain validation (D15).

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
**Fixed**: `author_voice` used to come back as the model's own generic
critical opinion ("As a film-literate critic, I approach...") rather
than an actual quote/statement from someone who made the film (director,
screenwriter), which is what that field is for (see Gone Girl's
Fincher/Flynn quotes). Added rule 11 to `SYSTEM_PROMPT` plus a GOOD/BAD
few-shot pair and a tool-schema `description` on the `author_voice`
property, all saying the same thing three ways: it's an attributed
quote or it's empty, never the drafter's own opinion. Not yet re-tested
live (see the 403 note below) — re-run `research_assist.py` on Citizen
Kane again once Groq access is confirmed, to check this actually changed
the output and didn't just move the problem.

**Still open, blocking live re-verification**: the Groq 403 ("Access
denied, check your network settings") that this session first hit and
then saw clear on its own **recurred** when re-tested from this same
Bash execution environment while investigating item 2 of the next task
below (calling the API directly, no SDK, same 403). This confirms it's
tied to this environment's outbound network path specifically, not a
one-off — the user's own machine works fine once their VPN is off. Any
task in this repo that needs a live Groq call (`research_assist.py`,
`calibrate_llm_external.py`, `run_eval.py --generator baseline-groq`)
may need to be run from the user's own machine, not assumed to work from
whatever shell is executing Claude Code. No reproduction steps beyond
"sometimes this specific egress blocks, sometimes it doesn't" — not
chased further, per the user's own earlier instruction to not block on
this.

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
python evals/calibrate_llm_external.py --truncate-chars 1500 --interval 4.2       # isolate the truncation confound
python evals/calibrate_llm_external.py --model llama-3.3-70b-versatile --sample-per-title 12  # isolate the model confound
python webapp/research_assist.py "<Title>" <year> [n_candidates=3]  # needs free GROQ_API_KEY (see D14)
pip install sentence-transformers && python evals/calibrate_nli_external.py --sample-per-title 20  # failed judge, see D15
pip install scikit-learn && python evals/train_spoiler_classifier.py       # best judge so far, see D15
python evals/calibrate_trained_classifier_internal.py             # validates it in-domain, not just IMDb reviews
python evals/run_eval.py --generator baseline-groq --judge substring  # Milestone 0 (default judge), needs GROQ_API_KEY
python evals/run_eval.py --generator baseline-groq --judge llm --show-leaks    # judges tested, none good enough -- see D15
python evals/run_eval.py --generator retrieval-groq --model llama-3.1-8b-instant --judge substring --show-leaks --save-briefs evals/results/retrieval_briefs.json  # Milestone 1 (D16), complete 20/20
python evals/run_eval.py --generator retrieval-groq --titles come_and_see_1985,tetsuo_1989  # re-run only specific titles, doesn't affect the >=15 gate
python evals/calibrate_similarity.py                               # calibrates SimilarityJudge, see D16 -- judge iteration is closed, kept for comparison only
```
`gh` CLI may need its full path (`C:\Program Files\GitHub CLI\gh.exe`) if not on PATH.
TMDB key/read token live in `.env` (gitignored) — used by `src/preshow/tmdb.py`
(D10) for the browse tier and catalogue posters. `GROQ_API_KEY` also belongs
in `.env` once obtained (same gitignored file, never commit it). For a public
deploy, add `UPSTASH_REDIS_REST_URL`/`UPSTASH_REDIS_REST_TOKEN` (free, no card,
console.upstash.com) so comments/movie-requests survive a redeploy — see D11;
without them the app still runs fine locally on plain files.
