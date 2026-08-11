"""Calibrates `LLMJudge` (evals/judge.py) against the same real IMDb
spoiler reviews used in calibrate_substring_external.py (D12), instead of
SubstringJudge -- the direct comparison the project's own next task
called for: does the LLM judge actually beat the free floor, on the same
human-labeled data?

Backend: Groq's free tier (already used for the Milestone 0 baseline
generator, `src/preshow/baseline_groq.py`) -- no cost, but genuinely
rate-limited: llama-3.1-8b-instant is capped at 30 requests/min, ~6,000
tokens/min, 1,000 requests/day (org-wide, shared with anything else using
this key today). A small model on purpose -- judge.py's own rationale:
"the task is a short binary entailment call and doesn't justify a large
one."

Consequence: testing all 7,657 reviews x ~2.7 labels/title (~20,000 calls)
is not possible for free in one day. This samples a fixed, stratified
subset instead -- up to SAMPLE_PER_TITLE reviews per title, split as
evenly as possible between spoiler-tagged and not, deterministic (fixed
seed) so a re-run is comparable. Report the sample size honestly; this is
a smaller, not a weaker, form of the same measurement -- same real human
text, same "entails ANY documented label" method and its one caveat
(D12: a real spoiler review can reveal a plot point this project didn't
document, counted here as a miss that isn't really the judge's fault).

Also truncates each review to a few hundred characters before sending it
to the model, to keep per-call tokens well under the TPM cap -- a spoiler
line is usually near the start of a review or repeated through it, but a
reveal appearing only very late in a long review could be missed by this
truncation specifically (separate from the labeling-coverage caveat
above). Said plainly, not hidden.

CLI flags let you isolate the two confounds the first run (--model
llama-3.1-8b-instant --truncate-chars 350, the defaults) couldn't
separate, WITHOUT re-running the same conditions twice or overwriting
that baseline result (output filename is derived from the config; the
exact-default config keeps writing to the original
llm_calibration_external.json):

  # truncation confound: same model, longer text (median review is ~800
  # chars per an offline sample -- 1500 covers most of the distribution
  # without blowing the 8B model's ~6,000 TPM cap at this pacing).
  # CONFIRMED live (D13 follow-up): recall 0.089 -> 0.356, precision
  # 0.471 -> 0.64, 95% CIs don't overlap -- truncation was suppressing
  # real recall, this is not sample noise.
  python evals/calibrate_llm_external.py --truncate-chars 1500 --interval 4.2

  # model confound: same 350-char truncation as the baseline, stronger
  # model. Sample size cut to fit the model's real observed daily budget
  # (see "Token budget math" below -- the naive estimate undershot this
  # by ~35%, and a live run hit DailyQuotaExhausted around call ~400/480).
  python evals/calibrate_llm_external.py --model llama-3.3-70b-versatile --sample-per-title 12

Token budget math (rough, ~4 chars/token): prompt overhead (JUDGE_PROMPT
template + one SpoilerLabel) is ~100-150 tokens; review text adds
truncate_chars/4 tokens. At the default 480 calls (180 reviews x ~2.7
labels/review), the naive estimate was ~190 tokens/call (~91k total,
"fits" under llama-3.3-70b-versatile's 100k TPD cap) -- WRONG in
practice: a live run hit DailyQuotaExhausted (see that exception's
docstring) around call ~400/480, meaning real usage is closer to
~250 tokens/call. --sample-per-title 12 above targets ~290 calls
(12 reviews/title x 9 titles x ~2.7 labels), leaving real margin instead
of running right up to the edge again. Note this makes it a SMALLER,
DIFFERENT sample than the 180-review baseline/truncation runs (SEED is
fixed, but changing --sample-per-title changes which reviews the
reservoir sampler keeps, not just how many -- see build_sample's
docstring) -- report it as its own n, don't imply it's the same 180
reviews.
llama-3.1-8b-instant has no documented TPD cap (only RPM/RPD/TPM), which
is why the truncation-confound command above keeps the model at the
default and raises --interval instead, to stay under the 8B model's
~6,000 TPM ceiling (475 tokens/call x 14 calls/min ~= 6,650, so
--interval 4.2 targets ~14 calls/min with margin) -- CONFIRMED live, that
one completed all 480 calls cleanly.
Re-verify current limits before trusting this math -- Groq's free-tier
caps are not contractual and can change; this project has already hit an
unrelated 403 from Groq/Cloudflare once (see CLAUDE.md), so budget in a
retry margin, not just the happy path. If a run hits the daily token cap
OR Groq's API is itself down/overloaded (503 "over capacity" -- observed
live on a --sample-per-title 12 attempt), it now stops cleanly and writes
a partial result (see DailyQuotaExhausted, ServiceUnavailable, and the
"partial"/"stop_reason"/"reviews_completed" output fields) instead of
silently padding the rest of the sample with retry-exhausted "NO"s.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import yaml
from groq import APIConnectionError, APITimeoutError, Groq, InternalServerError, RateLimitError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import LLMJudge  # noqa: E402
from preshow import tmdb  # noqa: E402
from preshow.env import read_env  # noqa: E402
from preshow.schemas import SpoilerLabel  # noqa: E402
from stats import wilson_interval  # noqa: E402

ZIP_PATH = ROOT / "evals" / "dataset" / "external" / "imbd_spoiler_dataset.zip"
TITLES_PATH = ROOT / "evals" / "dataset" / "titles.yaml"
LABELS_DIR = ROOT / "evals" / "dataset" / "spoilers"

DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_SAMPLE_PER_TITLE = 20  # up to this many reviews per title, split pos/neg
DEFAULT_TRUNCATE_CHARS = 350
DEFAULT_INTERVAL_S = 2.4  # ~25 calls/min, under the 30 RPM / ~6k TPM caps
MAX_RETRIES = 4
SEED = 20260728  # fixed: reproducible sample across re-runs (same reviews
# get picked regardless of model/truncation config, so results stay
# comparable across confound-isolation runs)


def load_labels_by_title() -> dict[str, list[SpoilerLabel]]:
    out: dict[str, list[SpoilerLabel]] = {}
    for p in sorted(LABELS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        out[p.stem] = [SpoilerLabel(**l) for l in (raw.get("labels") or [])]
    return out


def resolve_imdb_ids() -> dict[str, str]:
    films = yaml.safe_load(TITLES_PATH.read_text(encoding="utf-8"))["films"]
    tt_by_title_id: dict[str, str] = {}
    for f in films:
        movie = tmdb.get_movie(f["tmdb_id"])
        imdb_id = movie.get("imdb_id") if movie else None
        if imdb_id:
            tt_by_title_id[f["title_id"]] = imdb_id
    return tt_by_title_id


def iter_reviews():
    z = zipfile.ZipFile(ZIP_PATH)
    with z.open("IMDB_reviews.json") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_sample(labels_by_title: dict, tt_by_title_id: dict, sample_per_title: int) -> list[dict]:
    """Streams the 950MB reviews file ONCE, bucketing by (title, is_spoiler),
    then reservoir-samples down to sample_per_title/2 per bucket -- avoids
    holding all 7,657 matching reviews in memory just to sample from them.
    SEED is fixed regardless of model/truncation, so at the SAME
    sample_per_title this is the exact same 180 reviews as the original
    baseline -- keep --sample-per-title at the default when isolating the
    model or truncation confound, so only one thing changes at a time.
    (Changing sample_per_title itself changes the reservoir draw sequence,
    not just its size -- it does not yield a superset of a smaller run.)"""
    title_id_by_tt = {tt: tid for tid, tt in tt_by_title_id.items()}
    rng = random.Random(SEED)
    per_class = max(1, sample_per_title // 2)
    buckets: dict[tuple[str, bool], list[dict]] = {}
    seen_counts: dict[tuple[str, bool], int] = {}

    for review in iter_reviews():
        title_id = title_id_by_tt.get(review.get("movie_id"))
        if title_id is None or not labels_by_title.get(title_id):
            continue
        truth = bool(review.get("is_spoiler"))
        key = (title_id, truth)
        seen_counts[key] = seen_counts.get(key, 0) + 1
        bucket = buckets.setdefault(key, [])
        if len(bucket) < per_class:
            bucket.append(review)
        else:
            j = rng.randint(0, seen_counts[key] - 1)
            if j < per_class:
                bucket[j] = review

    sample = [r for bucket in buckets.values() for r in bucket]
    rng.shuffle(sample)
    return sample


CALL_TIMEOUT_S = 30.0  # without this, a stalled connection hangs the
# whole run forever with no exception and no output -- observed live
# (a run silently stopped printing progress partway through, no traceback)


class EvaluationAborted(Exception):
    """Base for conditions where continuing the loop in main() would
    silently corrupt the rest of the run rather than genuinely fail --
    see the two subclasses below. Caught in one place in main() so a run
    always stops cleanly and writes whatever it completed, instead of
    padding the tail with retry-exhausted "NO"s that look like real
    judgments but aren't."""


class DailyQuotaExhausted(EvaluationAborted):
    """Raised instead of retrying when Groq reports a PER-DAY (not
    per-minute) rate limit. Observed live: a per-minute limit clears in
    seconds and the normal backoff-and-retry loop below handles it fine,
    but a per-day limit reported "try again in 1-4 minutes" -- far longer
    than this loop's backoff (maxes at ~19s over MAX_RETRIES=4). Retrying
    anyway just burns MAX_RETRIES failures per remaining label, silently
    defaulting every one of them to "NO" -- corrupting the tail of the
    run with fake non-matches that reflect an exhausted quota, not the
    model's actual judgment, while looking like a normal completed run.
    Better to stop the whole evaluation cleanly and report how far it got
    (see main()) than let that happen quietly."""


class ServiceUnavailable(EvaluationAborted):
    """Raised when Groq's API itself is down/overloaded (503 "over
    capacity") for several consecutive calls in a row, not just one blip.
    A single 503 is worth a normal backoff-and-retry (transient); several
    in a row across DIFFERENT calls means the model is genuinely down for
    everyone right now (see https://groqstatus.com) and every remaining
    call in this run would fail the same way -- same silent-corruption
    risk as DailyQuotaExhausted, so it gets the same clean-stop
    treatment instead of grinding through the rest of the sample as
    all-"NO"."""


CONSECUTIVE_EXHAUSTION_LIMIT = 3  # this many DIFFERENT calls in a row each
# burning all MAX_RETRIES attempts on InternalServerError means a real
# outage, not a blip -- stop rather than silently default the rest of
# the run to "NO"


def make_groq_client_fn(model: str, interval_s: float):
    client = Groq(api_key=read_env("GROQ_API_KEY"), timeout=CALL_TIMEOUT_S)
    last_call = [0.0]
    consecutive_exhaustions = [0]

    def client_fn(prompt: str) -> str:
        wait = interval_s - (time.monotonic() - last_call[0])
        if wait > 0:
            time.sleep(wait)
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=5,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                last_call[0] = time.monotonic()
                consecutive_exhaustions[0] = 0
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                if "per day" in str(e):
                    raise DailyQuotaExhausted(str(e)) from e
                last_call[0] = time.monotonic()
                backoff = interval_s * (2**attempt)
                print(
                    f"  [RateLimitError] attempt {attempt + 1}/{MAX_RETRIES}, "
                    f"backing off {backoff:.1f}s: {e}",
                    file=sys.stderr,
                )
                time.sleep(backoff)
            except (APITimeoutError, APIConnectionError, InternalServerError) as e:
                last_call[0] = time.monotonic()
                backoff = interval_s * (2**attempt)
                print(
                    f"  [{type(e).__name__}] attempt {attempt + 1}/{MAX_RETRIES}, "
                    f"backing off {backoff:.1f}s: {e}",
                    file=sys.stderr,
                )
                time.sleep(backoff)
        consecutive_exhaustions[0] += 1
        print(f"  exhausted {MAX_RETRIES} retries, counting as NO", file=sys.stderr)
        if consecutive_exhaustions[0] >= CONSECUTIVE_EXHAUSTION_LIMIT:
            raise ServiceUnavailable(
                f"{CONSECUTIVE_EXHAUSTION_LIMIT} consecutive calls each exhausted all "
                f"{MAX_RETRIES} retries -- {model} looks genuinely down, not just slow "
                "for one call. Check https://groqstatus.com."
            )
        return "NO"  # exhausted retries -- treat as a non-match, not a crash

    return client_fn


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Groq model id (default: {DEFAULT_MODEL})")
    p.add_argument(
        "--truncate-chars",
        type=int,
        default=DEFAULT_TRUNCATE_CHARS,
        help=f"chars/review sent to the model (default: {DEFAULT_TRUNCATE_CHARS})",
    )
    p.add_argument(
        "--sample-per-title",
        type=int,
        default=DEFAULT_SAMPLE_PER_TITLE,
        help=f"max reviews/title, split pos/neg (default: {DEFAULT_SAMPLE_PER_TITLE}) -- keep at "
        "the default when isolating the model or truncation confound (see docstring)",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help=f"min seconds between LLM calls, tune to the model's TPM cap (default: {DEFAULT_INTERVAL_S})",
    )
    return p.parse_args(argv)


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"missing {ZIP_PATH}", file=sys.stderr)
        return 1

    args = parse_args()
    is_default_config = (
        args.model == DEFAULT_MODEL
        and args.truncate_chars == DEFAULT_TRUNCATE_CHARS
        and args.sample_per_title == DEFAULT_SAMPLE_PER_TITLE
    )

    labels_by_title = load_labels_by_title()
    tt_by_title_id = resolve_imdb_ids()

    print("building stratified sample (streaming the 950MB reviews file once)...")
    sample = build_sample(labels_by_title, tt_by_title_id, args.sample_per_title)
    n_pos = sum(1 for r in sample if r.get("is_spoiler"))
    print(f"sampled {len(sample)} reviews ({n_pos} spoiler-tagged, {len(sample) - n_pos} not)")

    title_id_by_tt = {tt: tid for tid, tt in tt_by_title_id.items()}
    judge = LLMJudge(client_fn=make_groq_client_fn(args.model, args.interval))

    tp = fp = tn = fn = 0
    n_calls = 0
    n_by_title: dict[str, int] = {}
    t0 = time.time()
    reviews_completed = 0
    stopped_early = False
    stop_reason = None

    for i, review in enumerate(sample):
        title_id = title_id_by_tt[review["movie_id"]]
        labels = labels_by_title[title_id]
        text = (review.get("review_text") or "")[: args.truncate_chars]
        truth = bool(review.get("is_spoiler"))

        try:
            pred = False
            for label in labels:
                n_calls += 1
                if judge.entails(text, label):
                    pred = True
                    # Don't break: every label call still gets cached/counted,
                    # but stopping early would under-count n_calls vs. reality
                    # if this run is ever resumed from the cache. Keep it simple.
        except DailyQuotaExhausted as e:
            print(
                f"\nDaily token quota exhausted for {args.model} after {i}/{len(sample)} "
                f"reviews ({n_calls} calls) -- stopping here instead of letting the rest "
                f"of the run silently default to 'NO' (see DailyQuotaExhausted's "
                f"docstring). Groq's message: {e}\n"
                "Re-run tomorrow once the quota resets, or with a smaller "
                "--sample-per-title so the full sample fits this model's daily budget.",
                file=sys.stderr,
            )
            stopped_early = True
            stop_reason = "daily_quota_exhausted"
            break
        except ServiceUnavailable as e:
            print(
                f"\n{e}\nStopping after {i}/{len(sample)} reviews ({n_calls} calls) "
                "instead of letting the rest of the run silently default to 'NO'. "
                "Re-run once Groq's status page shows it's back.",
                file=sys.stderr,
            )
            stopped_early = True
            stop_reason = "service_unavailable"
            break

        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
        n_by_title[title_id] = n_by_title.get(title_id, 0) + 1
        reviews_completed += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(sample)} reviews, {n_calls} LLM calls, {elapsed:.0f}s elapsed", flush=True)

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision_ci = wilson_interval(tp, tp + fp)
    recall_ci = wilson_interval(tp, tp + fn)

    result = {
        "judge": f"llm ({args.model}, Groq free tier)",
        "source": "IMDB Spoiler Dataset (Misra), stratified sample restricted to "
        "this project's 20 titles.yaml movies",
        "n": n,
        "n_llm_calls": n_calls,
        "n_titles_matched": len(n_by_title),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 3),
        "precision_ci_95": [round(v, 3) for v in precision_ci] if precision_ci else None,
        "recall": round(recall, 3),
        "recall_ci_95": [round(v, 3) for v in recall_ci] if recall_ci else None,
        "reviews_per_title": n_by_title,
        "config": {
            "model": args.model,
            "truncate_chars": args.truncate_chars,
            "sample_per_title": args.sample_per_title,
            "interval_s": args.interval,
        },
        "partial": stopped_early,
        "stop_reason": stop_reason,
        "reviews_completed": reviews_completed,
        "reviews_sampled": len(sample),
        "note": (
            f"Sampled (seed={SEED}, up to {args.sample_per_title} reviews/title, "
            "balanced spoiler/non-spoiler) rather than the full 7,657-review "
            "external set used for SubstringJudge in D12 -- Groq's free tier "
            "caps this key at 1,000 requests/day, and each review costs "
            "len(that movie's labels) calls. Review text truncated to "
            f"{args.truncate_chars} chars per call. A reveal appearing only "
            "later in a long review could still be missed by truncation "
            "specifically, separate from D12's labeling-coverage caveat (a "
            "real spoiler review can reveal a plot point this project didn't "
            "document as a SpoilerLabel, counted here as a miss that isn't "
            "the judge's fault either). "
            "*_ci_95 fields are 95% Wilson score intervals (Wilson, 1927) on "
            "the point estimate -- with n this small, treat the point "
            "estimate as illustrative and the interval as the real answer."
            + (
                f" PARTIAL RUN ({stop_reason}): stopped after "
                f"{reviews_completed}/{len(sample)} reviews -- all numbers above "
                "reflect only the reviews actually completed, not the full sample. "
                + (
                    "Re-run tomorrow (daily quota resets) or with a smaller "
                    "--sample-per-title for a complete result."
                    if stop_reason == "daily_quota_exhausted"
                    else "Re-run once Groq's status page (groqstatus.com) shows "
                    f"{args.model} is healthy again."
                )
                if stopped_early
                else ""
            )
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if is_default_config:
        out_name = "llm_calibration_external.json"  # the original D13 baseline
    else:
        model_slug = args.model.replace(".", "").replace("-", "_")
        partial_suffix = f"_PARTIAL_{stop_reason}" if stopped_early else ""
        out_name = f"llm_calibration_external_{model_slug}_t{args.truncate_chars}{partial_suffix}.json"
    out_path = ROOT / "evals" / "results" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
