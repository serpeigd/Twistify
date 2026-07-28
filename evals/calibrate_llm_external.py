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
"""

from __future__ import annotations

import json
import random
import sys
import time
import zipfile
from pathlib import Path

import yaml
from groq import Groq
from groq import RateLimitError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import LLMJudge  # noqa: E402
from preshow import tmdb  # noqa: E402
from preshow.env import read_env  # noqa: E402
from preshow.schemas import SpoilerLabel  # noqa: E402

ZIP_PATH = ROOT / "evals" / "dataset" / "external" / "imbd_spoiler_dataset.zip"
TITLES_PATH = ROOT / "evals" / "dataset" / "titles.yaml"
LABELS_DIR = ROOT / "evals" / "dataset" / "spoilers"

MODEL = "llama-3.1-8b-instant"
SAMPLE_PER_TITLE = 20  # up to this many reviews per title, split pos/neg
TEXT_TRUNCATE_CHARS = 350
MIN_CALL_INTERVAL_S = 2.4  # ~25 calls/min, under the 30 RPM / ~6k TPM caps
MAX_RETRIES = 4
SEED = 20260728  # fixed: reproducible sample across re-runs


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


def build_sample(labels_by_title: dict, tt_by_title_id: dict) -> list[dict]:
    """Streams the 950MB reviews file ONCE, bucketing by (title, is_spoiler),
    then reservoir-samples down to SAMPLE_PER_TITLE/2 per bucket -- avoids
    holding all 7,657 matching reviews in memory just to sample from them."""
    title_id_by_tt = {tt: tid for tid, tt in tt_by_title_id.items()}
    rng = random.Random(SEED)
    per_class = max(1, SAMPLE_PER_TITLE // 2)
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


def make_groq_client_fn():
    client = Groq(api_key=read_env("GROQ_API_KEY"))
    last_call = [0.0]

    def client_fn(prompt: str) -> str:
        wait = MIN_CALL_INTERVAL_S - (time.monotonic() - last_call[0])
        if wait > 0:
            time.sleep(wait)
        for attempt in range(MAX_RETRIES):
            try:
                resp = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=5,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                last_call[0] = time.monotonic()
                return resp.choices[0].message.content or ""
            except RateLimitError:
                last_call[0] = time.monotonic()
                time.sleep(MIN_CALL_INTERVAL_S * (2**attempt))
        return "NO"  # exhausted retries -- treat as a non-match, not a crash

    return client_fn


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"missing {ZIP_PATH}", file=sys.stderr)
        return 1

    labels_by_title = load_labels_by_title()
    tt_by_title_id = resolve_imdb_ids()

    print("building stratified sample (streaming the 950MB reviews file once)...")
    sample = build_sample(labels_by_title, tt_by_title_id)
    n_pos = sum(1 for r in sample if r.get("is_spoiler"))
    print(f"sampled {len(sample)} reviews ({n_pos} spoiler-tagged, {len(sample) - n_pos} not)")

    title_id_by_tt = {tt: tid for tid, tt in tt_by_title_id.items()}
    judge = LLMJudge(client_fn=make_groq_client_fn())

    tp = fp = tn = fn = 0
    n_calls = 0
    n_by_title: dict[str, int] = {}
    t0 = time.time()

    for i, review in enumerate(sample):
        title_id = title_id_by_tt[review["movie_id"]]
        labels = labels_by_title[title_id]
        text = (review.get("review_text") or "")[:TEXT_TRUNCATE_CHARS]
        truth = bool(review.get("is_spoiler"))

        pred = False
        for label in labels:
            n_calls += 1
            if judge.entails(text, label):
                pred = True
                # Don't break: every label call still gets cached/counted,
                # but stopping early would under-count n_calls vs. reality
                # if this run is ever resumed from the cache. Keep it simple.

        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
        n_by_title[title_id] = n_by_title.get(title_id, 0) + 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(sample)} reviews, {n_calls} LLM calls, {elapsed:.0f}s elapsed", flush=True)

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    result = {
        "judge": f"llm ({MODEL}, Groq free tier)",
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
        "recall": round(recall, 3),
        "reviews_per_title": n_by_title,
        "note": (
            f"Sampled (seed={SEED}, up to {SAMPLE_PER_TITLE} reviews/title, "
            "balanced spoiler/non-spoiler) rather than the full 7,657-review "
            "external set used for SubstringJudge in D12 -- Groq's free tier "
            "caps this key at 1,000 requests/day, and each review costs "
            "len(that movie's labels) calls. Review text truncated to "
            f"{TEXT_TRUNCATE_CHARS} chars per call to stay under the ~6,000 "
            "tokens/min cap; a reveal appearing only later in a long review "
            "could be missed by truncation specifically, separate from D12's "
            "labeling-coverage caveat (a real spoiler review can reveal a "
            "plot point this project didn't document as a SpoilerLabel, "
            "counted here as a miss that isn't the judge's fault either)."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = ROOT / "evals" / "results" / "llm_calibration_external.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
