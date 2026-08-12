"""Calibrates `NLIJudge` (evals/judge.py) against the same real IMDb
spoiler reviews used in calibrate_substring_external.py (D12) and
calibrate_llm_external.py (D13) -- the local, no-API alternative
discussed once both of LLMJudge's confounds (truncation, model size)
turned out resolved but still not good enough to trust a leakage_rate
(D13's follow-up: recall ceiling ~=0.35-0.4 with llama-3.1-8b-instant).

Why this is worth trying as a genuinely different approach, not just
another LLM config: NLIJudge has no per-call cost and no rate limit
(runs fully offline after a one-time ~140MB model download), so unlike
every LLMJudge run in this project so far, it can be calibrated against
the FULL 7,657-review external set in one go -- same statistical power
as SubstringJudge's D12 result, not a small stratified sample forced by
a free-tier budget.

It also produces a CONTINUOUS entailment score rather than a binary
yes/no, which means the decision threshold can be tuned against this
same calibration data instead of picked blind -- this script sweeps a
range of thresholds and reports precision/recall/CI at each, per D13's
own principle ("in spoiler safety, RECALL rules -- optimize toward
recall, report the precision you pay for it").

Real caveat, stated plainly (see NLIJudge's docstring for the full
version): formal NLI entailment is a stricter bar than "could a viewer
infer the spoiler from this text", which is what actually matters here.
A review that hints at a twist without stating it outright may score low
even though a human would call it a leak. If the sweep below shows poor
recall even at low thresholds, that mismatch -- not a lack of tuning --
is the likely reason, and is itself a useful, honest finding.

Requires `pip install sentence-transformers` (pulls in torch +
transformers) and evals/dataset/external/imbd_spoiler_dataset.zip (see
calibrate_substring_external.py's docstring for how to get it).

Speed note, learned the hard way: CrossEncoder inference on this
project's dev machine ran far slower than a synthetic benchmark
suggested and got WORSE, not better, after truncating text and batching
calls per review (0.2 -> 0.3 reviews/s) -- while CPU time reported by
the OS stayed near zero for a run that took over an hour of wall clock.
That mismatch (high wall time, ~no CPU time) points at something
external stalling each inference call, not a compute-bound cost -- the
leading suspect is real-time antivirus scanning the HuggingFace
cache/torch DLLs on every model forward pass, a documented Windows +
PyTorch problem. If you hit the same thing: add an antivirus exclusion
for the HF cache (`~/.cache/huggingface`) and your Python
site-packages directory, then re-benchmark with --sample-per-title
before trusting any full-set time estimate on your machine.

--truncate-chars (default 1500, matching the LLMJudge truncation
experiment for rough comparability) still helps regardless of the above,
since inference cost also scales with sequence length. If the full
7,657-review set still isn't practical on your hardware,
--sample-per-title does the SAME stratified reservoir sampling as
calibrate_llm_external.py (same SEED, same per-title/per-class balancing)
-- unlike a naive head-of-file --limit, which is badly biased here: the
external dataset's reviews are grouped by movie and (empirically) sorted
spoiler-tagged-first within each movie's block, so the first N matching
reviews in file order skew almost entirely toward one title and one
class.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from external_dataset import ZIP_PATH, iter_reviews, load_labels_by_title, resolve_imdb_ids  # noqa: E402
from judge import NLIJudge  # noqa: E402
from stats import wilson_interval  # noqa: E402

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"
DEFAULT_TRUNCATE_CHARS = 1500
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEED = 20260728  # same seed as calibrate_llm_external.py's build_sample --
# not directly comparable review-for-review (different reservoir logic
# instantiation) but keeps the "fixed, reproducible sample" convention


def build_sample(labels_by_title: dict, tt_by_title_id: dict, sample_per_title: int) -> list[dict]:
    """Stratified reservoir sample, same logic and SEED convention as
    calibrate_llm_external.py's build_sample -- up to sample_per_title/2
    reviews per (title, is_spoiler) bucket, streamed once. Exists here
    because a naive head-of-file --limit is badly biased on this dataset:
    reviews are grouped by movie and, empirically, sorted
    spoiler-tagged-first within each movie's block."""
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


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace cross-encoder id (default: {DEFAULT_MODEL})")
    p.add_argument(
        "--sample-per-title",
        type=int,
        default=None,
        help="if set, use a stratified reservoir sample (up to this many reviews/title, "
        "balanced spoiler/non-spoiler -- same method as calibrate_llm_external.py) instead "
        "of the full ~7,657-review set. Use this if the full set isn't practical on your "
        "hardware (default: no sampling, process everything)",
    )
    p.add_argument(
        "--truncate-chars",
        type=int,
        default=DEFAULT_TRUNCATE_CHARS,
        help=f"chars/review sent to the model -- inference cost scales with sequence length, "
        f"so this controls both accuracy and speed (default: {DEFAULT_TRUNCATE_CHARS})",
    )
    return p.parse_args(argv)


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"missing {ZIP_PATH} -- see calibrate_substring_external.py's docstring", file=sys.stderr)
        return 1

    args = parse_args()
    labels_by_title = load_labels_by_title()
    tt_by_title_id = resolve_imdb_ids()
    title_id_by_tt = {tt: tid for tid, tt in tt_by_title_id.items()}

    if args.sample_per_title:
        print(f"building stratified sample (seed={SEED}, up to {args.sample_per_title}/title)...")
        reviews = build_sample(labels_by_title, tt_by_title_id, args.sample_per_title)
        print(f"sampled {len(reviews)} reviews")
    else:
        print("no --sample-per-title given -- processing the full matching set (this can take a while)")
        reviews = [
            r
            for r in iter_reviews()
            if title_id_by_tt.get(r.get("movie_id")) and labels_by_title.get(title_id_by_tt[r["movie_id"]])
        ]
        print(f"{len(reviews)} matching reviews")

    print(f"loading {args.model} (one-time download if not already cached)...")
    judge = NLIJudge(model_name=args.model)

    scored: list[tuple[float, bool]] = []
    n_by_title: dict[str, int] = {}
    t0 = time.time()

    for review in reviews:
        title_id = title_id_by_tt[review["movie_id"]]
        labels = labels_by_title[title_id]

        text = (review.get("review_text") or "")[: args.truncate_chars]
        truth = bool(review.get("is_spoiler"))
        max_prob = judge.max_entailment_prob(text, labels)
        scored.append((max_prob, truth))
        n_by_title[title_id] = n_by_title.get(title_id, 0) + 1

        if len(scored) % 200 == 0:
            elapsed = time.time() - t0
            rate = len(scored) / elapsed if elapsed else 0.0
            remaining = (len(reviews) - len(scored)) / rate if rate else float("inf")
            print(
                f"  {len(scored)}/{len(reviews)} reviews scored, {elapsed:.0f}s elapsed "
                f"({rate:.2f}/s, ~{remaining / 60:.0f}min left)",
                flush=True,
            )

    sweep = []
    for th in THRESHOLDS:
        tp = fp = tn = fn = 0
        for prob, truth in scored:
            pred = prob >= th
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and truth:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision_ci = wilson_interval(tp, tp + fp)
        recall_ci = wilson_interval(tp, tp + fn)
        sweep.append(
            {
                "threshold": th,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(precision, 3),
                "precision_ci_95": [round(v, 3) for v in precision_ci] if precision_ci else None,
                "recall": round(recall, 3),
                "recall_ci_95": [round(v, 3) for v in recall_ci] if recall_ci else None,
            }
        )

    result = {
        "judge": f"nli ({args.model})",
        "source": "IMDB Spoiler Dataset (Misra), reviews restricted to this "
        "project's 20 titles.yaml movies",
        "n": len(scored),
        "n_titles_matched": len(n_by_title),
        "reviews_per_title": n_by_title,
        "sample_per_title": args.sample_per_title,
        "truncate_chars": args.truncate_chars,
        "threshold_sweep": sweep,
        "note": (
            "Positive = review tagged is_spoiler=true by a real IMDb user; judge "
            "score is max(entailment_prob(text, label)) over ALL of this movie's "
            "documented SpoilerLabels (canonical + paraphrases), same 'entails ANY' "
            "method as D12/D13. Same labeling-coverage caveat as those: a review can "
            "be a real spoiler for a plot point not in our label set, counted here as "
            "a false negative regardless of threshold. "
            + (
                f"Stratified sample (seed={SEED}, up to {args.sample_per_title}/title), "
                "not the full external set -- see --sample-per-title."
                if args.sample_per_title
                else "Ran against the full external set, no sampling."
            )
            + " *_ci_95 fields are 95% Wilson "
            "score intervals (Wilson, 1927; evals/stats.py). See NLIJudge's docstring "
            "for the formal-entailment-vs-'could a viewer infer this' task mismatch "
            "this approach carries."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = ROOT / "evals" / "results" / "nli_calibration_external.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
