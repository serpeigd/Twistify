"""Trains and evaluates a classifier directly on this project's own
external labels (IMDB Spoiler Dataset, restricted to the 9 of 20
titles.yaml movies with review coverage -- 7,657 reviews, 2,197
spoiler-tagged / 5,460 not), instead of relying on an off-the-shelf
model. SubstringJudge (D12: recall=0.0), LLMJudge (D13 + follow-ups:
recall ceiling ~=0.35-0.4), and NLIJudge (off-the-shelf cross-encoder:
recall ~=0.0, formal-entailment task mismatch) all fell short or
underperformed -- see docs/DESIGN.md for the full account of each.

Framing change from the other three, deliberate: SubstringJudge/
LLMJudge/NLIJudge all score "does this text entail ANY of THIS movie's
documented SpoilerLabels" -- a label-matching proxy for the real
question. This script trains directly on the dataset's own review-level
is_spoiler label instead: "does this text sound like it reveals a plot
point", full stop, no per-label matching required. That sidesteps D12's
labeling-coverage caveat entirely (a review can leak a plot point never
documented as a SpoilerLabel -- irrelevant here, there's no label list to
miss) and is arguably closer to what a real leak judge needs to decide:
is this text safe to show pre-viewing, not does it match one of a
handful of hand-picked spoilers for this specific film.

Model: TF-IDF + Logistic Regression (class_weight="balanced" for the
~2.5:1 imbalance). Deliberately NOT a transformer -- NLIJudge's own
calibration run just showed CPU transformer inference is impractical on
this project's dev hardware (over an hour for a few hundred reviews even
after optimization). A linear bag-of-words model is a well-established
strong baseline for exactly this kind of review-level spoiler
classification (see judge.py's own citation of Wan et al. 2019's
Goodreads spoiler task). Trains on the full 7,657-review set in seconds.

Evaluation: GROUPED k-fold BY TITLE (one fold per title with review
coverage, i.e. leave-one-title-out), not a random split. A random split
would let the model memorize movie-specific vocabulary (character names,
phrases recurring across that movie's own reviews) and inflate its score
without proving it generalizes to a movie it has never seen labels for
-- which is the only case that matters, since production use is always
on movies with no training labels. Every review's reported prediction
comes from the fold where its own title was held out of training
entirely, so the aggregated numbers below are a genuine held-out
estimate, not a fit-and-report-on-the-same-data number.

Also trains one more model on ALL the data (not held out) and persists
it to evals/models/spoiler_classifier.joblib (gitignored, regenerable --
same reasoning as evals/results/) for actual use as
`judge.TrainedClassifierJudge`, wired into run_eval.py via
`--judge trained-classifier`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from external_dataset import ZIP_PATH, iter_reviews, load_labels_by_title, resolve_imdb_ids  # noqa: E402
from stats import wilson_interval  # noqa: E402

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_dataset() -> tuple[list[str], list[bool], list[str]]:
    labels_by_title = load_labels_by_title()
    tt_by_title_id = resolve_imdb_ids()
    title_id_by_tt = {tt: tid for tid, tt in tt_by_title_id.items()}

    texts, truths, groups = [], [], []
    for review in iter_reviews():
        title_id = title_id_by_tt.get(review.get("movie_id"))
        if title_id is None or not labels_by_title.get(title_id):
            continue
        texts.append(review.get("review_text") or "")
        truths.append(bool(review.get("is_spoiler")))
        groups.append(title_id)
    return texts, truths, groups


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"missing {ZIP_PATH} -- see calibrate_substring_external.py's docstring", file=sys.stderr)
        return 1

    print("loading reviews (streaming the 950MB zip once)...")
    texts, truths, groups = load_dataset()
    n_pos = sum(truths)
    titles = sorted(set(groups))
    print(f"{len(texts)} reviews across {len(titles)} titles ({n_pos} spoiler-tagged, {len(texts) - n_pos} not)")

    gkf = GroupKFold(n_splits=len(titles))
    oof_prob = [None] * len(texts)
    per_title_n = {t: 0 for t in titles}

    import numpy as np

    texts_arr = np.array(texts, dtype=object)
    truths_arr = np.array(truths, dtype=bool)
    groups_arr = np.array(groups, dtype=object)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(texts_arr, truths_arr, groups_arr)):
        held_out_title = groups_arr[test_idx[0]]
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english", min_df=2)
        X_train = vectorizer.fit_transform(texts_arr[train_idx])
        X_test = vectorizer.transform(texts_arr[test_idx])

        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X_train, truths_arr[train_idx])
        probs = clf.predict_proba(X_test)[:, list(clf.classes_).index(True)]

        for i, p in zip(test_idx, probs):
            oof_prob[i] = float(p)
        per_title_n[held_out_title] = len(test_idx)
        print(f"  fold {fold + 1}/{len(titles)}: held out {held_out_title} ({len(test_idx)} reviews)")

    scored = list(zip(oof_prob, truths))
    assert all(p is not None for p, _ in scored)

    # Per-title breakdown at a fixed threshold -- Fight Club alone is 32%
    # of this dataset (2,480/7,657), so an aggregate number could hide
    # "it only works on Fight Club" behind a title-weighted average.
    # Checking this per title, not just in aggregate, is the whole point
    # of doing grouped CV instead of a random split.
    PER_TITLE_THRESHOLD = 0.5
    per_title_stats: dict[str, dict] = {}
    for title in titles:
        tp = fp = tn = fn = 0
        for prob, truth, g in zip(oof_prob, truths, groups):
            if g != title:
                continue
            pred = prob >= PER_TITLE_THRESHOLD
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and truth:
                fn += 1
            else:
                tn += 1
        recall = tp / (tp + fn) if (tp + fn) else None
        precision = tp / (tp + fp) if (tp + fp) else None
        per_title_stats[title] = {
            "n": tp + fp + tn + fn,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": round(recall, 3) if recall is not None else None,
            "precision": round(precision, 3) if precision is not None else None,
        }

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

    # A final model trained on ALL data (not held out -- the grouped CV
    # above already gave the honest generalization estimate; this one is
    # for actual use). Also doubles as the n-gram inspection below:
    # sanity-check that it learned something plausible (spoiler-ish
    # language), not an artifact.
    full_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english", min_df=2)
    X_full = full_vectorizer.fit_transform(texts_arr)
    full_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    full_clf.fit(X_full, truths_arr)
    feature_names = full_vectorizer.get_feature_names_out()
    coefs = full_clf.coef_[0]
    top_pos = sorted(zip(coefs, feature_names), reverse=True)[:20]
    top_neg = sorted(zip(coefs, feature_names))[:20]

    import joblib

    model_path = ROOT / "evals" / "models" / "spoiler_classifier.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": full_vectorizer, "model": full_clf}, model_path)
    print(f"wrote {model_path} -- use judge.TrainedClassifierJudge.from_artifact() to load it")

    result = {
        "judge": "trained-classifier (TF-IDF + LogisticRegression)",
        "source": "IMDB Spoiler Dataset (Misra), reviews restricted to this "
        "project's 20 titles.yaml movies",
        "n": len(scored),
        "n_titles": len(titles),
        "reviews_per_title": per_title_n,
        "eval_method": "grouped k-fold by title (leave-one-title-out), out-of-fold predictions only",
        "per_title_at_threshold_0.5": per_title_stats,
        "threshold_sweep": sweep,
        "top_spoiler_ngrams": [f"{w} ({c:.2f})" for c, w in top_pos],
        "top_non_spoiler_ngrams": [f"{w} ({c:.2f})" for c, w in top_neg],
        "note": (
            "Positive = review tagged is_spoiler=true by a real IMDb user. Unlike "
            "SubstringJudge/LLMJudge/NLIJudge, this predicts is_spoiler directly from "
            "text (no per-SpoilerLabel matching), trained on this project's own "
            "2,197-positive/5,460-negative external labels. Every prediction is "
            "out-of-fold (the review's own title was excluded from that fold's "
            "training set), so this is a genuine held-out estimate of generalization "
            "to unseen movies, not a fit-then-score-the-same-data number. "
            "*_ci_95 fields are 95% Wilson score intervals (Wilson, 1927; "
            "evals/stats.py)."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = ROOT / "evals" / "results" / "trained_classifier_external.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
