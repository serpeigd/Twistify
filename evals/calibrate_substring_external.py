"""Calibration of SubstringJudge against real, human-written spoiler
reviews (IMDB Spoiler Dataset, Misra) instead of the project's own LLM
paraphrases (see calibrate_substring.py for that internal version).

Why this fixes the gap the internal calibration admits to: the internal
one tests the judge against paraphrases written by the same kind of
system (an LLM) that wrote the ground truth. This one tests it against
IMDb users' own words -- text the judge, the generator, and the ground
truth researcher never saw or influenced.

Task mismatch, stated plainly (see evals/judge.py's docstring for the
general version of this problem): the dataset labels a whole REVIEW as
spoiler/not-spoiler. It does not say *which* plot point a spoiler review
reveals. This script treats "reveals ANY of this movie's known
SpoilerLabels" as the positive criterion, which introduces one specific,
named noise source:

  A review can be tagged is_spoiler=true while revealing a plot point
  that isn't in our SpoilerLabel set for that movie (we only label the
  headline twists per film, not every plot beat). That review then counts
  as a false negative if the judge correctly finds none of OUR labels in
  it -- a miss that reflects incomplete label coverage, not necessarily a
  judge failure. This inflates the false-negative count (lowers measured
  recall) versus the judge's true performance on the spoilers we actually
  documented. Report this alongside the number, not instead of it.

Requires evals/dataset/external/imbd_spoiler_dataset.zip -- a manual
download from https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset
(free Kaggle account, no API key). Gitignored: ~570k reviews, not ours to
redistribute. Streams the reviews file line-by-line (it's ~950MB
uncompressed) instead of loading it whole.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from external_dataset import ZIP_PATH, iter_reviews, load_labels_by_title, resolve_imdb_ids  # noqa: E402
from judge import SubstringJudge  # noqa: E402
from stats import wilson_interval  # noqa: E402


def main() -> int:
    if not ZIP_PATH.exists():
        print(f"missing {ZIP_PATH} -- see this script's docstring", file=sys.stderr)
        return 1

    labels_by_title = load_labels_by_title()
    tt_by_title_id = resolve_imdb_ids()
    title_id_by_tt = {tt: tid for tid, tt in tt_by_title_id.items()}

    judge = SubstringJudge()
    tp = fp = tn = fn = 0
    n_by_title: dict[str, int] = {}

    for review in iter_reviews():
        title_id = title_id_by_tt.get(review.get("movie_id"))
        if title_id is None:
            continue
        labels = labels_by_title.get(title_id) or []
        if not labels:
            continue

        text = review.get("review_text") or ""
        truth = bool(review.get("is_spoiler"))
        pred = any(judge.entails(text, label) for label in labels)

        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
        n_by_title[title_id] = n_by_title.get(title_id, 0) + 1

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision_ci = wilson_interval(tp, tp + fp)
    recall_ci = wilson_interval(tp, tp + fn)

    result = {
        "judge": judge.name,
        "source": "IMDB Spoiler Dataset (Misra), reviews restricted to this "
        "project's 20 titles.yaml movies",
        "n": n,
        "n_titles_matched": len(n_by_title),
        "n_titles_total": len(tt_by_title_id),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 3),
        "precision_ci_95": [round(v, 3) for v in precision_ci] if precision_ci else None,
        "recall": round(recall, 3),
        "recall_ci_95": [round(v, 3) for v in recall_ci] if recall_ci else None,
        "reviews_per_title": n_by_title,
        "note": (
            "Positive = review tagged is_spoiler=true by a real IMDb user; "
            "judge is credited if it flags ANY of this movie's documented "
            "SpoilerLabels in the review text. Caveat: a spoiler review can "
            "reveal a plot point not in our label set (we only document "
            "headline twists, not every beat) -- that case counts as a "
            "false negative here even though the judge didn't fail on a "
            "spoiler we actually track. This likely understates true "
            "recall on our documented labels specifically. Independent of "
            "that caveat, this is real human-written text the judge, "
            "generator, and ground-truth researcher never saw. "
            "recall_ci_95 is the 95% Wilson score interval (Wilson, 1927); "
            "precision_ci_95 is null because precision is undefined here "
            "(tp+fp=0)."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = ROOT / "evals" / "results" / "substring_calibration_external.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
