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
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import SubstringJudge  # noqa: E402
from preshow import tmdb  # noqa: E402
from preshow.schemas import SpoilerLabel  # noqa: E402

ZIP_PATH = ROOT / "evals" / "dataset" / "external" / "imbd_spoiler_dataset.zip"
TITLES_PATH = ROOT / "evals" / "dataset" / "titles.yaml"
LABELS_DIR = ROOT / "evals" / "dataset" / "spoilers"


def load_labels_by_title() -> dict[str, list[SpoilerLabel]]:
    out: dict[str, list[SpoilerLabel]] = {}
    for p in sorted(LABELS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        out[p.stem] = [SpoilerLabel(**l) for l in (raw.get("labels") or [])]
    return out


def resolve_imdb_ids() -> dict[str, str]:
    """title_id -> IMDb "tt..." id, via the tmdb_id already resolved in
    titles.yaml (D10). Skips (and reports) any title TMDB doesn't have an
    imdb_id for."""
    films = yaml.safe_load(TITLES_PATH.read_text(encoding="utf-8"))["films"]
    tt_by_title_id: dict[str, str] = {}
    missing: list[str] = []
    for f in films:
        movie = tmdb.get_movie(f["tmdb_id"])
        imdb_id = movie.get("imdb_id") if movie else None
        if imdb_id:
            tt_by_title_id[f["title_id"]] = imdb_id
        else:
            missing.append(f["title_id"])
    if missing:
        print(f"warning: no imdb_id resolved for {missing}", file=sys.stderr)
    return tt_by_title_id


def iter_reviews():
    """Yields {"movie_id": "tt...", "is_spoiler": bool, "review_text": str}
    for every line in IMDB_reviews.json without loading the whole 950MB
    file into memory."""
    z = zipfile.ZipFile(ZIP_PATH)
    with z.open("IMDB_reviews.json") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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
        "recall": round(recall, 3),
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
            "generator, and ground-truth researcher never saw."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = ROOT / "evals" / "results" / "substring_calibration_external.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
