"""Shared loaders for the IMDB Spoiler Dataset (Misra) external
calibration, used by every script that touches it:
calibrate_substring_external.py (D12), calibrate_llm_external.py (D13),
calibrate_nli_external.py (D15), train_spoiler_classifier.py (D15).

Extracted after these four functions had drifted into four separate
copy-pasted copies (byte-identical except one script's
`resolve_imdb_ids` had grown a diagnostic the other three lacked) --
simplification pass, no behavior change intended other than that
diagnostic now being available everywhere.

Requires evals/dataset/external/imbd_spoiler_dataset.zip -- a manual
download from https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset
(free Kaggle account, no API key). Gitignored: ~570k reviews, not ours to
redistribute. See calibrate_substring_external.py's own docstring for
the full rationale (why an external benchmark, why this one).
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import yaml

from preshow import tmdb
from preshow.schemas import SpoilerLabel

ROOT = Path(__file__).resolve().parent.parent
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
