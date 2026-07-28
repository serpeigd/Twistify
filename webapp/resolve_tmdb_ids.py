"""One-time helper: resolve every title in evals/dataset/titles.yaml to a
TMDB id and write it back into the file, so the demo can show a real
poster/synopsis for all 20 measurement-set titles, not just the 7
hand-researched ones (see D10 in docs/DESIGN.md).

Edits the YAML as text (insert one `tmdb_id:` line per entry) instead of
parsing + re-dumping it, so the file's hand-written header comment and
formatting survive untouched. Safe to re-run: an entry that already has
a `tmdb_id` line is left alone.

Also pre-warms content/_tmdb_cache/ with each resolved movie's full
record (poster, overview, genres), same idea as prewarm_translations.py.

    python webapp/resolve_tmdb_ids.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from preshow import tmdb  # noqa: E402

TITLES_FILE = ROOT / "evals" / "dataset" / "titles.yaml"

_ENTRY_START = re.compile(r"^(\s*)- title_id:\s*(\S+)\s*$")
_STRATUM_LINE = re.compile(r"^\s*stratum:")


def main() -> None:
    # The parsed structure (not the raw text) is the source of truth for
    # "does this entry already have a tmdb_id" -- checking raw lines with
    # a forward scan can't see a tmdb_id line that comes *after* the
    # stratum line it would insert at, which silently re-inserted a
    # duplicate on every re-run in an earlier version of this script.
    raw = yaml.safe_load(TITLES_FILE.read_text(encoding="utf-8"))
    by_id = {f["title_id"]: f for f in raw["films"]}

    lines = TITLES_FILE.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    current_id: str | None = None

    for line in lines:
        m = _ENTRY_START.match(line)
        if m:
            current_id = m.group(2)

        out.append(line)

        if current_id and _STRATUM_LINE.match(line) and not by_id[current_id].get("tmdb_id"):
            case = by_id[current_id]
            indent = re.match(r"^(\s*)", line).group(1)
            match = tmdb.best_match(case["title"], case["year"])
            if match is None:
                print(f"no match  {current_id}")
                continue
            out.append(f"{indent}tmdb_id: {match['tmdb_id']}\n")
            tmdb.get_movie(match["tmdb_id"])  # pre-warm the cache
            print(f"resolved  {current_id:28s} -> {match['tmdb_id']} ({match['title']}, {match['year']})")
        elif current_id and _STRATUM_LINE.match(line) and by_id[current_id].get("tmdb_id"):
            print(f"skip      {current_id} (already has tmdb_id)")

    TITLES_FILE.write_text("".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
