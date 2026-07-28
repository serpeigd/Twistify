"""One-time helper: translate every researched title, plus the browse-tier
TMDB overview of every NOT-yet-researched title, to Spanish and cache the
result to content/_translations/ -- so nobody viewing the live demo has to
wait on the free MyMemory API (see src/preshow/translate.py). It's slow
and rate-limited on an uncached title, but only ever runs once per title.

This directory is committed to git, deliberately not gitignored: it's a
deterministic, regenerable build artifact (like a compiled asset), not a
secret or local scratch state. A free host's filesystem is usually
ephemeral (wiped on redeploy/restart -- see D11) and MyMemory itself is
unreliable from a shared hosting IP, so translating live, on the deployed
server, on a visitor's first request is exactly the slow/silently-English
failure mode this script exists to avoid. Committing the output means the
deploy serves the same instant, correct Spanish content local dev does,
with MyMemory never in the request path at all for anything already
covered here.

Safe to re-run: already-cached titles are skipped. Re-run (and commit the
result) after adding/editing a researched title, resolving a new tmdb_id,
or deleting a cache file to force a re-translation. Before a deploy,
re-running this is the equivalent of a build step.

    python webapp/prewarm_translations.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preshow import tmdb  # noqa: E402
from preshow.content import load_all  # noqa: E402
from preshow.schemas import TitleCase  # noqa: E402
from preshow.translate import translate_pack_dump, translate_text  # noqa: E402

CONTENT_DIR = ROOT / "content" / "researched"
TITLES_PATH = ROOT / "evals" / "dataset" / "titles.yaml"
TRANSLATIONS_DIR = ROOT / "content" / "_translations"


def load_cases() -> dict[str, TitleCase]:
    raw = yaml.safe_load(TITLES_PATH.read_text(encoding="utf-8"))
    return {f["title_id"]: TitleCase(kind="film", **f) for f in raw["films"]}


def prewarm_packs() -> None:
    packs = load_all(CONTENT_DIR)
    if not packs:
        print(f"No researched titles found under {CONTENT_DIR}")
        return

    for title_id, pack in packs.items():
        cache_path = TRANSLATIONS_DIR / f"{title_id}.json"
        if cache_path.exists():
            print(f"skip  {title_id} (already cached)")
            continue

        print(f"...   {title_id}", end="", flush=True)
        t0 = time.time()
        translated, ok = translate_pack_dump(pack.public_dump(seen=True))
        elapsed = time.time() - t0
        if not ok:
            print(f"  FAILED in {elapsed:.1f}s (MyMemory down/rate-limited) -- not cached, will retry")
            continue
        cache_path.write_text(
            json.dumps(translated, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  done in {elapsed:.1f}s")


def prewarm_browse_overviews() -> None:
    """The other half of the not-researched catalogue (D10): its TMDB
    overview is what's shown instead of a researched deep-dive, and it
    needs the same Spanish coverage the researched titles get, or the ES
    toggle silently leaves it in English."""
    packs = load_all(CONTENT_DIR) if CONTENT_DIR.exists() else {}
    cases = load_cases()
    unresearched = {tid: c for tid, c in cases.items() if tid not in packs and c.tmdb_id}

    for title_id, case in unresearched.items():
        cache_path = TRANSLATIONS_DIR / f"browse_{case.tmdb_id}.json"
        if cache_path.exists():
            print(f"skip  {title_id} browse overview (already cached)")
            continue

        movie = tmdb.get_movie(case.tmdb_id)
        overview = movie.get("overview") if movie else None
        if not overview:
            print(f"skip  {title_id} browse overview (no TMDB overview)")
            continue

        print(f"...   {title_id} browse overview", end="", flush=True)
        t0 = time.time()
        translated = translate_text(overview)
        elapsed = time.time() - t0
        if translated == overview:
            print(f"  FAILED in {elapsed:.1f}s (MyMemory down/rate-limited) -- not cached, will retry")
            continue
        cache_path.write_text(
            json.dumps({"overview_es": translated}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  done in {elapsed:.1f}s")


def main() -> None:
    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
    prewarm_packs()
    prewarm_browse_overviews()


if __name__ == "__main__":
    main()
