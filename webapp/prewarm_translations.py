"""One-time helper: translate every curated title to Spanish and cache the
result to content/_translations/, so nobody viewing the live demo has to
wait on the free MyMemory API (see src/preshow/translate.py) -- it's slow
and rate-limited on an uncached title, but only ever runs once per title.

Safe to re-run: already-cached titles are skipped. Re-run after adding or
editing a curated title, or after deleting its cache file to force a
re-translation.

    python webapp/prewarm_translations.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preshow.content import load_all  # noqa: E402
from preshow.translate import translate_pack_dump  # noqa: E402

CONTENT_DIR = ROOT / "content" / "curated"
TRANSLATIONS_DIR = ROOT / "content" / "_translations"


def main() -> None:
    packs = load_all(CONTENT_DIR)
    if not packs:
        print(f"No curated titles found under {CONTENT_DIR}")
        return

    TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    main()
