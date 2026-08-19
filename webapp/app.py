"""Twistify — local app. Free, no network, no ANTHROPIC_API_KEY.

WHAT THIS MAKES VISIBLE (this is a portfolio piece, not a movie blog):

1. Real context partition (D3). Post-viewing fields aren't hidden with
   CSS: they never leave the server until the client declares it has seen
   the work. Opening devtools reveals nothing.
2. Per-claim grounding. Every factual claim shows whether it has a source
   or not. The gaps are visible on purpose.
3. Live leak verification, with its error bar. Pre-viewing content runs
   through the same detector `evals/run_eval.py` uses, and is reported
   alongside the judge's MEASURED recall -- because "0 leaks" from a judge
   with 0.0 recall doesn't mean "there are no leaks."

EXPLICIT, KNOWINGLY RISKY DECISION (2026-08-19, user's own instruction,
after being warned): "+ Suggest a movie" auto-publishes straight to
content/researched/ with NO human or AI review step -- see
_auto_publish_suggestion() below. This directly contradicts D14's own
review gate (content/_drafts/ existing specifically because "an LLM can
still misread retrieved text correctly-cited") and the SAME session's
own finding that 4/13 titles in one batch leaked a real spoiler into a
"spoiler-free" field before that gate caught them. The user was told
this plainly and chose speed anyway. Not silently done -- if this ever
needs revisiting, that's the reason it looked the way it does.

Run:
    pip install fastapi "uvicorn[standard]"
    python webapp/app.py
    # http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

import yaml
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from judge import SubstringJudge  # noqa: E402
from preshow import kv_store, tmdb  # noqa: E402
from preshow.content import ContentPack, load_all, strip_post_show  # noqa: E402
from preshow.schemas import SpoilerLabel, TitleCase  # noqa: E402
from preshow.translate import translate_pack_dump, translate_text  # noqa: E402

DATA = ROOT / "evals" / "dataset"
CONTENT_DIR = ROOT / "content" / "researched"
COMMENTS_FILE = ROOT / "content" / "comments.json"
MOVIE_REQUESTS_FILE = ROOT / "content" / "movie_requests.json"
TRANSLATIONS_DIR = ROOT / "content" / "_translations"
CALIBRATION_FILE = ROOT / "evals" / "results" / "substring_calibration.json"

app = FastAPI(title="Twistify")
judge = SubstringJudge()


def load_cases() -> dict[str, TitleCase]:
    raw = yaml.safe_load((DATA / "titles.yaml").read_text(encoding="utf-8"))
    return {f["title_id"]: TitleCase(kind="film", **f) for f in raw["films"]}


def load_labels(title_id: str) -> list[SpoilerLabel]:
    p = DATA / "spoilers" / f"{title_id}.yaml"
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [SpoilerLabel(**l) for l in (raw.get("labels") or [])]


def load_calibration() -> dict | None:
    if CALIBRATION_FILE.exists():
        return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    return None


CASES = load_cases()
PACKS: dict[str, ContentPack] = load_all(CONTENT_DIR) if CONTENT_DIR.exists() else {}

# Demo-only titles: a researched ContentPack in content/researched/ with no
# matching entry in evals/dataset/titles.yaml -- e.g. one added on request
# (The Odyssey, 2026-08-18) rather than through the stratified-sample
# measurement track. Synthesizes a TitleCase purely for display (title/year/
# tmdb_id come from the pack itself -- see ContentPack's docstring);
# stratum="mainstream" here is a DISPLAY grouping only, not a stratified-
# sample claim. Kept separate from CASES on purpose -- evals/run_eval.py
# reads titles.yaml directly and never sees this dict, so a demo-only title
# can't affect the measurement track (dataset size, the >=15-labeled gate,
# stratum breakdowns, anything). Requires pack.title/pack.year to be set
# (added alongside this feature) -- older packs without them are silently
# skipped here, not shown as demo-only (correct: no real title/year to show).
DEMO_ONLY_CASES: dict[str, TitleCase] = {
    tid: TitleCase(
        kind="film",
        title_id=tid,
        title=pack.title,
        year=pack.year,
        stratum="mainstream",
        tmdb_id=pack.tmdb_id,
        notes="Demo-only title, not part of the measurement track's stratified sample.",
    )
    for tid, pack in PACKS.items()
    if tid not in CASES and pack.title and pack.year
}
CALIBRATION = load_calibration()


def load_translated_dump(title_id: str, pack: ContentPack) -> tuple[dict, bool]:
    """Spanish version of a researched pack, translated once and cached to
    disk so the free MyMemory API is only ever hit the first time a title
    is viewed in Spanish. Returns (dump, fully_translated). Anything short
    of every field translating -- including a partial run where some
    fields succeeded and others got rate-limited back to English -- is NOT
    cached, so the next request retries the whole title from scratch
    instead of leaving a half-Spanish, half-English file marked as done
    forever. See src/preshow/translate.py for the all-or-nothing rationale
    and what's excluded from translation."""
    cache_path = TRANSLATIONS_DIR / f"{title_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), True
    translated, ok = translate_pack_dump(pack.public_dump(seen=True))
    if ok:
        TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(translated, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return translated, ok


def load_translated_browse(tmdb_id: int, browse: dict) -> dict:
    """Spanish version of a browse-tier (TMDB) overview, same cache-first
    shape as load_translated_dump above: `webapp/prewarm_translations.py`
    pre-generates and commits `browse_{tmdb_id}.json` for the catalogue's
    not-yet-researched titles, so this is a cache hit in production. Falls
    back to a live (uncached, best-effort) translation for any tmdb_id the
    prewarm script hasn't covered yet, e.g. a freshly-suggested title."""
    cache_path = TRANSLATIONS_DIR / f"browse_{tmdb_id}.json"
    if cache_path.exists():
        overview_es = json.loads(cache_path.read_text(encoding="utf-8"))["overview_es"]
        return {**browse, "overview": overview_es, "auto_translated": True}

    overview = browse.get("overview")
    if not overview:
        return {**browse, "auto_translated": False}
    translated = translate_text(overview)
    ok = translated != overview
    if ok:
        TRANSLATIONS_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"overview_es": translated}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return {**browse, "overview": translated if ok else browse["overview"], "auto_translated": ok}


def verify_pre_show(pack: ContentPack, labels: list[SpoilerLabel]) -> dict:
    """Runs the pre-viewing surface through the leak detector.

    NOTE (redesign): the UI no longer shows this verdict to the user. A
    "no leaks" badge communicates a guarantee the measurement doesn't back
    up: the judge has 0.0 recall (it can't see paraphrases). Showing it in
    green was misleading. The endpoint stays to audit the pipeline
    (evals/, /api/stats), not to sell it on the entry page. The
    `judge_recall` field always rides along with the verdict so no one
    reads it without its error bar.
    """
    hits = []
    for loc, text in pack.pre_show_text():
        for label in labels:
            if judge.entails(text, label):
                hits.append(
                    {
                        "label_id": label.id,
                        "severity": label.severity,
                        "where": loc,
                        "evidence": text[:200],
                    }
                )
    return {
        "leaks": hits,
        "leaked": bool(hits),
        "core_leaked": any(h["severity"] == "core" for h in hits),
        "n_labels_checked": len(labels),
        "n_surface_blocks": len(pack.pre_show_text()),
        "judge": judge.name,
        "judge_recall": (CALIBRATION or {}).get("recall"),
        "judge_n": (CALIBRATION or {}).get("n"),
    }


# ------------------------------------------------------------------ comments


def read_comments() -> dict[str, list[dict]]:
    return kv_store.read_json_blob("twistify:comments", COMMENTS_FILE, {})


def write_comments(data: dict) -> None:
    kv_store.write_json_blob("twistify:comments", COMMENTS_FILE, data)


# ------------------------------------------------------------------ endpoints


@app.get("/api/catalogue")
def api_catalogue():
    out = []
    for tid, case in {**CASES, **DEMO_ONLY_CASES}.items():
        pack = PACKS.get(tid)
        browse = tmdb.get_movie(case.tmdb_id) if case.tmdb_id else None
        out.append(
            {
                "title_id": tid,
                "title": case.title,
                "year": case.year,
                "stratum": case.stratum,
                "notes": case.notes,
                "researched": pack is not None,
                "completeness": pack.completeness()["pct"] if pack else 0,
                "n_spoilers": len(load_labels(tid)),
                "director": pack.director if pack else None,
                "themes": pack.themes if pack else [],
                "awards_count": len(pack.critical_consensus.awards) if pack else 0,
                "poster_url": browse["poster_url"] if browse else None,
            }
        )
    return sorted(out, key=lambda x: (-x["completeness"], x["title"]))


@app.get("/api/film/{title_id}")
def api_film(title_id: str, seen: bool = False, lang: str = "en"):
    case = CASES.get(title_id) or DEMO_ONLY_CASES.get(title_id)
    if case is None:
        raise HTTPException(404, f"Unknown title: {title_id}")

    pack = PACKS.get(title_id)
    labels = load_labels(title_id)

    if pack is None:
        # Browse-tier fallback (D10): no cited deep-dive exists yet, but if
        # we know the TMDB id, show a real poster/synopsis instead of an
        # empty placeholder. TMDB's overview is marketing copy, same
        # spoiler-safety profile as the researched tier's own pre-show
        # text -- it is NOT run through the leak detector.
        browse = tmdb.get_movie(case.tmdb_id) if case.tmdb_id else None
        if browse and lang == "es":
            browse = load_translated_browse(case.tmdb_id, browse)
        return {
            "case": case.model_dump(),
            "researched": False,
            "content": None,
            "verification": None,
            "grounding": None,
            "completeness": None,
            "seen": seen,
            "auto_translated": False,
            "browse": browse,
        }

    needs, grounded = pack.grounding()
    auto_translated = False
    if lang == "es":
        translated_dump, auto_translated = load_translated_dump(title_id, pack)
        content = strip_post_show(translated_dump, seen)
    else:
        content = pack.public_dump(seen=seen)
    return {
        "case": case.model_dump(),
        "researched": True,
        "content": content,
        "verification": verify_pre_show(pack, labels),
        "grounding": {
            "needs_source": needs,
            "has_source": grounded,
            "pct": round(100 * grounded / needs) if needs else None,
        },
        "completeness": pack.completeness(),
        "seen": seen,
        "n_comments": len(read_comments().get(title_id, [])),
        "auto_translated": auto_translated,
    }


@app.get("/api/search")
def api_search(q: str = ""):
    """Live TMDB search (D10) -- the browse tier that lets the catalogue
    reach effectively all of TMDB instead of only the 20 measurement-set
    titles + 7 researched ones. Never raises: tmdb.search_movies() returns
    an empty list if the query is blank or no TMDB token is configured."""
    return tmdb.search_movies(q)


def _public_comment(c: dict, owner_token: str) -> dict:
    return {
        "id": c["id"],
        "author": c["author"],
        "text": c["text"],
        "at": c["at"],
        "spoilers": c["spoilers"],
        "edited": bool(c.get("edited")),
        "mine": bool(owner_token) and c.get("owner_token") == owner_token,
    }


@app.get("/api/film/{title_id}/comments")
def api_get_comments(title_id: str, owner_token: str = ""):
    data = read_comments()
    lst = data.get(title_id, [])
    changed = False
    for c in lst:
        if "id" not in c:
            c["id"] = uuid.uuid4().hex[:12]
            changed = True
    if changed:
        write_comments(data)
    return [_public_comment(c, owner_token) for c in lst]


@app.post("/api/film/{title_id}/comments")
def api_post_comment(title_id: str, payload: dict = Body(...)):
    text = (payload.get("text") or "").strip()
    author = (payload.get("author") or "anonymous").strip()[:40]
    owner_token = (payload.get("owner_token") or "").strip()[:64]
    if not text:
        raise HTTPException(400, "Empty comment")
    if title_id not in CASES and title_id not in DEMO_ONLY_CASES:
        raise HTTPException(404, "Unknown title")

    data = read_comments()
    entry = {
        "id": uuid.uuid4().hex[:12],
        "author": author,
        "text": text[:2000],
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spoilers": bool(payload.get("spoilers")),
        "owner_token": owner_token,
    }
    data.setdefault(title_id, []).append(entry)
    write_comments(data)
    return _public_comment(entry, owner_token)


@app.put("/api/film/{title_id}/comments/{comment_id}")
def api_edit_comment(title_id: str, comment_id: str, payload: dict = Body(...)):
    text = (payload.get("text") or "").strip()
    owner_token = (payload.get("owner_token") or "").strip()
    if not text:
        raise HTTPException(400, "Empty comment")

    data = read_comments()
    for c in data.get(title_id, []):
        if c.get("id") == comment_id:
            if not owner_token or c.get("owner_token") != owner_token:
                raise HTTPException(403, "You can't edit this comment")
            c["text"] = text[:2000]
            c["edited"] = True
            write_comments(data)
            return _public_comment(c, owner_token)
    raise HTTPException(404, "Comment not found")


@app.delete("/api/film/{title_id}/comments/{comment_id}")
def api_delete_comment(title_id: str, comment_id: str, owner_token: str = ""):
    data = read_comments()
    lst = data.get(title_id, [])
    for i, c in enumerate(lst):
        if c.get("id") == comment_id:
            if not owner_token or c.get("owner_token") != owner_token:
                raise HTTPException(403, "You can't delete this comment")
            lst.pop(i)
            write_comments(data)
            return {"ok": True}
    raise HTTPException(404, "Comment not found")


_AUTO_PUBLISH_IN_PROGRESS: set[str] = set()


def _auto_publish_suggestion(title: str, year: int, tmdb_id: int | None) -> None:
    """Researches and publishes a suggested title with NO review step --
    see the module docstring's 2026-08-19 note for why this exists in
    this exact shape and what it knowingly gives up. Runs as a
    FastAPI BackgroundTask (after the /api/requests response is already
    sent), so this function must never raise into the request/response
    cycle -- every failure mode here is caught and logged, not
    propagated. Needs GROQ_API_KEY; silently no-ops without one (same
    as the rest of the app's optional-integration pattern, e.g. TMDB)."""
    import research_assist  # noqa: PLC0415 -- lazy: heavy (groq client),

    # only needed for this one background path, not every app.py request
    title_id = research_assist.slugify(title, year)
    if title_id in CASES or title_id in PACKS or title_id in _AUTO_PUBLISH_IN_PROGRESS:
        return  # already researched, or already being researched (dedupe concurrent suggestions)
    _AUTO_PUBLISH_IN_PROGRESS.add(title_id)
    try:
        pack_dict = research_assist.draft_best_of(title, year, n=1, save_incrementally=True)
        if tmdb_id and not pack_dict.get("tmdb_id"):
            pack_dict["tmdb_id"] = tmdb_id
        out_path = CONTENT_DIR / f"{title_id}.json"
        out_path.write_text(json.dumps(pack_dict, indent=2, ensure_ascii=False), encoding="utf-8")

        pack = ContentPack(**pack_dict)
        PACKS[title_id] = pack
        if title_id not in CASES and pack.title and pack.year:
            DEMO_ONLY_CASES[title_id] = TitleCase(
                kind="film",
                title_id=title_id,
                title=pack.title,
                year=pack.year,
                stratum="mainstream",
                tmdb_id=pack.tmdb_id,
                notes="Auto-published from a user suggestion, no review step -- see app.py's module docstring.",
            )
        print(f"[auto-publish] {title_id}: published, no review (user's explicit 2026-08-19 decision)")
    except Exception as e:  # noqa: BLE001 -- background task, must never crash the process
        print(f"[auto-publish] {title_id}: FAILED -- {type(e).__name__}: {e}")
    finally:
        _AUTO_PUBLISH_IN_PROGRESS.discard(title_id)


@app.post("/api/requests")
def api_post_movie_request(payload: dict = Body(...), background_tasks: BackgroundTasks = None):
    """Captures a title someone couldn't find in the catalogue, and (see
    the module docstring's 2026-08-19 note) kicks off unreviewed
    auto-publishing in the background if TMDB can resolve a year for it.
    The HTTP response returns immediately either way -- research takes
    1-4+ minutes (Groq pacing, see research_assist.py), nowhere close to
    a request timeout.
    """
    title = (payload.get("title") or "").strip()[:200]
    note = (payload.get("note") or "").strip()[:300]
    tmdb_id = payload.get("tmdb_id")
    tmdb_id = int(tmdb_id) if isinstance(tmdb_id, (int, str)) and str(tmdb_id).isdigit() else None
    if not title:
        raise HTTPException(400, "Empty title")

    requests_list = kv_store.read_json_blob("twistify:movie_requests", MOVIE_REQUESTS_FILE, [])
    requests_list.append(
        {
            "title": title,
            "note": note,
            "tmdb_id": tmdb_id,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    kv_store.write_json_blob("twistify:movie_requests", MOVIE_REQUESTS_FILE, requests_list)

    # Resolve a confirmed title/year before handing off to research --
    # prefer the tmdb_id the client already resolved via autocomplete
    # (D10) over the free-typed title, since TMDB's own title/year is
    # what wikipedia.find_page_title needs to work reliably.
    resolved_title, resolved_year, resolved_tmdb_id = title, None, tmdb_id
    movie = tmdb.get_movie(tmdb_id) if tmdb_id else None
    if movie is None:
        matches = tmdb.search_movies(title)
        movie = matches[0] if matches else None
    if movie and movie.get("year"):
        resolved_title, resolved_year, resolved_tmdb_id = movie["title"], movie["year"], movie["tmdb_id"]

    if resolved_year and background_tasks is not None:
        background_tasks.add_task(_auto_publish_suggestion, resolved_title, resolved_year, resolved_tmdb_id)

    return {"ok": True}


@app.get("/api/stats")
def api_stats():
    """Global panel. The metric that matters isn't how many entries exist,
    but what fraction of the factual content is grounded."""
    researched = list(PACKS.values())
    total_needs = total_grounded = 0
    for p in researched:
        n, g = p.grounding()
        total_needs += n
        total_grounded += g

    leaked = 0
    for tid, pack in PACKS.items():
        if verify_pre_show(pack, load_labels(tid))["leaked"]:
            leaked += 1

    return {
        "n_titles": len(CASES),
        "n_researched": len(researched),
        "n_spoiler_labelled": sum(1 for t in CASES if load_labels(t)),
        "grounded_fact_rate": round(total_grounded / total_needs, 3) if total_needs else None,
        "n_sourced_claims": total_needs,
        "leakage_rate": round(leaked / len(researched), 3) if researched else None,
        "calibration": CALIBRATION,
        "avg_completeness": round(
            sum(p.completeness()["pct"] for p in researched) / len(researched)
        )
        if researched
        else 0,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
