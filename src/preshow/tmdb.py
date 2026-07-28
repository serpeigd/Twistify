"""Thin client for TMDB's free API -- the BROWSE tier only (poster,
synopsis, year for effectively any title), not a source for the
researched tier's cited claims. See D10 in docs/DESIGN.md for why this is
a separate tier and never conflated with `content/researched/*.json`.

Stdlib HTTP only (no `requests` dependency, same reasoning as
translate.py). Every call is cached to disk under content/_tmdb_cache/
(gitignored) -- TMDB's free tier is generous, but there's no reason to
re-fetch a search or a movie twice.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .env import read_env

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "content" / "_tmdb_cache"

_BASE_URL = "https://api.themoviedb.org/3"
_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."


def _get(path: str, params: dict | None = None) -> dict | None:
    token = read_env("TMDB_READ_ACCESS_TOKEN")
    if not token:
        return None
    query = urllib.parse.urlencode(params or {})
    url = f"{_BASE_URL}{path}" + (f"?{query}" if query else "")
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}", "accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None


def _poster_url(poster_path: str | None) -> str | None:
    return f"{_IMAGE_BASE}{poster_path}" if poster_path else None


def _to_result(m: dict) -> dict:
    return {
        "tmdb_id": m.get("id"),
        "imdb_id": m.get("imdb_id"),
        "title": m.get("title") or m.get("original_title") or "",
        "year": int(d[:4]) if (d := (m.get("release_date") or "")) and d[:4].isdigit() else None,
        "overview": m.get("overview") or "",
        "poster_url": _poster_url(m.get("poster_path")),
        "vote_average": m.get("vote_average"),
    }


def _cache_key(*parts: str) -> str:
    raw = "_".join(parts).lower()
    return "".join(c if c.isalnum() else "_" for c in raw)[:100]


def search_movies(query: str) -> list[dict]:
    """Live search against TMDB, cached per query string. Returns up to 12
    browse-tier results: [{tmdb_id, title, year, overview, poster_url,
    vote_average}, ...] -- empty list if the query is blank or no
    TMDB_READ_ACCESS_TOKEN is configured (never raises)."""
    query = (query or "").strip()
    if not query:
        return []

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"search_{_cache_key(query)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    data = _get("/search/movie", {"query": query, "include_adult": "false"})
    if data is None:
        return []
    results = [_to_result(m) for m in data.get("results", [])[:12]]
    cache_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def get_movie(tmdb_id: int) -> dict | None:
    """Full browse-tier record for one title, cached by tmdb_id."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"movie_{tmdb_id}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if "imdb_id" in cached:
            return cached
        # Cache predates the imdb_id field (see D12) -- refetch once to
        # migrate instead of returning a permanently incomplete record.

    data = _get(f"/movie/{tmdb_id}")
    if data is None:
        return None
    result = _to_result(data)
    result["genres"] = [g["name"] for g in data.get("genres", [])]
    cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def get_director(tmdb_id: int) -> str | None:
    """Director's name via TMDB's credits endpoint -- the base /movie/{id}
    response (used by get_movie) doesn't include crew. Cached separately
    since it's a different endpoint, not part of _to_result's shape."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"credits_{tmdb_id}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached.get("director")

    data = _get(f"/movie/{tmdb_id}/credits")
    director = None
    if data:
        for c in data.get("crew", []):
            if c.get("job") == "Director":
                director = c.get("name")
                break
    cache_path.write_text(json.dumps({"director": director}, ensure_ascii=False), encoding="utf-8")
    return director


def best_match(title: str, year: int) -> dict | None:
    """Resolves a (title, year) pair -- e.g. from titles.yaml -- to a TMDB
    record. Exact-year match preferred; falls back to the top search hit
    so a title that's off by a year (re-release, regional date) still
    resolves instead of silently returning nothing."""
    results = search_movies(title)
    if not results:
        return None
    for r in results:
        if r["year"] == year:
            return r
    return results[0]
