"""Thin client for Wikipedia's public MediaWiki API -- read-only, CC BY-SA,
stdlib only (same reasoning as tmdb.py/translate.py: no new dependency, no
key required). Used by webapp/research_assist.py to fetch real, citable
article text instead of asking an LLM to draft from parametric memory,
which is exactly the failure mode D6/D7 exist to avoid.

Not a general-purpose Wikipedia client: just enough to (a) resolve a movie
title+year to the right article (disambiguating from a remake/short story
of the same name) and (b) pull its plain-text extract, split into the
sections a research draft actually needs (plot, production, reception,
accolades).
"""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = ROOT / "content" / "_wikipedia_cache"
_API = "https://en.wikipedia.org/w/api.php"
_USER_AGENT = "Twistify-research-assist/1.0 (portfolio project; https://github.com/serpeigd/Twistify)"


def _ssl_context() -> ssl.SSLContext | None:
    """Some Windows Python installs verify Wikipedia's cert chain against
    a stale local trust store and reject it as expired, even though the
    OS's own certificate store (what curl/schannel uses) accepts it fine
    -- a local-environment gap, not a real expired certificate. `certifi`
    ships an independently up-to-date CA bundle and is already present
    here as a transitive dependency of the `groq` package; reuse it
    instead of adding a new direct dependency. Falls back to Python's
    default context (fine on most systems) if certifi isn't installed."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None

# Wikipedia's plaintext extract marks headings as "== Heading ==" (level 2)
# or "=== Sub ===" (level 3+). This is deliberately fuzzy (substring match
# on lowercased heading) rather than an exact heading list, because
# articles don't share one fixed structure -- some have "Critical
# response", others "Reception", others split "Box office" out from
# "Release".
_SECTION_KEYWORDS = {
    "plot": ["plot"],
    "production": ["production", "development", "filming", "writing"],
    "reception": ["reception", "critical response", "critical reaction"],
    "accolades": ["accolades", "awards"],
}


def _cache_key(*parts: str) -> str:
    raw = "_".join(parts).lower()
    return "".join(c if c.isalnum() else "_" for c in raw)[:100]


def _get(params: dict) -> dict | None:
    query = urllib.parse.urlencode({**params, "format": "json"})
    url = f"{_API}?{query}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15, context=_ssl_context()) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return None


def find_page_title(title: str, year: int | None) -> str | None:
    """Resolves a movie title (+ optional year) to the exact Wikipedia
    page title, trying the disambiguated forms first -- "Dune (2021
    film)" -- so a remake or same-titled novel doesn't win instead."""
    candidates = []
    if year:
        candidates.append(f"{title} ({year} film)")
    candidates += [f"{title} (film)", title]

    for candidate in candidates:
        data = _get({"action": "query", "list": "search", "srsearch": candidate, "srlimit": 1})
        hits = ((data or {}).get("query") or {}).get("search") or []
        if hits:
            return hits[0]["title"]
    return None


def fetch_article(page_title: str) -> dict | None:
    """Full plain-text extract for one article, cached to disk. Returns
    {"title", "url", "extract"} or None on any failure."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(page_title)}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    data = _get(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": page_title,
            "redirects": 1,
        }
    )
    pages = ((data or {}).get("query") or {}).get("pages") or {}
    page = next(iter(pages.values()), None)
    if not page or "extract" not in page or not page["extract"]:
        return None

    resolved_title = page.get("title", page_title)
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(resolved_title.replace(" ", "_"))
    result = {"title": resolved_title, "url": url, "extract": page["extract"]}
    cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


_HEADING_RE = re.compile(r"^(={2,4})\s*(.+?)\s*\1$", re.MULTILINE)


def split_sections(extract: str) -> dict[str, str]:
    """Splits a plaintext extract into {heading: body}, plus a "_lead"
    key for the text before the first heading. Headings are kept as
    written (not normalized) -- callers match by keyword, see
    relevant_sections()."""
    matches = list(_HEADING_RE.finditer(extract))
    sections: dict[str, str] = {"_lead": extract[: matches[0].start()].strip() if matches else extract.strip()}
    for i, m in enumerate(matches):
        heading = m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(extract)
        body = extract[start:end].strip()
        if body:
            sections[heading] = (sections.get(heading, "") + "\n" + body).strip()
    return sections


def relevant_sections(extract: str) -> dict[str, str]:
    """Picks out plot/production/reception/accolades text by fuzzy
    heading match -- the subset a research draft actually needs, instead
    of the full article (cast lists, references, etc.)."""
    sections = split_sections(extract)
    out: dict[str, str] = {}
    if sections.get("_lead"):
        out["overview"] = sections["_lead"]
    for label, keywords in _SECTION_KEYWORDS.items():
        for heading, body in sections.items():
            if any(kw in heading.lower() for kw in keywords):
                out[label] = (out.get(label, "") + "\n\n" + body).strip()
    return out
