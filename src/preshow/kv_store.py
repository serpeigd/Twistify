"""Tiny JSON-blob persistence layer, backed by Upstash Redis's free REST
API when configured, falling back to a local JSON file otherwise.

Why: most free hosting tiers (Render, etc.) wipe the filesystem on every
restart/redeploy, so content/comments.json and content/movie_requests.json
would silently reset in production. Upstash's free tier (no card; REST
API, so no persistent connection to manage -- a good fit for a small box
that may spin down between requests) fixes that without adding a database
dependency or an ORM: same read/write-a-JSON-blob shape the app already
had, just backed by a key that survives restarts.

Local dev needs no Upstash account at all: without
UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN in .env, this falls
back to the exact same local-file behavior the app always had.

Free signup: https://console.upstash.com -- create a Redis database, the
"REST API" tab shows both values to put in .env (gitignored, same pattern
as TMDB_READ_ACCESS_TOKEN / GROQ_API_KEY). Not tested against a live
Upstash instance while writing this (no account existed yet) -- verify
the first read/write once real credentials are in place.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .env import read_env


def _upstash_creds() -> tuple[str, str] | None:
    url = read_env("UPSTASH_REDIS_REST_URL")
    token = read_env("UPSTASH_REDIS_REST_TOKEN")
    if url and token:
        return url.rstrip("/"), token
    return None


def _get(url: str, token: str, key: str) -> str | None:
    req = urllib.request.Request(
        f"{url}/get/{key}", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("result")


def _set(url: str, token: str, key: str, value: str) -> None:
    req = urllib.request.Request(
        f"{url}/set/{key}",
        data=value.encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("result") != "OK":
        raise RuntimeError(f"Upstash SET did not confirm success: {data}")


def read_json_blob(key: str, file_path: Path, default: Any) -> Any:
    """Best-effort read: falls back to `default` on any Upstash failure --
    comments/requests failing to load should show an empty state, not
    crash the page."""
    creds = _upstash_creds()
    if creds:
        url, token = creds
        try:
            raw = _get(url, token, key)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            return default
        return json.loads(raw) if raw else default
    if file_path.exists():
        return json.loads(file_path.read_text(encoding="utf-8"))
    return default


def write_json_blob(key: str, file_path: Path, value: Any) -> None:
    """Raises on failure when Upstash is configured -- a write the caller
    thinks succeeded (e.g. posting a comment) but silently didn't persist
    would be worse than a visible error."""
    creds = _upstash_creds()
    if creds:
        url, token = creds
        _set(url, token, key, json.dumps(value, ensure_ascii=False))
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
