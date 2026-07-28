"""Reads a single variable from the environment, falling back to a
plain-text .env parse -- no python-dotenv dependency for a couple of
variables. Shared by tmdb.py and baseline_groq.py so both look in the
same one place (repo-root .env, gitignored, never committed)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT / ".env"


def read_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() == name:
            return val.strip()
    return None
