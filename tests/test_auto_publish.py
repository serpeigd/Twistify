"""Tests for webapp/app.py's "+ Suggest a movie" auto-publish path
(_auto_publish_suggestion / POST /api/requests) -- added 2026-08-19
alongside the feature itself, an explicit, knowingly-risky decision to
skip any review step (see app.py's module docstring). No network: Groq
is mocked via monkeypatching research_assist.draft_best_of, TMDB via
monkeypatching preshow.tmdb's functions app.py calls.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))
sys.path.insert(0, str(ROOT / "evals"))

import pytest  # noqa: E402

import app  # noqa: E402
import research_assist  # noqa: E402
from preshow import tmdb  # noqa: E402


def _fake_pack(title: str, year: int, **_kwargs) -> dict:
    tid = research_assist.slugify(title, year)
    return {
        "title_id": tid,
        "story": "fake story",
        "context_bullets": [],
        "before_watching": [],
        "author_voice": [],
        "emotional_temperature": "calm",
        "why_now": "because",
        "metaphors": [],
        "intertextual_refs": [],
        "production_trivia": [],
        "scene_analysis": [],
        "critical_consensus": {"summary": None, "scores": [], "awards": []},
        "strengths": [],
        "weaknesses": [],
        "verdict": "fine",
        "useless_fact": None,
        "fun_facts": [],
        "questions": [],
        "debate_prompts": [],
        "cta": "watch it",
        "sources": [],
        "director": "Fake Director",
        "themes": ["Identity"],
        "title": title,
        "year": year,
        "tmdb_id": 999999,
    }


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(research_assist, "draft_best_of", _fake_pack)
    monkeypatch.setattr(app, "CONTENT_DIR", tmp_path)  # never touch the real content/researched/
    monkeypatch.setattr(app, "PACKS", {}, raising=False)
    monkeypatch.setattr(app, "DEMO_ONLY_CASES", {}, raising=False)
    monkeypatch.setattr(app, "CASES", {}, raising=False)
    app._AUTO_PUBLISH_IN_PROGRESS.clear()
    # Don't actually persist a movie-request row for these tests.
    monkeypatch.setattr(app.kv_store, "read_json_blob", lambda *a, **k: [])
    monkeypatch.setattr(app.kv_store, "write_json_blob", lambda *a, **k: None)
    return TestClient(app.app)


def test_resolvable_tmdb_id_auto_publishes_with_no_review(client, monkeypatch):
    monkeypatch.setattr(tmdb, "get_movie", lambda tmdb_id: {"title": "Fake Film", "year": 2020, "tmdb_id": tmdb_id})

    resp = client.post("/api/requests", json={"title": "Fake Film", "tmdb_id": 42})
    assert resp.status_code == 200

    tid = research_assist.slugify("Fake Film", 2020)
    assert tid in app.PACKS
    assert tid in app.DEMO_ONLY_CASES
    assert app.DEMO_ONLY_CASES[tid].year == 2020
    assert (app.CONTENT_DIR / f"{tid}.json").exists()
    assert tid not in app._AUTO_PUBLISH_IN_PROGRESS  # cleared after completion


def test_unresolvable_title_does_not_auto_publish(client, monkeypatch):
    monkeypatch.setattr(tmdb, "get_movie", lambda tmdb_id: None)
    monkeypatch.setattr(tmdb, "search_movies", lambda query: [])

    resp = client.post("/api/requests", json={"title": "Totally Unknown Nonexistent Movie"})
    assert resp.status_code == 200
    assert app.PACKS == {}
    assert app.DEMO_ONLY_CASES == {}


def test_already_researched_title_is_not_re_researched(client, monkeypatch):
    monkeypatch.setattr(tmdb, "get_movie", lambda tmdb_id: {"title": "Fake Film", "year": 2020, "tmdb_id": tmdb_id})
    tid = research_assist.slugify("Fake Film", 2020)
    app.CASES[tid] = object()  # already in the measurement track -- shouldn't matter which type

    def _boom(*a, **k):
        raise AssertionError("draft_best_of should not be called for an already-researched title")

    monkeypatch.setattr(research_assist, "draft_best_of", _boom)

    resp = client.post("/api/requests", json={"title": "Fake Film", "tmdb_id": 42})
    assert resp.status_code == 200
    assert tid not in app.PACKS
