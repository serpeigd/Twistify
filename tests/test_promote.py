"""Tests for webapp/promote.py -- the draft -> researched gate. Offline, no
network; drafts are synthetic dicts, directories are tmp_path."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))
sys.path.insert(0, str(ROOT / "evals"))

import promote  # noqa: E402
from preshow.schemas import SpoilerLabel  # noqa: E402

LABEL = SpoilerLabel(
    id="twist", canonical="The narrator is the killer", severity="core", paraphrases=[]
)


def _draft(depth: int = 20, *, sourced: bool = True, story: str = "A quiet film about a town.", **over) -> dict:
    """`depth` items spread over metaphors + fun_facts (both count as depth)."""
    src = "https://en.wikipedia.org/wiki/X" if sourced else None
    half = depth // 2
    d = {
        "title_id": "x_2000",
        "story": story,
        "title": "X",
        "year": 2000,
        "metaphors": [{"text": f"m{i}", "source_id": src, "kind": "fact"} for i in range(half)],
        "fun_facts": [{"lead": f"l{i}", "text": f"t{i}", "source_id": src} for i in range(depth - half)],
        "debate_prompts": ["Is it A, or B?"],
    }
    d.update(over)
    return d


def _run(draft, *, labels=(LABEL,), existing=None, in_titles_yaml=True):
    return promote.evaluate(draft, list(labels), in_titles_yaml=in_titles_yaml, existing=existing)


def test_rich_grounded_draft_passes():
    r = _run(_draft(20))
    assert r.passed and r.depth == 20 and not r.warnings


def test_thin_draft_is_blocked_on_depth():
    r = _run(_draft(6))
    assert not r.passed
    assert any(f.startswith("depth:") for f in r.failures)


def test_depth_between_floor_and_warn_level_passes_with_warning():
    r = _run(_draft(14))
    assert r.passed and any(w.startswith("depth:") for w in r.warnings)


def test_unsourced_claims_are_blocked_on_grounding():
    r = _run(_draft(20, sourced=False))
    assert any(f.startswith("grounding:") for f in r.failures)


def test_regression_vs_live_version_is_blocked():
    r = _run(_draft(14), existing=_draft(20))
    assert r.prior_depth == 20
    assert any(f.startswith("regression:") for f in r.failures)


def test_literal_spoiler_in_pre_show_surface_is_blocked():
    r = _run(_draft(20, story="Turns out the narrator is the killer, sadly."))
    assert r.leaks and r.leaks[0]["where"] == "story"
    assert any(f.startswith("leak floor:") for f in r.failures)


def test_spoiler_in_post_show_field_is_not_a_leak():
    # Post-viewing fields are SUPPOSED to contain spoilers.
    d = _draft(20)
    d["metaphors"][0]["text"] = "The narrator is the killer"
    assert _run(d).passed


def test_no_labels_warns_instead_of_silently_passing():
    r = _run(_draft(20), labels=())
    assert r.passed and any("no documented spoiler labels" in w for w in r.warnings)


def test_title_outside_titles_yaml_needs_title_and_year():
    d = _draft(20)
    del d["title"], d["year"]
    assert _run(d, in_titles_yaml=True).passed
    assert any(f.startswith("schema:") for f in _run(d, in_titles_yaml=False).failures)


def test_invalid_draft_is_blocked_not_raised():
    r = _run(_draft(20, metaphors="not a list"))
    assert any(f.startswith("schema:") for f in r.failures)


def _dirs(tmp_path, draft, live=None, cache=False):
    drafts, researched, trans = tmp_path / "d", tmp_path / "r", tmp_path / "t"
    for p in (drafts, researched, trans):
        p.mkdir()
    (drafts / "x_2000.json").write_text(json.dumps(draft), encoding="utf-8")
    if live is not None:
        (researched / "x_2000.json").write_text(json.dumps(live), encoding="utf-8")
    if cache:
        (trans / "x_2000.json").write_text("{}", encoding="utf-8")
    return dict(drafts_dir=drafts, researched_dir=researched, translations_dir=trans,
                log_path=tmp_path / "log.jsonl", in_titles_yaml=True, labels=[LABEL])


def test_not_reviewed_writes_nothing(tmp_path):
    kw = _dirs(tmp_path, _draft(20), cache=True)
    _, outcome = promote.promote("x_2000", reviewed=False, **kw)
    assert outcome == "needs-review"
    assert not (kw["researched_dir"] / "x_2000.json").exists()
    assert (kw["translations_dir"] / "x_2000.json").exists()


def test_reviewed_promotes_logs_and_drops_stale_translation(tmp_path):
    kw = _dirs(tmp_path, _draft(20), live=_draft(6), cache=True)
    _, outcome = promote.promote("x_2000", reviewed=True, **kw)
    assert outcome == "promoted"
    assert promote.depth(json.loads((kw["researched_dir"] / "x_2000.json").read_text())) == 20
    assert not (kw["translations_dir"] / "x_2000.json").exists()
    row = json.loads(kw["log_path"].read_text().splitlines()[0])
    assert (row["title_id"], row["depth"], row["prior_depth"]) == ("x_2000", 20, 6)


def test_blocked_draft_is_not_written_even_when_reviewed(tmp_path):
    kw = _dirs(tmp_path, _draft(6), cache=True)
    _, outcome = promote.promote("x_2000", reviewed=True, **kw)
    assert outcome == "blocked"
    assert not (kw["researched_dir"] / "x_2000.json").exists()
    assert (kw["translations_dir"] / "x_2000.json").exists()


def test_allow_regression_overrides_only_the_regression_check(tmp_path):
    kw = _dirs(tmp_path, _draft(14), live=_draft(20))
    assert promote.promote("x_2000", reviewed=True, **kw)[1] == "blocked"
    assert promote.promote("x_2000", reviewed=True, allow_regression=True, **kw)[1] == "promoted"


def test_allow_regression_does_not_waive_the_depth_floor(tmp_path):
    kw = _dirs(tmp_path, _draft(6), live=_draft(20))
    assert promote.promote("x_2000", reviewed=True, allow_regression=True, **kw)[1] == "blocked"
