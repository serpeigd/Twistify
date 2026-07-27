"""Tests for the harness against outputs with PLANTED leaks.

We know the answer up front. If the metric doesn't reproduce it, the metric
is broken — and you find that out now, not after 200 API calls spent and a
number in the README that means nothing.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import SubstringJudge, calibrate  # noqa: E402
from metrics import aggregate, evaluate_case, grounding  # noqa: E402
from preshow.generator import make_brief  # noqa: E402
from preshow.schemas import Claim, SpoilerLabel  # noqa: E402

LABEL = SpoilerLabel(
    id="x_dead",
    canonical="the protagonist has been dead the whole movie",
    paraphrases=["the protagonist isn't alive", "one of them is a ghost"],
    severity="core",
)


def test_clean_brief_no_leak():
    b = make_brief("t1", [("A psychological thriller set in Philadelphia.", "wiki#1")])
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    assert not r.leaked


def test_literal_leak_detected():
    b = make_brief("t1", [("Heads up: the protagonist has been dead the whole movie.", "wiki#1")])
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    assert r.leaked and r.core_leaked
    assert r.leaks[0].where == "context_bullets[0]"


def test_leak_in_voiceover_is_counted():
    """Regression: it's easy to only measure the bullets and forget the
    script."""
    b = make_brief("t1", [("Nothing suspicious.", "wiki#1")],
                   voiceover="One of them is a ghost.")
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    assert r.leaked
    assert "script" in r.leaks[0].where


def test_substring_judge_misses_paraphrase():
    """EXPECTED AND DOCUMENTED FAILURE.

    This test doesn't check that the system works; it checks that we know
    WHERE it doesn't. It's the quantitative justification for the LLM
    judge.
    """
    b = make_brief("t1", [("The perspective you're seeing this from is a lie.", "w#1")])
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    assert not r.leaked  # <- real leak the baseline doesn't see


def test_grounding_ignores_interpretations():
    claims = [
        Claim(text="It was filmed in Philadelphia.", kind="fact", source_id="w#3"),
        Claim(text="It was filmed in Madrid.", kind="fact", source_id=None),
        Claim(text="The color red marks the supernatural.", kind="interpretation"),
    ]
    total, grounded = grounding(claims)
    assert (total, grounded) == (2, 1)


def test_empty_brief_scores_perfectly_on_safety():
    """The most important test in this file.

    An empty brief has 0% leakage. If your README only reports
    leakage_rate, the optimal system is to say nothing. That's why
    richness ALWAYS rides alongside it.
    """
    b = make_brief("t1", [])
    r = evaluate_case(b, [LABEL], "mainstream", SubstringJudge())
    agg = aggregate([r])
    assert agg["overall"]["leakage_rate"] == 0.0
    assert agg["overall"]["richness_claims_per_case"] == 0.0


def test_judge_calibration_reports_recall():
    data = [
        ("the protagonist isn't alive", LABEL, True),   # TP
        ("a story about grief", LABEL, False),   # TN
        ("nothing you're seeing is what it seems", LABEL, True),  # FN
    ]
    cal = calibrate(SubstringJudge(), data)
    assert cal.recall == 0.5
    assert cal.precision == 1.0


def test_strata_are_reported_separately():
    leaky = make_brief("t1", [("the protagonist isn't alive", "w#1")])
    clean = make_brief("t2", [("A 90s thriller.", "w#2")])
    rs = [
        evaluate_case(leaky, [LABEL], "longtail", SubstringJudge()),
        evaluate_case(clean, [LABEL], "mainstream", SubstringJudge()),
    ]
    agg = aggregate(rs)
    assert agg["longtail"]["leakage_rate"] == 1.0
    assert agg["mainstream"]["leakage_rate"] == 0.0
