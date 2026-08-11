"""Tests for HybridJudge (D15's correction in docs/DESIGN.md) -- the fix
for TrainedClassifierJudge misfiring on this project's actual short,
terse generator output (confirmed live: leakage_rate=0.95 on things like
"Black screen" and cast credits, not real leaks).

No scikit-learn needed here (unlike test_trained_classifier_judge.py) --
HybridJudge only routes by word count between two injected judges, so
this tests the ROUTING LOGIC with simple stubs, not real classifiers.
Runs in CI.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import HybridJudge  # noqa: E402
from preshow.schemas import SpoilerLabel  # noqa: E402

LABEL = SpoilerLabel(id="x", canonical="irrelevant to this test", severity="core")


class _AlwaysJudge:
    """Stub that always returns a fixed verdict, so tests can tell which
    of the two injected judges actually handled a given text."""

    def __init__(self, verdict: bool):
        self._verdict = verdict

    def entails(self, text: str, label: SpoilerLabel) -> bool:
        return self._verdict


def test_short_text_goes_to_short_judge():
    hybrid = HybridJudge(_AlwaysJudge(True), _AlwaysJudge(False), min_words=15)
    assert hybrid.entails("Black screen", LABEL) is True  # short_judge's verdict


def test_long_text_goes_to_long_judge():
    hybrid = HybridJudge(_AlwaysJudge(False), _AlwaysJudge(True), min_words=15)
    long_text = " ".join(["word"] * 20)
    assert hybrid.entails(long_text, LABEL) is True  # long_judge's verdict


def test_boundary_is_inclusive_at_min_words():
    hybrid = HybridJudge(_AlwaysJudge(False), _AlwaysJudge(True), min_words=5)
    exactly_five = "one two three four five"
    assert hybrid.entails(exactly_five, LABEL) is True  # routed to long_judge


def test_confirmed_false_positives_from_the_live_run_route_to_short_judge():
    """Regression test for the exact bug D15 documents: these phrases are
    real output from a live run.py --judge trained-classifier run, and
    all previously scored 0.30-0.47 (misfiring at threshold 0.3). With
    the default min_words=15 they must all route to the short judge
    (here, a stub that always says "no leak") instead of the classifier."""
    from judge import DEFAULT_MIN_WORDS_FOR_CLASSIFIER

    confirmed_false_positives = [
        "Black screen",
        "It stars Amy Adams as a linguist",
        "The film stars Daniel Kaluuya, Allison Williams, and Bradley Whitford.",
        "Parasite premiered at the 2019 Cannes Film Festival, where it won the Palme d'Or.",
        "We're all just trying to find our place in the world",
    ]
    hybrid = HybridJudge(_AlwaysJudge(False), _AlwaysJudge(True))  # default min_words
    for text in confirmed_false_positives:
        assert len(text.split()) < DEFAULT_MIN_WORDS_FOR_CLASSIFIER, (
            f"{text!r} is >= the word threshold -- this test's premise (these were "
            "confirmed short-text false positives) no longer holds, re-check D15"
        )
        assert hybrid.entails(text, LABEL) is False  # short_judge's verdict, not the classifier's
