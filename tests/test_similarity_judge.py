"""Tests for SimilarityJudge (D16 in docs/DESIGN.md).

Built after D16's human spot-check of Milestone 1 found a real leak
(Los cronocrímenes: "his other selves") that SubstringJudge missed --
zero shared substring with the documented paraphrase. This judge uses
sentence-embedding cosine similarity against the SPECIFIC label's own
canonical + paraphrases instead of literal matching.

Uses pytest.importorskip so CI (which only installs pydantic/pytest/pyyaml,
see .github/workflows/tests.yml) skips this file cleanly -- same pattern
as test_trained_classifier_judge.py. Runs for real wherever
sentence-transformers is already installed, e.g. this dev machine
(downloads the small all-MiniLM-L6-v2 model on first run if not cached).
"""

import sys
from pathlib import Path

import pytest

st = pytest.importorskip("sentence_transformers")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import SimilarityJudge  # noqa: E402
from preshow.schemas import SpoilerLabel  # noqa: E402

LABEL = SpoilerLabel(
    id="tc_hector_becomes_his_own_attacker",
    canonical="Hector travels an hour back in time by accident and, to avoid altering "
    "what has already happened, ends up becoming the 'bandaged man' who attacks his own "
    "past self.",
    severity="core",
    paraphrases=[
        "the mysterious masked attacker from the start turns out to be the protagonist "
        "himself, just from later on",
        "there are two, even three, versions of the same man coexisting on the same day",
    ],
)


def test_catches_the_real_leak_substring_missed():
    """Regression test for the exact D16 finding: this is real output
    from a live retrieval-groq run, and SubstringJudge confirmed missed
    it (zero literal substring overlap with the documented paraphrase)."""
    judge = SimilarityJudge()
    assert judge.entails("One man must stop his other selves", LABEL)


def test_rejects_clean_unrelated_text():
    judge = SimilarityJudge()
    assert not judge.entails(
        "A comet streaks across the sky, the nights are always the darkest before the dawn",
        LABEL,
    )


def test_max_similarity_is_a_cosine_score():
    judge = SimilarityJudge()
    sim = judge.max_similarity("One man must stop his other selves", LABEL)
    assert -1.0 <= sim <= 1.0


def test_threshold_controls_the_decision():
    text = "One man must stop his other selves"
    assert SimilarityJudge(threshold=0.01).entails(text, LABEL)
    assert not SimilarityJudge(threshold=0.99).entails(text, LABEL)
