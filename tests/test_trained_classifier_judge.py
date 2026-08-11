"""Tests for TrainedClassifierJudge (D15 in docs/DESIGN.md).

Uses pytest.importorskip so CI (which only installs pydantic/pytest/pyyaml,
see .github/workflows/tests.yml) skips this file cleanly instead of failing
-- scikit-learn is an opt-in dependency for this one judge, not a core
requirement for the offline harness. Runs for real wherever scikit-learn
is already installed, e.g. this dev machine.

Doesn't use the real persisted model artifact (evals/models/, gitignored,
needs the external Kaggle dataset to build) -- fits a tiny throwaway
vectorizer+model on a few toy examples instead, via the same constructor
signature the real artifact loads into. This tests the WIRING
(TfidfVectorizer -> LogisticRegression -> spoiler_prob -> threshold ->
entails), not the calibrated numbers reported in D15, which live in
evals/results/trained_classifier_external.json (gitignored) and the docs.
"""

import sys
from pathlib import Path

import pytest

sklearn = pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import TrainedClassifierJudge  # noqa: E402
from preshow.schemas import SpoilerLabel  # noqa: E402

LABEL = SpoilerLabel(id="x", canonical="irrelevant to this judge", severity="core")


def _toy_judge(threshold: float = 0.5) -> TrainedClassifierJudge:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = [
        "the twist ending reveals he was dead the whole time",
        "the killer turns out to be the narrator himself",
        "a beautifully shot, atmospheric thriller",
        "great performances, highly recommend this film",
    ]
    truths = [True, True, False, False]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression()
    model.fit(X, truths)
    return TrainedClassifierJudge(vectorizer, model, threshold=threshold)


def test_spoiler_prob_is_a_probability():
    judge = _toy_judge()
    p = judge.spoiler_prob("the twist ending reveals he was dead")
    assert 0.0 <= p <= 1.0


def test_entails_ignores_the_label_argument():
    """Different labels, same text -> same verdict, since this judge
    predicts "is this spoiler-y text" from the text alone (see class
    docstring) -- unlike Substring/LLM/NLIJudge, which check entailment
    against the specific label."""
    judge = _toy_judge()
    text = "the twist ending reveals he was dead the whole time"
    other_label = SpoilerLabel(id="y", canonical="something completely different", severity="minor")
    assert judge.entails(text, LABEL) == judge.entails(text, other_label)


def test_threshold_controls_the_decision():
    judge_strict = _toy_judge(threshold=0.99)
    judge_loose = _toy_judge(threshold=0.01)
    text = "the twist ending reveals he was dead the whole time"
    assert judge_loose.entails(text, LABEL)
    assert not judge_strict.entails(text, LABEL)


def test_from_artifact_raises_a_clear_error_when_missing():
    with pytest.raises(FileNotFoundError, match="train_spoiler_classifier"):
        TrainedClassifierJudge.from_artifact(path=ROOT / "evals" / "models" / "does_not_exist.joblib")
