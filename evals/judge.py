"""Spoiler-leak judges.

THE POINT THAT SEPARATES THIS PROJECT FROM A DEMO
--------------------------------------------------
You're about to use an LLM to decide whether your LLM leaked a spoiler.
That moves the trust problem, it doesn't solve it: now your metric depends
on a non-deterministic, unvalidated component.

The answer isn't "use a better model." It's CALIBRATE THE JUDGE against
human-labeled ground truth, report its precision/recall, and present your
leak numbers with that error bar attached.

Ground truth available without labeling anything yourself:
  - TV Tropes Movies (Boyd-Graber et al., 2013): ~16k sentences, ~50%
    spoiler. Balanced, short, film-oriented. The best starting point.
  - IMDB Spoiler Dataset (Misra, arXiv:2212.06034): movie reviews.
  - Goodreads / UCSD Book Graph (Wan et al., 2019): 1.3M reviews,
    sentence-level labels, ~3% positive. For when you add books.

Watch out for the task mismatch: those datasets classify "IS this sentence
a spoiler?" You need "does this sentence REVEAL this specific spoiler?",
which is entailment, not classification. Calibration gives you an upper
bound on reliability, not an exact validation. Say so in the writeup —
knowing how to make that distinction is part of what the project
demonstrates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from preshow.schemas import SpoilerLabel


class SubstringJudge:
    """Deliberately bad baseline. Offline, deterministic, free.

    Exists for two reasons:
      1. The harness tests run with no network, no API key.
      2. It's the floor you measure the LLM judge against. If the LLM
         doesn't clearly beat this, you're not paying for anything useful.

    Expected failure: it doesn't detect paraphrases, which is how a real
    leak happens.
    """

    name = "substring"

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[^\w\s]", "", s.lower())

    def entails(self, text: str, label: SpoilerLabel) -> bool:
        t = self._norm(text)
        needles = [label.canonical, *label.paraphrases]
        return any(self._norm(n) in t for n in needles if n.strip())


JUDGE_PROMPT = """\
You are a spoiler-leak verifier. Answer ONLY with YES or NO.

PROTECTED SPOILER (must not be inferable):
{spoiler}

TEXT TO EVALUATE (pre-viewing promotional material):
{text}

Question: could a viewer who does NOT know the work infer the protected
spoiler by reading this text?

Strict criterion: also answer "YES" if the text hints at it in a way that
anticipates the reveal. Exclusion criterion: "NO" if the text only
generates generic intrigue without pointing at the spoiler's specific
content.
"""


class LLMJudge:
    """Real judge. `client_fn(prompt) -> str` is injected so it can be
    mocked.

    Uses a small model: the task is a short binary entailment call and
    doesn't justify a large one. Cost matters: it's len(surface) x
    len(labels) calls per case, i.e. the most expensive component of the
    whole pipeline if left unchecked.
    """

    name = "llm"

    def __init__(self, client_fn, cache: dict | None = None):
        self._client = client_fn
        self._cache = cache if cache is not None else {}

    def entails(self, text: str, label: SpoilerLabel) -> bool:
        key = (text, label.id)
        if key not in self._cache:
            raw = self._client(JUDGE_PROMPT.format(spoiler=label.canonical, text=text))
            self._cache[key] = raw.strip().upper().startswith("YES")
        return self._cache[key]


class NLIJudge:
    """Local NLI/entailment classifier. No per-call cost, no rate limit,
    no API key, runs fully offline after the model's one-time download
    (~140MB) -- sidesteps every Groq free-tier failure mode LLMJudge hit
    (network 403s, daily quota, transient 503s). Needs
    `pip install sentence-transformers` (pulls in torch + transformers).

    Uses a pretrained cross-encoder (default: cross-encoder/nli-deberta-v3-small,
    trained on SNLI+MultiNLI) to score entailment_prob(text, claim) directly,
    rather than asking an LLM to say YES/NO. Unlike LLMJudge, this gives a
    continuous score -- calibrate the DECISION THRESHOLD against real data
    (see evals/calibrate_nli_external.py) instead of picking one blind.

    Real caveat, not hidden: formal NLI entailment (SNLI/MultiNLI's
    definition -- "does the hypothesis logically follow from the premise")
    is a STRICTER bar than "could a viewer infer the spoiler by reading
    this text", which is what LLMJudge's prompt actually asks and what
    this project cares about. A promotional blurb that HINTS at a twist
    without stating it outright may score low on formal entailment even
    though a human would call it a leak. This is a real, structural
    reason this judge could underperform on recall specifically -- report
    it plainly if the calibration shows it, don't paper over it.
    """

    name = "nli"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", threshold: float = 0.5):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        id2label = {v.lower(): k for k, v in self._model.config.id2label.items()}
        self._entailment_idx = id2label["entailment"]
        self.threshold = threshold

    def entailment_prob(self, text: str, label: SpoilerLabel) -> float:
        """Max entailment probability across canonical + paraphrases --
        same "any phrasing counts" logic as SubstringJudge's needles, but
        scored instead of substring-matched."""
        return self.max_entailment_prob(text, [label])

    def max_entailment_prob(self, text: str, labels: list[SpoilerLabel]) -> float:
        """Same as entailment_prob but batches EVERY hypothesis (canonical +
        paraphrases) across ALL given labels into one predict() call --
        calling predict() once per label was needlessly slow: on real
        (long) review text, CrossEncoder inference cost scales with
        sequence length, and one predict() call per label multiplies
        that cost by len(labels) for no benefit, since the model doesn't
        share any work across separate calls anyway. Batching lets
        sentence-transformers process all pairs together instead."""
        import torch

        hypotheses = [h for label in labels for h in (label.canonical, *label.paraphrases)]
        logits = self._model.predict([(text, h) for h in hypotheses])
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, self._entailment_idx]
        return float(probs.max())

    def entails(self, text: str, label: SpoilerLabel) -> bool:
        return self.entailment_prob(text, label) >= self.threshold


DEFAULT_CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "spoiler_classifier.joblib"
DEFAULT_CLASSIFIER_THRESHOLD = 0.3  # not 0.5 -- see docs/DESIGN.md D15: at
# 0.3 this classifier catches 89% of real spoiler reveals (95% CI
# 0.876-0.902) at the cost of precision (0.357) -- deliberately favors
# recall per this project's own stated principle in `calibrate()` below
# ("a false positive makes you rewrite an innocent bullet; a false
# negative publishes the spoiler" -- not a symmetric cost)


class TrainedClassifierJudge:
    """TF-IDF + Logistic Regression, trained directly on this project's
    own external labels (evals/train_spoiler_classifier.py) instead of
    an off-the-shelf model. The first judge in this project to clear a
    real recall bar: 0.889 at threshold 0.3, evaluated with grouped
    k-fold by title (leave-one-title-out) on the full 7,657-review
    external set -- beats SubstringJudge (recall 0.0), LLMJudge (ceiling
    ~=0.35-0.4), and NLIJudge (~0, task mismatch). See docs/DESIGN.md D15
    for the full comparison and caveats.

    Different framing from the other three judges, deliberately: this
    does NOT check whether `text` entails `label` specifically. It
    predicts "does this text sound like it reveals a plot point" from
    the text alone (`label` is accepted for interface compatibility with
    detect_leaks()/calibrate() but ignored) -- trained on the external
    dataset's own review-level is_spoiler label, not a per-label
    entailment proxy. That sidesteps the labeling-coverage caveat
    Substring/LLM/NLIJudge all carry (a review can leak a plot point
    outside our documented SpoilerLabel set), at the cost of not being
    able to say WHICH spoiler was leaked, only that something was.

    Needs a persisted model artifact -- build one with
    `python evals/train_spoiler_classifier.py` (needs
    evals/dataset/external/, see D12, and `pip install scikit-learn`).
    """

    name = "trained-classifier"

    def __init__(self, vectorizer, model, threshold: float = DEFAULT_CLASSIFIER_THRESHOLD):
        self._vectorizer = vectorizer
        self._model = model
        self._true_idx = list(model.classes_).index(True)
        self.threshold = threshold

    @classmethod
    def from_artifact(cls, path=None, threshold: float = DEFAULT_CLASSIFIER_THRESHOLD) -> "TrainedClassifierJudge":
        import joblib

        p = Path(path) if path else DEFAULT_CLASSIFIER_PATH
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found -- run `python evals/train_spoiler_classifier.py` first "
                "to build it (needs evals/dataset/external/, see D12 in docs/DESIGN.md)."
            )
        bundle = joblib.load(p)
        return cls(bundle["vectorizer"], bundle["model"], threshold=threshold)

    def spoiler_prob(self, text: str) -> float:
        x = self._vectorizer.transform([text])
        return float(self._model.predict_proba(x)[0, self._true_idx])

    def entails(self, text: str, label: SpoilerLabel) -> bool:  # noqa: ARG002 -- label unused, see class docstring
        return self.spoiler_prob(text) >= self.threshold


DEFAULT_MIN_WORDS_FOR_CLASSIFIER = 15  # see docs/DESIGN.md D15's correction:
# a live run's confirmed false positives (real generator output, not a
# synthetic test) topped out at 14 words -- "Parasite premiered at the
# 2019 Cannes Film Festival, where it won the Palme d'Or." -- while this
# judge's validated-good negatives (real researched-content bullets)
# started at 7 words. There is NO clean word-count line between "false
# positive" and "handled correctly" in that overlap range; 15 is picked
# to sit just above every observed false positive, not because word
# count is a principled signal here (it isn't, see HybridJudge's
# docstring) -- it's the most defensible cutoff the actual evidence
# supports, not a guarantee against a false positive at word 16.


class HybridJudge:
    """Routes each (text, label) pair by word count: short text goes to
    `short_judge`, longer text to `long_judge`. Built specifically
    because TrainedClassifierJudge (D15) turned out to have no reliable
    signal on this project's actual generator output, which writes
    short, terse statements across EVERY field (context_bullets,
    author_voice, script lines) -- not just the obviously fragment-like
    `script[].on_screen_text`/`voiceover`. Field-based routing (e.g.
    "trust the classifier on context_bullets, not on script lines") was
    considered and rejected: a live run's false positives included
    `author_voice`/`context_bullets` text just as often as script
    fragments. Word count is a real but blunt proxy, not a fix for the
    underlying cause (a bag-of-words classifier trained on full review
    sentences doesn't generalize to this generator's terse register) --
    see docs/DESIGN.md D15's correction for the un-sugarcoated version.
    Test with `--show-leaks` after swapping this in; don't assume it's
    airtight just because it clears the specific false positives already
    found.
    """

    name = "hybrid"

    def __init__(self, short_judge, long_judge, min_words: int = DEFAULT_MIN_WORDS_FOR_CLASSIFIER):
        self._short = short_judge
        self._long = long_judge
        self._min_words = min_words

    def entails(self, text: str, label: SpoilerLabel) -> bool:
        judge = self._long if len(text.split()) >= self._min_words else self._short
        return judge.entails(text, label)


DEFAULT_SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.3  # calibrated (evals/calibrate_similarity.py,
# same internal held-out-paraphrase dataset as SubstringJudge's own D7
# calibration): recall=0.87 (95% CI 0.799-0.918), precision=0.856 (95% CI
# 0.784-0.907) at this threshold -- the best recall/precision balance in
# the sweep, and recall-favoring per this project's own "recall rules"
# principle (see calibrate()'s docstring below). Update this constant if
# a re-run picks a different value, don't let the two drift apart.


class SimilarityJudge:
    """Sentence-embedding cosine similarity between `text` and the
    SPECIFIC label's own canonical + paraphrases -- per-label, same
    framing as SubstringJudge/LLMJudge (unlike TrainedClassifierJudge,
    D15, which ignores `label` entirely).

    Built to catch what D16's human spot-check of Milestone 1 found
    SubstringJudge missing: Los cronocrímenes' generated "his other
    selves" vs. its documented "two, even three, versions of the same
    man coexisting" -- zero shared substring, so SubstringJudge (and,
    tested directly, TF-IDF cosine similarity: 0.023, no real signal)
    both missed it. A cross-encoder NLI entailment check (NLIJudge's own
    model, already downloaded) was tried on this same pair too and gave
    a WORSE, backwards result (0.026 "entailment" for the real leak,
    0.775 for a genuinely unrelated clean sentence -- unusable). A plain
    bi-encoder sentence embedding, compared by cosine similarity (NOT
    entailment classification), is what actually separated the two:
    0.525 for the real leak vs. 0.035-0.19 for clean/unrelated text in
    that same spot-check. This class turns that into something
    calibrated rather than a one-off anecdote -- see
    evals/calibrate_similarity.py.

    Bi-encoders embed each text ONCE and compare via cosine similarity,
    unlike NLIJudge's cross-encoder (which reprocesses the full pair
    through the transformer every time and was impractically slow on
    long text, D15) -- this stays fast even on this project's dev
    hardware, since the texts being compared here are always short
    (a claim/script line vs. a label's own paraphrase, not a full
    review).
    """

    name = "similarity"

    def __init__(self, model_name: str = DEFAULT_SIMILARITY_MODEL, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.threshold = threshold

    def max_similarity(self, text: str, label: SpoilerLabel) -> float:
        """Max cosine similarity between `text` and any of the label's
        canonical + paraphrases -- same "any phrasing counts" logic as
        SubstringJudge's needles."""
        needles = [label.canonical, *label.paraphrases]
        embeddings = self._model.encode([text, *needles], normalize_embeddings=True)
        text_vec, needle_vecs = embeddings[0], embeddings[1:]
        return float((needle_vecs @ text_vec).max())

    def entails(self, text: str, label: SpoilerLabel) -> bool:
        return self.max_similarity(text, label) >= self.threshold


@dataclass(frozen=True)
class Calibration:
    judge: str
    n: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def summary(self) -> dict:
        return {
            "judge": self.judge,
            "n": self.n,
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
        }


def calibrate(judge, labeled: list[tuple[str, SpoilerLabel, bool]]) -> Calibration:
    """`labeled`: (text, spoiler, does it really reveal it?), hand-labeled.

    In spoiler safety, RECALL rules. A false positive makes you rewrite an
    innocent bullet; a false negative publishes the spoiler. Optimize the
    threshold toward recall and report the precision you pay for it.
    """
    tp = fp = tn = fn = 0
    for text, label, truth in labeled:
        pred = judge.entails(text, label)
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
        else:
            tn += 1
    return Calibration(
        judge=getattr(judge, "name", "unknown"),
        n=len(labeled),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )
