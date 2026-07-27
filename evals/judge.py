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
