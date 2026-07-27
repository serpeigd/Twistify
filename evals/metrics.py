"""Metrics.

Three, and all three are needed at once:

  leakage_rate      -> how many spoilers escape?              (safety)
  grounded_rate     -> how many facts have a source?           (truthfulness)
  richness          -> how much does it actually say?          (usefulness)

The first two without the third are a trap: a system that returns an empty
brief has 0% leakage and 100% grounding. The safety/richness trade-off IS
the project. A README that only reports leakage is hiding half the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from preshow.schemas import Claim, PreShowBrief, SpoilerLabel


@dataclass(frozen=True)
class LeakageHit:
    label_id: str
    severity: str
    where: str
    evidence: str


@dataclass(frozen=True)
class CaseResult:
    title_id: str
    stratum: str
    leaks: list[LeakageHit]
    n_facts: int
    n_grounded_facts: int
    n_claims: int

    @property
    def leaked(self) -> bool:
        return len(self.leaks) > 0

    @property
    def core_leaked(self) -> bool:
        return any(h.severity == "core" for h in self.leaks)


def brief_surface(brief: PreShowBrief) -> list[tuple[str, str]]:
    """All the text a viewer would get to see/hear, with its location.

    Watch out: this includes the script. It's easy to forget that the
    voiceover is also leak surface and only measure the bullets.
    """
    out: list[tuple[str, str]] = []
    for i, c in enumerate(brief.context_bullets):
        out.append((f"context_bullets[{i}]", c.text))
    for i, c in enumerate(brief.author_voice):
        out.append((f"author_voice[{i}]", c.text))
    out.append(("emotional_temperature", brief.emotional_temperature))
    out.append(("why_now", brief.why_now))
    for i, b in enumerate(brief.script):
        out.append((f"script[{i}].on_screen_text", b.on_screen_text))
        out.append((f"script[{i}].voiceover", b.voiceover))
    return [(loc, txt) for loc, txt in out if txt.strip()]


def detect_leaks(
    brief: PreShowBrief,
    labels: list[SpoilerLabel],
    judge,
) -> list[LeakageHit]:
    """`judge` is any object with .entails(text, label) -> bool.

    Injected so you can swap the cheap judge (substring) for the expensive
    one (LLM) WITHOUT touching the metric, and so you can compare both on
    the same set. If the metric and the judge were coupled you couldn't
    measure how much better the expensive judge is than the cheap one —
    which is exactly the data that justifies paying for it.
    """
    hits: list[LeakageHit] = []
    for loc, text in brief_surface(brief):
        for label in labels:
            if judge.entails(text, label):
                hits.append(
                    LeakageHit(
                        label_id=label.id,
                        severity=label.severity,
                        where=loc,
                        evidence=text[:200],
                    )
                )
    return hits


def grounding(claims: list[Claim]) -> tuple[int, int]:
    """(total facts, facts with a source). Interpretations don't count:
    demanding a source for a personal reading of the work makes no sense."""
    facts = [c for c in claims if c.kind == "fact"]
    return len(facts), sum(1 for c in facts if c.source_id is not None)


def evaluate_case(
    brief: PreShowBrief,
    labels: list[SpoilerLabel],
    stratum: str,
    judge,
) -> CaseResult:
    n_facts, n_grounded = grounding(brief.all_claims)
    return CaseResult(
        title_id=brief.title_id,
        stratum=stratum,
        leaks=detect_leaks(brief, labels, judge),
        n_facts=n_facts,
        n_grounded_facts=n_grounded,
        n_claims=len(brief.all_claims),
    )


def aggregate(results: list[CaseResult]) -> dict:
    """Overall aggregate + breakdown by stratum.

    The breakdown is NOT optional. The project's hypothesis is that the
    baseline looks fine on mainstream titles and falls apart on long-tail
    ones. If you only report the average, that signal disappears.
    """

    def _block(rs: list[CaseResult]) -> dict:
        if not rs:
            return {}
        facts = sum(r.n_facts for r in rs)
        grounded = sum(r.n_grounded_facts for r in rs)
        return {
            "n_cases": len(rs),
            "leakage_rate": round(sum(r.leaked for r in rs) / len(rs), 3),
            "core_leakage_rate": round(sum(r.core_leaked for r in rs) / len(rs), 3),
            "grounded_fact_rate": round(grounded / facts, 3) if facts else None,
            "richness_claims_per_case": round(
                sum(r.n_claims for r in rs) / len(rs), 2
            ),
        }

    return {
        "overall": _block(results),
        "mainstream": _block([r for r in results if r.stratum == "mainstream"]),
        "longtail": _block([r for r in results if r.stratum == "longtail"]),
    }
