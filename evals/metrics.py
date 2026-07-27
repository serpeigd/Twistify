"""Métricas.

Tres, y las tres son necesarias a la vez:

  leakage_rate      -> ¿cuántos spoilers se escapan?          (seguridad)
  grounded_rate     -> ¿cuántos hechos tienen fuente?         (veracidad)
  richness          -> ¿cuánto dice realmente?                (utilidad)

Las dos primeras sin la tercera son una trampa: un sistema que devuelve un
brief vacío tiene 0% de fuga y 100% de grounding. El trade-off entre
seguridad y riqueza ES el proyecto. Un README que solo reporta leakage está
ocultando la mitad del resultado.
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
    """Todo el texto que un espectador llegaría a ver/oír, con su ubicación.

    Ojo: incluye el guion. Es fácil olvidarse de que el voiceover también es
    superficie de fuga y medir solo los bullets.
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
    """`judge` es cualquier objeto con .entails(text, label) -> bool.

    Se inyecta para poder intercambiar el juez barato (subcadena) por el caro
    (LLM) SIN tocar la métrica, y para poder comparar ambos sobre el mismo
    conjunto. Si la métrica y el juez estuvieran acoplados no podrías medir
    cuánto mejora el juez caro respecto al barato — que es justo el dato que
    justifica pagarlo.
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
    """(hechos totales, hechos con fuente). Las interpretaciones no cuentan:
    exigirle fuente a una lectura personal de la obra no tiene sentido."""
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
    """Agregado global + desglose por estrato.

    El desglose NO es opcional. La hipótesis del proyecto es que el baseline
    parece funcionar en mainstream y se hunde en cola larga. Si solo reportas
    la media, esa señal desaparece.
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
