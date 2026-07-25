"""Interfaz de generación.

PRINCIPIO: el harness de evaluación se construye y se prueba contra un
generador FALSO antes de tocar ninguna API.

Motivo: si escribes el harness y el generador real a la vez, cuando la métrica
salga rara no sabrás si el generador es malo o la métrica está mal. Un fake
determinista con fugas *plantadas* te da un test con respuesta conocida:
sabes que hay exactamente 2 fugas, así que si tu métrica no dice 2, la métrica
está rota. Esto es lo que hace que puedas confiar en el número del README.
"""

from __future__ import annotations

from typing import Protocol

from .schemas import Claim, DeepDive, PreShowBrief, ScriptBlock, SourceDoc, TitleCase


class Generator(Protocol):
    """Inyectable. El runner de evals no sabe si detrás hay un LLM o un dict."""

    name: str

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief: ...

    def deep_dive(self, case: TitleCase, corpus: list[SourceDoc]) -> DeepDive: ...


class ScriptedFakeGenerator:
    """Devuelve salidas fijadas por título. Cero red, cero coste, determinista.

    Se usa en tests/test_metrics.py para verificar que las métricas cuentan lo
    que dicen contar.
    """

    name = "fake"

    def __init__(self, canned: dict[str, PreShowBrief]):
        self._canned = canned

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief:
        return self._canned[case.title_id]

    def deep_dive(self, case: TitleCase, corpus: list[SourceDoc]) -> DeepDive:
        raise NotImplementedError


def make_brief(
    title_id: str,
    facts: list[tuple[str, str | None]],
    voiceover: str = "",
) -> PreShowBrief:
    """Helper para construir briefs sintéticos en tests.

    `facts` son tuplas (texto, source_id). source_id=None => hecho sin fundamentar.
    """
    return PreShowBrief(
        title_id=title_id,
        context_bullets=[
            Claim(text=t, kind="fact", source_id=s) for t, s in facts[:3]
        ],
        author_voice=[],
        emotional_temperature="como abrir una nevera a oscuras",
        why_now="",
        script=[
            ScriptBlock(
                start_s=0,
                end_s=15,
                on_screen_text="",
                voiceover=voiceover,
                visual_direction="",
            )
        ],
    )
