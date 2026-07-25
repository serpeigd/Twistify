"""Generador de DEMO. No llama a ningún LLM, no cuesta nada, no es el baseline
real (ese es `baseline.py`, que sí necesita ANTHROPIC_API_KEY y gasta crédito).

Construye un PreShowBrief a partir de plantillas fijas y solo de los campos
seguros que ya tenemos gratis en el propio dataset (título, año, estrato,
notas de `titles.yaml`). Nunca toca `evals/dataset/spoilers/*.yaml`: ese
fichero es precisamente lo que un brief pre-visionado no debe filtrar.

Sirve para tener algo que ver y verificar en una UI sin gastar crédito real.
No confundir con una medición del sistema -- el `leakage_rate` de esto es
0.0 por construcción (no hay retrieval real detrás), no porque el sistema
"funcione".
"""

from __future__ import annotations

from .schemas import Claim, DeepDive, PreShowBrief, ScriptBlock, SourceDoc, TitleCase

_STRATUM_VOICE = {
    "mainstream": "una de esas películas de las que todo el mundo tiene una opinión, aunque no la haya visto",
    "longtail": "una de esas películas que casi nadie ha visto pero todo el que la ha visto no la olvida",
}


class DemoGenerator:
    """Generator (Protocol) de demo. Determinista: mismo título -> mismo brief."""

    name = "demo"

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief:
        voice = _STRATUM_VOICE.get(case.stratum, "una película que merece verse sin saber nada de antemano")

        context_bullets = [
            Claim(
                text=f"{case.title} se estrenó en {case.year}.",
                kind="fact",
                source_id=None,
            ),
        ]
        if case.notes:
            context_bullets.append(
                Claim(text=case.notes, kind="interpretation", source_id=None)
            )

        return PreShowBrief(
            title_id=case.title_id,
            context_bullets=context_bullets[:3],
            author_voice=[
                Claim(text=voice, kind="interpretation", source_id=None),
            ],
            emotional_temperature="como abrir una puerta sin saber qué hay detrás",
            why_now=f"{case.title} ({case.year}) — de las que conviene ver sin haber leído nada antes.",
            script=[
                ScriptBlock(
                    start_s=0,
                    end_s=15,
                    on_screen_text=case.title,
                    voiceover=voice,
                    visual_direction="plano fijo, sin música hasta el segundo 10",
                )
            ],
        )

    def deep_dive(self, case: TitleCase, corpus: list[SourceDoc]) -> DeepDive:
        raise NotImplementedError("La demo solo cubre pre_show; deep_dive no existe todavía")
