"""Generador baseline del Hito 0.

Una sola llamada a la API de Anthropic, SIN retrieval: el modelo escribe el
brief desde su propia memoria parametrica. Es la version corregida del
prompt original (ver seccion 2 de CLAUDE.md): misma ausencia de retrieval,
pero salida tipada via tool use en vez de markdown con semaforo autoetiquetado,
medida por el harness en vez de prometida en el propio texto generado.

No pretende ser bueno. Sirve para medir el punto de partida real -- si el
leakage_rate sale alto en cola larga, esa es la justificacion cuantitativa
del Hito 1 (retrieval), no una suposicion de diseno.
"""

from __future__ import annotations

import os

import anthropic

from .schemas import Claim, DeepDive, PreShowBrief, ScriptBlock, SourceDoc, TitleCase

SYSTEM_PROMPT = """\
Escribes material promocional PRE-VISIONADO (spoiler-free) para una película.
Quien lo lea todavía no la ha visto.

Reglas:
- No reveles el desenlace, giros de trama, quién es el asesino/villano, ni
  ningún hecho que solo tenga sentido después de ver la película.
- context_bullets y author_voice: máximo 3 cada uno. Cada uno es un Claim con
  kind="fact" (verificable, sobre producción/reparto/premisa) o
  kind="interpretation" (tu lectura, no necesita fuente). Si es "fact", pon
  source_id a null -- no tienes fuentes reales, esto es un baseline sin
  retrieval, no inventes un identificador de fuente falso.
- emotional_temperature: una frase que capture el tono (metáfora sensorial).
- why_now: por qué verla ahora, sin destripar nada.
- script: 1-3 bloques con tiempos en segundos, texto en pantalla, voiceover y
  dirección visual. El voiceover también es spoiler-surface: las mismas
  reglas aplican ahí.

Llama a la tool emit_preshow_brief con el resultado. No escribas nada fuera
de la tool call.
"""

_CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "kind": {"type": "string", "enum": ["fact", "interpretation"]},
        "source_id": {"type": ["string", "null"]},
    },
    "required": ["text", "kind", "source_id"],
}

_TOOL = {
    "name": "emit_preshow_brief",
    "description": "Emite el brief pre-visionado en el formato requerido.",
    "input_schema": {
        "type": "object",
        "properties": {
            "context_bullets": {"type": "array", "maxItems": 3, "items": _CLAIM_SCHEMA},
            "author_voice": {"type": "array", "maxItems": 3, "items": _CLAIM_SCHEMA},
            "emotional_temperature": {"type": "string"},
            "why_now": {"type": "string"},
            "script": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_s": {"type": "integer"},
                        "end_s": {"type": "integer"},
                        "on_screen_text": {"type": "string"},
                        "voiceover": {"type": "string"},
                        "visual_direction": {"type": "string"},
                    },
                    "required": [
                        "start_s",
                        "end_s",
                        "on_screen_text",
                        "voiceover",
                        "visual_direction",
                    ],
                },
            },
        },
        "required": [
            "context_bullets",
            "author_voice",
            "emotional_temperature",
            "why_now",
            "script",
        ],
    },
}


class AnthropicBaselineGenerator:
    """Generator (Protocol) sin retrieval. `corpus` se ignora a propósito:
    ese es justo el punto que este baseline existe para medir."""

    name = "baseline"

    def __init__(self, client: anthropic.Anthropic | None = None, model: str = "claude-sonnet-5"):
        self._client = client or anthropic.Anthropic()
        self._model = model

    def pre_show(self, case: TitleCase, corpus: list[SourceDoc]) -> PreShowBrief:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "emit_preshow_brief"},
            messages=[
                {
                    "role": "user",
                    "content": f"Título: {case.title} ({case.year})",
                }
            ],
        )
        block = next(b for b in resp.content if b.type == "tool_use")
        data = block.input
        return PreShowBrief(
            title_id=case.title_id,
            context_bullets=[Claim(**c) for c in data["context_bullets"]],
            author_voice=[Claim(**c) for c in data["author_voice"]],
            emotional_temperature=data["emotional_temperature"],
            why_now=data["why_now"],
            script=[ScriptBlock(**b) for b in data["script"]],
        )

    def deep_dive(self, case: TitleCase, corpus: list[SourceDoc]) -> DeepDive:
        raise NotImplementedError("Hito 0 solo cubre pre_show; deep_dive llega en un hito posterior")
