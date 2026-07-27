"""Contrato de datos del CONTENIDO curado (lo que la app muestra).

Distinto de `schemas.py`: aquel modela lo que un generador PRODUCE y el
harness MIDE. Esto modela lo que un humano/investigación ha curado y la app
SIRVE. Se mantienen separados a propósito -- mezclarlos haría imposible
distinguir "esto lo escribió un modelo" de "esto vino de una fuente".

Principio heredado de D2 (ver docs/DESIGN.md): el esquema PERMITE estados
incompletos. Un `SourcedText` sin `source_id` es válido y se renderiza como
"sin fuente" en la UI. Si el esquema lo rechazara, la carencia sería una
excepción en vez de un dato visible -- y el hueco visible ES el producto:
demuestra dónde el contenido está fundamentado y dónde no.

PARTICIÓN POR FASE (D3): `PRE_SHOW_FIELDS` y `POST_SHOW_FIELDS` no son
documentación, son la frontera que la API usa para decidir qué envía al
navegador antes de que el usuario declare haber visto la obra. Un campo
post-visionado no se filtra con CSS: no sale del servidor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# La frontera de spoilers. La API nunca envía POST_SHOW_FIELDS salvo que el
# cliente declare explícitamente haber visto la obra.
PRE_SHOW_FIELDS = (
    "story",
    "context_bullets",
    "before_watching",
    "author_voice",
    "emotional_temperature",
    "why_now",
)

POST_SHOW_FIELDS = (
    "metaphors",
    "intertextual_refs",
    "production_trivia",
    "scene_analysis",
    "critical_consensus",
    "strengths",
    "weaknesses",
    "verdict",
    "useless_fact",
    "fun_facts",
)

# Ni pre ni post: son preguntas/debate, no revelan trama por sí mismas.
NEUTRAL_FIELDS = ("questions", "debate_prompts", "cta", "sources", "director", "themes")


class SourcedText(BaseModel):
    """Texto con procedencia opcional. `source_id` ausente => sin fundamentar,
    y la UI lo marca como tal en vez de ocultarlo."""

    text: str
    source_id: str | None = None
    kind: Literal["fact", "interpretation"] | None = None

    @property
    def is_grounded(self) -> bool:
        return self.kind == "interpretation" or bool(self.source_id)


class SceneNote(BaseModel):
    scene: str
    text: str
    source_id: str | None = None


class FactBullet(BaseModel):
    """Dato con titular en negrita + explicación. Usado en 'antes de verla'
    (spoiler-free) y en 'datos curiosos' (post-visionado)."""

    lead: str
    text: str
    source_id: str | None = None


class Score(BaseModel):
    source: str
    value: str
    url: str | None = None


class CriticalConsensus(BaseModel):
    summary: str | None = None
    scores: list[Score] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)


class ContentPack(BaseModel):
    """Paquete completo de una obra. Todo opcional salvo el id: una ficha a
    medio curar es un estado legítimo y la UI debe poder mostrarla."""

    title_id: str

    # --- Fase 1: pre-visionado (spoiler-free) ---
    story: str | None = None
    context_bullets: list[str] = Field(default_factory=list)
    before_watching: list[FactBullet] = Field(default_factory=list)
    author_voice: list[SourcedText] = Field(default_factory=list)
    emotional_temperature: str | None = None
    why_now: str | None = None

    # --- Fase 2: post-visionado (aquí los spoilers son el producto) ---
    metaphors: list[SourcedText] = Field(default_factory=list)
    intertextual_refs: list[SourcedText] = Field(default_factory=list)
    production_trivia: list[SourcedText] = Field(default_factory=list)
    scene_analysis: list[SceneNote] = Field(default_factory=list)

    # --- Fase 3: crítica ---
    critical_consensus: CriticalConsensus = Field(default_factory=CriticalConsensus)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    verdict: str | None = None
    useless_fact: str | None = None
    fun_facts: list[FactBullet] = Field(default_factory=list)

    # --- Fase 4: engagement ---
    questions: list[str] = Field(default_factory=list)
    debate_prompts: list[str] = Field(default_factory=list)
    cta: str | None = None

    sources: list[str] = Field(default_factory=list)

    # --- Metadatos de catálogo (para filtrar/ordenar sin spoilers) ---
    director: str | None = None
    themes: list[str] = Field(default_factory=list)

    # ---- Métricas de completitud / fundamentación ----

    def all_sourced(self) -> list[SourcedText]:
        as_st = lambda b: SourcedText(text=b.text, source_id=b.source_id, kind="fact")
        return [
            *self.author_voice,
            *self.metaphors,
            *self.intertextual_refs,
            *self.production_trivia,
            *[as_st(b) for b in self.before_watching],
            *[as_st(b) for b in self.fun_facts],
        ]

    def grounding(self) -> tuple[int, int]:
        """(afirmaciones que requieren fuente, cuántas la tienen).

        Las interpretaciones no cuentan: exigirle fuente a una lectura
        personal de la obra no tiene sentido (misma regla que evals/metrics.py).
        """
        needs = [s for s in self.all_sourced() if s.kind != "interpretation"]
        return len(needs), sum(1 for s in needs if s.source_id)

    def completeness(self) -> dict:
        """Qué secciones tienen contenido. Los huecos se muestran, no se
        disimulan: un campo vacío significa 'no encontramos fuente', que es
        información útil, no un fallo a esconder."""
        sections = {
            "story": bool(self.story),
            "before_watching": bool(self.before_watching),
            "author_voice": bool(self.author_voice),
            "fun_facts": bool(self.fun_facts),
            "metaphors": bool(self.metaphors),
            "intertextual_refs": bool(self.intertextual_refs),
            "production_trivia": bool(self.production_trivia),
            "scene_analysis": bool(self.scene_analysis),
            "critical_consensus": bool(self.critical_consensus.scores),
            "strengths": bool(self.strengths),
            "weaknesses": bool(self.weaknesses),
            "questions": bool(self.questions),
        }
        filled = sum(sections.values())
        return {
            "sections": sections,
            "filled": filled,
            "total": len(sections),
            "pct": round(100 * filled / len(sections)),
        }

    def pre_show_text(self) -> list[tuple[str, str]]:
        """Superficie pre-visionado como (ubicación, texto), para pasarla por
        el detector de fugas. Mismo criterio que evals/metrics.brief_surface:
        si el usuario lo puede leer antes de ver la obra, es superficie."""
        out: list[tuple[str, str]] = []
        if self.story:
            out.append(("story", self.story))
        for i, b in enumerate(self.context_bullets):
            out.append((f"context_bullets[{i}]", b))
        for i, b in enumerate(self.before_watching):
            out.append((f"before_watching[{i}].lead", b.lead))
            out.append((f"before_watching[{i}]", b.text))
        for i, a in enumerate(self.author_voice):
            out.append((f"author_voice[{i}]", a.text))
        if self.emotional_temperature:
            out.append(("emotional_temperature", self.emotional_temperature))
        if self.why_now:
            out.append(("why_now", self.why_now))
        return [(loc, t) for loc, t in out if t.strip()]

    def public_dump(self, seen: bool) -> dict:
        """Serializa para el cliente. Si `seen` es False, los campos
        post-visionado NO se incluyen -- se omiten en el servidor, no se
        ocultan en el navegador. Es la diferencia entre una garantía y una
        sugerencia (ver D3 en docs/DESIGN.md)."""
        data = self.model_dump()
        if not seen:
            for field in POST_SHOW_FIELDS:
                data.pop(field, None)
        return data


def load_pack(path: Path) -> ContentPack:
    return ContentPack(**json.loads(path.read_text(encoding="utf-8")))


def load_all(directory: Path) -> dict[str, ContentPack]:
    packs = {}
    for p in sorted(directory.glob("*.json")):
        pack = load_pack(p)
        packs[pack.title_id] = pack
    return packs
