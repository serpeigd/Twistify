"""Contrato de datos del sistema.

DECISIÓN DE DISEÑO CENTRAL
--------------------------
Un `Claim` puede existir sin fuente. Esto parece un bug: ¿por qué no forzar
`source_id: str` y que Pydantic rechace lo no fundamentado?

Porque no podrías medirlo. Si el parser rechaza el estado inválido, el fallo
se convierte en una excepción en vez de en una métrica, y lo único que
aprendes es "petó". Necesitas que el modelo PUEDA producir una afirmación sin
fuente para poder contar con qué frecuencia lo hace, en qué tipo de títulos y
si tu intervención lo reduce.

La validación no vive en el esquema. Vive en el verificador, que es una etapa
explícita del pipeline con su propia métrica.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SpoilerTier(str, Enum):
    """Semáforo, pero asignado al CORPUS, no al output.

    En el prompt original el modelo se autoetiquetaba. Aquí el tier se decide
    sobre los documentos recuperados, antes de generar. El generador
    pre-experiencia solo recibe GREEN. La seguridad es una propiedad del
    contexto, no una instrucción que el modelo puede ignorar.
    """

    GREEN = "green"  # seguro pre-experiencia
    AMBER = "amber"  # ambiguo -> se trata como RED por defecto
    RED = "red"      # solo post-experiencia


Origin = Literal["tmdb", "omdb", "wikipedia", "openlibrary", "model_memory"]


class SourceDoc(BaseModel):
    """Un chunk recuperado. `model_memory` existe para el baseline (Hito 0):
    permite representar el caso 'esto salió de los pesos, no de una fuente'."""

    source_id: str
    origin: Origin
    text: str
    tier: SpoilerTier
    url: str | None = None
    section: str | None = None


class Claim(BaseModel):
    """Unidad atómica de contenido.

    `kind` separa lo verificable de lo opinable. Una interpretación sin fuente
    es legítima (es tu lectura de la obra). Un hecho sin fuente es basura.
    Mezclarlos en el mismo campo, como hacía el prompt original, hace imposible
    aplicar reglas distintas.
    """

    text: str
    kind: Literal["fact", "interpretation"]
    source_id: str | None = None
    quote: str | None = Field(
        default=None,
        description="Fragmento literal de la fuente que soporta el claim.",
    )

    @property
    def is_grounded(self) -> bool:
        return self.kind == "interpretation" or self.source_id is not None


class ScriptBlock(BaseModel):
    """Bloque temporal del guion.

    Datos, no markdown. Si algún día encadenas TTS + montaje automático,
    necesitas `start_s`/`end_s` como números, no una celda de tabla.
    """

    start_s: int
    end_s: int
    on_screen_text: str
    voiceover: str
    visual_direction: str


class PreShowBrief(BaseModel):
    """FASE 1. Se genera con contexto GREEN exclusivamente."""

    title_id: str
    context_bullets: list[Claim] = Field(max_length=3)
    author_voice: list[Claim] = Field(max_length=3)
    emotional_temperature: str  # metáfora sensorial: siempre interpretación
    why_now: str
    script: list[ScriptBlock]

    @property
    def all_claims(self) -> list[Claim]:
        return [*self.context_bullets, *self.author_voice]


class DeepDive(BaseModel):
    """FASE 2+3. Contexto completo. Aquí los spoilers son el producto."""

    title_id: str
    metaphors: list[Claim]
    intertextual_refs: list[Claim]
    production_trivia: list[Claim]
    critical_consensus: list[Claim]
    strengths: list[Claim]
    weaknesses: list[Claim]
    verdict: str

    @property
    def all_claims(self) -> list[Claim]:
        return [
            *self.metaphors,
            *self.intertextual_refs,
            *self.production_trivia,
            *self.critical_consensus,
            *self.strengths,
            *self.weaknesses,
        ]


class SpoilerLabel(BaseModel):
    """Ground truth etiquetado a mano. Un hecho que NUNCA debe aparecer
    en un PreShowBrief.

    `paraphrases` existe porque la fuga real casi nunca es literal. El modelo
    no escribe "el protagonista está muerto"; escribe "una revelación final
    reconfigura todo lo que has visto". Si tu detector solo busca subcadenas,
    tu tasa de fuga medida será cero y será mentira.
    """

    id: str
    canonical: str
    paraphrases: list[str] = Field(default_factory=list)
    severity: Literal["core", "major", "minor"]


class TitleCase(BaseModel):
    """Un caso del set de evaluación."""

    title_id: str
    title: str
    year: int
    kind: Literal["film", "book"]
    stratum: Literal["mainstream", "longtail"]
    tmdb_id: int | None = None
    notes: str | None = None
