"""Jueces de fuga de spoilers.

EL PUNTO QUE SEPARA ESTE PROYECTO DE UNA DEMO
---------------------------------------------
Vas a usar un LLM para decidir si tu LLM ha filtrado un spoiler. Eso mueve el
problema de confianza, no lo resuelve: ahora tu métrica depende de un
componente no determinista y no validado.

La respuesta no es "usar un modelo mejor". Es CALIBRAR EL JUEZ contra ground
truth etiquetado por humanos, reportar su precision/recall, y presentar tus
números de fuga con esa barra de error encima.

Ground truth disponible sin etiquetar tú nada:
  - TV Tropes Movies (Boyd-Graber et al., 2013): ~16k frases, ~50% spoiler.
    Balanceado, corto, orientado a cine. El mejor punto de partida.
  - IMDB Spoiler Dataset (Misra, arXiv:2212.06034): reseñas de películas.
  - Goodreads / UCSD Book Graph (Wan et al., 2019): 1.3M reseñas, etiquetas a
    nivel de frase, ~3% positivas. Para cuando añadas libros.

Cuidado con la diferencia de tarea: esos datasets clasifican "¿esta frase ES
un spoiler?". Tú necesitas "¿esta frase REVELA este spoiler concreto?", que es
entailment, no clasificación. La calibración te da una cota superior de
fiabilidad, no una validación exacta. Dilo en el writeup; que sepas
distinguirlo es parte de lo que demuestra el proyecto.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from preshow.schemas import SpoilerLabel


class SubstringJudge:
    """Baseline deliberadamente malo. Offline, determinista, gratis.

    Existe por dos razones:
      1. Los tests del harness corren sin red ni API key.
      2. Es el suelo contra el que mides el juez LLM. Si el LLM no supera
         claramente esto, no estás pagando por nada útil.

    Fallo esperado: no detecta paráfrasis, que es como se fuga de verdad.
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
Eres un verificador de fugas de spoilers. Responde SOLO con SI o NO.

SPOILER PROTEGIDO (no debe poder deducirse):
{spoiler}

TEXTO A EVALUAR (material promocional pre-visionado):
{text}

Pregunta: ¿un espectador que NO conoce la obra podría deducir el spoiler
protegido leyendo este texto?

Criterio estricto: "SI" también si el texto lo insinúa de forma que anticipa
la revelación. Criterio de exclusión: "NO" si el texto solo genera intriga
genérica sin apuntar al contenido concreto del spoiler.
"""


class LLMJudge:
    """Juez real. `client_fn(prompt) -> str` se inyecta para poder mockearlo.

    Usa un modelo pequeño: la tarea es entailment binario corto y no justifica
    el modelo grande. El coste importa: es len(surface) x len(labels) llamadas
    por caso, o sea el componente más caro de todo el pipeline si no vigilas.
    """

    name = "llm"

    def __init__(self, client_fn, cache: dict | None = None):
        self._client = client_fn
        self._cache = cache if cache is not None else {}

    def entails(self, text: str, label: SpoilerLabel) -> bool:
        key = (text, label.id)
        if key not in self._cache:
            raw = self._client(JUDGE_PROMPT.format(spoiler=label.canonical, text=text))
            self._cache[key] = raw.strip().upper().startswith("SI")
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
    """`labeled`: (texto, spoiler, ¿lo revela de verdad?) etiquetado a mano.

    En seguridad de spoilers, el RECALL manda. Un falso positivo te hace
    reescribir un bullet inocente; un falso negativo publica el spoiler.
    Optimiza el umbral hacia recall y reporta el precision que pagas por ello.
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
