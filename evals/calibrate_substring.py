"""Calibración del SubstringJudge contra un set derivado del propio ground
truth (evals/dataset/spoilers/*.yaml), no contra un benchmark externo.

Por qué no un benchmark externo: TV Tropes Movies (Boyd-Graber et al. 2013)
no tiene descarga directa sin contactar a los autores; el IMDB Spoiler
Dataset (Misra) vive en Kaggle y exige cuenta/API key -- una credencial del
usuario que este script no debe pedir ni gestionar. Ver README, sección
"Benchmarks para calibrar el juez".

Qué mide esto en su lugar: por cada SpoilerLabel, sus paráfrasis y su
canonical son positivos (SÍ revelan ese spoiler); paráfrasis de OTROS
labels/títulos son negativos (no deberían activarlo), más un puñado de
frases neutras de marketing genérico como negativos adicionales.

Limitación explícita: es autoevaluación parcial -- las paráfrasis las
escribió el mismo tipo de sistema (LLM) que generó el ground truth (ver D7
en docs/DESIGN.md). Sirve para tener un número real y reproducible con lo
que hay disponible gratis, no sustituye la calibración contra un benchmark
independiente. No reportar esto como equivalente a esa calibración.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import SubstringJudge, calibrate  # noqa: E402
from preshow.schemas import SpoilerLabel  # noqa: E402

DATA = ROOT / "evals" / "dataset" / "spoilers"

NEUTRAL_SENTENCES = [
    "Una historia ambientada en una ciudad que nunca duerme.",
    "El reparto incluye a varios actores reconocidos por su trabajo previo.",
    "La fotografía y la banda sonora han recibido elogios de la crítica.",
    "Se rodó en varias localizaciones durante el año de producción.",
    "Es una película que mezcla drama y momentos de tensión.",
    "El director ya había trabajado antes en el mismo género.",
]


def load_all_labels() -> list[tuple[str, SpoilerLabel]]:
    out = []
    for p in sorted(DATA.glob("*.yaml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for l in raw.get("labels") or []:
            out.append((p.stem, SpoilerLabel(**l)))
    return out


def build_dataset() -> list[tuple[str, SpoilerLabel, bool]]:
    """Split train/held-out por label: la mitad de las paráfrasis se usan como
    'needles' del juez (lo que ya conoce), la otra mitad como texto de prueba
    que el juez NUNCA vio. Sin este split, evaluar el juez contra sus propias
    needles da precision=recall=1.0 por construcción -- no mide nada, solo
    confirma que una cadena se contiene a sí misma. Con el split, medimos si
    detectar una needle conocida generaliza a una paráfrasis nueva del MISMO
    spoiler, que es la pregunta real (y donde se espera que falle: por eso
    existe test_substring_judge_misses_paraphrase)."""
    all_labels = load_all_labels()
    data: list[tuple[str, SpoilerLabel, bool]] = []

    for i, (title_id, label) in enumerate(all_labels):
        paras = label.paraphrases
        split = max(1, len(paras) // 2)
        train, heldout = paras[:split], paras[split:]
        stripped = SpoilerLabel(
            id=label.id,
            canonical=label.canonical,
            paraphrases=train,
            severity=label.severity,
        )

        # Positivos reales: paráfrasis que el juez NO tiene como needle.
        for h in heldout:
            data.append((h, stripped, True))

        # Negativos: paráfrasis de un label distinto (no debería activarse).
        other = all_labels[(i + 7) % len(all_labels)]
        if other[1].paraphrases:
            data.append((other[1].paraphrases[0], stripped, False))

    # Negativos adicionales: frases neutras de marketing genérico.
    for i, sentence in enumerate(NEUTRAL_SENTENCES):
        _, label = all_labels[i * 3 % len(all_labels)]
        stripped = SpoilerLabel(
            id=label.id,
            canonical=label.canonical,
            paraphrases=label.paraphrases[: max(1, len(label.paraphrases) // 2)],
            severity=label.severity,
        )
        data.append((sentence, stripped, False))

    return data


def main() -> int:
    dataset = build_dataset()
    cal = calibrate(SubstringJudge(), dataset)
    result = cal.summary()
    result["tp"], result["fp"], result["tn"], result["fn"] = cal.tp, cal.fp, cal.tn, cal.fn
    result["nota"] = (
        "Calibrado contra el propio ground truth del proyecto (paráfrasis "
        "LLM vs LLM), no contra TV Tropes Movies ni IMDB Spoiler Dataset "
        "(ambos requieren descarga autenticada). No es una calibración "
        "independiente."
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = ROOT / "evals" / "results" / "substring_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
