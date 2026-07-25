"""Runner. `python evals/run_eval.py --generator fake`

El generador se inyecta. El runner no sabe si detrás hay un LLM, un stub o
una versión futura del pipeline: por eso puedes comparar Hito 0 vs Hito 1 vs
Hito 2 con el MISMO código de medición. Si cambias el harness entre hitos,
tus comparaciones no valen nada.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import SubstringJudge  # noqa: E402
from metrics import aggregate, evaluate_case  # noqa: E402
from preshow.schemas import SpoilerLabel, TitleCase  # noqa: E402

DATA = ROOT / "evals" / "dataset"


def load_cases() -> list[TitleCase]:
    raw = yaml.safe_load((DATA / "titles.yaml").read_text())
    return [TitleCase(kind="film", **f) for f in raw["films"]]


def load_labels(title_id: str) -> list[SpoilerLabel]:
    p = DATA / "spoilers" / f"{title_id}.yaml"
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text()) or {}
    return [SpoilerLabel(**l) for l in (raw.get("labels") or [])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="fake")
    ap.add_argument("--out", default="evals/results/latest.json")
    args = ap.parse_args()

    cases = load_cases()
    labelled = [c for c in cases if load_labels(c.title_id)]

    # Puerta de calidad del dataset. Sin ground truth suficiente, la métrica
    # es ruido con dos decimales. Mejor fallar ruidosamente que reportar 0.0.
    if len(labelled) < 15:
        print(
            f"BLOQUEADO: solo {len(labelled)}/{len(cases)} títulos etiquetados.\n"
            f"Etiqueta al menos 15 antes de correr evals. Un leakage_rate\n"
            f"calculado sobre {len(labelled)} títulos no significa nada.",
            file=sys.stderr,
        )
        return 1

    if args.generator == "fake":
        print("El generador 'fake' solo sirve para tests. Implementa baseline.")
        return 1

    raise NotImplementedError(
        "TODO(tú): conecta aquí tu generador del Hito 0 y corre las 20 fichas."
    )


if __name__ == "__main__":
    raise SystemExit(main())
