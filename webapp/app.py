"""App web local de DEMO. Gratis, sin red, sin ANTHROPIC_API_KEY.

Muestra, por cada uno de los 20 títulos ya etiquetados, un PreShowBrief
generado por plantilla (DemoGenerator, ver src/preshow/demo_generator.py) y
lo evalúa en vivo contra el ground truth real (evals/dataset/spoilers) con
el mismo harness que usa run_eval.py -- SubstringJudge, evaluate_case.

Esto NO es el sistema real (ese necesita baseline.py + retrieval real y sí
cuesta crédito). Es la parte gratis del proyecto hecha visible.

Correr:
    pip install fastapi "uvicorn[standard]"
    python webapp/app.py
    # abre http://127.0.0.1:8000
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from judge import SubstringJudge  # noqa: E402
from metrics import evaluate_case  # noqa: E402
from preshow.demo_generator import DemoGenerator  # noqa: E402
from preshow.schemas import SpoilerLabel, TitleCase  # noqa: E402

DATA = ROOT / "evals" / "dataset"

app = FastAPI(title="Pre-Show Reels — demo gratis")
generator = DemoGenerator()
judge = SubstringJudge()


def load_cases() -> list[TitleCase]:
    raw = yaml.safe_load((DATA / "titles.yaml").read_text(encoding="utf-8"))
    return [TitleCase(kind="film", **f) for f in raw["films"]]


def load_labels(title_id: str) -> list[SpoilerLabel]:
    p = DATA / "spoilers" / f"{title_id}.yaml"
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [SpoilerLabel(**l) for l in (raw.get("labels") or [])]


_CASES = {c.title_id: c for c in load_cases()}


@app.get("/api/titles")
def api_titles():
    return [
        {
            "title_id": c.title_id,
            "title": c.title,
            "year": c.year,
            "stratum": c.stratum,
        }
        for c in _CASES.values()
    ]


@app.get("/api/brief/{title_id}")
def api_brief(title_id: str):
    case = _CASES.get(title_id)
    if case is None:
        raise HTTPException(404, f"Título desconocido: {title_id}")

    labels = load_labels(title_id)
    brief = generator.pre_show(case, corpus=[])
    result = evaluate_case(brief, labels, case.stratum, judge)

    return {
        "case": {"title": case.title, "year": case.year, "stratum": case.stratum},
        "brief": brief.model_dump(),
        "eval": {
            "leaked": result.leaked,
            "core_leaked": result.core_leaked,
            "n_facts": result.n_facts,
            "n_grounded_facts": result.n_grounded_facts,
            "n_claims": result.n_claims,
            "n_labels_checked": len(labels),
            "leaks": [l.__dict__ for l in result.leaks],
        },
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
