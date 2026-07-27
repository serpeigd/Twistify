"""Pre-Show Reels — app local. Gratis, sin red, sin ANTHROPIC_API_KEY.

QUÉ HACE VISIBLE (esto es una pieza de portfolio, no un blog de cine):

1. Partición de contexto real (D3). Los campos post-visionado no se ocultan
   con CSS: no salen del servidor hasta que el cliente declara haber visto la
   obra. Abrir devtools no revela nada.
2. Fundamentación por afirmación. Cada dato factual muestra si tiene fuente
   o no. Los huecos son visibles a propósito.
3. Verificación de fuga en vivo, con su barra de error. El contenido
   pre-visionado se pasa por el mismo detector que usa `evals/run_eval.py`,
   y se reporta junto al recall MEDIDO del juez -- porque "0 fugas" con un
   juez de recall 0.0 no significa "no hay fugas".

Correr:
    pip install fastapi "uvicorn[standard]"
    python webapp/app.py
    # http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

import yaml
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from judge import SubstringJudge  # noqa: E402
from preshow.content import ContentPack, load_all  # noqa: E402
from preshow.schemas import SpoilerLabel, TitleCase  # noqa: E402

DATA = ROOT / "evals" / "dataset"
CONTENT_DIR = ROOT / "content" / "curated"
COMMENTS_FILE = ROOT / "content" / "comments.json"
CALIBRATION_FILE = ROOT / "evals" / "results" / "substring_calibration.json"

app = FastAPI(title="Pre-Show Reels")
judge = SubstringJudge()


def load_cases() -> dict[str, TitleCase]:
    raw = yaml.safe_load((DATA / "titles.yaml").read_text(encoding="utf-8"))
    return {f["title_id"]: TitleCase(kind="film", **f) for f in raw["films"]}


def load_labels(title_id: str) -> list[SpoilerLabel]:
    p = DATA / "spoilers" / f"{title_id}.yaml"
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [SpoilerLabel(**l) for l in (raw.get("labels") or [])]


def load_calibration() -> dict | None:
    if CALIBRATION_FILE.exists():
        return json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
    return None


CASES = load_cases()
PACKS: dict[str, ContentPack] = load_all(CONTENT_DIR) if CONTENT_DIR.exists() else {}
CALIBRATION = load_calibration()


def verify_pre_show(pack: ContentPack, labels: list[SpoilerLabel]) -> dict:
    """Pasa la superficie pre-visionado por el detector de fugas.

    NOTA (rediseño): la UI ya NO muestra este veredicto al usuario. Un badge
    "sin fugas" comunica una garantía que la medición no respalda: el juez
    tiene recall 0.0 (no ve paráfrasis). Presentarlo en verde era engañoso.
    El endpoint se mantiene para auditar el pipeline (evals/, /api/stats),
    no para venderlo en la ficha. El campo `judge_recall` acompaña siempre al
    veredicto para que nadie lo lea sin su barra de error.
    """
    hits = []
    for loc, text in pack.pre_show_text():
        for label in labels:
            if judge.entails(text, label):
                hits.append(
                    {
                        "label_id": label.id,
                        "severity": label.severity,
                        "where": loc,
                        "evidence": text[:200],
                    }
                )
    return {
        "leaks": hits,
        "leaked": bool(hits),
        "core_leaked": any(h["severity"] == "core" for h in hits),
        "n_labels_checked": len(labels),
        "n_surface_blocks": len(pack.pre_show_text()),
        "judge": judge.name,
        "judge_recall": (CALIBRATION or {}).get("recall"),
        "judge_n": (CALIBRATION or {}).get("n"),
    }


# ---------------------------------------------------------------- comentarios


def read_comments() -> dict[str, list[dict]]:
    if COMMENTS_FILE.exists():
        return json.loads(COMMENTS_FILE.read_text(encoding="utf-8"))
    return {}


def write_comments(data: dict) -> None:
    COMMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    COMMENTS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ------------------------------------------------------------------ endpoints


@app.get("/api/catalogue")
def api_catalogue():
    out = []
    for tid, case in CASES.items():
        pack = PACKS.get(tid)
        out.append(
            {
                "title_id": tid,
                "title": case.title,
                "year": case.year,
                "stratum": case.stratum,
                "notes": case.notes,
                "curated": pack is not None,
                "completeness": pack.completeness()["pct"] if pack else 0,
                "n_spoilers": len(load_labels(tid)),
                "director": pack.director if pack else None,
                "themes": pack.themes if pack else [],
                "awards_count": len(pack.critical_consensus.awards) if pack else 0,
            }
        )
    return sorted(out, key=lambda x: (-x["completeness"], x["title"]))


@app.get("/api/film/{title_id}")
def api_film(title_id: str, seen: bool = False):
    case = CASES.get(title_id)
    if case is None:
        raise HTTPException(404, f"Título desconocido: {title_id}")

    pack = PACKS.get(title_id)
    labels = load_labels(title_id)

    if pack is None:
        return {
            "case": case.model_dump(),
            "curated": False,
            "content": None,
            "verification": None,
            "grounding": None,
            "completeness": None,
            "seen": seen,
        }

    needs, grounded = pack.grounding()
    return {
        "case": case.model_dump(),
        "curated": True,
        "content": pack.public_dump(seen=seen),
        "verification": verify_pre_show(pack, labels),
        "grounding": {
            "needs_source": needs,
            "has_source": grounded,
            "pct": round(100 * grounded / needs) if needs else None,
        },
        "completeness": pack.completeness(),
        "seen": seen,
        "n_comments": len(read_comments().get(title_id, [])),
    }


def _public_comment(c: dict, owner_token: str) -> dict:
    return {
        "id": c["id"],
        "author": c["author"],
        "text": c["text"],
        "at": c["at"],
        "spoilers": c["spoilers"],
        "edited": bool(c.get("edited")),
        "mine": bool(owner_token) and c.get("owner_token") == owner_token,
    }


@app.get("/api/film/{title_id}/comments")
def api_get_comments(title_id: str, owner_token: str = ""):
    data = read_comments()
    lst = data.get(title_id, [])
    changed = False
    for c in lst:
        if "id" not in c:
            c["id"] = uuid.uuid4().hex[:12]
            changed = True
    if changed:
        write_comments(data)
    return [_public_comment(c, owner_token) for c in lst]


@app.post("/api/film/{title_id}/comments")
def api_post_comment(title_id: str, payload: dict = Body(...)):
    text = (payload.get("text") or "").strip()
    author = (payload.get("author") or "anónimo").strip()[:40]
    owner_token = (payload.get("owner_token") or "").strip()[:64]
    if not text:
        raise HTTPException(400, "Comentario vacío")
    if title_id not in CASES:
        raise HTTPException(404, "Título desconocido")

    data = read_comments()
    entry = {
        "id": uuid.uuid4().hex[:12],
        "author": author,
        "text": text[:2000],
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spoilers": bool(payload.get("spoilers")),
        "owner_token": owner_token,
    }
    data.setdefault(title_id, []).append(entry)
    write_comments(data)
    return _public_comment(entry, owner_token)


@app.put("/api/film/{title_id}/comments/{comment_id}")
def api_edit_comment(title_id: str, comment_id: str, payload: dict = Body(...)):
    text = (payload.get("text") or "").strip()
    owner_token = (payload.get("owner_token") or "").strip()
    if not text:
        raise HTTPException(400, "Comentario vacío")

    data = read_comments()
    for c in data.get(title_id, []):
        if c.get("id") == comment_id:
            if not owner_token or c.get("owner_token") != owner_token:
                raise HTTPException(403, "No puedes editar este comentario")
            c["text"] = text[:2000]
            c["edited"] = True
            write_comments(data)
            return _public_comment(c, owner_token)
    raise HTTPException(404, "Comentario no encontrado")


@app.delete("/api/film/{title_id}/comments/{comment_id}")
def api_delete_comment(title_id: str, comment_id: str, owner_token: str = ""):
    data = read_comments()
    lst = data.get(title_id, [])
    for i, c in enumerate(lst):
        if c.get("id") == comment_id:
            if not owner_token or c.get("owner_token") != owner_token:
                raise HTTPException(403, "No puedes eliminar este comentario")
            lst.pop(i)
            write_comments(data)
            return {"ok": True}
    raise HTTPException(404, "Comentario no encontrado")


@app.get("/api/stats")
def api_stats():
    """Panel global. La métrica que importa no es cuántas fichas hay, sino
    qué proporción del contenido factual está fundamentado."""
    curated = list(PACKS.values())
    total_needs = total_grounded = 0
    for p in curated:
        n, g = p.grounding()
        total_needs += n
        total_grounded += g

    leaked = 0
    for tid, pack in PACKS.items():
        if verify_pre_show(pack, load_labels(tid))["leaked"]:
            leaked += 1

    return {
        "n_titles": len(CASES),
        "n_curated": len(curated),
        "n_spoiler_labelled": sum(1 for t in CASES if load_labels(t)),
        "grounded_fact_rate": round(total_grounded / total_needs, 3) if total_needs else None,
        "n_sourced_claims": total_needs,
        "leakage_rate": round(leaked / len(curated), 3) if curated else None,
        "calibration": CALIBRATION,
        "avg_completeness": round(
            sum(p.completeness()["pct"] for p in curated) / len(curated)
        )
        if curated
        else 0,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    import os

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
