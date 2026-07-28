"""Runner. `python evals/run_eval.py --generator fake`

The generator gets injected. The runner doesn't know whether there's an
LLM, a stub, or a future version of the pipeline behind it: that's why you
can compare Milestone 0 vs Milestone 1 vs Milestone 2 with the SAME
measurement code. If you change the harness between milestones, your
comparisons are worthless.
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

def _load_generator(name: str):
    if name == "baseline":
        from preshow.baseline import AnthropicBaselineGenerator

        return AnthropicBaselineGenerator()
    if name == "baseline-groq":
        from preshow.baseline_groq import GroqBaselineGenerator

        return GroqBaselineGenerator()
    return None

DATA = ROOT / "evals" / "dataset"


def load_cases() -> list[TitleCase]:
    raw = yaml.safe_load((DATA / "titles.yaml").read_text(encoding="utf-8"))
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

    # Dataset quality gate. Without enough ground truth, the metric is noise
    # with two decimal places. Better to fail loudly than report 0.0.
    if len(labelled) < 15:
        print(
            f"BLOCKED: only {len(labelled)}/{len(cases)} titles labeled.\n"
            f"Label at least 15 before running evals. A leakage_rate\n"
            f"computed over {len(labelled)} titles means nothing.",
            file=sys.stderr,
        )
        return 1

    if args.generator == "fake":
        print("The 'fake' generator is for tests only. Use --generator baseline.")
        return 1

    generator = _load_generator(args.generator)
    if generator is None:
        print(f"Unknown generator: {args.generator}", file=sys.stderr)
        return 1

    judge = SubstringJudge()
    results = []
    for case in labelled:
        labels = load_labels(case.title_id)
        brief = generator.pre_show(case, corpus=[])
        results.append(evaluate_case(brief, labels, case.stratum, judge))
        print(f"  {case.title_id}: {'LEAK' if results[-1].leaked else 'ok'}", file=sys.stderr)

    agg = aggregate(results)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    print(json.dumps(agg, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
