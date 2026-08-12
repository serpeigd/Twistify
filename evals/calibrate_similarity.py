"""Calibrates `SimilarityJudge` (evals/judge.py) -- built after D16's
human spot-check of Milestone 1 found a real leak (Los cronocrímenes:
"his other selves") that both `SubstringJudge` (D12, zero shared
substring with the documented paraphrase) and a quick TF-IDF cosine
check (0.023 similarity, no real signal) missed. A bi-encoder sentence
embedding, tested by hand against that exact pair, showed real
separation (0.525 for the true match vs. 0.035-0.19 for clean/unrelated
text). This script turns that one-off check into a proper calibration.

Reuses `calibrate_substring.py`'s `build_dataset()` on purpose, not a
new dataset: it's the RIGHT methodology for this judge's framing
(per-label, like SubstringJudge -- unlike TrainedClassifierJudge's
label-agnostic "is this spoiler-y text" design, D15). That function
already does the correct thing for a per-label judge: half of each
label's paraphrases are held out as genuine positives (text the judge
never saw as a "needle"), so this measures whether recognizing one
phrasing of a spoiler generalizes to a genuinely different phrasing of
the SAME spoiler -- exactly the question a leak like Los cronocrímenes'
raises. Same partial-self-evaluation caveat that script documents
(paraphrases are LLM-written, same kind of system as the ground truth,
D7) -- not an independent benchmark, but the same one already used to
calibrate SubstringJudge, so the two numbers are directly comparable.

Needs `pip install sentence-transformers` (already a dependency from
D15's NLIJudge experiment). Unlike NLIJudge, this should run fast even
on this project's dev hardware: bi-encoder embeddings are computed once
per text and compared by cosine similarity, not reprocessed per pair
through a cross-encoder -- see SimilarityJudge's docstring for why that
distinction mattered here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from calibrate_substring import build_dataset  # noqa: E402
from judge import DEFAULT_SIMILARITY_MODEL, SimilarityJudge  # noqa: E402
from stats import wilson_interval  # noqa: E402

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main() -> int:
    dataset = build_dataset()
    n_pos = sum(1 for _, _, truth in dataset if truth)
    print(f"{len(dataset)} examples ({n_pos} positive, {len(dataset) - n_pos} negative), "
          "same set used to internally calibrate SubstringJudge")

    judge = SimilarityJudge(model_name=DEFAULT_SIMILARITY_MODEL)
    print(f"scoring with {DEFAULT_SIMILARITY_MODEL}...")
    scored = []
    for i, (text, label, truth) in enumerate(dataset):
        scored.append((judge.max_similarity(text, label), truth, text))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(dataset)}", flush=True)

    sweep = []
    for th in THRESHOLDS:
        tp = fp = tn = fn = 0
        for sim, truth, _text in scored:
            pred = sim >= th
            if pred and truth:
                tp += 1
            elif pred and not truth:
                fp += 1
            elif not pred and truth:
                fn += 1
            else:
                tn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision_ci = wilson_interval(tp, tp + fp)
        recall_ci = wilson_interval(tp, tp + fn)
        sweep.append(
            {
                "threshold": th,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": round(precision, 3),
                "precision_ci_95": [round(v, 3) for v in precision_ci] if precision_ci else None,
                "recall": round(recall, 3),
                "recall_ci_95": [round(v, 3) for v in recall_ci] if recall_ci else None,
            }
        )

    result = {
        "judge": f"similarity ({DEFAULT_SIMILARITY_MODEL})",
        "source": "evals/calibrate_substring.py's build_dataset() -- this project's own 20 "
        "titles' held-out SpoilerLabel paraphrases + neutral marketing sentences",
        "n": len(dataset),
        "threshold_sweep": sweep,
        "note": (
            "Same dataset/methodology as SubstringJudge's own internal calibration "
            "(calibrate_substring.py) -- directly comparable to that judge's numbers. "
            "*_ci_95 fields are 95% Wilson score intervals (Wilson, 1927; evals/stats.py)."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = ROOT / "evals" / "results" / "similarity_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
