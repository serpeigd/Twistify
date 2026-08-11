"""Validates `TrainedClassifierJudge` against the project's OWN domain,
not the one it was trained on -- the real gap this script exists to
check.

D15 (docs/DESIGN.md) calibrated this judge against the IMDB Spoiler
Dataset: real IMDb REVIEWS, informal first-person prose, often long. But
the judge's actual job is scoring `PreShowBrief` surface text: short
promotional bullets and voiceover lines, a genuinely different register.

FIRST ATTEMPT AT THIS SCRIPT REUSED calibrate_substring.py's
build_dataset() and got it wrong -- worth stating so the mistake doesn't
get repeated. That dataset's "negatives" are paraphrases of a DIFFERENT
movie's spoiler, used to test whether a PER-LABEL judge (Substring/LLM/
NLIJudge) wrongly matches label A's needle against label B's text. But
TrainedClassifierJudge doesn't do per-label matching (see its docstring
in evals/judge.py) -- it scores "is this spoiler-revealing text", full
stop. Movie B's spoiler IS spoiler-revealing text, just not for movie A
-- so it's a legitimate positive for THIS judge's actual task, not a
false positive. Scoring against the wrong dataset made the judge look
much worse than it is.

Correct dataset for this judge's actual task:
  - POSITIVES: every SpoilerLabel canonical + paraphrase across all 20
    titles (evals/dataset/spoilers/*.yaml) -- all of them, no per-label
    held-out split, since there's no "needle" concept here.
  - NEGATIVES: real, human-reviewed, genuinely spoiler-free pre-viewing
    text -- context_bullets/before_watching/why_now from the 8
    hand-researched titles (content/researched/*.json, D6/D7 cited-source
    ground truth) plus calibrate_substring.py's NEUTRAL_SENTENCES. This
    is real target-domain text, not invented for this script.

Same partial-self-evaluation caveat as calibrate_substring.py: labels
and researched content were LLM-drafted (with human review), not an
independent benchmark. Still a same-register check, which the external
IMDb calibration in D15 was not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from calibrate_substring import NEUTRAL_SENTENCES, load_all_labels  # noqa: E402
from judge import TrainedClassifierJudge  # noqa: E402
from stats import wilson_interval  # noqa: E402

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RESEARCHED_DIR = ROOT / "content" / "researched"


def build_positives() -> list[str]:
    out = []
    for _title_id, label in load_all_labels():
        out.append(label.canonical)
        out.extend(label.paraphrases)
    return out


def build_negatives() -> list[str]:
    out = list(NEUTRAL_SENTENCES)
    for p in sorted(RESEARCHED_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out.extend(d.get("context_bullets") or [])
        out.extend(b.get("text", "") for b in (d.get("before_watching") or []))
        if d.get("why_now"):
            out.append(d["why_now"])
    return [t for t in out if t.strip()]


def main() -> int:
    positives = build_positives()
    negatives = build_negatives()
    dataset = [(t, True) for t in positives] + [(t, False) for t in negatives]
    print(f"{len(dataset)} examples ({len(positives)} real spoiler sentences, "
          f"{len(negatives)} real spoiler-free pre-viewing sentences)")

    judge = TrainedClassifierJudge.from_artifact()
    scored = [(judge.spoiler_prob(text), truth, text) for text, truth in dataset]

    sweep = []
    for th in THRESHOLDS:
        tp = fp = tn = fn = 0
        for prob, truth, _text in scored:
            pred = prob >= th
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

    default_threshold = 0.3
    false_positives = [t for p, truth, t in scored if p >= default_threshold and not truth]
    false_negatives = [t for p, truth, t in scored if p < default_threshold and truth]

    result = {
        "judge": "trained-classifier (validated against IN-DOMAIN brief-style text, not IMDb reviews)",
        "source": "positives: all 277 SpoilerLabel canonicals/paraphrases across this project's "
        "20 titles; negatives: real spoiler-free pre-viewing text from the 8 hand-researched "
        "titles (content/researched/) + calibrate_substring.py's neutral marketing sentences",
        "n": len(dataset),
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "threshold_sweep": sweep,
        "false_positives_at_default_threshold_0.3": false_positives,
        "false_negatives_at_default_threshold_0.3": false_negatives,
        "note": (
            "Task-matched to what TrainedClassifierJudge actually predicts (general "
            "is_spoiler, not per-label entailment) -- see this script's docstring for why "
            "an earlier version of this script (reusing calibrate_substring.py's "
            "cross-label-negative dataset) mismeasured this judge. Same "
            "partial-self-evaluation caveat as calibrate_substring.py (labels and "
            "researched content were LLM-drafted with human review, not an independent "
            "benchmark) -- but a same-register check, which D15's external IMDb "
            "calibration was not."
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = ROOT / "evals" / "results" / "trained_classifier_internal.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
