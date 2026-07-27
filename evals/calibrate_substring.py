"""Calibration of SubstringJudge against a set derived from the project's
own ground truth (evals/dataset/spoilers/*.yaml), not against an external
benchmark.

Why not an external benchmark: TV Tropes Movies (Boyd-Graber et al. 2013)
has no direct download without contacting the authors; the IMDB Spoiler
Dataset (Misra) lives on Kaggle and requires an account/API key -- a user
credential this script must not request or manage. See the README, section
"Benchmarks to calibrate the judge."

What this measures instead: for each SpoilerLabel, its paraphrases and its
canonical are positives (they DO reveal that spoiler); paraphrases from
OTHER labels/titles are negatives (they shouldn't trigger it), plus a
handful of neutral, generic marketing sentences as additional negatives.

Explicit limitation: this is partial self-evaluation -- the paraphrases
were written by the same kind of system (an LLM) that generated the ground
truth (see D7 in docs/DESIGN.md). It gives you a real, reproducible number
with what's freely available, it does not replace calibration against an
independent benchmark. Don't report this as equivalent to that calibration.
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
    "A story set in a city that never sleeps.",
    "The cast includes several actors known for their previous work.",
    "The cinematography and score have received praise from critics.",
    "It was filmed in several locations during the year of production.",
    "It's a movie that mixes drama with moments of tension.",
    "The director had already worked in the same genre before.",
]


def load_all_labels() -> list[tuple[str, SpoilerLabel]]:
    out = []
    for p in sorted(DATA.glob("*.yaml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        for l in raw.get("labels") or []:
            out.append((p.stem, SpoilerLabel(**l)))
    return out


def build_dataset() -> list[tuple[str, SpoilerLabel, bool]]:
    """Train/held-out split per label: half of the paraphrases are used as
    the judge's 'needles' (what it already knows), the other half as test
    text the judge NEVER saw. Without this split, evaluating the judge
    against its own needles gives precision=recall=1.0 by construction --
    it measures nothing, it just confirms a string contains itself. With
    the split, we measure whether detecting a known needle generalizes to
    a new paraphrase of the SAME spoiler, which is the real question (and
    where it's expected to fail: that's why
    test_substring_judge_misses_paraphrase exists)."""
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

        # Real positives: paraphrases the judge does NOT have as a needle.
        for h in heldout:
            data.append((h, stripped, True))

        # Negatives: paraphrases from a different label (shouldn't trigger).
        other = all_labels[(i + 7) % len(all_labels)]
        if other[1].paraphrases:
            data.append((other[1].paraphrases[0], stripped, False))

    # Additional negatives: neutral, generic marketing sentences.
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
    result["note"] = (
        "Calibrated against the project's own ground truth (LLM "
        "paraphrases vs. LLM), not against TV Tropes Movies or the IMDB "
        "Spoiler Dataset (both require authenticated download). Not an "
        "independent calibration."
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    out_path = ROOT / "evals" / "results" / "substring_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
