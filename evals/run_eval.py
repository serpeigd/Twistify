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

from judge import HybridJudge, LLMJudge, SimilarityJudge, SubstringJudge, TrainedClassifierJudge  # noqa: E402
from metrics import aggregate, evaluate_case  # noqa: E402
from preshow.schemas import SpoilerLabel, TitleCase  # noqa: E402

def _load_generator(name: str, model: str | None = None):
    if name == "baseline":
        from preshow.baseline import AnthropicBaselineGenerator

        return AnthropicBaselineGenerator()
    if name == "baseline-groq":
        from preshow.baseline_groq import GroqBaselineGenerator

        kwargs = {} if model is None else {"model": model}
        return GroqBaselineGenerator(**kwargs)
    if name == "retrieval-groq":
        # Milestone 1 (see docs/DESIGN.md D3): real GREEN-tier retrieval
        # instead of an empty corpus. Same measurement code as
        # baseline-groq -- the comparison between the two IS Milestone 1's
        # result. Needs network access to en.wikipedia.org in addition to
        # GROQ_API_KEY. Default model (llama-3.3-70b-versatile) has a
        # 100K TPD cap that's repeatedly blocked a full 20-title run --
        # --model llama-3.1-8b-instant has a 500K TPD cap (checked live
        # against Groq's own rate-limits page) and was mostly unused
        # today, since every run so far hit the 70B model specifically.
        from preshow.retrieval_groq import GroqRetrievalGenerator

        kwargs = {} if model is None else {"model": model}
        return GroqRetrievalGenerator(**kwargs)
    return None


def _load_judge(name: str, threshold: float | None):
    if name == "substring":
        return SubstringJudge()
    if name == "trained-classifier":
        # KNOWN BROKEN on this project's actual generator output as of the
        # day this was wired in -- see D15's correction in docs/DESIGN.md.
        # It's well-calibrated on full sentences (IMDb reviews, this
        # project's own SpoilerLabel paraphrases) but the baseline
        # generators write short, terse text in EVERY field, not just
        # script[] fragments -- a register this judge was never trained
        # or validated on. A live run scored leakage_rate=0.95 on
        # completely benign text (cast credits, stage directions,
        # production trivia) -- confirmed via --show-leaks it's not a
        # threshold problem, every flagged phrase landed in a narrow
        # 0.30-0.47 band regardless of content. DO NOT use this to report
        # a real leakage_rate -- use --judge hybrid instead.
        print(
            "WARNING: --judge trained-classifier misfires on this project's "
            "generator output (leakage_rate=0.95 in testing, confirmed to be "
            "judge noise, not real leaks -- see D15's correction in "
            "docs/DESIGN.md). Use --judge hybrid instead for a usable reading.",
            file=sys.stderr,
        )
        kwargs = {} if threshold is None else {"threshold": threshold}
        return TrainedClassifierJudge.from_artifact(**kwargs)
    if name == "hybrid":
        # ALSO NOT GOOD ENOUGH, confirmed live -- see D15's second
        # correction in docs/DESIGN.md. Routing short text (< 15 words)
        # to SubstringJudge cut leakage_rate from 0.95 to 0.6, but the
        # remaining flagged text is STILL essentially all false positives
        # (mood/theme/production sentences, not real spoilers) -- e.g. a
        # neutral CONTROL sentence invented for this check scored 0.252,
        # barely below the flagged ones. The classifier has no real
        # discriminating signal on this generator's writing style at ANY
        # length; routing by word count only changes how OFTEN it fires,
        # not whether its verdicts mean anything. Kept available for
        # comparison, not for reporting.
        print(
            "WARNING: --judge hybrid reduces but does NOT fix "
            "trained-classifier's misfiring (leakage_rate=0.6 in testing was "
            "still mostly false positives on mood/theme/production text, not "
            "real leaks -- see D15's second correction in docs/DESIGN.md). "
            "Try --judge similarity instead.",
            file=sys.stderr,
        )
        kwargs = {} if threshold is None else {"threshold": threshold}
        return HybridJudge(SubstringJudge(), TrainedClassifierJudge.from_artifact(**kwargs))
    if name == "similarity":
        # ALSO NOT GOOD ENOUGH, confirmed live -- see D16's final
        # correction in docs/DESIGN.md. Calibrated well on held-out
        # paraphrases (recall=0.87/precision=0.856), and did catch the
        # one confirmed leak it was built for (Los cronocrímenes), but a
        # live run immediately showed it can't separate real leaks from
        # generic movie content in THIS generator's writing style: a
        # false positive ("The film stars Bruce Willis as a child
        # psychologist...", the film's own public premise) scored 0.561
        # -- HIGHER than the confirmed real leak's 0.525. No threshold
        # separates those two. Sixth judge in this project to calibrate
        # well and still fail live -- see D16 for why judge iteration
        # stops here. Kept available for comparison only.
        print(
            "WARNING: --judge similarity also fails on this project's actual "
            "generator output despite good offline calibration -- a live run "
            "scored a false positive HIGHER than the one confirmed real leak "
            "it was built to catch (0.561 vs 0.525, no threshold separates "
            "them). See D16's final correction in docs/DESIGN.md. Judge "
            "iteration is closed -- use --judge substring (the default) for "
            "reporting.",
            file=sys.stderr,
        )
        kwargs = {} if threshold is None else {"threshold": threshold}
        return SimilarityJudge(**kwargs)
    if name == "llm":
        # The one calibrated alternative never tested against this
        # project's actual generator output. D13's external calibration
        # (real IMDb reviews) found recall~=0.35-0.4 with generous
        # truncation -- modest, but from genuine language understanding,
        # not bag-of-words, so it's a real question whether it
        # generalizes to short fragments better than TrainedClassifierJudge
        # did (it might not -- untested, that's the point of trying it).
        # Needs GROQ_API_KEY in .env. Paced/retried the same way
        # evals/calibrate_llm_external.py is, including its
        # DailyQuotaExhausted/ServiceUnavailable safety nets -- see
        # main()'s try/except around the per-case loop below.
        from calibrate_llm_external import DEFAULT_INTERVAL_S, DEFAULT_MODEL, make_groq_client_fn

        return LLMJudge(client_fn=make_groq_client_fn(DEFAULT_MODEL, DEFAULT_INTERVAL_S))
    raise ValueError(f"unknown judge: {name}")

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
    ap.add_argument(
        "--model",
        default=None,
        help="Groq model override for --generator baseline-groq/retrieval-groq "
        "(default: each generator's own default, llama-3.3-70b-versatile). "
        "Its 100K TPD cap has repeatedly blocked a full 20-title run --try "
        "llama-3.1-8b-instant (500K TPD, checked live on Groq's rate-limits "
        "page) if that keeps happening",
    )
    ap.add_argument("--out", default="evals/results/latest.json")
    ap.add_argument(
        "--judge",
        default="substring",
        choices=["substring", "trained-classifier", "hybrid", "similarity", "llm"],
        help="substring (default -- offline, known recall=0.0, but the only "
        "judge in this project confirmed NOT to produce false leak "
        "verdicts on real generator output; judge iteration is closed, "
        "see D16 in docs/DESIGN.md -- use --save-briefs + a human read "
        "for anything substring can't see, not a different judge). "
        "trained-classifier/hybrid/similarity/llm are all kept for "
        "comparison only -- each calibrated well offline and each failed "
        "live for a different reason (D15, D16); none fit to report a "
        "real leakage_rate",
    )
    ap.add_argument(
        "--judge-threshold",
        type=float,
        default=None,
        help="decision threshold for --judge trained-classifier (default: "
        "TrainedClassifierJudge's own default, 0.3 -- see D15)",
    )
    ap.add_argument(
        "--show-leaks",
        action="store_true",
        help="print each detected leak's location and evidence text to stderr -- "
        "without this, a high leakage_rate can't be diagnosed (which field, what "
        "text) from this script's output alone",
    )
    ap.add_argument(
        "--save-briefs",
        default=None,
        help="write every generated PreShowBrief in full to this JSON path, keyed by "
        "title_id -- a leakage_rate of 0.0 from a judge with a known blind spot "
        "(e.g. --judge substring, recall=0.0 by construction, D12) proves nothing "
        "on its own; this is what lets a human actually read the text and check "
        "for a leak no judge here can reliably see",
    )
    ap.add_argument(
        "--titles",
        default=None,
        help="comma-separated title_ids to process, instead of every labeled title -- "
        "for finishing a partial run (e.g. a slow paced generator hitting a "
        "rate limit) without re-spending quota on titles that already succeeded. "
        "Does NOT affect the >=15-labeled-titles gate below (still checked against "
        "the full dataset) -- the resulting aggregate here only covers the given "
        "titles, merge with a prior --save-briefs run by hand for the full picture",
    )
    args = ap.parse_args()

    cases = load_cases()
    labelled = [c for c in cases if load_labels(c.title_id)]

    # Dataset quality gate. Without enough ground truth, the metric is noise
    # with two decimal places. Better to fail loudly than report 0.0. Checked
    # against the FULL labelled set, before --titles filtering, on purpose --
    # see that flag's help text.
    if len(labelled) < 15:
        print(
            f"BLOCKED: only {len(labelled)}/{len(cases)} titles labeled.\n"
            f"Label at least 15 before running evals. A leakage_rate\n"
            f"computed over {len(labelled)} titles means nothing.",
            file=sys.stderr,
        )
        return 1

    if args.titles:
        wanted = set(args.titles.split(","))
        labelled = [c for c in labelled if c.title_id in wanted]
        missing = wanted - {c.title_id for c in labelled}
        if missing:
            print(f"WARNING: --titles named unknown/unlabeled title_ids: {sorted(missing)}", file=sys.stderr)

    if args.generator == "fake":
        print("The 'fake' generator is for tests only. Use --generator baseline.")
        return 1

    generator = _load_generator(args.generator, args.model)
    if generator is None:
        print(f"Unknown generator: {args.generator}", file=sys.stderr)
        return 1

    judge = _load_judge(args.judge, args.judge_threshold)
    results = []
    briefs: dict[str, dict] = {}
    stopped_early = False
    for case in labelled:
        try:
            labels = load_labels(case.title_id)
            brief = generator.pre_show(case, corpus=[])
            if args.save_briefs:
                briefs[case.title_id] = brief.model_dump()
            results.append(evaluate_case(brief, labels, case.stratum, judge))
        except Exception as e:  # noqa: BLE001 -- --judge llm can raise
            # calibrate_llm_external.EvaluationAborted (DailyQuotaExhausted/
            # ServiceUnavailable) mid-case; stop cleanly and aggregate
            # whatever completed instead of losing the whole run to a
            # crash with no output. Broad except is deliberate: this is
            # the harness's outermost loop, and ANY judge/generator
            # failure here should stop the run cleanly rather than lose
            # already-computed results, not just the two named exceptions.
            print(
                f"\nStopped after {len(results)}/{len(labelled)} cases: {type(e).__name__}: {e}\n"
                "Aggregating what completed instead of losing it to a crash.",
                file=sys.stderr,
            )
            stopped_early = True
            break
        print(f"  {case.title_id}: {'LEAK' if results[-1].leaked else 'ok'}", file=sys.stderr)
        if args.show_leaks:
            for hit in results[-1].leaks:
                print(f"    [{hit.severity}] {hit.where}: {hit.evidence!r}", file=sys.stderr)

    agg = aggregate(results)
    agg["judge"] = {"name": getattr(judge, "name", args.judge), "threshold": getattr(judge, "threshold", None)}
    agg["partial"] = stopped_early
    agg["n_cases_completed"] = len(results)

    if args.save_briefs:
        briefs_path = ROOT / args.save_briefs
        briefs_path.parent.mkdir(parents=True, exist_ok=True)
        briefs_path.write_text(json.dumps(briefs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {len(briefs)} briefs to {briefs_path}", file=sys.stderr)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(agg, indent=2, ensure_ascii=False))

    print(json.dumps(agg, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
