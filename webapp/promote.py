"""Gate between content/_drafts/ and content/researched/.

Until now promoting a draft was a manual copy plus an eyeball check, and that
is how The Odyssey shipped at 6 depth items (hand-researched titles: 20-26)
without any check noticing. This makes the bar explicit and repeatable:

  hard checks (any failure blocks the promotion)
    - schema: the draft parses as a ContentPack; a title outside titles.yaml
      also needs title/year (the app silently skips it otherwise)
    - depth: >= MIN_DEPTH items across the five open-ended fields
    - grounding: >= MIN_GROUNDED_PCT of claims that need a source have one
    - regression: depth doesn't drop vs. the version already promoted
    - leak floor: SubstringJudge finds no documented spoiler label in the
      pre-show surface. recall=0.0 (D12), so passing proves nothing -- it only
      catches the literal case.

  human review (the check that actually caught every real leak so far)
    - the pre-show surface is printed next to the title's documented spoiler
      labels; the write only happens with --reviewed, i.e. after someone has
      read both. No automated judge substitutes for this (D15/D16).

Promoting invalidates the title's cached ES translation: the app serves
content/_translations/ unconditionally, so a stale cache would keep showing
the OLD text in Spanish. Re-run webapp/prewarm_translations.py afterwards.

    python webapp/promote.py <title_id>              # checks + print review material
    python webapp/promote.py <title_id> --reviewed   # ... and write if all pass
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals"))

from judge import SubstringJudge  # noqa: E402
from preshow.content import ContentPack  # noqa: E402
from preshow.schemas import SpoilerLabel  # noqa: E402

DRAFTS_DIR = ROOT / "content" / "_drafts"
RESEARCHED_DIR = ROOT / "content" / "researched"
TRANSLATIONS_DIR = ROOT / "content" / "_translations"
TITLES_PATH = ROOT / "evals" / "dataset" / "titles.yaml"
SPOILERS_DIR = ROOT / "evals" / "dataset" / "spoilers"
PROMOTIONS_LOG = ROOT / "content" / "promotions.jsonl"  # committed audit trail (evals/results/ is gitignored)

# The audit that found the thin packs flagged <=10 and the hand-researched
# titles run 20-26. 12 sits just above the thin cluster (5-9 items) and
# below the weakest title that was accepted as fine (Barbie / Anatomy of a
# Fall, 12); WARN_DEPTH is where "near hand-researched" starts.
DEPTH_FIELDS = ("metaphors", "intertextual_refs", "production_trivia", "scene_analysis", "fun_facts")
MIN_DEPTH = 12
WARN_DEPTH = 18
MIN_GROUNDED_PCT = 90


@dataclass
class Report:
    title_id: str
    depth: int = 0
    prior_depth: int | None = None
    grounded: int = 0
    needs_source: int = 0
    leaks: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def depth(pack: dict) -> int:
    return sum(len(pack.get(k) or []) for k in DEPTH_FIELDS)


def known_title_ids(titles_path: Path = TITLES_PATH) -> set[str]:
    raw = yaml.safe_load(titles_path.read_text(encoding="utf-8"))
    return {f["title_id"] for f in raw["films"]}


def load_labels(title_id: str, spoilers_dir: Path = SPOILERS_DIR) -> list[SpoilerLabel]:
    p = spoilers_dir / f"{title_id}.yaml"
    if not p.exists():
        return []
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return [SpoilerLabel(**label) for label in (raw.get("labels") or [])]


def evaluate(
    draft: dict,
    labels: list[SpoilerLabel],
    *,
    in_titles_yaml: bool,
    existing: dict | None = None,
) -> Report:
    """Pure: runs every hard check, touches no files."""
    r = Report(title_id=draft.get("title_id", "?"))

    try:
        pack = ContentPack(**draft)
    except Exception as e:  # noqa: BLE001 -- pydantic's message is the useful part
        r.failures.append(f"schema: draft doesn't parse as a ContentPack ({e})")
        return r

    if not in_titles_yaml and not (pack.title and pack.year):
        r.failures.append("schema: title/year missing and the title isn't in titles.yaml -- the app would skip it")

    r.depth = depth(draft)
    if r.depth < MIN_DEPTH:
        r.failures.append(f"depth: {r.depth} items across {len(DEPTH_FIELDS)} fields, need >= {MIN_DEPTH}")
    elif r.depth < WARN_DEPTH:
        r.warnings.append(f"depth: {r.depth} is above the floor but under {WARN_DEPTH} (hand-researched titles run 20-26)")

    r.needs_source, r.grounded = pack.grounding()
    if r.needs_source == 0:
        r.failures.append("grounding: no claims that need a source -- nothing to verify")
    elif 100 * r.grounded / r.needs_source < MIN_GROUNDED_PCT:
        r.failures.append(f"grounding: {r.grounded}/{r.needs_source} sourced, need >= {MIN_GROUNDED_PCT}%")

    if existing is not None:
        r.prior_depth = depth(existing)
        if r.depth < r.prior_depth:
            r.failures.append(f"regression: depth {r.depth} < {r.prior_depth} in the version already promoted")

    if not labels:
        r.warnings.append("leak floor: no documented spoiler labels for this title -- nothing was checked")
    judge = SubstringJudge()
    for loc, text in pack.pre_show_text():
        for label in labels:
            if judge.entails(text, label):
                r.leaks.append({"label_id": label.id, "severity": label.severity, "where": loc})
    if r.leaks:
        r.failures.append(f"leak floor: {len(r.leaks)} literal spoiler hit(s) in the pre-show surface: {r.leaks}")

    return r


def review_material(draft: dict, labels: list[SpoilerLabel]) -> str:
    """What a reviewer has to read before saying --reviewed: everything a
    viewer sees BEFORE declaring they've seen the film, next to the spoilers
    that must not be in it."""
    pack = ContentPack(**draft)
    out = ["--- PRE-SHOW SURFACE (must be spoiler-free) ---"]
    out += [f"[{loc}] {text}" for loc, text in pack.pre_show_text()]
    out.append("--- DOCUMENTED SPOILERS (must not be inferable from the above) ---")
    out += [f"[{label.severity}] {label.canonical}" for label in labels] or ["(none documented)"]
    return "\n".join(out)


def promote(
    title_id: str,
    *,
    reviewed: bool,
    dry_run: bool = False,
    allow_regression: bool = False,
    drafts_dir: Path = DRAFTS_DIR,
    researched_dir: Path = RESEARCHED_DIR,
    translations_dir: Path = TRANSLATIONS_DIR,
    log_path: Path = PROMOTIONS_LOG,
    in_titles_yaml: bool | None = None,
    labels: list[SpoilerLabel] | None = None,
) -> tuple[Report, str]:
    """Returns (report, outcome) where outcome is one of
    'promoted', 'dry-run', 'needs-review', 'blocked'."""
    draft_path = drafts_dir / f"{title_id}.json"
    live_path = researched_dir / f"{title_id}.json"
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    existing = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else None
    if in_titles_yaml is None:
        in_titles_yaml = title_id in known_title_ids()
    if labels is None:
        labels = load_labels(title_id)

    report = evaluate(draft, labels, in_titles_yaml=in_titles_yaml, existing=existing)
    if allow_regression:
        report.failures = [f for f in report.failures if not f.startswith("regression:")]

    if not report.passed:
        return report, "blocked"
    if dry_run:
        return report, "dry-run"
    if not reviewed:
        return report, "needs-review"

    researched_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(draft_path, live_path)
    stale = translations_dir / f"{title_id}.json"
    if stale.exists():
        stale.unlink()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "title_id": title_id,
            "depth": report.depth,
            "prior_depth": report.prior_depth,
            "grounded": report.grounded,
            "needs_source": report.needs_source,
            "n_labels": len(labels),
            "warnings": report.warnings,
        }) + "\n")
    return report, "promoted"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console chokes on non-cp1252 text
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("title_id")
    ap.add_argument("--reviewed", action="store_true",
                    help="I read the pre-show surface against the documented spoilers")
    ap.add_argument("--dry-run", action="store_true", help="run the checks, write nothing")
    ap.add_argument("--allow-regression", action="store_true",
                    help="permit promoting a draft shallower than the live version")
    args = ap.parse_args()

    if not (DRAFTS_DIR / f"{args.title_id}.json").exists():
        print(f"no draft at {DRAFTS_DIR / (args.title_id + '.json')}", file=sys.stderr)
        return 1

    report, outcome = promote(
        args.title_id, reviewed=args.reviewed, dry_run=args.dry_run,
        allow_regression=args.allow_regression,
    )
    prior = f" (live: {report.prior_depth})" if report.prior_depth is not None else ""
    print(f"{report.title_id}: depth {report.depth}{prior}, grounded {report.grounded}/{report.needs_source}")
    for w in report.warnings:
        print(f"  warn: {w}")
    for f in report.failures:
        print(f"  FAIL: {f}")

    if outcome == "blocked":
        return 1
    if outcome in ("needs-review", "dry-run"):
        draft = json.loads((DRAFTS_DIR / f"{args.title_id}.json").read_text(encoding="utf-8"))
        print(review_material(draft, load_labels(args.title_id)))
        if outcome == "needs-review":
            print("\nchecks passed; not written. Re-run with --reviewed after reading the above.")
            return 2
        return 0
    print(f"promoted -> content/researched/{args.title_id}.json "
          f"(stale ES cache removed; re-run webapp/prewarm_translations.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
