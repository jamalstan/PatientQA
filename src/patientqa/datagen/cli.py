"""Command-line interface for the synthetic data pipeline.

    uv run python -m patientqa.datagen generate --count 60 --seed 2026 --out manifest.jsonl
    uv run python -m patientqa.datagen starters manifest.jsonl --limit 8
    uv run python -m patientqa.datagen validate manifest.jsonl
    uv run python -m patientqa.datagen taxonomy
    uv run python -m patientqa.datagen research10 --out manifests/research10.jsonl
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from patientqa.datagen.pipeline import PipelineConfig, generate_manifest
from patientqa.datagen.schemas import ManifestEntry, parse_manifest_line, parse_starters_line
from patientqa.datagen.seeds import load_bundled_seedbank
from patientqa.datagen.starters import StartersConfig, generate_starters
from patientqa.datagen.taxonomy import OBJECTIVE_CLASSES, TEMPLATES
from patientqa.datagen.validate import validate_manifest


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="python -m patientqa.datagen",
        description="Synthetic patient/scenario generation (DESIGN.md §6-§7)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a manifest.jsonl")
    generate.add_argument("-n", "--count", type=int, default=60, help="number of calls")
    generate.add_argument("--seed", type=int, default=2026, help="base RNG seed")
    generate.add_argument(
        "--out", type=Path, default=Path("manifest.jsonl"), help="output JSONL path"
    )

    research = subparsers.add_parser(
        "research10", help="write the curated high-value 10-call research campaign"
    )
    research.add_argument("--seed", type=int, default=20260820, help="base RNG seed")
    research.add_argument(
        "--out",
        type=Path,
        default=Path("manifests/research10.jsonl"),
        help="output JSONL path",
    )
    generate.add_argument(
        "--elaboration",
        choices=("auto", "llm", "template"),
        default="auto",
        help="auto = LLM if a Cerebras key is configured, else template",
    )
    generate.add_argument(
        "--seed-source",
        choices=("bundled", "hf"),
        default="bundled",
        help="bundled seed files or Hugging Face enrichment (needs: uv add datasets)",
    )
    generate.add_argument("--model", default="gpt-oss-120b", help="Cerebras model id")
    generate.add_argument(
        "--retries", type=int, default=3, help="redraw attempts per failing entry"
    )

    validate = subparsers.add_parser("validate", help="validate an existing manifest")
    validate.add_argument("path", type=Path, help="manifest.jsonl to check")

    starters = subparsers.add_parser(
        "starters", help="generate conversation starters for an existing manifest"
    )
    starters.add_argument("path", type=Path, help="manifest.jsonl to read")
    starters.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSONL (default: <manifest>.starters.jsonl beside the manifest)",
    )
    starters.add_argument("--count", type=int, default=3, help="candidate openers per call")
    starters.add_argument(
        "--limit", type=int, default=None, help="only the first N manifest entries"
    )
    starters.add_argument(
        "--elaboration",
        choices=("auto", "llm", "template"),
        default="auto",
        help="auto = LLM if a Cerebras key is configured, else template",
    )
    starters.add_argument("--model", default="gpt-oss-120b", help="Cerebras model id")
    starters.add_argument(
        "--retries", type=int, default=2, help="LLM retries per entry before template fallback"
    )
    starters.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="seconds between LLM calls (free-tier rate-limit spacing)",
    )

    subparsers.add_parser("taxonomy", help="list objective classes and templates")

    args = parser.parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    if args.command == "research10":
        return _research10(args)
    if args.command == "starters":
        return _starters(args)
    if args.command == "validate":
        return _validate(args.path)
    return _taxonomy()


def _generate(args: argparse.Namespace) -> int:
    config = PipelineConfig(
        count=args.count,
        base_seed=args.seed,
        seed_source=args.seed_source,
        elaboration=args.elaboration,
        out=args.out,
        retries=args.retries,
        model=args.model,
    )
    report = generate_manifest(config)

    print(f"wrote {report.generated}/{report.requested} entries to {config.out}")
    print(f"  base seed      : {report.base_seed}")
    print(f"  elaboration    : {report.elaboration_counts}")
    print(f"  class coverage : {report.class_coverage}")
    print(f"  behaviors      : {report.behavior_coverage}")
    print(f"  techniques     : {report.technique_coverage}")
    if report.template_fallbacks:
        print(f"  template fallbacks (slots): {report.template_fallbacks}")
    for drop in report.drops:
        print(f"  dropped slot {drop.index}: {drop.reason}")
    print(f"  report         : {config.out.with_suffix('.report.json')}")
    if report.generated == 0:
        print("generated nothing — check the drop reasons above", file=sys.stderr)
        return 1
    return 0


def _research10(args: argparse.Namespace) -> int:
    from patientqa.datagen.research_campaign import write_research_10

    entries = write_research_10(args.out, base_seed=args.seed)
    starter_args = argparse.Namespace(
        path=args.out,
        out=args.out.parent / f"{args.out.stem}.starters.jsonl",
        count=3,
        limit=None,
        elaboration="template",
        model="gpt-oss-120b",
        retries=0,
        pause=0.0,
    )
    result = _starters(starter_args)
    if result:
        return result
    print(f"\ncurated research campaign: {len(entries)} isolated behaviors")
    for entry in entries:
        print(f"  {entry.call_id}: {entry.test_intent.behavior}")
    return 0


def _load_manifest(path: Path) -> list[ManifestEntry] | None:
    """Parse a manifest JSONL; prints an actionable error and returns None if bad."""
    entries: list[ManifestEntry] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(parse_manifest_line(line))
        except ValueError as exc:
            print(f"{path}:{line_number}: {exc}", file=sys.stderr)
            return None
    return entries


def _starters(args: argparse.Namespace) -> int:
    entries = _load_manifest(args.path)
    if entries is None:
        return 1
    if not entries:
        print(f"{args.path} contains no entries", file=sys.stderr)
        return 1
    if args.limit is not None:
        entries = entries[: max(0, args.limit)]

    out = args.out or args.path.parent / f"{args.path.stem}.starters.jsonl"
    config = StartersConfig(
        count=args.count,
        elaboration=args.elaboration,
        model=args.model,
        out=out,
        retries=args.retries,
        pause_seconds=args.pause,
    )
    report = generate_starters(entries, config)

    print(f"wrote {report.generated}/{report.requested} starter sets to {config.out}")
    print(f"  elaboration    : {report.elaboration_counts}")
    for fallback in report.fallbacks:
        print(f"  template fallback {fallback.call_id}: {fallback.reason}")
    print(f"  report         : {config.out.with_suffix('.report.json')}")

    first = parse_starters_line(config.out.read_text(encoding="utf-8").splitlines()[0])
    print(f"\nfirst set ({first.call_id}):")
    for starter in first.starters:
        print(f"  [{starter.angle}] {starter.text}")

    if report.generated == 0:
        print("generated nothing", file=sys.stderr)
        return 1
    return 0


def _validate(path: Path) -> int:
    bank = load_bundled_seedbank()
    entries = _load_manifest(path)
    if entries is None:
        return 1
    violations = validate_manifest(entries, today=date.today(), drug_lexicon=bank.drug_lexicon)
    if violations:
        for call_id, problems in violations.items():
            for problem in problems:
                print(f"{call_id}: {problem}", file=sys.stderr)
        print(f"{len(violations)} of {len(entries)} entries failed validation", file=sys.stderr)
        return 1
    print(f"ok: {len(entries)} entries, no violations")
    return 0


def _taxonomy() -> int:
    by_class: dict[str, list] = {}
    for tpl in TEMPLATES:
        by_class.setdefault(tpl.objective_class, []).append(tpl)
    for cls in OBJECTIVE_CLASSES:
        templates = by_class.get(cls.id, [])
        print(f"{cls.id}  (weight {cls.weight:.2f}) — hunts: {cls.failure_mode}")
        for tpl in templates:
            constraints = []
            if tpl.min_age > 18 or tpl.max_age < 100:
                constraints.append(f"age {tpl.min_age}-{tpl.max_age}")
            if tpl.language_tag:
                constraints.append(f"language={tpl.language_tag}")
            suffix = f"  [{', '.join(constraints)}]" if constraints else ""
            print(f"  - {tpl.type}: {tpl.goal}{suffix}")
    print(f"\n{len(TEMPLATES)} templates across {len(OBJECTIVE_CLASSES)} classes")
    return 0
