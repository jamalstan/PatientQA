"""Generation pipeline (DESIGN §6.3): sample -> elaborate -> validate -> manifest.jsonl.

Failures never kill the batch: an entry that fails post-validation is redrawn
(up to ``retries`` attempts with a shifted derived seed); an LLM elaboration
failure is retried on the same seed and finally falls back to the deterministic
template elaborator, so a flaky API can at worst cost prose quality, never a
call slot. Whatever survives is written atomically, one JSON object per line,
plus a sidecar ``.report.json`` describing the run.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from patientqa.datagen.elaborate import ElaborationError, Elaborator, TemplateElaborator
from patientqa.datagen.sampling import Sampler
from patientqa.datagen.schemas import ManifestEntry
from patientqa.datagen.seeds import SeedBank, load_bundled_seedbank
from patientqa.datagen.taxonomy import OBJECTIVE_CLASSES
from patientqa.datagen.validate import validate_entry

_LLM_RETRIES = 2  # same-seed retries before falling back to template elaboration


@dataclass
class PipelineConfig:
    count: int = 60
    base_seed: int = 2026
    seed_source: str = "bundled"  # "bundled" | "hf"
    elaboration: str = "auto"  # "auto" | "llm" | "template"
    out: Path = Path("manifest.jsonl")
    retries: int = 3  # redraw attempts for entries failing post-validation
    model: str = "gpt-oss-120b"


@dataclass
class DropRecord:
    index: int
    attempts: int
    reason: str


@dataclass
class GenerationReport:
    requested: int
    generated: int = 0
    base_seed: int = 0
    out: Path | None = None
    elaboration_counts: dict[str, int] = field(default_factory=dict)
    class_coverage: dict[str, int] = field(default_factory=dict)
    behavior_coverage: dict[str, int] = field(default_factory=dict)
    technique_coverage: dict[str, int] = field(default_factory=dict)
    template_fallbacks: list[int] = field(default_factory=list)  # slot indexes
    drops: list[DropRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "generated": self.generated,
            "base_seed": self.base_seed,
            "out": str(self.out) if self.out else None,
            "elaboration_counts": self.elaboration_counts,
            "class_coverage": self.class_coverage,
            "behavior_coverage": self.behavior_coverage,
            "technique_coverage": self.technique_coverage,
            "template_fallbacks": self.template_fallbacks,
            "drops": [drop.__dict__ for drop in self.drops],
        }


def allocate_classes(count: int) -> list[str]:
    """Deterministic stratified allocation of objective classes across slots.

    Largest-remainder apportionment by class weight, every class guaranteed at
    least one slot when ``count >= len(classes)``, then interleaved so adjacent
    calls never run the same class twice in a row.
    """
    classes = sorted(OBJECTIVE_CLASSES, key=lambda c: -c.weight)
    raw = [count * c.weight for c in classes]
    allocation = [int(r) for r in raw]
    shortfall = count - sum(allocation)
    remainders = sorted(range(len(classes)), key=lambda i: raw[i] - allocation[i], reverse=True)
    for i in remainders[:shortfall]:
        allocation[i] += 1

    # enforce the >= 1 slot guarantee by stealing from the largest buckets
    for i in range(len(allocation)):
        if allocation[i] == 0:
            donor = max(range(len(allocation)), key=lambda j: allocation[j])
            if allocation[donor] <= 1:
                break
            allocation[donor] -= 1
            allocation[i] += 1

    # A 12-call submission should exercise all three explicit security probes,
    # not leave prompt injection or PHI extraction to random chance. Preserve
    # one slot for every other class, then move surplus slots into this class.
    if count >= len(classes) + 2:
        security = next(i for i, cls in enumerate(classes) if cls.id == "adversarial_security")
        while allocation[security] < 3:
            donors = [
                i for i, cls in enumerate(classes)
                if cls.id != "adversarial_security" and allocation[i] > 1
            ]
            if not donors:
                break
            donor = max(donors, key=lambda i: allocation[i])
            allocation[donor] -= 1
            allocation[security] += 1

    queues = {cls.id: n for cls, n in zip(classes, allocation, strict=True)}
    weight_order = {cls.id: position for position, cls in enumerate(classes)}
    order: list[str] = []
    previous: str | None = None
    while any(n > 0 for n in queues.values()):
        # greedy: always take from the largest remaining queue that isn't the
        # previous class, so adjacent calls never repeat a class (ties go to
        # the heavier class, keeping the heavy hitters front-loaded)
        candidates = [cid for cid, n in queues.items() if n > 0 and cid != previous]
        if not candidates:
            candidates = [cid for cid, n in queues.items() if n > 0]
        pick = max(candidates, key=lambda cid: (queues[cid], -weight_order[cid]))
        queues[pick] -= 1
        order.append(pick)
        previous = pick
    return order


def resolve_elaborator(mode: str, *, model: str) -> Elaborator:
    """Build the primary elaborator; 'auto' means LLM iff a Cerebras key exists."""
    if mode == "template":
        return TemplateElaborator()
    if mode in ("llm", "auto"):
        from patientqa.datagen.elaborate import LlmElaborator

        try:
            from patientqa.config import get_secret

            api_key = get_secret("cerebras", "api_key")
        except Exception:
            if mode == "llm":
                raise
            return TemplateElaborator()
        return LlmElaborator(api_key, model=model)
    known = "auto, llm, template"
    raise ValueError(f"unknown elaboration mode {mode!r}; known: {known}")


def load_seedbank(source: str) -> SeedBank:
    if source == "bundled":
        return load_bundled_seedbank()
    if source == "hf":
        from patientqa.datagen.seeds import load_hf_seedbank

        return load_hf_seedbank()
    raise ValueError(f"unknown seed source {source!r}; known: bundled, hf")


def generate_manifest(
    config: PipelineConfig,
    *,
    seedbank: SeedBank | None = None,
    elaborator: Elaborator | None = None,
    today: date | None = None,
) -> GenerationReport:
    """Run the full pipeline; writes ``config.out`` and a ``.report.json`` sidecar."""
    today = today or date.today()
    bank = seedbank or load_seedbank(config.seed_source)
    primary = elaborator or resolve_elaborator(config.elaboration, model=config.model)
    template = TemplateElaborator()

    sampler = Sampler(
        bank, base_seed=config.base_seed, class_allocation=allocate_classes(config.count),
        generated_on=today,
    )

    entries: list[ManifestEntry] = []
    taken_names: set[str] = set()
    report = GenerationReport(requested=config.count, base_seed=config.base_seed, out=config.out)

    for index in range(config.count):
        accepted: ManifestEntry | None = None
        last_violations: list[str] = []

        for attempt in range(config.retries + 1):
            seed = sampler.draw(index, attempt=attempt, taken_names=frozenset(taken_names))
            try:
                candidate = primary.elaborate(seed)
            except ElaborationError:
                candidate = _elaborate_with_llm_retries(primary, template, seed, index, report)

            last_violations = validate_entry(
                candidate, today=today, drug_lexicon=bank.drug_lexicon
            )
            if not last_violations:
                accepted = candidate
                break

        if accepted is None:
            report.drops.append(
                DropRecord(
                    index=index,
                    attempts=config.retries + 1,
                    reason="; ".join(last_violations) or "unknown violation",
                )
            )
            continue

        entries.append(accepted)
        taken_names.add(accepted.persona.name.lower())

    report.generated = len(entries)
    for entry in entries:
        report.elaboration_counts[entry.elaboration] = (
            report.elaboration_counts.get(entry.elaboration, 0) + 1
        )
        class_id = _class_of(entry)
        report.class_coverage[class_id] = report.class_coverage.get(class_id, 0) + 1
        behavior = entry.test_intent.behavior if entry.test_intent else entry.objective.type
        report.behavior_coverage[behavior] = report.behavior_coverage.get(behavior, 0) + 1
        for technique in entry.objective.adversarial.techniques:
            report.technique_coverage[technique] = (
                report.technique_coverage.get(technique, 0) + 1
            )

    _write_manifest(config.out, entries)
    _write_report(config.out, report)
    return report


# ---- internals -----------------------------------------------------------------


def _elaborate_with_llm_retries(
    primary: Elaborator, template: TemplateElaborator, seed, index: int, report: GenerationReport
) -> ManifestEntry:
    """Primary (LLM) failed once: retry the same seed, then template-fallback."""
    for _ in range(_LLM_RETRIES):
        try:
            return primary.elaborate(seed)
        except ElaborationError:
            continue
    report.template_fallbacks.append(index)
    return template.elaborate(seed)


def _class_of(entry: ManifestEntry) -> str:
    from patientqa.datagen.taxonomy import TEMPLATES

    for tpl in TEMPLATES:
        if tpl.type == entry.objective.type:
            return tpl.objective_class
    return "unknown"


def _write_manifest(out: Path, entries: list[ManifestEntry]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.model_dump_json() + "\n")
    os.replace(tmp, out)


def _write_report(out: Path, report: GenerationReport) -> None:
    sidecar = out.with_suffix(".report.json")
    sidecar.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
