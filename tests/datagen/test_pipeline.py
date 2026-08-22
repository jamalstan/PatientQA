"""End-to-end pipeline tests (offline: template elaboration, no network)."""

from __future__ import annotations

import json
from datetime import date

from patientqa.datagen.elaborate import ElaborationError, TemplateElaborator
from patientqa.datagen.pipeline import (
    PipelineConfig,
    allocate_classes,
    generate_manifest,
)
from patientqa.datagen.schemas import parse_manifest_line
from patientqa.datagen.taxonomy import OBJECTIVE_CLASSES
from patientqa.datagen.validate import validate_manifest

TODAY = date(2026, 8, 17)


def _config(tmp_path, **overrides) -> PipelineConfig:
    defaults = dict(
        count=24,
        base_seed=2026,
        elaboration="template",
        out=tmp_path / "manifest.jsonl",
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def test_full_run_produces_valid_manifest(bank, tmp_path) -> None:
    config = _config(tmp_path)
    report = generate_manifest(config, seedbank=bank, today=TODAY)

    assert report.generated == config.count
    assert report.drops == []
    lines = config.out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == config.count

    entries = [parse_manifest_line(line) for line in lines]
    assert validate_manifest(entries, today=TODAY, drug_lexicon=bank.drug_lexicon) == {}

    report_json = json.loads(config.out.with_suffix(".report.json").read_text("utf-8"))
    assert report_json["generated"] == config.count
    assert report_json["base_seed"] == config.base_seed
    assert sum(report_json["behavior_coverage"].values()) == config.count
    assert all(entry.test_intent.intentional for entry in entries if entry.test_intent)


def test_deterministic_output_is_byte_identical(bank, tmp_path) -> None:
    first = _config(tmp_path, out=tmp_path / "a.jsonl")
    second = _config(tmp_path, out=tmp_path / "b.jsonl")
    generate_manifest(first, seedbank=bank, today=TODAY)
    generate_manifest(second, seedbank=bank, today=TODAY)
    assert first.out.read_bytes() == second.out.read_bytes()


def test_coverage_spans_all_classes(bank, tmp_path) -> None:
    report = generate_manifest(_config(tmp_path, count=24), seedbank=bank, today=TODAY)
    assert set(report.class_coverage) == {c.id for c in OBJECTIVE_CLASSES}
    assert all(count >= 1 for count in report.class_coverage.values())


def test_allocate_classes_guarantees_and_interleaves() -> None:
    for count in (10, 14, 60):  # >= len(OBJECTIVE_CLASSES), so every class gets a slot
        allocation = allocate_classes(count)
        assert len(allocation) == count
        assert set(allocation) == {c.id for c in OBJECTIVE_CLASSES}
        for left, right in zip(allocation, allocation[1:], strict=False):
            assert left != right, f"adjacent repeat in {allocation}"
    assert allocate_classes(2) == ["identity_phi", "conversational_stress"]
    assert allocate_classes(12).count("adversarial_security") == 3


class _AlwaysInvalid:
    """Returns entries that fail validation (unknown drug) for specific slots."""

    def __init__(self, bad_slots: set[int]) -> None:
        self.bad_slots = bad_slots
        self._inner = TemplateElaborator()

    def elaborate(self, seed):
        entry = self._inner.elaborate(seed)
        if seed.index in self.bad_slots:
            entry.persona.medications = ["not a real drug"]
        return entry


def test_persistently_invalid_slots_are_dropped_not_fatal(bank, tmp_path) -> None:
    config = _config(tmp_path, count=6, retries=2)
    report = generate_manifest(
        config, seedbank=bank, elaborator=_AlwaysInvalid({1, 3}), today=TODAY
    )
    assert report.generated == 4
    assert {drop.index for drop in report.drops} == {1, 3}
    assert len(config.out.read_text(encoding="utf-8").splitlines()) == 4
    # surviving entries are still clean
    entries = [parse_manifest_line(line) for line in config.out.read_text("utf-8").splitlines()]
    assert validate_manifest(entries, today=TODAY, drug_lexicon=bank.drug_lexicon) == {}


class _FlakyLlm:
    """Raises like a flaky LLM for the first call on each seed, then works
    (returns the template entry) — exercises the same-seed retry path."""

    def __init__(self, never_recover: bool = False) -> None:
        self.never_recover = never_recover
        self._seen: set[int] = set()
        self._inner = TemplateElaborator()

    def elaborate(self, seed):
        if not self.never_recover and seed.seed in self._seen:
            return self._inner.elaborate(seed)
        self._seen.add(seed.seed)
        raise ElaborationError("simulated LLM outage")


def test_flaky_llm_recovers_without_fallback(bank, tmp_path) -> None:
    report = generate_manifest(
        _config(tmp_path, count=5), seedbank=bank, elaborator=_FlakyLlm(), today=TODAY
    )
    assert report.generated == 5
    assert report.template_fallbacks == []
    assert report.drops == []


def test_dead_llm_falls_back_to_template(bank, tmp_path) -> None:
    report = generate_manifest(
        _config(tmp_path, count=5), seedbank=bank, elaborator=_FlakyLlm(True), today=TODAY
    )
    assert report.generated == 5
    assert report.template_fallbacks == [0, 1, 2, 3, 4]
    entries = [
        parse_manifest_line(line)
        for line in report.out.read_text(encoding="utf-8").splitlines()
    ]
    assert all(entry.elaboration == "template" for entry in entries)
