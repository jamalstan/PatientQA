"""Curated 2026 research campaign: exact behaviors and reproducible artifacts."""

from __future__ import annotations

import json
from datetime import date

from patientqa.campaign import behavior_runtime
from patientqa.datagen.research_campaign import (
    RESEARCH_10_BEHAVIORS,
    build_research_10,
    write_research_10,
)
from patientqa.datagen.schemas import parse_manifest_line
from patientqa.datagen.validate import validate_manifest

TODAY = date(2026, 8, 20)


def test_research_10_has_exactly_one_unique_intent_per_call(bank) -> None:
    entries = build_research_10(generated_on=TODAY, seedbank=bank)

    assert len(entries) == 10
    assert tuple(entry.test_intent.behavior for entry in entries) == RESEARCH_10_BEHAVIORS
    assert all(entry.test_intent.intentional for entry in entries)
    assert all(entry.test_intent.isolation == "single_behavior" for entry in entries)
    assert len({entry.call_id for entry in entries}) == 10
    assert validate_manifest(entries, today=TODAY, drug_lexicon=bank.drug_lexicon) == {}


def test_runtime_delivery_exists_for_transport_behaviors(bank) -> None:
    entries = {entry.objective.type: entry for entry in build_research_10(
        generated_on=TODAY, seedbank=bank
    )}

    assert behavior_runtime(entries["long_silence"]).response_delays_s == (5.0, 5.0)
    assert behavior_runtime(entries["backchannel_during_readback"]).barge_lines
    assert behavior_runtime(entries["third_party_interruption"]).barge_voice_id
    assert behavior_runtime(entries["degraded_audio_digits"]).audio_transform is not None


def test_write_research_10_is_deterministic_and_reports_coverage(bank, tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_research_10(first, generated_on=TODAY, seedbank=bank)
    write_research_10(second, generated_on=TODAY, seedbank=bank)

    assert first.read_bytes() == second.read_bytes()
    entries = [parse_manifest_line(line) for line in first.read_text("utf-8").splitlines()]
    assert len(entries) == 10
    report = json.loads(first.with_suffix(".report.json").read_text("utf-8"))
    assert report["one_behavior_per_call"] is True
    assert tuple(report["behaviors"]) == RESEARCH_10_BEHAVIORS
