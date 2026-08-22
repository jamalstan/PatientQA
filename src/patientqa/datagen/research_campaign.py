"""Curated one-factor-at-a-time campaign based on 2026 voice-agent research."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import date
from pathlib import Path

from patientqa.datagen.elaborate import TemplateElaborator
from patientqa.datagen.sampling import Sampler
from patientqa.datagen.schemas import ManifestEntry
from patientqa.datagen.seeds import SeedBank, load_bundled_seedbank
from patientqa.datagen.taxonomy import template_by_type
from patientqa.datagen.validate import validate_manifest

RESEARCH_10_BEHAVIORS: tuple[str, ...] = (
    "clean_scheduling_baseline",
    "self_correction_once",
    "long_silence",
    "barge_in",
    "backchannel_during_readback",
    "third_party_interruption",
    "degraded_audio_digits",
    "spanish_heavy_call",
    "cancel_reschedule_rollback",
    "refuse_dob",
)


def build_research_10(
    *,
    base_seed: int = 20260820,
    generated_on: date | None = None,
    seedbank: SeedBank | None = None,
) -> list[ManifestEntry]:
    """Build the fixed ten-behavior campaign with deterministic personas."""

    generated_on = generated_on or date.today()
    bank = seedbank or load_bundled_seedbank()
    templates = [template_by_type(type_id) for type_id in RESEARCH_10_BEHAVIORS]
    sampler = Sampler(
        bank,
        base_seed=base_seed,
        class_allocation=[template.objective_class for template in templates],
        generated_on=generated_on,
    )
    elaborator = TemplateElaborator()
    entries: list[ManifestEntry] = []
    taken_names: set[str] = set()
    for index, template in enumerate(templates):
        wants_spanish = template.type == "spanish_heavy_call"
        seed = None
        for attempt in range(100):
            candidate = sampler.draw(
                index,
                attempt=attempt,
                taken_names=frozenset(taken_names),
            )
            language_matches = (
                candidate.language_tag == "spanish"
                if wants_spanish
                else candidate.language_tag == "english"
            )
            if 45 <= candidate.age <= 65 and language_matches:
                seed = candidate
                break
        if seed is None:
            raise ValueError(f"could not draw neutral demographics for {template.type}")
        phrasing_style = "spanish_heavy" if wants_spanish else "precise"
        phrasing = next(
            item for item in bank.phrasings if item.style == phrasing_style
        )
        seed = replace(
            seed,
            template=template,
            objective_class=template.objective_class,
            disposition="brisk and businesslike",
            phrasing=phrasing,
        )
        entry = elaborator.elaborate(seed)
        entry.call_id = f"research10-{index + 1:02d}"
        entries.append(entry)
        taken_names.add(entry.persona.name.lower())

    violations = validate_manifest(
        entries,
        today=generated_on,
        drug_lexicon=bank.drug_lexicon,
    )
    if violations:
        raise ValueError(f"curated research campaign failed validation: {violations}")
    return entries


def write_research_10(
    out: Path,
    *,
    base_seed: int = 20260820,
    generated_on: date | None = None,
    seedbank: SeedBank | None = None,
) -> list[ManifestEntry]:
    """Atomically write the curated manifest and its compact coverage report."""

    entries = build_research_10(
        base_seed=base_seed,
        generated_on=generated_on,
        seedbank=seedbank,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.model_dump_json() + "\n")
    os.replace(tmp, out)
    report = {
        "campaign": "high_value_research_10",
        "base_seed": base_seed,
        "generated": len(entries),
        "one_behavior_per_call": True,
        "behaviors": [entry.test_intent.behavior for entry in entries if entry.test_intent],
    }
    out.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return entries
