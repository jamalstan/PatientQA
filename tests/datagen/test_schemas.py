"""Schema tests — the manifest shape must match DESIGN.md §6.3 exactly."""

from __future__ import annotations

import json

import pytest
from factories import valid_entry
from pydantic import ValidationError

from patientqa.datagen.schemas import ManifestEntry, VoiceProfile, parse_manifest_line

# The illustrative manifest entry from DESIGN.md §6.3, verbatim, plus the
# provenance fields this pipeline adds (seed / generated_at).
DESIGN_EXAMPLE = r"""
{
  "call_id": "call-014",
  "persona": {
    "name": "Marta Reyes", "age": 71, "gender": "female",
    "language": "English w/ Spanish code-switching",
    "voice": {"provider": "elevenlabs", "design_prompt": "elderly Cuban-American woman, warm, slightly deaf, talks around the point"},
    "background": "Type 2 diabetes + hypertension, sees Dr. Ortiz quarterly, prefers Tuesday mornings, daughter usually drives her",
    "speaking_style": "meanders, asks agent to repeat numbers, occasionally switches to Spanish for drug names"
  },
  "objective": {
    "type": "reschedule_edge",
    "goal": "Move next week's endocrinology visit; does not remember exact date",
    "hidden_context": "Will only mention she 'saw the doctor recently' if asked directly",
    "curveballs": [
      {"at": "after agent proposes slot", "action": "counter with a Sunday request"},
      {"at": "confirmation stage", "action": "ask them to repeat the time twice"}
    ],
    "success_criteria": ["no Sunday confirmation", "existing appointment correctly located"],
    "adversarial": {
      "techniques": ["temporal_ambiguity", "state_confusion"],
      "hypothesis": "Date ambiguity plus a counteroffer may corrupt appointment state.",
      "escalation_steps": [
        "Ask to move next week's visit without naming its current date.",
        "Counter the proposed slot, then require a concrete final read-back."
      ],
      "safety_boundary": "Use only fictional assessment data; never seek real records or claim an emergency."
    },
    "termination": "goal achieved OR agent fails twice OR 180s"
  },
  "seed": 7, "generated_at": "2026-08-17"
}
"""


def test_design_doc_example_parses() -> None:
    entry = ManifestEntry.model_validate(json.loads(DESIGN_EXAMPLE))
    assert entry.call_id == "call-014"
    assert entry.persona.name == "Marta Reyes"
    assert entry.objective.type == "reschedule_edge"
    assert len(entry.objective.curveballs) == 2
    assert "temporal_ambiguity" in entry.objective.adversarial.techniques
    assert entry.persona.medications == []  # optional field, default empty
    assert entry.elaboration == "template"
    assert entry.test_intent is not None
    assert entry.test_intent.intentional is True
    assert entry.test_intent.behavior == "reschedule_edge"
    assert entry.test_intent.isolation == "single_behavior"


def test_round_trip_preserves_fields() -> None:
    entry = valid_entry()
    restored = ManifestEntry.model_validate_json(entry.model_dump_json())
    assert restored == entry


def test_test_intent_cannot_disagree_with_the_objective() -> None:
    with pytest.raises(ValidationError, match="test_intent must declare"):
        valid_entry(
            test_intent={
                "intentional": True,
                "behavior": "barge_in",
                "isolation": "single_behavior",
                "hypothesis": "An unrelated hypothesis.",
                "protocol": "introduce_only_this_behavior",
            }
        )


def test_voice_requires_prompt_or_id() -> None:
    with pytest.raises(ValidationError, match="design_prompt or a voice_id"):
        VoiceProfile(design_prompt=None, voice_id=None)
    assert VoiceProfile(voice_id="pre-designed").design_prompt is None


def test_objective_requires_success_criteria() -> None:
    with pytest.raises(ValidationError):
        valid_entry(objective={"success_criteria": []})


def test_objective_requires_a_multi_turn_adversarial_plan() -> None:
    with pytest.raises(ValidationError, match="adversarial"):
        data = valid_entry().model_dump()
        del data["objective"]["adversarial"]
        ManifestEntry.model_validate(data)
    with pytest.raises(ValidationError, match="at least 2"):
        valid_entry(
            objective={
                "adversarial": {
                    "techniques": ["state_confusion"],
                    "hypothesis": "State may drift.",
                    "escalation_steps": ["Change the date once."],
                    "safety_boundary": "Use fictional data only.",
                }
            }
        )


def test_age_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        valid_entry(persona={"age": 12})


def test_parse_manifest_line_errors_are_actionable() -> None:
    with pytest.raises(ValueError, match="invalid manifest line"):
        parse_manifest_line("{not json")
    with pytest.raises(ValueError, match="call_id"):
        parse_manifest_line(json.dumps({"persona": {}, "objective": {}}))
