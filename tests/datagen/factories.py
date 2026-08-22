"""Helpers for building test entries without going through sampling."""

from __future__ import annotations

import copy
from typing import Any

from patientqa.datagen.schemas import (
    Curveball,
    ManifestEntry,
    Objective,
    Persona,
    Starter,
    StarterSet,
    VoiceProfile,
)

VALID_PERSONA: dict[str, Any] = {
    "name": "Marta Reyes",
    "age": 71,
    "gender": "female",
    "language": "English w/ Spanish code-switching",
    "voice": {
        "provider": "elevenlabs",
        "design_prompt": (
            "elderly Cuban-American woman, warm, slightly deaf, talks around the point"
        ),
    },
    "background": (
        "Type 2 diabetes + hypertension, sees Dr. Ortiz quarterly, "
        "prefers Tuesday mornings, daughter usually drives her"
    ),
    "speaking_style": "meanders, asks agent to repeat numbers, occasionally switches to Spanish",
    "medications": ["metformin", "lisinopril"],
    "identity": {
        "date_of_birth": "1955-03-12",
        "callback_number": "+14155550137",
        "insurance_plan": "Blue Shield PPO",
    },
}

VALID_OBJECTIVE: dict[str, Any] = {
    "type": "reschedule_edge",
    "goal": "Move next week's endocrinology visit; does not remember exact date",
    "hidden_context": "Will only mention she 'saw the doctor recently' if asked directly",
    "curveballs": [
        {"at": "after agent proposes slot", "action": "counter with a Sunday request"},
        {"at": "confirmation stage", "action": "ask them to repeat the time twice"},
    ],
    "secondary_asks": [
        "before hanging up, ask them to read the final appointment details back",
        "ask where exactly the office is and where to park once you get there",
        "ask what identification and paperwork you should bring to the visit",
        "ask how early you should arrive before the appointment time",
    ],
    "success_criteria": ["no Sunday confirmation", "existing appointment correctly located"],
    "adversarial": {
        "techniques": ["temporal_ambiguity", "state_confusion"],
        "hypothesis": "Ambiguous dates may corrupt the final appointment state.",
        "escalation_steps": [
            "Ask to move next week's visit without naming its current date.",
            "Counter the first slot, then require a concrete final read-back.",
        ],
        "safety_boundary": (
            "Use only fictional assessment data; do not seek real records or claim an emergency."
        ),
    },
    "termination": "goal achieved OR agent fails twice OR 180s",
}


VALID_STARTERS: list[dict[str, str]] = [
    {
        "angle": "direct",
        "text": "Hi, I'd like to move my endocrinology visit to sometime next week.",
    },
    {
        "angle": "chatty",
        "text": (
            "Hi there! This is Marta — I see Dr. Ortiz, and I need to move my "
            "visit next week if I can."
        ),
    },
    {
        "angle": "vague",
        "text": (
            "Hi, um, my daughter said I should call about my doctor "
            "appointment... something about next week?"
        ),
    },
]


def valid_starter_set(**overrides: Any) -> StarterSet:
    """A fully-valid starter set for ``valid_entry``'s call; kwargs override any field."""
    data: dict[str, Any] = {
        "call_id": "call-014",
        "starters": copy.deepcopy(VALID_STARTERS),
        "elaboration": "template",
        "model": "",
        "generated_at": "2026-08-17",
    }
    data.update(overrides)
    return StarterSet.model_validate(data)


def valid_entry(**overrides: Any) -> ManifestEntry:
    """A fully-valid manifest entry; kwargs override any field."""
    entry: dict[str, Any] = {
        "call_id": "call-014",
        "persona": copy.deepcopy(VALID_PERSONA),
        "objective": copy.deepcopy(VALID_OBJECTIVE),
        "seed": 12345,
        "generated_at": "2026-08-17",
        "elaboration": "template",
    }
    for key, value in overrides.items():
        if key in ("persona", "objective"):
            entry[key].update(value)
        else:
            entry[key] = value
    return ManifestEntry.model_validate(entry)


__all__ = [
    "VALID_OBJECTIVE",
    "VALID_PERSONA",
    "VALID_STARTERS",
    "valid_entry",
    "valid_starter_set",
    "Curveball",
    "Objective",
    "Persona",
    "Starter",
    "StarterSet",
    "VoiceProfile",
]
