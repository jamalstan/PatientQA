"""Cerebras integration: the gpt-oss-120b patient brain (DESIGN.md §3.2)."""

from patientqa.cerebras.client import (
    AGENT,
    DEFAULT_SYSTEM_PROMPT,
    PATIENT,
    CerebrasSettings,
    PatientBrain,
    Turn,
    extract_say,
    render_history,
)

__all__ = [
    "AGENT",
    "PATIENT",
    "CerebrasSettings",
    "DEFAULT_SYSTEM_PROMPT",
    "PatientBrain",
    "Turn",
    "extract_say",
    "render_history",
]
