"""ElevenLabs integration: telephony-native patient TTS (DESIGN.md §3.3)."""

from patientqa.elevenlabs.client import (
    FLASH_MODEL,
    ULAW_8000,
    ElevenLabsSettings,
    PatientVoice,
)

__all__ = [
    "FLASH_MODEL",
    "ULAW_8000",
    "ElevenLabsSettings",
    "PatientVoice",
]
