"""ElevenLabs TTS integration (DESIGN.md §3.3).

Flash v2.5 synthesis (~75 ms TTFT) emitted as 8 kHz μ-law — the exact frame
format Twilio Media Streams expects — so the hot path moves bytes without
resampling.

Per-persona voices are created offline via Voice Design when the persona
pipeline (DESIGN.md §6) lands; this module only needs "text in, μ-law bytes
out" for the voice IDs the manifest will carry.
"""

from dataclasses import dataclass
from pathlib import Path

from elevenlabs.client import ElevenLabs

from patientqa.config import get_secret

FLASH_MODEL = "eleven_flash_v2_5"
ULAW_8000 = "ulaw_8000"


@dataclass(frozen=True)
class ElevenLabsSettings:
    api_key: str
    model_id: str = FLASH_MODEL
    output_format: str = ULAW_8000

    @classmethod
    def from_secrets(cls, path: Path | None = None) -> "ElevenLabsSettings":
        return cls(api_key=get_secret("elevenlabs", "api_key", path=path))


class PatientVoice:
    """Speaks patient utterances as μ-law bytes ready for the outbound track."""

    def __init__(self, settings: ElevenLabsSettings, client: ElevenLabs | None = None) -> None:
        self._settings = settings
        self._client = client or ElevenLabs(api_key=settings.api_key)

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Speak ``text`` as 8 kHz μ-law audio; forward to Twilio as-is."""
        chunks = self._client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=self._settings.model_id,
            output_format=self._settings.output_format,
        )
        return b"".join(chunks)
