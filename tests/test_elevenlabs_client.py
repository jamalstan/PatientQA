from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from patientqa.elevenlabs.client import ElevenLabsSettings, PatientVoice


def test_settings_default_to_flash_and_ulaw(secrets_file: Path) -> None:
    settings = ElevenLabsSettings.from_secrets(secrets_file)
    assert settings.api_key == "test-elevenlabs"
    assert settings.model_id == "eleven_flash_v2_5"
    assert settings.output_format == "ulaw_8000"


def test_synthesize_requests_ulaw_and_joins_chunks() -> None:
    requests: list[dict[str, Any]] = []

    def fake_convert(**kwargs: Any) -> Iterator[bytes]:
        requests.append(kwargs)
        return iter([b"\x00\x01", b"\xff"])

    fake = SimpleNamespace(text_to_speech=SimpleNamespace(convert=fake_convert))
    voice = PatientVoice(ElevenLabsSettings(api_key="k"), client=fake)

    audio = voice.synthesize("I need an appointment", voice_id="voice-7")

    assert audio == b"\x00\x01\xff"
    assert requests == [
        {
            "voice_id": "voice-7",
            "text": "I need an appointment",
            "model_id": "eleven_flash_v2_5",
            "output_format": "ulaw_8000",
        }
    ]
