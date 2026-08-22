"""Deepgram primary-STT tests (DESIGN.md §3.4): settings, URL contract,
and socket-message parsing. No network — payloads are fixtures."""

from pathlib import Path

from patientqa.deepgram.client import (
    DeepgramSettings,
    build_deepgram_url,
    parse_deepgram_message,
)
from patientqa.stt import Final, Partial


def test_settings_load_from_secrets(secrets_file: Path) -> None:
    settings = DeepgramSettings.from_secrets(
        secrets_file, language="es", keywords=("metformin",)
    )
    assert settings.api_key == "test-deepgram"
    assert settings.model == "nova-3"
    assert settings.language == "es"
    assert settings.keywords == ("metformin",)
    assert settings.endpointing_ms == 250


def test_url_carries_every_setting_as_query_params() -> None:
    settings = DeepgramSettings(
        api_key="k",
        language="en",
        keywords=("Dr. Ortiz", "metformin"),
    )
    url = build_deepgram_url(settings, base_url="wss://api.deepgram.com/v1/listen")
    assert url == (
        "wss://api.deepgram.com/v1/listen"
        "?model=nova-3"
        "&language=en"
        "&encoding=mulaw"
        "&sample_rate=8000"
        "&channels=1"
        "&endpointing=250"
        "&interim_results=true"
        "&punctuate=true"
        "&smart_format=true"
        "&keywords=Dr.+Ortiz"
        "&keywords=metformin"
    )


def test_parse_interim_result_is_partial() -> None:
    message = {
        "type": "Results",
        "is_final": False,
        "channel": {"alternatives": [{"transcript": "What day"}]},
    }
    assert parse_deepgram_message(message) == Partial(text="What day")


def test_parse_final_result_is_final() -> None:
    message = {
        "type": "Results",
        "is_final": True,
        "speech_final": True,
        "channel": {"alternatives": [{"transcript": "What day works for you?"}]},
    }
    assert parse_deepgram_message(message) == Final(text="What day works for you?")


def test_parse_ignores_empty_transcripts_and_control_messages() -> None:
    empty = {
        "type": "Results",
        "is_final": True,
        "channel": {"alternatives": [{"transcript": ""}]},
    }
    assert parse_deepgram_message(empty) is None
    assert parse_deepgram_message({"type": "Metadata"}) is None
    assert parse_deepgram_message({"type": "UtteranceEnd"}) is None
    assert parse_deepgram_message({"type": "Results"}) is None  # no channel
