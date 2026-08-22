"""ElevenLabs Scribe fallback-STT tests (DESIGN.md §3.4/§5.3): the verified
URL recipe, message parsing, and the session_started echo discipline."""

from pathlib import Path

from patientqa.elevenlabs.scribe import (
    ScribeSettings,
    build_scribe_url,
    parse_scribe_message,
    verify_scribe_session,
    wrap_scribe_audio,
)
from patientqa.stt import Final, Partial


def test_settings_reuse_the_elevenlabs_key(secrets_file: Path) -> None:
    settings = ScribeSettings.from_secrets(
        secrets_file, language_code="en", secondary_languages=("es",)
    )
    assert settings.api_key == "test-elevenlabs"
    assert settings.model_id == "scribe_v2_realtime"
    assert settings.audio_format == "ulaw_8000"
    assert settings.secondary_languages == ("es",)


def test_url_is_the_verified_recipe() -> None:
    settings = ScribeSettings(api_key="k", language_code="en", keyterms=("Dr. Ortiz",))
    url = build_scribe_url(settings)
    assert url == (
        "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
        "?model_id=scribe_v2_realtime"
        "&audio_format=ulaw_8000"
        "&language_code=en"
        "&keyterms=Dr.+Ortiz"
        "&commit_strategy=vad"
        "&vad_threshold=0.5"
        "&vad_silence_threshold_secs=0.5"
        "&min_speech_duration_ms=250"
        "&filter_background_audio=false"
    )


def test_url_carries_secondary_languages() -> None:
    settings = ScribeSettings(api_key="k", language_code="es", secondary_languages=("en",))
    assert "language_code=es" in build_scribe_url(settings)
    assert "secondary_languages=en" in build_scribe_url(settings)


def test_parse_maps_partials_and_committed_transcripts() -> None:
    assert parse_scribe_message(
        {"message_type": "partial_transcript", "text": "What day"}
    ) == Partial(text="What day")
    assert parse_scribe_message(
        {"message_type": "committed_transcript", "text": "What day works?"}
    ) == Final(text="What day works?")
    assert parse_scribe_message(
        {"message_type": "final_transcript_with_timestamps", "text": "Hello?"}
    ) == Final(text="Hello?")


def test_parse_ignores_empty_commits_and_session_events() -> None:
    assert parse_scribe_message({"message_type": "committed_transcript", "text": ""}) is None
    assert parse_scribe_message({"message_type": "session_started"}) is None


def test_session_echo_matching_request_is_clean() -> None:
    settings = ScribeSettings(api_key="k")
    echo = {
        "message_type": "session_started",
        "config": {
            "audio_format": "ulaw_8000",
            "language_code": "en",
            "model_id": "scribe_v2_realtime",
            "commit_strategy": "vad",
        },
    }
    assert verify_scribe_session(echo, settings) == []


def test_session_echo_reports_silently_dropped_params() -> None:
    settings = ScribeSettings(api_key="k")
    echo = {
        "message_type": "session_started",
        "config": {
            "audio_format": "pcm_16000",  # server ignored ulaw_8000
            "language_code": None,  # param never applied
            "model_id": "scribe_v2_realtime",
            "commit_strategy": "vad",
        },
    }
    problems = verify_scribe_session(echo, settings)
    assert len(problems) == 2
    assert any("audio_format" in p for p in problems)
    assert any("language_code" in p for p in problems)


def test_session_echo_without_config_is_reported() -> None:
    assert verify_scribe_session({"message_type": "session_started"}, ScribeSettings(api_key="k"))


def test_session_echo_accepts_v2_vad_commit_strategy_bool() -> None:
    """The live v2 endpoint echoes no ``commit_strategy`` string — VAD mode
    shows up as ``vad_commit_strategy: true`` (verified against production)."""
    settings = ScribeSettings(api_key="k")
    echo = {
        "message_type": "session_started",
        "config": {
            "audio_format": "ulaw_8000",
            "language_code": "en",
            "model_id": "scribe_v2_realtime",
            "vad_commit_strategy": True,
        },
    }
    assert verify_scribe_session(echo, settings) == []
    echo["config"]["vad_commit_strategy"] = False  # server dropped the vad request
    problems = verify_scribe_session(echo, settings)
    assert len(problems) == 1
    assert "vad_commit_strategy" in problems[0]


def test_session_echo_with_no_commit_strategy_shape_is_reported() -> None:
    echo = {
        "message_type": "session_started",
        "config": {
            "audio_format": "ulaw_8000",
            "language_code": "en",
            "model_id": "scribe_v2_realtime",
        },
    }
    problems = verify_scribe_session(echo, ScribeSettings(api_key="k"))
    assert len(problems) == 1
    assert "commit strategy" in problems[0]


def test_wrap_scribe_audio_is_the_protocol_message() -> None:
    import base64
    import json

    wrapped = json.loads(wrap_scribe_audio(b"\xff" * 160))
    assert wrapped == {
        "message_type": "input_audio_chunk",
        "audio_base_64": base64.b64encode(b"\xff" * 160).decode("ascii"),
        "commit": False,
        "sample_rate": 8000,
    }
    # manual-commit mode flags the final chunk
    assert json.loads(wrap_scribe_audio(b"\x00", commit=True))["commit"] is True
