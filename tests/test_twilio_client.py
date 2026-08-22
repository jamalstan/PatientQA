from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from patientqa.twilio.client import (
    TwilioClient,
    TwilioSettings,
    build_stream_twiml,
    media_clear_message,
    parse_allowed_numbers,
)


def test_media_clear_message_flushes_playback_buffer() -> None:
    # §5.3 abort sequence, final step: one JSON message on the media socket.
    assert media_clear_message() == '{"event":"clear"}'


def test_settings_load_from_secrets(secrets_file: Path) -> None:
    settings = TwilioSettings.from_secrets(secrets_file)
    assert settings.api_key_sid == "SKtest"
    assert settings.account_sid == "ACtest"
    assert settings.from_number == "+15550001111"
    assert settings.allowed_numbers == ("+15550002222", "+15550003333")
    assert settings.live_test_number == "+15550002222"


def test_from_secrets_rejects_live_number_outside_allowlist(tmp_path: Path) -> None:
    path = tmp_path / "secrets.toml"
    path.write_text(
        "[twilio]\n"
        "api_key_sid = 'SKtest'\n"
        "api_key_secret = 'shhh'\n"
        "account_sid = 'ACtest'\n"
        "from_number = '+15550001111'\n"
        "allowed_numbers = ['+15550002222']\n"
        "live_test_number = '+15550009999'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="live_test_number"):
        TwilioSettings.from_secrets(path)


def test_parse_allowed_numbers_from_env_string() -> None:
    assert parse_allowed_numbers("+15550002222, +15550003333") == (
        "+15550002222",
        "+15550003333",
    )


def test_parse_allowed_numbers_rejects_non_e164() -> None:
    with pytest.raises(ValueError, match="not E.164"):
        parse_allowed_numbers(["+15550002222", "916-513-3194"])


def test_parse_allowed_numbers_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_allowed_numbers(" , ")


def test_stream_twiml_connects_and_escapes_url() -> None:
    twiml = build_stream_twiml("wss://host.ngrok.app/media?call=1&leg=2")
    assert twiml.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<Response><Connect><Stream" in twiml
    assert 'url="wss://host.ngrok.app/media?call=1&amp;leg=2"' in twiml


class _FakeCalls:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.created.append(kwargs)
        return SimpleNamespace(sid="CAtest123")


class _FakeRestClient:
    def __init__(self) -> None:
        self.calls = _FakeCalls()


def test_place_call_defaults_to_live_test_number(secrets_file: Path) -> None:
    settings = TwilioSettings.from_secrets(secrets_file)
    rest = _FakeRestClient()
    client = TwilioClient(settings, rest_client=rest)

    sid = client.place_call(stream_url="wss://example.invalid/media?call=1")

    assert sid == "CAtest123"
    (call,) = rest.calls.created
    assert call["to"] == settings.live_test_number
    assert call["from_"] == settings.from_number
    assert call["time_limit"] == 300
    assert 'url="wss://example.invalid/media?call=1"' in call["twiml"]


def test_place_call_accepts_explicit_allowed_number(secrets_file: Path) -> None:
    settings = TwilioSettings.from_secrets(secrets_file)
    rest = _FakeRestClient()
    client = TwilioClient(settings, rest_client=rest)

    client.place_call(
        stream_url="wss://example.invalid/media", to=settings.allowed_numbers[1]
    )

    (call,) = rest.calls.created
    assert call["to"] == settings.allowed_numbers[1]


def test_place_call_refuses_numbers_outside_allowlist(secrets_file: Path) -> None:
    client = TwilioClient(TwilioSettings.from_secrets(secrets_file))
    with pytest.raises(ValueError, match="allowlist"):
        client.place_call(stream_url="wss://example.invalid/media", to="+15559999999")


def test_place_call_enforces_server_side_duration_ceiling(secrets_file: Path) -> None:
    rest = _FakeRestClient()
    client = TwilioClient(TwilioSettings.from_secrets(secrets_file), rest_client=rest)
    client.place_call("wss://example.invalid/media", max_duration_s=180.5)
    assert rest.calls.created[0]["time_limit"] == 181
    with pytest.raises(ValueError, match="300"):
        client.place_call("wss://example.invalid/media", max_duration_s=301)
