"""Shared fixtures for the integration tests: a fully-populated secrets.toml."""

from pathlib import Path

import pytest

SECRETS_TOML = """\
[elevenlabs]
api_key = "test-elevenlabs"

[cerebras]
api_key = "test-cerebras"

[stt]
provider = "deepgram"

[deepgram]
api_key = "test-deepgram"

[twilio]
api_key_sid = "SKtest"
api_key_secret = "shhh"
account_sid = "ACtest"
from_number = "+15550001111"
allowed_numbers = ["+15550002222", "+15550003333"]
live_test_number = "+15550002222"

[github]
api_key = "test-github"
"""

_PROVIDER_ENV_VARS = (
    "ELEVENLABS_API_KEY",
    "CEREBRAS_API_KEY",
    "DEEPGRAM_API_KEY",
    "STT_PROVIDER",
    "TWILIO_API_KEY_SID",
    "TWILIO_API_KEY_SECRET",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_FROM_NUMBER",
    "TWILIO_ALLOWED_NUMBERS",
    "TWILIO_LIVE_TEST_NUMBER",
)


@pytest.fixture(autouse=True)
def _no_provider_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep provider env vars from leaking into tests through get_secret's override."""
    for name in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def secrets_file(tmp_path: Path) -> Path:
    path = tmp_path / "secrets.toml"
    path.write_text(SECRETS_TOML, encoding="utf-8")
    return path
