from pathlib import Path

import pytest

from patientqa.config import find_secrets_file, get_secret, load_secrets

SECRETS_TOML = """\
[elevenlabs]
api_key = "test-elevenlabs"

[twilio]
api_key_sid = "SKtest"
api_key_secret = "shhh"
"""


def _write_secrets(directory: Path) -> Path:
    path = directory / "secrets.toml"
    path.write_text(SECRETS_TOML, encoding="utf-8")
    return path


def test_load_secrets_from_explicit_path(tmp_path: Path) -> None:
    assert load_secrets(_write_secrets(tmp_path))["elevenlabs"]["api_key"] == "test-elevenlabs"


def test_find_secrets_file_walks_up_parents(tmp_path: Path) -> None:
    _write_secrets(tmp_path)
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_secrets_file(nested) == tmp_path / "secrets.toml"


def test_find_secrets_file_errors_with_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="secrets.example.toml"):
        find_secrets_file(tmp_path)


def test_get_secret_returns_nested_value(tmp_path: Path) -> None:
    _write_secrets(tmp_path)
    assert get_secret("twilio", "api_key_sid", path=tmp_path / "secrets.toml") == "SKtest"


def test_get_secret_env_var_overrides_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_secrets(tmp_path)
    monkeypatch.setenv("TWILIO_API_KEY_SID", "from-env")
    assert get_secret("twilio", "api_key_sid", path=tmp_path / "secrets.toml") == "from-env"


def test_get_secret_missing_key_raises_with_hint(tmp_path: Path) -> None:
    _write_secrets(tmp_path)
    with pytest.raises(KeyError, match=r"\[deepgram\] api_key"):
        get_secret("deepgram", "api_key", path=tmp_path / "secrets.toml")


def test_repo_secrets_file_loads_when_present() -> None:
    """Skips on fresh checkouts (CI, contributors); validates the local file when present."""
    repo_root = Path(__file__).resolve().parents[1]
    if not (repo_root / "secrets.toml").is_file():
        pytest.skip("secrets.toml not created in this checkout")
    for section, values in load_secrets().items():
        assert values, f"[{section}] is empty in secrets.toml"
        for key, value in values.items():
            assert str(value).strip(), f"[{section}] {key} is empty in secrets.toml"
