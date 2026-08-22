"""STT provider-agnostic layer tests (DESIGN.md §3.4, §5.3): event model,
per-persona language pinning, keyword biasing, and provider selection."""

from pathlib import Path

import pytest

from patientqa.deepgram.client import DeepgramSettings
from patientqa.elevenlabs.scribe import ScribeSettings
from patientqa.stt import (
    PROVIDER_DEEPGRAM,
    Final,
    Partial,
    build_stt_settings,
    persona_keywords,
    persona_languages,
    stt_provider,
)


def test_provider_defaults_to_deepgram_when_section_missing(tmp_path: Path) -> None:
    path = tmp_path / "secrets.toml"
    path.write_text("[deepgram]\napi_key = 'k'\n", encoding="utf-8")
    assert stt_provider(path) == PROVIDER_DEEPGRAM


def test_provider_reads_selection_from_secrets(secrets_file: Path) -> None:
    assert stt_provider(secrets_file) == PROVIDER_DEEPGRAM


def test_provider_rejects_unknown_name(tmp_path: Path) -> None:
    path = tmp_path / "secrets.toml"
    path.write_text("[stt]\nprovider = 'whisper'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provider"):
        stt_provider(path)


def test_english_persona_pins_english_only() -> None:
    assert persona_languages("English") == ("en", ())
    assert persona_languages("") == ("en", ())


def test_code_switching_persona_adds_spanish_secondary() -> None:
    assert persona_languages("English w/ Spanish code-switching") == ("en", ("es",))


def test_spanish_heavy_persona_flips_primary() -> None:
    assert persona_languages("Spanish-heavy call with English drug names") == ("es", ("en",))


def test_persona_keywords_take_explicit_terms_and_doctor_names() -> None:
    persona = {
        "stt_keywords": ["metformin", "Dr. Ortiz"],
        "background": "Sees Dr. Ortiz quarterly; Doctor Mendez once. Metformin daily.",
    }
    assert persona_keywords(persona) == ("metformin", "Dr. Ortiz", "Doctor Mendez")


def test_persona_keywords_deduplicate() -> None:
    persona = {"stt_keywords": ["metformin"], "background": "metformin mentions"}
    assert persona_keywords(persona) == ("metformin",)


def test_build_stt_settings_deepgram_pinned_to_persona(secrets_file: Path) -> None:
    persona = {
        "language": "English w/ Spanish code-switching",
        "stt_keywords": ["metformin"],
    }
    settings = build_stt_settings(persona, path=secrets_file)
    assert isinstance(settings, DeepgramSettings)
    assert settings.api_key == "test-deepgram"
    assert settings.language == "en"
    assert settings.keywords == ("metformin",)


def test_build_stt_settings_scribe_fallback_uses_elevenlabs_key(tmp_path: Path) -> None:
    path = tmp_path / "secrets.toml"
    path.write_text(
        "[elevenlabs]\napi_key = 'test-elevenlabs'\n"
        "[stt]\nprovider = 'scribe'\n",
        encoding="utf-8",
    )
    settings = build_stt_settings(
        {"language": "Spanish-heavy call with English drug names"}, path=path
    )
    assert isinstance(settings, ScribeSettings)
    assert settings.api_key == "test-elevenlabs"
    assert settings.language_code == "es"
    assert settings.secondary_languages == ("en",)


def test_event_kinds_are_distinct() -> None:
    assert Partial(text="Tue") != Final(text="Tue")
    assert Final(text="Tuesday.").text == "Tuesday."
