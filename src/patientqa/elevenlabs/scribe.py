"""ElevenLabs Scribe v2 Realtime — the fallback STT (DESIGN.md §3.4, §5.3).

Documented fallback, not the primary: Scribe bills 330 credits/min from the
same pool as TTS (~59K credits for a 180-min campaign — roughly the entire
§8 budget), while Deepgram runs inside a $200 signup credit and endpointing
tuned to the §4 latency budget. If the Deepgram credit ever dries up, the
swap is this module + ``[stt] provider = "scribe"`` in secrets.toml — zero
format work, because Scribe also accepts ``ulaw_8000`` natively.

The URL recipe below is the one verified against the API reference during
the §5.3 survey (2026-08), *including* the correction the survey made: with
a semantic gate catching false turn-ends, the VAD silence window can stay
tight (~0.5 s) instead of the 0.8–1.0 s a gateless pipeline needs to avoid
committing mid-sentence pauses.
"""

import base64
import json
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patientqa.config import get_secret

DEFAULT_MODEL = "scribe_v2_realtime"
DEFAULT_BASE_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


@dataclass(frozen=True)
class ScribeSettings:
    """Scribe Realtime socket settings; all ride the URL as query params."""

    api_key: str
    model_id: str = DEFAULT_MODEL
    audio_format: str = "ulaw_8000"  # Twilio Media Streams wire format, native
    language_code: str = "en"  # ISO 639-1/639-3; pinned per persona (§5.3)
    secondary_languages: tuple[str, ...] = field(default_factory=tuple)
    commit_strategy: str = "vad"  # or "manual"; vad commits on silence
    vad_threshold: float = 0.5
    vad_silence_threshold_secs: float = 0.5  # tight: the semantic gate backstops us
    min_speech_duration_ms: int = 250
    filter_background_audio: bool = False
    keyterms: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_secrets(
        cls,
        path: Path | None = None,
        *,
        language_code: str = "en",
        secondary_languages: tuple[str, ...] = (),
        keyterms: tuple[str, ...] = (),
    ) -> "ScribeSettings":
        return cls(
            api_key=get_secret("elevenlabs", "api_key", path=path),
            language_code=language_code,
            secondary_languages=secondary_languages,
            keyterms=keyterms,
        )


def build_scribe_url(settings: ScribeSettings, base_url: str = DEFAULT_BASE_URL) -> str:
    """The verified query-param recipe: format, pinned language, VAD commit."""
    params: list[tuple[str, str]] = [
        ("model_id", settings.model_id),
        ("audio_format", settings.audio_format),
        ("language_code", settings.language_code),
    ]
    params.extend(("secondary_languages", code) for code in settings.secondary_languages)
    params.extend(("keyterms", term) for term in settings.keyterms)
    params.extend(
        [
            ("commit_strategy", settings.commit_strategy),
            ("vad_threshold", str(settings.vad_threshold)),
            ("vad_silence_threshold_secs", str(settings.vad_silence_threshold_secs)),
            ("min_speech_duration_ms", str(settings.min_speech_duration_ms)),
            (
                "filter_background_audio",
                "true" if settings.filter_background_audio else "false",
            ),
        ]
    )
    return base_url + "?" + urllib.parse.urlencode(params)


def wrap_scribe_audio(
    ulaw: bytes, commit: bool = False, sample_rate: int = 8000
) -> str:
    """One μ-law chunk as Scribe's only client→server protocol message.

    Audio rides the socket JSON-encoded (``input_audio_chunk``), never as raw
    binary frames — the server answers those with ``input_error`` /
    ``"Message must be a valid protocol message"`` and closes. ``commit=True``
    requests an immediate segment commit (manual strategy); with ``vad`` the
    server commits on silence and the flag stays False.
    """
    return json.dumps(
        {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(ulaw).decode("ascii"),
            "commit": commit,
            "sample_rate": sample_rate,
        }
    )


def parse_scribe_message(message: Mapping[str, Any]):
    """One Scribe socket message → :data:`patientqa.stt.SttEvent` or ``None``.

    ``partial_transcript`` → :class:`~patientqa.stt.Partial` (may still
    change); ``final_transcript``/``committed_transcript`` (with or without
    the ``_with_timestamps`` suffix) → :class:`~patientqa.stt.Final`. Only
    committed transcripts ever reach the brain (§5.3). Field names are
    accepted tolerantly because the API is young.
    """
    from patientqa.stt import Final, Partial

    kind = message.get("message_type") or message.get("type") or ""
    text = str(message.get("text") or message.get("transcript") or "").strip()
    if kind == "partial_transcript":
        return Partial(text=text)
    if kind.startswith(("final_transcript", "committed_transcript")):
        return Final(text=text) if text else None
    return None


def verify_scribe_session(
    message: Mapping[str, Any], settings: ScribeSettings
) -> list[str]:
    """Check the ``session_started`` config echo against what we asked for.

    The §5.3 survey's debugging discipline: Scribe can accept a connection
    while silently dropping URL parameters, and the echo is the only place
    the truth shows. Returns a list of problems — empty means the session is
    configured as intended.
    """
    problems: list[str] = []
    config = message.get("config")
    if not isinstance(config, Mapping):
        return [f"session_started has no config echo: {message!r}"]
    for key, expected in (
        ("audio_format", settings.audio_format),
        ("language_code", settings.language_code),
        ("model_id", settings.model_id),
    ):
        actual = config.get(key)
        if actual != expected:
            problems.append(
                f"session config {key}={actual!r} but requested {expected!r}"
            )
    # The commit strategy echoes under two shapes: the AsyncAPI-documented
    # ``commit_strategy`` string, or v2's ``vad_commit_strategy`` boolean.
    if "commit_strategy" in config:
        actual = config.get("commit_strategy")
        if actual != settings.commit_strategy:
            problems.append(
                f"session config commit_strategy={actual!r} "
                f"but requested {settings.commit_strategy!r}"
            )
    elif config.get("vad_commit_strategy") is not None:
        want_vad = settings.commit_strategy == "vad"
        if bool(config["vad_commit_strategy"]) is not want_vad:
            problems.append(
                f"session config vad_commit_strategy={config['vad_commit_strategy']!r} "
                f"but requested commit_strategy={settings.commit_strategy!r}"
            )
    else:
        problems.append("session config echoes no commit strategy at all")
    return problems
