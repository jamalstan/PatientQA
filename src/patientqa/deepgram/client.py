"""Deepgram Nova-3 streaming STT — primary (DESIGN.md §3.4, §5.3).

Chosen for the raw socket's tunable silence endpointing (100–300 ms — the
§4 turn-end budget) and native 8 kHz μ-law intake, so inbound Twilio frames
are forwarded byte-exact with no resampling. Nova-3 is phone-conversational
by training, which matters: transcripts feed the semantic gate and the
success-criteria checks, not just human ears.

This module is the settings/URL/parser layer; the live WebSocket pump lands
with the call-loop stage, exactly like the other providers.
"""

import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patientqa.config import get_secret

DEFAULT_MODEL = "nova-3"
DEFAULT_BASE_URL = "wss://api.deepgram.com/v1/listen"
#: §4 budget: ~150 ms endpointing + final transcript, 150–350 ms total.
DEFAULT_ENDPOINTING_MS = 250


@dataclass(frozen=True)
class DeepgramSettings:
    """Everything the raw streaming socket needs, persona-pinned (§5.3)."""

    api_key: str
    model: str = DEFAULT_MODEL
    language: str = "en"  # pinned per persona — never auto-detect
    encoding: str = "mulaw"  # Twilio Media Streams wire format
    sample_rate: int = 8000
    channels: int = 1
    endpointing_ms: int = DEFAULT_ENDPOINTING_MS
    interim_results: bool = True
    punctuate: bool = True
    smart_format: bool = True
    keywords: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_secrets(
        cls,
        path: Path | None = None,
        *,
        language: str = "en",
        keywords: tuple[str, ...] = (),
        endpointing_ms: int = DEFAULT_ENDPOINTING_MS,
    ) -> "DeepgramSettings":
        return cls(
            api_key=get_secret("deepgram", "api_key", path=path),
            language=language,
            keywords=keywords,
            endpointing_ms=endpointing_ms,
        )


def build_deepgram_url(
    settings: DeepgramSettings, base_url: str = DEFAULT_BASE_URL
) -> str:
    """The streaming-WS URL with every setting as a query parameter.

    Deterministic parameter order keeps the URL assertable in tests, and
    building it explicitly is the discipline the §5.3 survey endorsed:
    settings that ride the URL are settings the server actually applied.
    """
    params: list[tuple[str, str]] = [
        ("model", settings.model),
        ("language", settings.language),
        ("encoding", settings.encoding),
        ("sample_rate", str(settings.sample_rate)),
        ("channels", str(settings.channels)),
        ("endpointing", str(settings.endpointing_ms)),
        ("interim_results", "true" if settings.interim_results else "false"),
        ("punctuate", "true" if settings.punctuate else "false"),
        ("smart_format", "true" if settings.smart_format else "false"),
    ]
    params.extend(("keywords", keyword) for keyword in settings.keywords)
    return base_url + "?" + urllib.parse.urlencode(params)


def parse_deepgram_message(message: Mapping[str, Any]):
    """One Deepgram socket message → :data:`patientqa.stt.SttEvent` or ``None``.

    ``Results`` messages map to :class:`~patientqa.stt.Final` when Deepgram
    finalized the segment (``is_final``) and to
    :class:`~patientqa.stt.Partial` otherwise; empty transcripts and
    non-transcription messages (``Metadata``, ``UtteranceEnd``, …) map to
    ``None``.
    """
    from patientqa.stt import Final, Partial

    if message.get("type") != "Results":
        return None
    try:
        transcript = str(message["channel"]["alternatives"][0]["transcript"] or "")
    except (KeyError, IndexError, TypeError):
        return None
    if not transcript.strip():
        return None
    if message.get("is_final"):
        return Final(text=transcript.strip())
    return Partial(text=transcript.strip())
