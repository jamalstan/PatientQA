"""STT event model + per-persona settings (DESIGN.md §3.4, §5.3).

Provider-agnostic on purpose: the call loop consumes :data:`SttEvent`s and
never knows whether Deepgram (primary) or ElevenLabs Scribe (fallback) is on
the other end of the socket. The language is **pinned per call from the
manifest persona** rather than left to auto-detection — open detection over
90+ languages misclassifies short, noisy phone fragments (§5.3 survey), and
the persona already knows what language the call is in.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from patientqa.config import get_secret
from patientqa.deepgram.client import DeepgramSettings
from patientqa.elevenlabs.scribe import ScribeSettings

PROVIDER_DEEPGRAM = "deepgram"
PROVIDER_SCRIBE = "scribe"
PROVIDERS = (PROVIDER_DEEPGRAM, PROVIDER_SCRIBE)


@dataclass(frozen=True)
class Partial:
    """An interim transcript — still allowed to change; never a turn trigger."""

    text: str


@dataclass(frozen=True)
class Final:
    """A committed transcript — the §5.3 input to the semantic gate."""

    text: str


SttEvent = Partial | Final

#: "Dr. Ortiz", "Doctor Mendez" — bias these so the words the success
#: criteria check against transcribe reliably (§3.4 keyword biasing).
_DOCTOR_NAME = re.compile(r"\b(?:Dr\.?|Doctor)\s+[A-Z][a-zà-ÿ]+")


def stt_provider(path: Path | None = None) -> str:
    """Which STT backend to use: ``[stt] provider`` in secrets, default Deepgram."""
    try:
        provider = get_secret("stt", "provider", path=path).strip().lower()
    except KeyError:
        return PROVIDER_DEEPGRAM
    if provider not in PROVIDERS:
        raise ValueError(
            f"secrets.toml [stt] provider must be one of {', '.join(PROVIDERS)}; "
            f"got {provider!r}."
        )
    return provider


def persona_languages(persona_language: str) -> tuple[str, tuple[str, ...]]:
    """Manifest persona language → (primary ISO 639-1, secondaries).

    "English w/ Spanish code-switching" → ``("en", ("es",))``;
    "Spanish-heavy call with English drug names" → ``("es", ("en",))``;
    anything else English-first — the challenge grades an English line (§6.5).
    """
    text = persona_language.lower()
    if text.startswith("spanish"):
        return "es", ("en",)
    if "spanish" in text:
        return "en", ("es",)
    return "en", ()


def persona_keywords(persona: Mapping[str, object]) -> tuple[str, ...]:
    """Transcription bias terms for one persona (§3.4/§5.3 pinning).

    Explicit ``stt_keywords`` on the manifest entry, plus any doctor names
    mentioned in the persona background.
    """
    keywords: list[str] = []
    explicit = persona.get("stt_keywords") or ()
    keywords.extend(str(term) for term in explicit)
    background = str(persona.get("background") or "")
    keywords.extend(match.strip() for match in _DOCTOR_NAME.findall(background))
    seen: set[str] = set()
    return tuple(kw for kw in keywords if not (kw in seen or seen.add(kw)))


def build_stt_settings(
    persona: Mapping[str, object] | None = None, path: Path | None = None
) -> DeepgramSettings | ScribeSettings:
    """Settings for whichever provider secrets select, pinned to the persona."""
    language, secondary = persona_languages(
        str((persona or {}).get("language") or "")
    )
    keywords = persona_keywords(persona or {})
    if stt_provider(path) == PROVIDER_SCRIBE:
        return ScribeSettings.from_secrets(
            path,
            language_code=language,
            secondary_languages=secondary,
            keyterms=keywords,
        )
    return DeepgramSettings.from_secrets(path, language=language, keywords=keywords)
