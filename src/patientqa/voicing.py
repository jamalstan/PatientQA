"""Casting a persona to an ElevenLabs voice (DESIGN.md §3.3).

Two modes:

- **library** (default, free): search ElevenLabs' shared voice library for a
  public voice matching the persona's age band, gender and design prompt;
  pick deterministically per bucket so every elderly Cuban-American woman in
  the campaign is voiced consistently. Results are cached on disk so a
  campaign run doesn't re-search (and stays stable if the library shifts).
- **design** (opt-in, ``--design-voices``): true Voice Design v3 —
  ``create_previews`` then ``create`` — a bespoke voice per persona from its
  ``design_prompt``. Preview generation bills only the preview text, but
  instantiating a voice can carry a substantial one-time credit fee on some
  plans, so it is never the default and any failure falls back to the
  library cast.

Every failure path degrades to a known gender-matched premade voice. A single
global fallback is unsafe: it previously paired female personas such as Nicole
with Will's male voice whenever a precise library search returned no results.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from patientqa.datagen.schemas import ManifestEntry

log = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path("tmp") / "voice_cast_cache.json"

#: Kept for ad-hoc callers with no persona. Campaigns use ``fallback_voice``.
DEFAULT_VOICE_ID = "bIHbv24MWmeRgasZH58o"

_FALLBACKS = {
    "male": (DEFAULT_VOICE_ID, "Will — Relaxed Optimist"),
    "female": ("EXAVITQu4vr4xnSDxMaL", "Sarah — Mature, Reassuring, Confident"),
    "nonbinary": ("SAz9YHcvj6GT2YYXdXww", "River — Relaxed, Neutral, Informative"),
}
_API_GENDERS = {"male": "male", "female": "female", "nonbinary": "neutral"}


@dataclass(frozen=True)
class CastVoice:
    voice_id: str
    name: str = ""
    origin: str = "gender_default"  # gender_default | library | designed | manifest
    gender: str = ""


def fallback_voice(entry: ManifestEntry) -> CastVoice:
    """A verified premade voice matching the manifest's persona gender."""
    gender = entry.persona.gender
    voice_id, name = _FALLBACKS[gender]
    return CastVoice(voice_id, name=name, origin="gender_default", gender=gender)


def _age_bucket(age: int) -> str:
    if age < 30:
        return "young"
    if age < 60:
        return "middle_aged"
    return "old"


def bucket_key(entry: ManifestEntry) -> str:
    """The casting bucket: enough to match a voice, coarse enough to cache."""
    gender = entry.persona.gender if entry.persona.gender in ("male", "female") else ""
    return f"{gender or 'any'}/{_age_bucket(entry.persona.age)}"


class VoiceCaster:
    """Persona → voice id, with a disk cache and graceful degradation."""

    def __init__(
        self,
        client: Any = None,
        *,
        cache_path: Path | None = None,
        design: bool = False,
    ) -> None:
        if client is None:
            from elevenlabs.client import ElevenLabs

            from patientqa.config import get_secret

            client = ElevenLabs(api_key=get_secret("elevenlabs", "api_key"))
        self._client = client
        self._cache_path = cache_path if cache_path is not None else DEFAULT_CACHE_PATH
        self._design = design
        self._bucket_cache = self._load_cache()

    # -- public ---------------------------------------------------------------

    def cast(self, entry: ManifestEntry) -> CastVoice:
        """Resolve the entry's voice; never raises."""
        if entry.persona.voice.voice_id:
            manifested = self._manifest_voice(entry)
            if manifested is not None:
                return manifested
        if self._design:
            designed = self._design_voice(entry)
            if designed is not None:
                return designed
        return self._library_voice(entry)

    # -- library casting ------------------------------------------------------

    def _library_voice(self, entry: ManifestEntry) -> CastVoice:
        key = bucket_key(entry)
        if key in self._bucket_cache:
            cached = CastVoice(**self._bucket_cache[key])
            if cached.gender == entry.persona.gender:
                return cached
            log.warning("discarding unverified or mismatched voice cache entry for %s", key)
        api_gender = _API_GENDERS[entry.persona.gender]
        age = _age_bucket(entry.persona.age)
        usable: list[Any] = []
        searches = (
            {"search": _search_text(entry), "gender": api_gender, "age": age},
            {"gender": api_gender, "age": age},
            {"gender": api_gender},
        )
        for filters in searches:
            try:
                voices = self._client.voices.search(page_size=10, **filters).voices
            except Exception as exc:
                log.warning("voice search failed for %s (%r)", key, exc)
                break
            usable = [
                voice
                for voice in voices
                if getattr(voice, "voice_id", "")
                and _voice_api_gender(voice) == api_gender
            ]
            if usable:
                break
        if not usable:
            log.warning("voice search returned no verified match for %s; using safe fallback", key)
            return fallback_voice(entry)
        pick = usable[entry.seed % len(usable)]
        cast = CastVoice(
            pick.voice_id,
            name=getattr(pick, "name", ""),
            origin="library",
            gender=entry.persona.gender,
        )
        self._remember(key, cast)
        return cast

    def _manifest_voice(self, entry: ManifestEntry) -> CastVoice | None:
        """Use an explicit voice ID only after its library gender is verified."""
        voice_id = entry.persona.voice.voice_id
        assert voice_id is not None
        expected = _API_GENDERS[entry.persona.gender]
        try:
            voices = self._client.voices.search(voice_ids=[voice_id], page_size=10).voices
        except Exception as exc:
            log.warning(
                "could not verify manifest voice for %s (%r); using safe casting",
                entry.call_id,
                exc,
            )
            return None
        match = next(
            (voice for voice in voices if getattr(voice, "voice_id", "") == voice_id),
            None,
        )
        actual = _voice_api_gender(match) if match is not None else ""
        if actual != expected:
            log.warning(
                "manifest voice gender mismatch for %s: expected %s, got %s",
                entry.call_id,
                expected,
                actual or "unverified",
            )
            return None
        return CastVoice(
            voice_id,
            name=getattr(match, "name", ""),
            origin="manifest",
            gender=entry.persona.gender,
        )

    # -- voice design (opt-in) --------------------------------------------------

    def _design_voice(self, entry: ManifestEntry) -> CastVoice | None:
        description = entry.persona.voice.design_prompt or f"{entry.persona.age} year old patient"
        try:
            previews = self._client.text_to_voice.create_previews(
                voice_description=description,
                text="Hello, I'm calling about my appointment.",
                seed=entry.seed,
            )
            preview = previews.previews[0]
            voice = self._client.text_to_voice.create(
                voice_name=f"sim-{entry.call_id}",
                voice_description=description,
                generated_voice_id=preview.generated_voice_id,
            )
            log.info("designed voice for %s: %s", entry.call_id, voice.voice_id)
            return CastVoice(
                voice.voice_id,
                name=f"sim-{entry.call_id}",
                origin="designed",
                gender=entry.persona.gender,
            )
        except Exception as exc:
            log.warning("voice design failed for %s (%r); falling back", entry.call_id, exc)
            return None

    # -- cache ------------------------------------------------------------------

    def _load_cache(self) -> dict[str, dict[str, str]]:
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return {}

    def _remember(self, key: str, cast: CastVoice) -> None:
        record = {
            "voice_id": cast.voice_id,
            "name": cast.name,
            "origin": cast.origin,
            "gender": cast.gender,
        }
        self._bucket_cache[key] = record
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._bucket_cache, indent=2, sort_keys=True), encoding="utf-8"
            )
        except OSError as exc:
            log.debug("voice cache write failed: %r", exc)


def _search_text(entry: ManifestEntry) -> str:
    """The search query: the persona's own design prompt, trimmed to its essence."""
    prompt = entry.persona.voice.design_prompt or ""
    return prompt[:120] if prompt else "warm natural patient voice"


def _voice_api_gender(voice: Any) -> str:
    labels = getattr(voice, "labels", None)
    return str(labels.get("gender", "")).lower() if isinstance(labels, dict) else ""
