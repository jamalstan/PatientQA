"""Persona voice casting is deterministic, cached, and failure-tolerant."""

from pathlib import Path

from factories import valid_entry

from patientqa.voicing import VoiceCaster, bucket_key, fallback_voice


class _Voice:
    def __init__(self, voice_id: str, name: str, gender: str = "female") -> None:
        self.voice_id = voice_id
        self.name = name
        self.labels = {"gender": gender}


class _Voices:
    def __init__(self, *, fail: bool = False, voices=None, precise_empty: bool = False) -> None:
        self.fail = fail
        self.returned = voices or [_Voice("voice-a", "A"), _Voice("voice-b", "B")]
        self.precise_empty = precise_empty
        self.calls = 0
        self.kwargs = []

    def search(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        if self.fail:
            raise RuntimeError("offline")
        voices = [] if self.precise_empty and kwargs.get("search") else self.returned
        return type("Result", (), {"voices": voices})()


class _Client:
    def __init__(self, *, fail: bool = False, voices=None, precise_empty: bool = False) -> None:
        self.voices = _Voices(
            fail=fail,
            voices=voices,
            precise_empty=precise_empty,
        )


def test_bucket_key_tracks_gender_and_age() -> None:
    assert bucket_key(valid_entry()) == "female/old"


def test_library_voice_is_cached_on_disk(tmp_path: Path) -> None:
    entry = valid_entry()
    client = _Client()
    cache = tmp_path / "voices.json"
    first = VoiceCaster(client, cache_path=cache).cast(entry)
    second = VoiceCaster(client, cache_path=cache).cast(entry)

    assert first == second
    assert first.voice_id in {"voice-a", "voice-b"}
    assert client.voices.calls == 1


def test_voice_search_failure_uses_known_default(tmp_path: Path) -> None:
    entry = valid_entry()
    cast = VoiceCaster(_Client(fail=True), cache_path=tmp_path / "v.json").cast(entry)
    assert cast == fallback_voice(entry)
    assert cast.gender == "female"


def test_empty_precise_search_retries_broad_gender_bucket(tmp_path: Path) -> None:
    client = _Client(precise_empty=True)

    cast = VoiceCaster(client, cache_path=tmp_path / "v.json").cast(valid_entry())

    assert cast.origin == "library"
    assert cast.gender == "female"
    assert client.voices.calls == 2
    assert "search" in client.voices.kwargs[0]
    assert "search" not in client.voices.kwargs[1]


def test_wrong_gender_library_result_is_rejected(tmp_path: Path) -> None:
    entry = valid_entry()
    client = _Client(voices=[_Voice("male-voice", "Wrong", gender="male")])

    cast = VoiceCaster(client, cache_path=tmp_path / "v.json").cast(entry)

    assert cast == fallback_voice(entry)
    assert client.voices.calls == 3


def test_nonbinary_fallback_is_neutral_voice(tmp_path: Path) -> None:
    entry = valid_entry(persona={"gender": "nonbinary"})

    cast = VoiceCaster(_Client(fail=True), cache_path=tmp_path / "v.json").cast(entry)

    assert cast.voice_id == "SAz9YHcvj6GT2YYXdXww"
    assert cast.gender == "nonbinary"


def test_unverified_cached_voice_is_recast(tmp_path: Path) -> None:
    cache = tmp_path / "v.json"
    cache.write_text(
        '{"female/old": {"voice_id": "old", "name": "Old", "origin": "library"}}',
        encoding="utf-8",
    )
    client = _Client()

    cast = VoiceCaster(client, cache_path=cache).cast(valid_entry())

    assert cast.gender == "female"
    assert cast.voice_id in {"voice-a", "voice-b"}
    assert client.voices.calls == 1


def test_manifest_voice_with_wrong_gender_is_rejected(tmp_path: Path) -> None:
    entry = valid_entry(persona={"voice": {"voice_id": "wrong-male"}})
    client = _Client(voices=[_Voice("wrong-male", "Wrong", gender="male")])

    cast = VoiceCaster(client, cache_path=tmp_path / "v.json").cast(entry)

    assert cast == fallback_voice(entry)
