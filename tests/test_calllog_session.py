import json
import wave
from array import array
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from patientqa.calllog.session import (
    AUDIO_INBOUND,
    AUDIO_PLAYED,
    CALL_CONNECTED,
    CALL_STARTED,
    SAMPLE_RATE,
    TURN_AGENT,
    TURN_PATIENT,
    CallSession,
    leg_placements,
    mixdown_wav,
    read_jsonl,
)
from patientqa.calllog.ulaw import ulaw_byte_to_pcm16


class FakeClock:
    """Deterministic monotonic + wall clocks the session reads."""

    def __init__(self) -> None:
        self.mono = 1000.0
        self.wall = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)

    def perf_counter(self) -> float:
        return self.mono

    def utc_now(self) -> datetime:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += timedelta(seconds=seconds)


def start_session(
    tmp_path: Path, call_id: str = "call-001", manifest: dict | None = None
) -> tuple[CallSession, FakeClock]:
    clock = FakeClock()
    session = CallSession.start(
        tmp_path,
        call_id,
        manifest if manifest is not None else {"persona": {"name": "Marta Reyes"}},
        utc_now=clock.utc_now,
        perf_counter=clock.perf_counter,
    )
    return session, clock


def test_start_creates_folder_layout(tmp_path: Path) -> None:
    intent = {
        "intentional": True,
        "behavior": "long_silence",
        "isolation": "single_behavior",
        "hypothesis": "A pause may cause premature turn completion.",
        "protocol": "introduce_only_this_behavior",
    }
    session, _ = start_session(
        tmp_path,
        manifest={"persona": {"name": "Marta Reyes"}, "test_intent": intent},
    )

    assert session.directory.name.startswith("call-001_20260817T120000Z")
    assert (session.directory / "audio").is_dir()
    assert session.inbound_path.read_bytes() == b""
    assert session.outbound_path.read_bytes() == b""

    meta = json.loads((session.directory / "meta.json").read_text(encoding="utf-8"))
    assert meta["call_id"] == "call-001"
    assert meta["manifest"]["persona"]["name"] == "Marta Reyes"
    assert meta["test_intent"] == intent
    assert meta["format_version"] == 1

    events = read_jsonl(session.session_path)
    assert [e["type"] for e in events] == [CALL_STARTED]
    assert events[0]["seq"] == 0
    assert events[0]["t_ms"] == 0
    assert events[0]["data"]["manifest"]["persona"]["name"] == "Marta Reyes"


def test_folder_name_is_sanitized_and_collision_safe(tmp_path: Path) -> None:
    clock = FakeClock()
    kwargs = {"utc_now": clock.utc_now, "perf_counter": clock.perf_counter}
    first = CallSession.start(tmp_path, "weird/call id!", **kwargs).directory.name
    second = CallSession.start(tmp_path, "weird/call id!", **kwargs).directory.name

    assert first == "weird-call-id_20260817T120000Z"
    assert second == "weird-call-id_20260817T120000Z-2"


def test_log_appends_ordered_events(tmp_path: Path) -> None:
    session, clock = start_session(tmp_path)
    clock.advance(1.5)
    session.log(TURN_AGENT, text="Clinic, how can I help?")
    clock.advance(0.4)
    session.log(TURN_PATIENT, text="Tuesday morning, please.", respond_ms=900)

    events = read_jsonl(session.session_path)
    assert [e["type"] for e in events] == [CALL_STARTED, TURN_AGENT, TURN_PATIENT]
    assert [e["seq"] for e in events] == [0, 1, 2]
    assert [e["t_ms"] for e in events] == [0, 1500, 1900]
    assert events[1]["wall"] == "2026-08-17T12:00:01.500+00:00"
    assert events[2]["data"]["respond_ms"] == 900


def test_append_legs_counts_bytes(tmp_path: Path) -> None:
    session, _ = start_session(tmp_path)
    assert session.append_inbound(b"\xff" * 160) == 160
    assert session.append_inbound(b"\xff" * 40) == 200
    assert session.append_outbound(b"\x80" * 80) == 80
    assert session.inbound_path.stat().st_size == 200
    assert session.outbound_path.stat().st_size == 80


def test_close_finalizes_mono_recording_placed_on_the_timeline(tmp_path: Path) -> None:
    session, clock = start_session(tmp_path)
    clock.advance(2.0)  # dial + ring before the stream connects
    session.log(CALL_CONNECTED, call_sid="CA0", stream_sid="MS0")
    clock.advance(0.5)
    session.append_inbound(b"\x80" * 4000)  # 0.5 s of loud agent audio at t=2.5 s
    clock.advance(3.0)  # call lasted 5.5 s total -> padding must win
    session.close("objective_achieved")

    with wave.open(str(session.recording_path), "rb") as wav:
        assert wav.getnchannels() == 1  # mono: both parties in both ears
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnframes() == 5500 * SAMPLE_RATE // 1000
        mono = array("h")
        mono.frombytes(wav.readframes(wav.getnframes()))

    loud = ulaw_byte_to_pcm16(0x80)
    silence = array("h", bytes(2 * 2500 * SAMPLE_RATE // 1000))
    assert mono[:2500 * SAMPLE_RATE // 1000] == silence  # ring time is silence
    assert mono[2500 * SAMPLE_RATE // 1000 : 3000 * SAMPLE_RATE // 1000] == array(
        "h", [loud] * 4000
    )  # agent leg anchored at its audio.inbound t_ms, not at 0
    assert mono[3000 * SAMPLE_RATE // 1000 :] == array(
        "h", bytes(2 * (5500 * SAMPLE_RATE // 1000 - 3000 * SAMPLE_RATE // 1000))
    )  # padded tail


def test_close_places_outbound_runs_at_their_event_times(tmp_path: Path) -> None:
    session, clock = start_session(tmp_path)
    clock.advance(1.0)
    session.append_inbound(b"\xff" * 8000)  # silent agent leg from t=1.0 s
    clock.advance(0.2)
    session.append_outbound(b"\x80" * 800)  # 0.1 s of patient audio at t=1.2 s
    session.log(AUDIO_PLAYED, audio_bytes=800)
    clock.advance(1.0)
    session.append_outbound(b"\x80" * 800)  # …and again at t=2.2 s
    session.log(AUDIO_PLAYED, audio_bytes=800)
    clock.advance(0.5)
    session.close("objective_achieved")

    with wave.open(str(session.recording_path), "rb") as wav:
        mono = array("h")
        mono.frombytes(wav.readframes(wav.getnframes()))

    loud = ulaw_byte_to_pcm16(0x80)
    assert mono[1200 * 8 : 1300 * 8] == array("h", [loud] * 800)
    assert mono[2200 * 8 : 2300 * 8] == array("h", [loud] * 800)
    assert mono[1300 * 8 : 2200 * 8] == array("h", bytes(2 * 900 * 8))  # gap is quiet


def test_mixdown_packs_outbound_runs_sequentially() -> None:
    # Two 1 s runs whose event times overlap by 90 %: synthesis outran
    # playback, but the line's single FIFO queue played them back-to-back —
    # the mixdown must not stack them on top of each other.
    wav_bytes = mixdown_wav(
        b"",
        b"\x80" * 8000 + b"\xff" * 8000,
        0,
        outbound_segments=[(0, 8000), (100, 8000)],
    )
    import io

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        mono = array("h")
        mono.frombytes(wav.readframes(wav.getnframes()))

    loud = ulaw_byte_to_pcm16(0x80)
    quiet = ulaw_byte_to_pcm16(0xFF)
    assert mono[:8000] == array("h", [loud] * 8000)  # first run at its event time
    assert mono[8000:16000] == array("h", [quiet] * 8000)  # second waits, at 1 s


def test_mixdown_saturates_when_legs_overlap() -> None:
    wav_bytes = mixdown_wav(
        b"\x80" * 100,  # +32124
        b"\x80" * 100,  # +32124 on top → clipped, not wrapped
        0,
        inbound_offset_ms=0,
        outbound_segments=[(0, 100)],
    )
    import io

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        mono = array("h")
        mono.frombytes(wav.readframes(wav.getnframes()))
    assert mono[:100] == array("h", [32767] * 100)


def test_leg_placements_prefers_the_exact_anchor_and_maps_runs() -> None:
    events = [
        {"type": CALL_STARTED, "t_ms": 0, "data": {}},
        {"type": CALL_CONNECTED, "t_ms": 6900, "data": {}},
        {"type": AUDIO_INBOUND, "t_ms": 6960, "data": {}},
        {"type": AUDIO_PLAYED, "t_ms": 13300, "data": {"audio_bytes": 24000}},
        {"type": AUDIO_PLAYED, "t_ms": 23000, "data": {"audio_bytes": 16000}},
    ]
    assert leg_placements(events) == (6960, [(13300, 24000), (23000, 16000)])


def test_leg_placements_falls_back_to_connect_then_zero() -> None:
    legacy = [{"type": CALL_CONNECTED, "t_ms": 9000, "data": {}}]
    assert leg_placements(legacy) == (9000, [])
    assert leg_placements([]) == (0, [])
    # Event objects (close()'s in-memory list) work like read_jsonl dicts
    from patientqa.calllog.session import Event

    events = [
        Event(seq=0, t_ms=0, wall="w", event_type=CALL_STARTED),
        Event(seq=1, t_ms=4100, wall="w", event_type=CALL_CONNECTED),
        Event(seq=2, t_ms=4200, wall="w", event_type=AUDIO_INBOUND),
        Event(seq=3, t_ms=5000, wall="w", event_type=AUDIO_PLAYED, data={"audio_bytes": 800}),
    ]
    assert leg_placements(events) == (4200, [(5000, 800)])


def test_close_writes_transcript_and_summary(tmp_path: Path) -> None:
    intent = {
        "intentional": True,
        "behavior": "long_silence",
        "isolation": "single_behavior",
        "hypothesis": "A pause may cause premature turn completion.",
        "protocol": "introduce_only_this_behavior",
    }
    session, clock = start_session(
        tmp_path,
        manifest={"persona": {"name": "Marta"}, "test_intent": intent},
    )
    clock.advance(1.0)
    session.log(TURN_AGENT, text="How can I help?")
    clock.advance(0.2)
    session.log("brain.reply", say="I need an appointment", latency_ms=200)
    session.log("tts.done", chars=21, audio_bytes=16000, latency_ms=140)
    clock.advance(0.2)
    session.log(TURN_PATIENT, text="I need an appointment", respond_ms=340)
    clock.advance(0.5)
    session.close("objective_achieved")

    transcript = json.loads(session.transcript_path.read_text(encoding="utf-8"))
    assert transcript["end_reason"] == "objective_achieved"
    assert [(t["role"], t["text"]) for t in transcript["turns"]] == [
        ("agent", "How can I help?"),
        ("patient", "I need an appointment"),
    ]
    assert transcript["turns"][1]["respond_ms"] == 340

    summary = json.loads(session.summary_path.read_text(encoding="utf-8"))
    assert summary["end_reason"] == "objective_achieved"
    assert summary["duration_ms"] == 1900
    assert summary["stats"]["agent_turns"] == 1
    assert summary["stats"]["patient_turns"] == 1
    assert summary["stats"]["respond_ms_avg"] == 340
    assert summary["stats"]["respond_ms_max"] == 340
    assert summary["manifest"]["persona"]["name"] == "Marta"
    assert summary["test_intent"] == intent


def test_close_is_idempotent(tmp_path: Path) -> None:
    session, _ = start_session(tmp_path)
    session.close("timeout")
    events_after_first = read_jsonl(session.session_path)
    session.close("completed")

    assert read_jsonl(session.session_path) == events_after_first
    summary = json.loads(session.summary_path.read_text(encoding="utf-8"))
    assert summary["end_reason"] == "timeout"


def test_context_manager_finalizes_abandoned_on_error(tmp_path: Path) -> None:
    clock = FakeClock()
    with pytest.raises(RuntimeError), CallSession.start(
        tmp_path,
        "call-002",
        utc_now=clock.utc_now,
        perf_counter=clock.perf_counter,
    ) as session:
        raise RuntimeError("boom")

    summary = json.loads(session.summary_path.read_text(encoding="utf-8"))
    assert summary["end_reason"] == "abandoned"
    assert session.recording_path.is_file()


def test_read_jsonl_is_tolerant(tmp_path: Path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text(
        '{"seq": 0, "t_ms": 0, "wall": "x", "type": "note", "data": {}}\n'
        "not json at all\n"
        "\n"
        '{"seq": 1, "t_ms": 5, "wall": "x", "type": "note", "data": {}}\n',
        encoding="utf-8",
    )
    assert [e["seq"] for e in read_jsonl(path)] == [0, 1]
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_mixdown_pads_to_requested_duration() -> None:
    wav_bytes = mixdown_wav(b"", b"", 1500)
    import io

    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getnframes() == 1500 * SAMPLE_RATE // 1000
