"""Offline tests for the call loop (DESIGN.md §5.4 wiring) — fakes only."""

import asyncio
import base64
import json
import time

import pytest

from patientqa.calllog.session import CallSession, read_jsonl
from patientqa.calllog.ulaw import pcm16_to_ulaw_bytes
from patientqa.callloop import (
    END_OF_TURN_MARK,
    CallLoop,
    is_noninteractive_utterance,
    is_obviously_incomplete,
    is_terminal_utterance,
    media_payload,
    validate_call_duration,
    validate_voice_assignment,
    wrap_clear,
    wrap_mark,
    wrap_media,
)
from patientqa.orchestrator import TurnResult
from patientqa.stt import Final, Partial
from patientqa.turns import TurnState

#: 20 ms of 8 kHz μ-law at a healthy amplitude (RMS ≫ the gate's 500).
SPEECH = pcm16_to_ulaw_bytes([6000] * 160)
SILENCE = b"\xff" * 160


def test_live_call_duration_has_a_five_minute_ceiling() -> None:
    assert validate_call_duration(180) == 180
    assert validate_call_duration(300) == 300
    with pytest.raises(ValueError, match="300"):
        validate_call_duration(300.01)
    with pytest.raises(ValueError, match="greater than 0"):
        validate_call_duration(0)


def test_voice_assignment_rejects_gender_mismatch() -> None:
    manifest = {
        "voice": {
            "voice_id": "male-voice",
            "persona_gender": "female",
            "voice_gender": "male",
            "gender_match": False,
        }
    }

    with pytest.raises(ValueError, match="voice gender mismatch"):
        validate_voice_assignment("male-voice", manifest)


def test_voice_assignment_accepts_matching_provenance() -> None:
    manifest = {
        "voice": {
            "voice_id": "female-voice",
            "persona_gender": "female",
            "voice_gender": "female",
            "gender_match": True,
        }
    }

    validate_voice_assignment("female-voice", manifest)


class InlineExecutor:
    """Runs engages synchronously — the loop under test stays single-threaded."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)
        return _DoneFuture()


class _DoneFuture:
    def done(self) -> bool:
        return True


class FakeSimulator:
    """Duck-typed PatientSimulator: gated replies + an opener path."""

    def __init__(self, say: str = "Tuesday morning works, thank you.", refuse: bool = False):
        self.say = say
        self.refuse = refuse  # semantic gate says backchannel → stay silent
        self.engaged: list[str] = []
        self.openers: list[str] = []
        self.unprompted_voice_ids: list[str | None] = []

    def respond_streaming(self, text, *, on_audio=None, is_stale=None):
        self.engaged.append(text)
        if self.refuse or is_stale is not None and is_stale():
            return None
        if on_audio is not None:
            on_audio(SPEECH)
        return TurnResult(text=self.say, audio=SPEECH, first_audio_ms=12)

    def speak_unprompted(self, text, *, on_audio=None, voice_id=None):
        self.openers.append(text)
        self.unprompted_voice_ids.append(voice_id)
        if on_audio is not None:
            on_audio(SPEECH)
        return TurnResult(text=text, audio=SPEECH, first_audio_ms=12)


class Harness:
    def __init__(self, tmp_path, **simulator_kwargs):
        self.sent: list[str] = []
        self.stt_frames: list[bytes] = []
        self.timers: list[tuple[float, object]] = []
        self.simulator = FakeSimulator(**simulator_kwargs)
        self.session = CallSession.start(tmp_path / "calls", "test-call")
        self.loop = CallLoop(
            self.simulator,
            self.session,
            stt_send=self.stt_frames.append,
            send_json=self.sent.append,
            executor=InlineExecutor(),
            call_later=lambda seconds, fn: self.timers.append((seconds, fn)),
            opener_after_s=5.0,
            max_call_s=180.0,
        )

    def start(self, sid: str = "MS1") -> None:
        start = {"streamSid": sid, "callSid": "CA1"}
        self.loop.on_twilio_message({"event": "start", "start": start})

    def media(self, ulaw: bytes) -> None:
        payload = base64.b64encode(ulaw).decode()
        self.loop.on_twilio_message({"event": "media", "media": {"payload": payload}})

    def mark(self) -> None:
        self.loop.on_twilio_message({"event": "mark", "mark": {"name": END_OF_TURN_MARK}})

    def run_timers(self) -> None:
        timers, self.timers = self.timers, []
        for _, fn in sorted(timers, key=lambda timer: timer[0]):  # time order
            fn()

    def run_timers_due(self, before: float) -> None:
        """Advance virtual time to ``before``; later timers stay scheduled."""
        timers, self.timers = self.timers, []
        for seconds, fn in sorted(timers, key=lambda timer: timer[0]):
            if seconds <= before:
                fn()
            else:
                self.timers.append((seconds, fn))

    def media_events(self) -> list[dict]:
        return [json.loads(m) for m in self.sent if json.loads(m)["event"] == "media"]

    def clear_events(self) -> list[dict]:
        return [json.loads(m) for m in self.sent if json.loads(m)["event"] == "clear"]


@pytest.fixture()
def harness(tmp_path):
    return Harness(tmp_path)


# -- message wrapping ----------------------------------------------------------


def test_wrap_media_round_trips_through_media_payload():
    message = json.loads(wrap_media(b"\x00\x01\x02", "MS1"))
    assert message["event"] == "media"
    assert message["streamSid"] == "MS1"
    assert media_payload(message) == b"\x00\x01\x02"


def test_media_payload_ignores_non_media_messages():
    assert media_payload({"event": "connected"}) is None
    assert media_payload({"event": "media", "media": {}}) is None


def test_wrap_clear_and_mark_shapes():
    assert json.loads(wrap_clear("MS2")) == {"event": "clear", "streamSid": "MS2"}
    mark = json.loads(wrap_mark("MS2"))
    assert mark["mark"]["name"] == END_OF_TURN_MARK


# -- start / media routing -----------------------------------------------------


def test_start_connects_logs_and_schedules_opener_and_cap(harness):
    harness.start()
    assert harness.loop.stream_sid == "MS1"
    assert harness.loop.call_sid == "CA1"
    delays = sorted(seconds for seconds, _ in harness.timers)
    # opener deadline, idle-nudge deadline, then the max-duration cap
    assert delays == [5.0, 10.0, 180.0]
    events = [line["type"] for line in read_jsonl(harness.session.session_path)]
    assert "call.connected" in events


def test_media_frames_record_forward_to_stt_and_send_nothing(harness):
    harness.start()
    for _ in range(3):
        harness.media(SILENCE)
    assert harness.stt_frames == [SILENCE] * 3
    assert harness.sent == []  # silence never triggers speech
    assert harness.loop.director.state is TurnState.LISTENING


def test_media_before_start_is_ignored(harness):
    harness.media(SPEECH)
    assert harness.stt_frames == []
    assert harness.sent == []


def test_final_boundary_classifiers_cover_live_regressions() -> None:
    assert is_noninteractive_utterance(
        "This call may be recorded for quality and training purposes."
    )
    assert is_noninteractive_utterance("Para español, oprima el dos.")
    assert is_noninteractive_utterance("Transfer you now. Thank you.")
    assert is_obviously_incomplete("I don't see any...")
    assert is_obviously_incomplete("Before I can do that, I need to")
    assert is_terminal_utterance("Goodbye.")


def test_noninteractive_and_incomplete_finals_never_engage(harness):
    harness.start()
    harness.loop.on_stt_event(Final("This call may be recorded for quality."))
    harness.loop.on_stt_event(Final("I don't see any..."))
    assert harness.simulator.engaged == []
    gates = [
        event["data"]["gate"]
        for event in read_jsonl(harness.session.session_path)
        if event["type"] == "stt.final"
    ]
    assert gates == ["noninteractive", "incomplete_heuristic"]


def test_split_finals_are_joined_before_one_reply(harness):
    harness.start()
    harness.loop.on_stt_event(Final("I don't see any..."))
    harness.loop.on_stt_event(Final("upcoming appointments in your record."))
    assert harness.simulator.engaged == [
        "I don't see any upcoming appointments in your record."
    ]


def test_terminal_final_stops_without_reply(harness):
    harness.start()
    harness.loop.on_stt_event(Final("Goodbye."))
    assert harness.loop.stopped
    assert harness.loop.end_reason == "agent_goodbye"
    assert harness.simulator.engaged == []


# -- the engage path ------------------------------------------------------------


def test_agent_final_engages_and_speaks_then_mark_ends_turn(harness):
    harness.start()
    harness.loop.on_stt_event(Final("Thanks for calling, how can I help you?"))
    assert harness.simulator.engaged == ["Thanks for calling, how can I help you?"]
    assert len(harness.media_events()) == 1
    assert harness.loop.director.state is TurnState.SPEAKING
    kinds = [json.loads(m)["event"] for m in harness.sent]
    assert kinds == ["media", "mark"]  # mark queued after the audio
    harness.mark()
    assert harness.loop.director.state is TurnState.LISTENING
    # turn.agent/turn.patient logging is the real simulator's job
    # (test_orchestrator.py); here we only assert the routing above.


def test_refused_gate_stays_silent(harness, tmp_path):
    refusing = Harness(tmp_path, refuse=True)
    refusing.start()
    refusing.loop.on_stt_event(Final("mm-hm"))
    assert refusing.simulator.engaged == ["mm-hm"]
    assert refusing.sent == []
    assert refusing.loop.director.state is TurnState.LISTENING


def test_partial_events_never_engage(harness):
    harness.start()
    harness.loop.on_stt_event(Partial("Thanks for call"))
    assert harness.simulator.engaged == []
    assert harness.sent == []


def test_final_waits_until_inbound_speech_really_ends(harness):
    harness.start()
    harness.media(SPEECH)
    harness.loop.on_stt_event(Final("How can I help you?"))
    assert harness.simulator.engaged == []

    for _ in range(5):
        harness.media(SILENCE)

    assert harness.simulator.engaged == ["How can I help you?"]


# -- turn-slot bookkeeping ------------------------------------------------------


def test_final_while_speaking_waits_for_mark_then_engages(harness):
    harness.start()
    harness.loop.on_stt_event(Final("How can I help?"))
    assert harness.loop.director.state is TurnState.SPEAKING
    harness.loop.on_stt_event(Final("Are you still there?"))  # during playback
    assert len(harness.simulator.engaged) == 1  # held, not dropped
    harness.mark()
    assert harness.simulator.engaged == ["How can I help?", "Are you still there?"]


def test_old_final_is_not_answered_after_playback_no_longer_needs_it(harness):
    harness.start()
    harness.loop.on_stt_event(Final("How can I help?"))
    harness.loop.on_stt_event(Final("A question that will be obsolete."))
    text, _ = harness.loop._pending_final
    harness.loop._pending_final = (text, time.monotonic() - 10)

    harness.mark()

    assert harness.simulator.engaged == ["How can I help?"]
    stale = [
        event
        for event in read_jsonl(harness.session.session_path)
        if event["type"] == "stt.final" and event["data"].get("gate") == "stale"
    ]
    assert len(stale) == 1


def test_lost_mark_is_released_by_the_playback_watchdog(harness):
    harness.start()
    harness.loop.on_stt_event(Final("How can I help?"))
    assert harness.loop.director.state is TurnState.SPEAKING
    # Pretend the mark echo never came: everything queued has long drained.
    harness.loop._last_send = time.monotonic() - 30
    harness.media(SILENCE)
    assert harness.loop.director.state is TurnState.LISTENING
    events = [line["type"] for line in read_jsonl(harness.session.session_path)]
    assert "note" in events  # the watchdog leaves a breadcrumb


# -- opener ---------------------------------------------------------------------


def test_opener_fires_when_agent_stays_silent(harness):
    harness.start()
    harness.run_timers_due(6.0)  # opener deadline passes; the 180 s cap does not
    assert harness.simulator.openers == [harness.loop.opener_text]
    kinds = [json.loads(m)["event"] for m in harness.sent]
    assert kinds == ["media", "mark"]
    assert harness.loop.director.state is TurnState.SPEAKING


def test_opener_skipped_when_agent_already_spoke(harness):
    harness.start()
    harness.loop.on_stt_event(Final("Hi, this is the clinic."))
    harness.run_timers()
    assert harness.simulator.openers == []
    assert len(harness.simulator.engaged) == 1


def test_opener_waits_while_agent_is_audibly_speaking(harness):
    harness.start()
    harness.media(SPEECH)
    harness.run_timers_due(6.0)
    assert harness.simulator.openers == []


def test_patient_goodbye_stops_after_playback_mark(tmp_path):
    ending = Harness(tmp_path, say="No, that's everything. Thank you, goodbye.")
    ending.start()
    ending.loop.on_stt_event(Final("Anything else?"))
    assert not ending.loop.stopped
    ending.mark()
    assert ending.loop.stopped
    assert ending.loop.end_reason == "patient_goodbye"


# -- barge-in and teardown -------------------------------------------------------


def test_sustained_agent_speech_aborts_playback(harness):
    harness.start()
    harness.loop.on_stt_event(Final("How can I help?"))
    assert harness.loop.director.state is TurnState.SPEAKING
    for _ in range(25):  # 500 ms of sustained speech crosses the 400 ms window
        harness.media(SPEECH)
    assert harness.clear_events()  # the §5.3 abort sequence ran
    assert harness.loop.director.state is TurnState.LISTENING
    # A stale generation's chunks must not leak after the interrupt.
    before = len(harness.sent)
    harness.loop._port_send_audio(SPEECH)
    assert len(harness.sent) == before


def test_scripted_patient_barge_in_speaks_during_agent_utterance(harness):
    harness.loop.scripted_barge_ins = ("Sorry — was that in the morning?",)
    harness.start()
    for _ in range(25):
        harness.media(SPEECH)

    # The remote agent is still speaking when the one-second arm fires.
    harness.run_timers_due(1.1)

    assert harness.simulator.openers == ["Sorry — was that in the morning?"]
    assert harness.clear_events()
    assert harness.loop.director.state is TurnState.SPEAKING


def test_scripted_second_speaker_uses_voice_override_and_logs_intent(harness):
    harness.loop.scripted_barge_ins = ("Friday works for us.",)
    harness.loop.scripted_barge_kind = "third_party_interruption"
    harness.loop.scripted_barge_voice_id = "other-voice"
    harness.start()
    for _ in range(25):
        harness.media(SPEECH)
    harness.run_timers_due(1.1)

    assert harness.simulator.unprompted_voice_ids == ["other-voice"]
    events = read_jsonl(harness.session.session_path)
    fired = [event for event in events if event["type"] == "behavior.fired"]
    assert fired[0]["data"]["behavior"] == "third_party_interruption"
    assert fired[0]["data"]["intentional"] is True


def test_scripted_long_silence_delays_response_and_logs_intent(harness):
    harness.loop.scripted_response_delays_s = (5.0,)
    harness.start()
    harness.loop.on_stt_event(Final("What day would you like?"))

    assert harness.simulator.engaged == []
    harness.run_timers_due(5.0)
    assert harness.simulator.engaged == ["What day would you like?"]
    events = read_jsonl(harness.session.session_path)
    fired = [event for event in events if event["type"] == "behavior.fired"]
    assert fired[0]["data"] == {
        "behavior": "long_silence",
        "intentional": True,
        "delay_s": 5.0,
    }


def test_stop_is_idempotent_and_bumps_the_epoch(harness):
    harness.start()
    harness.loop.stop("max_duration")
    assert harness.loop.stopped and harness.loop.end_reason == "max_duration"
    epoch = harness.loop.director.generations.current
    harness.loop.stop("stream_stopped")  # second stop is a no-op
    assert harness.loop.end_reason == "max_duration"
    assert harness.loop.director.generations.current == epoch


def test_stopped_loop_ignores_finals_and_media(harness):
    harness.start()
    harness.loop.stop("stream_stopped")
    harness.loop.on_stt_event(Final("hello?"))
    harness.media(SPEECH)
    assert harness.simulator.engaged == []
    assert harness.stt_frames == []


def test_stt_wiring_encodes_per_provider():
    """Scribe audio rides JSON protocol messages; Deepgram takes raw frames."""
    from patientqa.callloop import stt_wiring
    from patientqa.deepgram.client import DeepgramSettings
    from patientqa.elevenlabs.scribe import ScribeSettings, wrap_scribe_audio

    deepgram = stt_wiring(DeepgramSettings(api_key="k"))
    assert deepgram.encode(SPEECH) == SPEECH

    scribe = stt_wiring(ScribeSettings(api_key="k"))
    assert scribe.encode(SPEECH) == wrap_scribe_audio(SPEECH)
    assert json.loads(scribe.encode(SPEECH))["message_type"] == "input_audio_chunk"


def test_date_context_pins_today_and_the_window():
    from datetime import date

    from patientqa.callloop import date_context

    ctx = date_context(date(2026, 8, 19))  # the Wednesday of the live test
    assert "Wednesday, August 19, 2026" in ctx
    assert "Wednesday, August 26, 2026" in ctx  # one week out, named explicitly
    assert "actual date" in ctx


def test_run_call_postprocesses_every_finalized_session(tmp_path, monkeypatch):
    import patientqa.analyze as analyze_module
    import patientqa.callloop as callloop_module

    processed = []

    async def fake_connected_call(**kwargs):
        session = kwargs["session"]
        session.close("test_complete")
        return session.directory

    def fake_postprocess(directory, *, judge):
        assert (directory / "call.json").is_file()
        processed.append((directory, judge))

    monkeypatch.setattr(callloop_module, "_run_connected_call", fake_connected_call)
    monkeypatch.setattr(analyze_module, "postprocess_session", fake_postprocess)

    result = asyncio.run(
        callloop_module.run_call(
            stream_url="wss://example.test/media",
            calls_root=tmp_path / "calls",
            call_id="auto-analysis",
            analysis_judge=False,
        )
    )

    assert processed == [(result, False)]


# -- the idle nudge (stalled-line recovery) -------------------------------------


def _stalled_harness(tmp_path):
    """A line whose turns the semantic gate refuses: we go quiet, line stalls."""
    harness = Harness(tmp_path, refuse=True)
    harness.start()
    harness.loop.on_stt_event(Final("So would you like fries with that, or..."))
    harness.run_timers_due(0)  # let the refused engage release the turn slot
    assert harness.simulator.openers == []
    assert harness.loop.director.state is TurnState.LISTENING
    return harness


def test_idle_nudge_recovers_a_stalled_line(tmp_path):
    from patientqa.callloop import NUDGE_TEXT

    harness = _stalled_harness(tmp_path)
    harness.run_timers_due(10.0)  # nudge deadline passes with nothing committable
    assert harness.simulator.openers == [NUDGE_TEXT]
    assert harness.loop.director.state is TurnState.SPEAKING
    kinds = [json.loads(m)["event"] for m in harness.sent]
    assert kinds == ["media", "mark"]  # the nudge is a normal spoken turn


def test_idle_nudge_is_capped_at_two(tmp_path):
    from patientqa.callloop import NUDGE_TEXT

    harness = _stalled_harness(tmp_path)
    harness.run_timers_due(10.0)  # nudge 1
    harness.mark()  # its turn ends → re-armed
    harness.run_timers_due(20.0)  # nudge 2
    harness.mark()
    harness.run_timers_due(30.0)  # cap reached: the line stays quiet
    assert harness.simulator.openers == [NUDGE_TEXT, NUDGE_TEXT]


def test_idle_nudge_holds_off_while_agent_is_mid_utterance(tmp_path):
    harness = _stalled_harness(tmp_path)
    for _ in range(3):
        harness.media(SPEECH)  # uncommitted agent audio is flowing
    harness.run_timers_due(10.0)
    assert harness.simulator.openers == []  # energy gate says: they're talking


def test_new_agent_activity_rearms_the_nudge(tmp_path):
    harness = _stalled_harness(tmp_path)
    harness.run_timers_due(5.0)  # halfway to the deadline, nothing due
    assert harness.simulator.openers == []
    harness.loop.on_stt_event(Final("Bro, it broke."))  # line wakes up → re-arm
    harness.run_timers_due(9.9)  # the old deadline's arm is now stale
    assert harness.simulator.openers == []
    assert any(seconds <= 10.0 for seconds, _ in harness.timers)  # fresh arm pending
