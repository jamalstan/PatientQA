import json
from collections.abc import Iterator
from pathlib import Path

from patientqa.calllog.session import read_jsonl
from patientqa.cerebras.client import AGENT, PATIENT, BrainReply, Turn, extract_reply
from patientqa.orchestrator import (
    PatientSimulator,
    build_patient_simulator,
    build_twilio_client,
)
from patientqa.turns import AGENT_TURN_BACKCHANNEL, AGENT_TURN_FINISHED
from patientqa.twilio.client import TwilioClient

REPLY = '{"agent_turn": "finished", "say": "Tuesday morning, please."}'


class FakeBrain:
    """Full brain fake: batch reply plus a token-dribbled streaming reply."""

    def __init__(self, reply_json: str = REPLY) -> None:
        self.reply_json = reply_json
        self.received: list[tuple[str, tuple[Turn, ...]]] = []

    def reply(self, system_prompt: str, history: list[Turn]) -> BrainReply:
        self.received.append((system_prompt, tuple(history)))
        return extract_reply(self.reply_json)

    def reply_stream(self, system_prompt: str, history: list[Turn]) -> Iterator[str]:
        self.received.append((system_prompt, tuple(history)))
        text = self.reply_json
        for i in range(0, len(text), 7):
            yield text[i : i + 7]


class StaleAfterOneDelta:
    """Streams one delta, then reports itself stale (a barge-in landed)."""

    def __init__(self) -> None:
        self.received: list[tuple[str, tuple[Turn, ...]]] = []

    def reply_stream(self, system_prompt: str, history: list[Turn]) -> Iterator[str]:
        self.received.append((system_prompt, tuple(history)))
        yield '{"agent_turn": '
        self.stale = True
        yield '"finished", "say": "Tuesday morning, please."}'


class FakeVoice:
    def __init__(self) -> None:
        self.received: list[tuple[str, str]] = []

    def synthesize(self, text: str, voice_id: str) -> bytes:
        self.received.append((text, voice_id))
        return text.encode("ascii")


def _simulator(brain=None) -> tuple[PatientSimulator, FakeBrain, FakeVoice]:
    brain = brain or FakeBrain()
    voice = FakeVoice()
    return PatientSimulator(brain=brain, voice=voice, voice_id="voice-7"), brain, voice


def test_respond_bridges_brain_to_voice() -> None:
    simulator, brain, voice = _simulator()
    result = simulator.respond("Clinic, how can I help you?")

    assert result is not None
    assert result.text == "Tuesday morning, please."
    assert result.audio == b"Tuesday morning, please."
    assert result.first_audio_ms is not None
    assert voice.received == [("Tuesday morning, please.", "voice-7")]
    system_prompt, history = brain.received[0]
    assert system_prompt  # the default persona rules ship a usable prompt
    assert history == (Turn(role=AGENT, content="Clinic, how can I help you?"),)


def test_history_accumulates_both_sides_of_the_dialogue() -> None:
    simulator, _, _ = _simulator()
    simulator.respond("Clinic, how can I help you?")
    simulator.respond("Is Tuesday at 10am okay?")

    assert simulator.history == (
        Turn(role=AGENT, content="Clinic, how can I help you?"),
        Turn(role=PATIENT, content="Tuesday morning, please."),
        Turn(role=AGENT, content="Is Tuesday at 10am okay?"),
        Turn(role=PATIENT, content="Tuesday morning, please."),
    )


def test_fixed_identity_response_bypasses_brain_and_enters_history() -> None:
    brain = FakeBrain()
    voice = FakeVoice()
    simulator = PatientSimulator(
        brain=brain,
        voice=voice,
        voice_id="v",
        fixed_responder=lambda text: (
            "My date of birth is May tenth, nineteen forty-five."
            if "birth" in text.lower()
            else None
        ),
    )

    result = simulator.respond("What is your date of birth?")

    assert result is not None
    assert result.text == "My date of birth is May tenth, nineteen forty-five."
    assert brain.received == []
    assert simulator.history[-1] == Turn(role=PATIENT, content=result.text)


def test_respond_returns_none_when_gate_says_backchannel() -> None:
    simulator, _, voice = _simulator(
        FakeBrain('{"agent_turn": "backchannel", "say": ""}')
    )

    assert simulator.respond("mm-hm") is None
    assert simulator.history == ()  # the utterance never became a dialogue turn
    assert voice.received == []  # and nothing was spoken


def test_respond_returns_none_on_incomplete_verdict() -> None:
    simulator, _, _ = _simulator(FakeBrain('{"agent_turn": "incomplete", "say": ""}'))
    assert simulator.respond("So what I'll do is—") is None
    assert simulator.history == ()


def test_respond_records_session_events(tmp_path: Path) -> None:
    from patientqa.calllog.session import CallSession

    session = CallSession.start(tmp_path, "call-042", {"persona": {"name": "Marta"}})
    simulator = PatientSimulator(
        brain=FakeBrain(), voice=FakeVoice(), voice_id="v", session=session
    )
    simulator.respond("Clinic, how can I help you?")

    events = read_jsonl(session.session_path)
    assert [e["type"] for e in events] == [
        "call.started",
        "stt.final",
        "turn.agent",
        "audio.played",
        "brain.reply",
        "tts.done",
        "turn.patient",
    ]
    assert events[1]["data"] == {
        "text": "Clinic, how can I help you?",
        "gate": AGENT_TURN_FINISHED,
    }
    assert events[2]["data"]["text"] == "Clinic, how can I help you?"
    assert events[4]["data"]["say"] == "Tuesday morning, please."
    assert events[4]["data"]["gate"] == AGENT_TURN_FINISHED
    assert events[5]["data"]["audio_bytes"] == len(b"Tuesday morning, please.")
    assert session.outbound_path.read_bytes() == b"Tuesday morning, please."

    session.close("completed")
    transcript = json.loads(session.transcript_path.read_text(encoding="utf-8"))
    assert [(t["role"], t["text"]) for t in transcript["turns"]] == [
        ("agent", "Clinic, how can I help you?"),
        ("patient", "Tuesday morning, please."),
    ]


def test_backchannel_is_logged_but_never_becomes_a_turn(tmp_path: Path) -> None:
    from patientqa.calllog.session import CallSession

    session = CallSession.start(tmp_path, "call-043")
    simulator = PatientSimulator(
        brain=FakeBrain('{"agent_turn": "backchannel", "say": ""}'),
        voice=FakeVoice(),
        voice_id="v",
        session=session,
    )
    assert simulator.respond("mm-hm") is None

    events = read_jsonl(session.session_path)
    assert [e["type"] for e in events] == ["call.started", "stt.final"]
    assert events[1]["data"]["gate"] == AGENT_TURN_BACKCHANNEL
    assert session.outbound_path.read_bytes() == b""


def test_respond_streaming_speaks_in_chunks() -> None:
    simulator, _, voice = _simulator()
    sent: list[bytes] = []

    result = simulator.respond_streaming(
        "What day works for you?", on_audio=sent.append
    )

    assert result is not None
    assert result.text == "Tuesday morning, please."
    assert b"".join(sent) == result.audio
    # The first speakable unit ("Tuesday morning,") went out before the rest.
    assert [text for text, _ in voice.received] == ["Tuesday morning,", "please."]


def test_respond_streaming_stops_the_instant_the_epoch_goes_stale() -> None:
    brain = StaleAfterOneDelta()
    simulator, _, _ = _simulator(brain)
    sent: list[bytes] = []

    result = simulator.respond_streaming(
        "Can you hold on a—", on_audio=sent.append, is_stale=lambda: getattr(brain, "stale", False)
    )

    assert result is None
    assert sent == []  # a cancelled generation never leaks audio
    # The agent's turn stays in history; only our reply to it was cancelled.
    assert simulator.history == (Turn(role=AGENT, content="Can you hold on a—"),)


def test_stale_during_blocking_tts_does_not_commit_phantom_patient_turn() -> None:
    simulator, _, _ = _simulator()
    stale = False

    def reject_audio(audio: bytes) -> bool:
        nonlocal stale
        stale = True
        return False

    result = simulator.respond_streaming(
        "Let me finish that thought.",
        on_audio=reject_audio,
        is_stale=lambda: stale,
    )

    assert result is None
    assert simulator.history == (Turn(role=AGENT, content="Let me finish that thought."),)


def test_build_patient_simulator_wires_all_providers(secrets_file: Path) -> None:
    simulator = build_patient_simulator("voice-7", path=secrets_file)
    assert isinstance(simulator, PatientSimulator)


def test_build_twilio_client_loads_settings(secrets_file: Path) -> None:
    assert isinstance(build_twilio_client(path=secrets_file), TwilioClient)
