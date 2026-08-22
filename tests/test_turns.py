"""Turn-taking core tests (DESIGN.md §5.3): state machine, epoch guard,
energy gate, speakable chunking, and the barge-in abort sequence."""

from collections.abc import Iterator
from pathlib import Path

from patientqa.calllog.session import CallSession, read_jsonl
from patientqa.calllog.ulaw import pcm16_to_ulaw_bytes
from patientqa.cerebras.client import Turn
from patientqa.orchestrator import PatientSimulator, TurnResult
from patientqa.turns import (
    BACKCHANNEL,
    BARGE_IN,
    SPEECH_ENDED,
    SPEECH_STARTED,
    EnergyGate,
    EnergyGateSettings,
    FillerPolicy,
    Generations,
    SpeakableBuffer,
    TurnDirector,
    TurnState,
    split_speakable,
)

FRAME_MS = 20
SILENCE = b"\xff" * (FRAME_MS * 8)  # μ-law silence: decodes to 0
SPEECH = pcm16_to_ulaw_bytes([8000] * (FRAME_MS * 8))  # loud, well above the gate


class FakeBrain:
    """Streaming brain whose reply JSON is fixed for the test."""

    def __init__(self, reply_json: str) -> None:
        self._reply_json = reply_json

    def reply_stream(self, system_prompt: str, history: list[Turn]) -> Iterator[str]:
        text = self._reply_json
        for i in range(0, len(text), 7):
            yield text[i : i + 7]


class PausingBrain:
    """Yields one delta, pauses (hook runs — a barge-in can land), continues."""

    def __init__(self, hook) -> None:
        self._hook = hook

    def reply_stream(self, system_prompt: str, history: list[Turn]) -> Iterator[str]:
        yield '{"agent_turn": '
        self._hook()
        yield '"finished", "say": "Tuesday morning, please."}'


class FakeVoice:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def synthesize(self, text: str, voice_id: str) -> bytes:
        self.chunks.append(text)
        return text.encode("ascii")


def _simulator(brain) -> PatientSimulator:
    return PatientSimulator(brain=brain, voice=FakeVoice(), voice_id="voice-7")


def _director(brain, **kwargs) -> TurnDirector:
    return TurnDirector(simulator=_simulator(brain), **kwargs)


# -- Generations (the epoch guard) ---------------------------------------------


def test_generation_bump_invalidates_captured_epoch() -> None:
    generations = Generations()
    token = generations.capture()
    assert generations.is_current(token)
    generations.bump()
    assert not generations.is_current(token)
    assert generations.is_current(generations.capture())


# -- EnergyGate (§5.3 delta 2) ---------------------------------------------------


def test_energy_gate_fires_barge_in_at_sustained_speech() -> None:
    gate = EnergyGate(EnergyGateSettings(sustained_ms=400, hangover_ms=100))
    assert gate.feed(SPEECH) == SPEECH_STARTED
    for _ in range(18):
        assert gate.feed(SPEECH) is None
    assert gate.feed(SPEECH) == BARGE_IN  # 20th frame crosses 400 ms
    for _ in range(4):
        assert gate.feed(SILENCE) is None  # hangover
    assert gate.feed(SILENCE) == SPEECH_ENDED


def test_energy_gate_calls_short_burst_a_backchannel() -> None:
    gate = EnergyGate(EnergyGateSettings(sustained_ms=400, hangover_ms=100))
    gate.feed(SPEECH)
    for _ in range(9):  # 200 ms of speech — below the sustained window
        gate.feed(SPEECH)
    for _ in range(4):
        assert gate.feed(SILENCE) is None
    assert gate.feed(SILENCE) == BACKCHANNEL


def test_energy_gate_ignores_pure_silence() -> None:
    gate = EnergyGate()
    for _ in range(50):
        assert gate.feed(SILENCE) is None


# -- SpeakableBuffer (§5.3 delta 3: never forward single tokens) ----------------


def test_first_chunk_releases_at_first_comma() -> None:
    buffer = SpeakableBuffer(min_words=5, max_words=15)
    assert buffer.feed("Hello, I need an appointment") == ["Hello,"]


def test_later_commas_wait_for_min_words() -> None:
    buffer = SpeakableBuffer(min_words=5, max_words=15)
    buffer.feed("Hello, ")  # first chunk leaves
    assert buffer.feed("so, I need an appointment") == []


def test_sentence_end_always_releases() -> None:
    buffer = SpeakableBuffer(min_words=8, max_words=15)
    assert buffer.feed("No. ") == ["No."]


def test_long_runs_hard_cut_at_max_words() -> None:
    text = " ".join(f"word{i}" for i in range(20))  # no punctuation at all
    chunks = split_speakable(text, min_words=5, max_words=15)
    assert all(len(chunk.split()) <= 15 for chunk in chunks)
    assert " ".join(chunks) == text


def test_split_speakable_joins_back_to_the_original() -> None:
    text = "Hello, I would like to book a visit with Dr. Ortiz, please. Tuesday works."
    chunks = split_speakable(text)
    assert chunks
    assert " ".join(chunks) == text


# -- FillerPolicy (§9) -----------------------------------------------------------


def test_filler_fires_once_after_threshold() -> None:
    policy = FillerPolicy(after_ms=700)
    assert not policy.should_fire(200, fired=False)
    assert policy.should_fire(750, fired=False)
    assert not policy.should_fire(900, fired=True)


# -- TurnDirector -----------------------------------------------------------------


def test_director_happy_path_speaks_and_releases_floor() -> None:
    brain = FakeBrain('{"agent_turn": "finished", "say": "Tuesday morning, please."}')
    sent: list[bytes] = []
    director = _director(brain, send_audio=sent.append)

    result = director.on_agent_final("What day works for you?")

    assert isinstance(result, TurnResult)
    assert result.text == "Tuesday morning, please."
    assert director.state is TurnState.SPEAKING
    assert sent and b"".join(sent) == result.audio
    director.on_playback_finished()
    assert director.state is TurnState.LISTENING


def test_director_backchannel_verdict_stays_silent(tmp_path: Path) -> None:
    brain = FakeBrain('{"agent_turn": "backchannel", "say": ""}')
    sent: list[bytes] = []
    session = CallSession.start(tmp_path, "call-1")
    simulator = PatientSimulator(brain=brain, voice=FakeVoice(), voice_id="v", session=session)
    director = TurnDirector(simulator=simulator, send_audio=sent.append, session=session)

    assert director.on_agent_final("mm-hm") is None
    assert director.state is TurnState.LISTENING
    assert sent == []
    assert simulator.history == ()  # never became a dialogue turn

    events = read_jsonl(session.session_path)
    finals = [e for e in events if e["type"] == "stt.final"]
    assert finals[-1]["data"] == {"text": "mm-hm", "gate": "backchannel"}
    assert not [e for e in events if e["type"] == "turn.agent"]


def test_director_barge_in_aborts_generation_mid_stream(tmp_path: Path) -> None:
    session = CallSession.start(tmp_path, "call-2")
    cleared: list[int] = []
    sent: list[bytes] = []

    director_holder: list[TurnDirector] = []

    def pause_and_barge() -> None:
        director = director_holder[0]
        for _ in range(20):  # 400 ms of sustained agent speech
            director.on_media_frame(SPEECH)

    simulator = PatientSimulator(
        brain=PausingBrain(pause_and_barge), voice=FakeVoice(), voice_id="v", session=session
    )
    director = TurnDirector(
        simulator=simulator,
        send_audio=sent.append,
        clear_playback=lambda: cleared.append(1),
        session=session,
    )
    director_holder.append(director)

    assert director.on_agent_final("Can you hold on a—") is None

    assert director.state is TurnState.LISTENING
    assert sent == []  # nothing leaked after the epoch bump
    assert cleared == [1]  # Twilio playback buffer flushed
    events = read_jsonl(session.session_path)
    barge = [e for e in events if e["type"] == "barge_in"]
    assert len(barge) == 1
    assert barge[0]["data"]["reason"] == "agent_continued_after_final"
    assert barge[0]["data"]["state"] == "thinking"


def test_director_backchannel_during_playback_keeps_playing() -> None:
    brain = FakeBrain('{"agent_turn": "finished", "say": "Tuesday morning, please."}')
    cleared: list[int] = []
    director = _director(brain, clear_playback=lambda: cleared.append(1))

    director.on_agent_final("Go ahead please.")
    assert director.state is TurnState.SPEAKING

    director.on_media_frame(SPEECH)  # short "mm-hm"-length burst begins
    for _ in range(9):
        director.on_media_frame(SPEECH)
    for _ in range(5):
        director.on_media_frame(SILENCE)  # ends at 200 ms — a backchannel

    assert cleared == []  # no abort
    assert director.state is TurnState.SPEAKING  # still our turn


def test_director_scripted_overtalk_interrupts_our_own_playback() -> None:
    brain = FakeBrain('{"agent_turn": "finished", "say": "Tuesday morning, please."}')
    director = _director(brain)
    director.on_agent_final("Go ahead please.")

    director.interrupt(reason="scripted_overtalk")

    assert director.state is TurnState.LISTENING
    # The bump means any still-running generation is now stale.
    assert not director.generations.is_current(0)


def test_director_logs_filler_near_miss(tmp_path: Path) -> None:
    brain = FakeBrain('{"agent_turn": "finished", "say": "Tuesday morning, please."}')
    session = CallSession.start(tmp_path, "call-3")
    simulator = PatientSimulator(brain=brain, voice=FakeVoice(), voice_id="v", session=session)
    director = TurnDirector(
        simulator=simulator, filler=FillerPolicy(after_ms=0), session=session
    )
    director.on_agent_final("Go ahead please.")
    events = read_jsonl(session.session_path)
    notes = [e["data"]["text"] for e in events if e["type"] == "note"]
    assert any(text.startswith("filler window missed") for text in notes)
