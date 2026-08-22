"""Turn-taking core (DESIGN.md §5.3) — deciding whose turn it is.

Three signals, layered:

1. **Signal VAD** — :class:`EnergyGate` watches the inbound μ-law track with a
   plain RMS gate. Silence alone cannot decide turns (~48–50% of human turn
   transitions overlap), so it only answers *when* speech happens.
2. **Semantic end-of-turn** — the brain classifies each committed agent
   utterance ``finished | backchannel | incomplete`` and the simulator refuses
   to speak unless the verdict is ``finished`` (§5.3 delta 1).
3. **Barge-in** — sustained agent speech during our playback aborts everything
   in flight; a short burst is a backchannel and playback continues (delta 2).

:class:`Generations` is the epoch counter from the turn-taking review: every
in-flight stream (brain deltas, TTS chunks, queued playback) captures an epoch
and abandons itself the moment :meth:`TurnDirector.interrupt` bumps it, so a
cancelled response can never leak audio. The abort sequence ends with Twilio's
``{"event": "clear"}`` (via the ``clear_playback`` port) which flushes queued
outbound audio.

Everything here is transport-free by design: the call loop (the FastAPI /
Media-Streams stage) feeds :class:`TurnDirector` frames and STT events and
wires its ports to real sockets. All logic is therefore testable offline.
"""

import enum
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from patientqa.calllog.session import BARGE_IN as BARGE_IN_EVENT
from patientqa.calllog.session import NOTE, CallSession
from patientqa.calllog.ulaw import ulaw_bytes_to_pcm

log = logging.getLogger(__name__)

# -- §5.3 delta 1: the semantic end-of-turn verdicts -------------------------
AGENT_TURN_FINISHED = "finished"
AGENT_TURN_BACKCHANNEL = "backchannel"
AGENT_TURN_INCOMPLETE = "incomplete"
VERDICTS = (AGENT_TURN_FINISHED, AGENT_TURN_BACKCHANNEL, AGENT_TURN_INCOMPLETE)


class TurnState(enum.Enum):
    """The simulator's side of the conversation (§5.3 state machine)."""

    LISTENING = "listening"  # agent owns the floor; STT events drive us
    THINKING = "thinking"  # agent turn committed; brain + TTS in flight
    SPEAKING = "speaking"  # our μ-law audio is queued on the outbound track


class Generations:
    """Monotonic epoch counter guarding every in-flight generation.

    ``interrupt()`` bumps the epoch; loops that captured the previous one see
    ``is_current(token)`` go false and abandon their work. One counter guards
    all three concurrent streams at once (brain deltas, TTS synthesis, queued
    Twilio playback), which is exactly the leak the review warned about.
    """

    def __init__(self) -> None:
        self._current = 0

    @property
    def current(self) -> int:
        return self._current

    def capture(self) -> int:
        """The epoch a new generation runs under."""
        return self._current

    def bump(self) -> int:
        """Cancel every in-flight generation; returns the new epoch."""
        self._current += 1
        return self._current

    def is_current(self, token: int) -> bool:
        return token == self._current


# -- §5.3 delta 2: energy gate on the inbound μ-law track --------------------

SPEECH_STARTED = "speech_started"  # inbound speech just began
BARGE_IN = "barge_in"  # run crossed the sustained-speech window
BACKCHANNEL = "backchannel"  # run ended before the window — listener noise
SPEECH_ENDED = "speech_ended"  # run that had crossed the window went quiet

#: Twilio Media Streams deliver 8 kHz μ-law, i.e. exactly 8 bytes per millisecond.
BYTES_PER_MS = 8


@dataclass(frozen=True)
class EnergyGateSettings:
    """RMS gate tuning for one inbound track (§5.3 delta 2 defaults)."""

    speech_rms: float = 500.0  # PCM16 RMS above which a frame counts as speech
    sustained_ms: int = 400  # speech this long is a barge-in, not a backchannel
    hangover_ms: int = 100  # silence this long ends a speech run


class EnergyGate:
    """Classifies speech runs on the agent's inbound μ-law track.

    ``feed`` takes one Media-Streams frame (any length) and returns at most
    one event: when a run crosses ``sustained_ms`` the gate fires BARGE_IN
    immediately — the abort sequence must not wait for the run to end — and
    when a run ends it reports BACKCHANNEL (short burst: "mm-hm", "sure") or
    SPEECH_ENDED (it had already qualified as an interruption).
    """

    def __init__(self, settings: EnergyGateSettings | None = None) -> None:
        self._settings = settings or EnergyGateSettings()
        self._speech_ms = 0
        self._silence_ms = 0.0
        self._in_speech = False
        self._fired = False  # BARGE_IN already fired for the current run

    @staticmethod
    def _rms(ulaw: bytes) -> float:
        pcm = ulaw_bytes_to_pcm(ulaw)
        if not len(pcm):
            return 0.0
        return (sum(sample * sample for sample in pcm) / len(pcm)) ** 0.5

    @property
    def speech_ms(self) -> int:
        """Duration of the current (or, after the run ends, last) speech run."""
        return self._speech_ms

    @property
    def in_speech(self) -> bool:
        """True while a speech run is open (last frame was above the gate)."""
        return self._in_speech

    def feed(self, ulaw: bytes) -> str | None:
        frame_ms = len(ulaw) / BYTES_PER_MS
        if self._rms(ulaw) >= self._settings.speech_rms:
            if not self._in_speech:
                self._in_speech = True
                self._speech_ms = 0
                self._fired = False
                self._silence_ms = 0.0
            self._speech_ms = int(self._speech_ms + frame_ms)
            self._silence_ms = 0.0
            if not self._fired and self._speech_ms >= self._settings.sustained_ms:
                self._fired = True
                return BARGE_IN
            return SPEECH_STARTED if self._speech_ms <= frame_ms else None

        if not self._in_speech:
            return None
        self._silence_ms += frame_ms
        if self._silence_ms < self._settings.hangover_ms:
            return None
        self._in_speech = False
        return SPEECH_ENDED if self._fired else BACKCHANNEL


# -- §5.3 delta 3 mechanics: speakable chunking of streamed text -------------

_SENTENCE = re.compile(r"^(.*?[.!?])(?:\s|$)")
_COMMA = re.compile(r"^(.*?,)(?:\s|$)")


@dataclass
class SpeakableBuffer:
    """Buckets streamed text into speakable chunks before it hits TTS.

    The review's rule: never forward single tokens to TTS — release at
    sentence ends always, at commas once the chunk carries ``min_words``
    (the *first* chunk may release at its first comma, however short, so
    speech starts on a natural unit), and hard-cut at ``max_words``.
    """

    min_words: int = 5
    max_words: int = 15
    _pending: str = ""
    _first: bool = True

    def feed(self, text: str) -> list[str]:
        self._pending += text
        return self._release()

    def flush(self) -> str:
        chunk, self._pending = self._pending.strip(), ""
        return chunk

    def _release(self) -> list[str]:
        chunks: list[str] = []
        while True:
            pending = self._pending.strip()
            if not pending:
                self._pending = ""
                break
            self._pending = pending
            if match := _SENTENCE.match(pending):
                cut = len(match.group(1))
            elif (match := _COMMA.match(pending)) and (
                self._first or len(match.group(1).split()) >= self.min_words
            ):
                cut = len(match.group(1))
            elif len(pending.split()) >= self.max_words:
                words = pending.split()
                cut = len(" ".join(words[: self.max_words]))
            else:
                break
            chunks.append(pending[:cut].strip())
            self._pending = pending[cut:]
            self._first = False
        return chunks


def split_speakable(text: str, *, min_words: int = 5, max_words: int = 15) -> list[str]:
    """One-shot form of :class:`SpeakableBuffer` (whole utterance in)."""
    buffer = SpeakableBuffer(min_words=min_words, max_words=max_words)
    chunks = buffer.feed(text)
    if tail := buffer.flush():
        chunks.append(tail)
    return chunks


# -- §9 mitigation: soft-timeout filler --------------------------------------


@dataclass(frozen=True)
class FillerPolicy:
    """Persona-consistent cover for slow brain+TTS turns (§9, ElevenLabs idea).

    ``should_fire`` is the pure decision the call loop consults while a turn
    is stuck in THINKING; the filler itself plays through the warm TTS socket,
    so wiring it is a call-loop concern. This stage logs the near-misses so
    the telemetry shows when it would have fired.
    """

    text: str = "Ah, let me think a second…"
    after_ms: int = 700
    once_per_turn: bool = True

    def should_fire(self, waiting_ms: int, fired: bool) -> bool:
        if fired and self.once_per_turn:
            return False
        return waiting_ms >= self.after_ms


# -- the state machine --------------------------------------------------------


@dataclass
class TurnDirector:
    """Owns the simulator's turn state; the call loop drives it.

    Ports keep it transport-free: ``send_audio`` forwards one μ-law chunk to
    the Media Streams socket, ``clear_playback`` sends Twilio's
    ``{"event": "clear"}``. The simulator (brain + voice + gate) is injected
    duck-typed — :class:`patientqa.orchestrator.PatientSimulator` in
    production, small fakes in tests.
    """

    simulator: Any
    send_audio: Callable[[bytes], bool | None] | None = None
    clear_playback: Callable[[], None] | None = None
    energy: EnergyGate = field(default_factory=EnergyGate)
    filler: FillerPolicy | None = FillerPolicy()
    session: CallSession | None = None

    def __post_init__(self) -> None:
        self.state = TurnState.LISTENING
        self.generations = Generations()

    # -- transport callbacks (the call loop calls these) ----------------------

    def on_media_frame(self, ulaw: bytes) -> str | None:
        """One inbound (agent-leg) μ-law frame: record it, watch the gate."""
        if self.session is not None:
            self.session.append_inbound(ulaw)
        event = self.energy.feed(ulaw)
        if event is None:
            return None
        if event == SPEECH_STARTED and self.state is TurnState.THINKING:
            # A final was committed, but the remote agent immediately resumed.
            # Cancel before the first blocking TTS request can become audible;
            # waiting for the 400 ms barge threshold creates avoidable overtalk.
            log.debug("energy gate: agent resumed while patient was thinking")
            self.interrupt(reason="agent_continued_after_final", speech_ms=self.energy.speech_ms)
        elif event == BARGE_IN and self.state is TurnState.SPEAKING:
            # Sustained agent speech while we hold/queue audio: abort (§5.3 delta 2).
            log.debug("energy gate: barge-in after %dms of speech", self.energy.speech_ms)
            self.interrupt(reason="agent_speech_during_playback", speech_ms=self.energy.speech_ms)
        elif event == BACKCHANNEL and self.state == TurnState.SPEAKING:
            # "mm-hm" from the agent — keep playing (§5.3 delta 2 refinement).
            log.debug("energy gate: agent backchannel during playback")
            if self.session is not None:
                self.session.log(NOTE, text="agent backchannel during playback")
        return event

    def on_agent_final(self, text: str) -> Any:
        """A committed agent transcript: gate → think → speak, epoch-guarded.

        Returns the simulator's TurnResult when the patient spoke, ``None``
        when the semantic gate said backchannel/incomplete. Preemptive
        generation (§5.3 delta 3) is the call loop's refinement: it may call
        this with the final *interim*, discarding via :meth:`interrupt`.
        """
        if self.state != TurnState.LISTENING:
            # A late final while we are mid-turn: treat as fuel for a barge-in
            # decision, never as a fresh turn.
            log.debug("agent final ignored mid-turn (state=%s)", self.state.value)
            return None
        log.debug("state: listening → thinking")
        self.state = TurnState.THINKING
        epoch = self.generations.capture()
        result = self.simulator.respond_streaming(
            text,
            on_audio=self._send,
            is_stale=lambda: not self.generations.is_current(epoch),
        )
        if result is None:
            log.debug("state: thinking → listening (semantic gate kept us silent)")
            self.state = TurnState.LISTENING
            return None
        if (
            self.filler is not None
            and result.first_audio_ms is not None
            and result.first_audio_ms >= self.filler.after_ms
        ):
            # Live injection needs the async loop; record the near-miss (§9).
            if self.session is not None:
                self.session.log(
                    NOTE, text=f"filler window missed: first audio {result.first_audio_ms}ms in"
                )
        self.state = TurnState.SPEAKING
        return result

    def on_playback_finished(self) -> None:
        """The outbound track drained; the patient's turn is over."""
        if self.state == TurnState.SPEAKING:
            log.debug("state: speaking → listening (turn ended)")
            self.state = TurnState.LISTENING

    # -- the abort sequence ----------------------------------------------------

    def interrupt(self, reason: str = "barge_in", speech_ms: int | None = None) -> None:
        """§5.3 abort: bump the epoch, clear playback, log, release the floor.

        Also the entry point for *scripted* curveballs (§7 overtalk): the
        call loop can interrupt its own playback mid-sentence with
        ``reason="scripted_overtalk"``.
        """
        self.generations.bump()
        if self.clear_playback is not None:
            self.clear_playback()
        log.info(
            "interrupt (reason=%s, was %s): epoch bumped, playback cleared",
            reason,
            self.state.value,
        )
        if self.session is not None:
            data: dict[str, Any] = {"reason": reason, "state": self.state.value}
            if speech_ms is not None:
                data["speech_ms"] = speech_ms
            self.session.log(BARGE_IN_EVENT, **data)
        self.state = TurnState.LISTENING

    # -- internals --------------------------------------------------------------

    def _send(self, audio: bytes) -> bool:
        if self.state is TurnState.LISTENING:
            return False
        if self.send_audio is not None:
            return self.send_audio(audio) is not False
        return True
