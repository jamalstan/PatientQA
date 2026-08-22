"""Orchestrator — the bridge between the integrations (DESIGN.md §2, §5.3).

Scaffold-level wiring: Twilio owns the call and the audio transport, and for
every committed agent utterance the simulator asks the brain for one gated
patient reply (§5.3 delta 1: the brain also classifies the agent's turn
``finished | backchannel | incomplete`` and we refuse to speak unless it is
``finished``) and speaks it as μ-law bytes.

``respond_streaming`` is the §5.3 delta-3 path the :class:`TurnDirector`
drives: brain deltas flow through :class:`~patientqa.cerebras.client.SayStream`
(the verdict closes within a few tokens, then the utterance streams) into
:class:`~patientqa.turns.SpeakableBuffer` (prosodic chunks, never single
tokens), each chunk synthesized and forwarded the moment it is speakable.
An ``is_stale`` callback — the director's epoch guard — stops the whole
chain the instant a barge-in lands, so a cancelled response can never leak
audio.

When a ``CallSession`` is attached, every turn's text, synthesized audio and
per-stage latency land in the session log (DESIGN.md §4) — the same events
the call viewer renders.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from patientqa.calllog.session import (
    AUDIO_PLAYED,
    BRAIN_REPLY,
    STT_FINAL,
    TTS_DONE,
    TURN_AGENT,
    TURN_PATIENT,
    CallSession,
)
from patientqa.cerebras.client import (
    AGENT,
    DEFAULT_SYSTEM_PROMPT,
    PATIENT,
    CerebrasSettings,
    PatientBrain,
    SayStream,
    Turn,
)
from patientqa.elevenlabs.client import ElevenLabsSettings, PatientVoice
from patientqa.turns import AGENT_TURN_FINISHED, SpeakableBuffer, split_speakable
from patientqa.twilio.client import TwilioClient, TwilioSettings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TurnResult:
    """One completed patient turn: the words, the audio, and the §4 timings."""

    text: str
    audio: bytes  # 8 kHz μ-law; forward to Twilio Media Streams as-is
    first_audio_ms: int | None = None  # committed transcript → first audible chunk


class PatientSimulator:
    """Bridges the brain and the voice around the running dialogue history."""

    def __init__(
        self,
        brain: PatientBrain,
        voice: PatientVoice,
        voice_id: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        session: CallSession | None = None,
        fixed_responder: Callable[[str], str | None] | None = None,
        audio_transform: Callable[[bytes], bytes] | None = None,
    ) -> None:
        self._brain = brain
        self._voice = voice
        self._voice_id = voice_id
        self._system_prompt = system_prompt
        self._session = session
        self._fixed_responder = fixed_responder
        self._audio_transform = audio_transform
        self._history: list[Turn] = []

    @property
    def history(self) -> tuple[Turn, ...]:
        """The dialogue so far, oldest turn first."""
        return tuple(self._history)

    # -- the two engagement paths ---------------------------------------------

    def respond(self, agent_utterance: str) -> TurnResult | None:
        """Agent speech → gated reply (Cerebras) → μ-law audio (ElevenLabs).

        Returns ``None`` when the semantic gate says the agent's turn is a
        backchannel or still incomplete — nothing is spoken and the utterance
        never becomes a dialogue turn.
        """
        return self._engage(agent_utterance, streaming=False)

    def respond_streaming(
        self,
        agent_utterance: str,
        *,
        on_audio: Callable[[bytes], bool | None] | None = None,
        is_stale: Callable[[], bool] | None = None,
    ) -> TurnResult | None:
        """Streaming engagement (§5.3 delta 3), driven by the TurnDirector.

        ``on_audio`` receives each chunk's μ-law bytes the moment they are
        synthesized; ``is_stale`` is polled between deltas — the instant it
        flips true the chain stops and ``None`` is returned (the epoch was
        bumped by a barge-in; playback was cleared elsewhere). The agent's
        utterance stays in the history: it was a real turn, only our reply
        to it was cancelled.
        """
        return self._engage(
            agent_utterance, streaming=True, on_audio=on_audio, is_stale=is_stale
        )

    def speak_unprompted(
        self,
        text: str,
        *,
        on_audio: Callable[[bytes], bool | None] | None = None,
        voice_id: str | None = None,
    ) -> TurnResult:
        """Speak ``text`` without waiting for an agent turn — the §6.5 opener.

        The conversation starter comes from the manifest's starters file (or
        the CLI for ad-hoc tests). It is synthesized chunk-by-chunk like a
        reply so speech starts early, and it lands in the history and the
        session log as a patient turn so the brain knows what it already said.
        """
        start = perf_counter()
        audio_parts: list[bytes] = []
        first_audio_ms: int | None = None
        for chunk in split_speakable(text):
            audio_parts.append(self._speak_chunk(chunk, on_audio, voice_id=voice_id))
            if first_audio_ms is None:
                first_audio_ms = int((perf_counter() - start) * 1000)
        audio = b"".join(audio_parts)
        tts_ms = int((perf_counter() - start) * 1000)
        self._history.append(Turn(role=PATIENT, content=text))
        if self._session is not None:
            self._session.log(
                TTS_DONE, chars=len(text), audio_bytes=len(audio), latency_ms=tts_ms
            )
            self._session.log(TURN_PATIENT, text=text, respond_ms=tts_ms)
        return TurnResult(text=text, audio=audio, first_audio_ms=first_audio_ms)

    # -- shared machinery -------------------------------------------------------

    def _engage(
        self,
        agent_utterance: str,
        *,
        streaming: bool,
        on_audio: Callable[[bytes], bool | None] | None = None,
        is_stale: Callable[[], bool] | None = None,
    ) -> TurnResult | None:
        session = self._session
        log.debug("engage start (streaming=%s, history=%d turns)", streaming, len(self._history))
        self._history.append(Turn(role=AGENT, content=agent_utterance))

        brain_start = perf_counter()
        if self._fixed_responder is not None:
            fixed = self._fixed_responder(agent_utterance)
            if fixed is not None:
                self._log_committed(agent_utterance, AGENT_TURN_FINISHED)
                return self._speak_fixed(
                    fixed,
                    started=brain_start,
                    on_audio=on_audio,
                    is_stale=is_stale,
                    agent_utterance=agent_utterance,
                )
        audio_parts: list[bytes] = []
        first_audio_ms: int | None = None
        tts_started: float | None = None

        if streaming:
            parser = SayStream()
            buffer = SpeakableBuffer()
            gate_logged = False
            for delta in self._brain.reply_stream(self._system_prompt, self._history):
                if is_stale is not None and is_stale():
                    if session is not None:
                        session.log(STT_FINAL, text=agent_utterance, gate="aborted")
                    return None  # epoch bumped: abandon without another word
                if parser.blocked:
                    break  # gate verdict is not "finished" — stop reading
                say_delta = parser.feed(delta)
                if not gate_logged and parser.verdict is not None:
                    gate_logged = True
                    self._log_committed(agent_utterance, parser.verdict)
                for chunk in buffer.feed(say_delta):
                    if tts_started is None:
                        tts_started = perf_counter()
                    audio_parts.append(self._speak_chunk(chunk, on_audio))
                    if is_stale is not None and is_stale():
                        if session is not None:
                            session.log(STT_FINAL, text=agent_utterance, gate="aborted")
                        return None
                    if first_audio_ms is None:
                        first_audio_ms = int((perf_counter() - brain_start) * 1000)
            reply = parser.finish()
            if not parser.blocked and reply.agent_turn == AGENT_TURN_FINISHED:
                if tail := buffer.flush():
                    if tts_started is None:
                        tts_started = perf_counter()
                    audio_parts.append(self._speak_chunk(tail, on_audio))
                    if is_stale is not None and is_stale():
                        if session is not None:
                            session.log(STT_FINAL, text=agent_utterance, gate="aborted")
                        return None
                    if first_audio_ms is None:
                        first_audio_ms = int((perf_counter() - brain_start) * 1000)
        else:
            reply = self._brain.reply(self._system_prompt, self._history)
            self._log_committed(agent_utterance, reply.agent_turn)

        brain_ms = int((perf_counter() - brain_start) * 1000)

        if reply.agent_turn != AGENT_TURN_FINISHED or not reply.say:
            # Backchannel / incomplete / empty reply: the floor is not ours.
            # The utterance never becomes a dialogue turn; the committed
            # transcript and its gate verdict are already logged above.
            self._history.pop()
            return None

        if not audio_parts:
            tts_started = perf_counter()
            audio_parts.append(self._speak_chunk(reply.say, on_audio))
            first_audio_ms = int((perf_counter() - brain_start) * 1000)
        tts_ms = int((perf_counter() - (tts_started or brain_start)) * 1000)
        audio = b"".join(audio_parts)

        self._history.append(Turn(role=PATIENT, content=reply.say))
        if session is not None:
            session.log(
                BRAIN_REPLY, say=reply.say, latency_ms=brain_ms, gate=reply.agent_turn
            )
            session.log(
                TTS_DONE, chars=len(reply.say), audio_bytes=len(audio), latency_ms=tts_ms
            )
            session.log(TURN_PATIENT, text=reply.say, respond_ms=brain_ms + tts_ms)
        return TurnResult(text=reply.say, audio=audio, first_audio_ms=first_audio_ms)

    def _speak_fixed(
        self,
        text: str,
        *,
        started: float,
        on_audio: Callable[[bytes], bool | None] | None,
        is_stale: Callable[[], bool] | None,
        agent_utterance: str,
    ) -> TurnResult | None:
        """Synthesize one deterministic identity response with epoch checks."""
        audio_parts: list[bytes] = []
        first_audio_ms: int | None = None
        for chunk in split_speakable(text):
            audio_parts.append(self._speak_chunk(chunk, on_audio))
            if is_stale is not None and is_stale():
                if self._session is not None:
                    self._session.log(STT_FINAL, text=agent_utterance, gate="aborted")
                return None
            if first_audio_ms is None:
                first_audio_ms = int((perf_counter() - started) * 1000)
        audio = b"".join(audio_parts)
        elapsed_ms = int((perf_counter() - started) * 1000)
        self._history.append(Turn(role=PATIENT, content=text))
        if self._session is not None:
            self._session.log(
                BRAIN_REPLY, say=text, latency_ms=0, gate=AGENT_TURN_FINISHED
            )
            self._session.log(
                TTS_DONE, chars=len(text), audio_bytes=len(audio), latency_ms=elapsed_ms
            )
            self._session.log(TURN_PATIENT, text=text, respond_ms=elapsed_ms)
        return TurnResult(text=text, audio=audio, first_audio_ms=first_audio_ms)

    def _log_committed(self, agent_utterance: str, gate: str) -> None:
        """Log the STT final with its gate verdict; promote to a turn when engaged."""
        log.debug("semantic gate verdict %s for %r", gate, agent_utterance[:60])
        if self._session is None:
            return
        self._session.log(STT_FINAL, text=agent_utterance, gate=gate)
        if gate == AGENT_TURN_FINISHED:
            self._session.log(TURN_AGENT, text=agent_utterance)

    def _speak_chunk(
        self,
        chunk: str,
        on_audio: Callable[[bytes], bool | None] | None,
        *,
        voice_id: str | None = None,
    ) -> bytes:
        started = perf_counter()
        audio = self._voice.synthesize(chunk, voice_id or self._voice_id)
        if self._audio_transform is not None:
            audio = self._audio_transform(audio)
        accepted = on_audio(audio) is not False if on_audio is not None else True
        if accepted and self._session is not None:
            self._session.append_outbound(audio)
            self._session.log(AUDIO_PLAYED, audio_bytes=len(audio))
        log.debug(
            "tts chunk: %d chars → %d bytes (%d ms)",
            len(chunk),
            len(audio),
            int((perf_counter() - started) * 1000),
        )
        return audio if accepted else b""


def build_twilio_client(path: Path | None = None) -> TwilioClient:
    """The Twilio integration, wired from secrets.toml."""
    return TwilioClient(TwilioSettings.from_secrets(path))


def build_patient_simulator(voice_id: str, path: Path | None = None) -> PatientSimulator:
    """The brain + voice integrations, wired from secrets.toml."""
    brain = PatientBrain(CerebrasSettings.from_secrets(path))
    voice = PatientVoice(ElevenLabsSettings.from_secrets(path))
    return PatientSimulator(brain=brain, voice=voice, voice_id=voice_id)
