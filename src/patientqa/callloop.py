"""The live call loop (DESIGN.md §2, §5.4's "call-loop stage").

One module, one job: bridge a Twilio Media Streams WebSocket to the
transport-free core everything else already implements —

* inbound ``media`` frames → session inbound leg + :class:`EnergyGate`
  (via :meth:`TurnDirector.on_media_frame`) + the STT socket,
* STT ``Final`` events → :meth:`TurnDirector.on_agent_final` on a worker
  thread (the brain + TTS calls are blocking),
* the director's ``send_audio`` port → outbound ``media`` messages,
* a trailing ``mark`` → :meth:`TurnDirector.on_playback_finished` — our
  turn ended, the floor returns to the agent.

All outbound Twilio traffic (media / clear / mark) funnels through one FIFO
queue drained by a single sender task, so a ``clear`` from a barge-in can
never overtake chunks synthesized before it. Engages run on a one-thread
executor: replies serialize in order, and the epoch guard
(:class:`patientqa.turns.Generations`) plus a state check in the send port
keep a cancelled generation from leaking audio after an interrupt.

:class:`CallLoop` is deliberately synchronous and socket-free — it takes
``send_json`` / ``stt_send`` / ``executor`` / ``call_later`` ports — so the
whole routing state machine is testable offline; :func:`run_call` is the
thin asyncio glue that wires those ports to real sockets and dials.
"""

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve as ws_serve

from patientqa.calllog.session import (
    BEHAVIOR_FIRED,
    CALL_CONNECTED,
    ERROR,
    NOTE,
    STT_FINAL,
    CallSession,
)
from patientqa.cerebras.client import (
    DEFAULT_SYSTEM_PROMPT,
    CerebrasSettings,
    PatientBrain,
)
from patientqa.deepgram.client import (
    DeepgramSettings,
    build_deepgram_url,
    parse_deepgram_message,
)
from patientqa.elevenlabs.client import ElevenLabsSettings, PatientVoice
from patientqa.elevenlabs.scribe import (
    ScribeSettings,
    build_scribe_url,
    parse_scribe_message,
    verify_scribe_session,
    wrap_scribe_audio,
)
from patientqa.logsetup import attach_call_log, detach_call_log
from patientqa.orchestrator import PatientSimulator, build_twilio_client
from patientqa.stt import Final, SttEvent, build_stt_settings
from patientqa.turns import (
    BACKCHANNEL,
    SPEECH_ENDED,
    SPEECH_STARTED,
    EnergyGate,
    TurnDirector,
    TurnState,
)

#: Sentinel mark name: Twilio echoes it once every queued chunk has played.
END_OF_TURN_MARK = "patient-turn-end"

#: Default §6.5-style opener for ad-hoc tests (manifests carry real starters).
DEFAULT_OPENER = "Hi, I'd like to schedule an appointment, please."

#: A premade ElevenLabs voice from this account's library ("Will — Relaxed
#: Optimist"); the persona pipeline's Voice Design replaces this per entry.
DEFAULT_VOICE_ID = "bIHbv24MWmeRgasZH58o"

# The campaign targets three minutes, but no caller-controlled option may keep
# a live assessment call open beyond five. Twilio receives the same deadline,
# so the ceiling still holds if this process or its event loop wedges.
DEFAULT_CALL_SECONDS = 180.0
MAX_CALL_SECONDS = 300.0


def validate_call_duration(seconds: float) -> float:
    """Validate and normalize the authorized live-call duration."""
    value = float(seconds)
    if not 0 < value <= MAX_CALL_SECONDS:
        raise ValueError(
            f"max_call_s must be greater than 0 and no more than {MAX_CALL_SECONDS:.0f}s"
        )
    return value


def validate_voice_assignment(voice_id: str, manifest: Mapping[str, Any]) -> None:
    """Reject a resolved campaign voice whose provenance says it mismatches.

    Older/ad-hoc manifests may omit the derived ``voice`` block and remain
    valid. Once that block exists, its ID and gender claim are binding.
    """
    voice = manifest.get("voice")
    if not isinstance(voice, Mapping):
        return
    recorded_id = str(voice.get("voice_id", ""))
    if recorded_id and recorded_id != voice_id:
        raise ValueError(
            f"voice assignment mismatch: run_call received {voice_id!r}, "
            f"metadata records {recorded_id!r}"
        )
    persona_gender = str(voice.get("persona_gender", ""))
    voice_gender = str(voice.get("voice_gender", ""))
    if voice.get("gender_match") is False or (
        persona_gender and voice_gender and persona_gender != voice_gender
    ):
        raise ValueError(
            f"voice gender mismatch: persona={persona_gender or '?'} "
            f"voice={voice_gender or '?'}"
        )

_FINAL_FRESHNESS_S = 4.0
_TRAILING_CONNECTOR = re.compile(
    r"(?:\b(?:and|but|or|because|so|then|if|when|before|after|to)|[-–—])\s*$",
    re.IGNORECASE,
)
_NONINTERACTIVE_PATTERNS = (
    "this call may be recorded",
    "para español",
    "para espanol",
    "oprima el dos",
    "press 2 for spanish",
    "transfer you now",
    "transferring you now",
    "thank you for call",
)


def normalize_final(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def is_noninteractive_utterance(text: str) -> bool:
    """Announcements/backchannels that do not call for a patient answer."""
    lowered = text.lower().strip()
    if any(pattern in lowered for pattern in _NONINTERACTIVE_PATTERNS):
        return True
    return normalize_final(text) in {
        "got it",
        "i see",
        "okay",
        "sure",
        "thank you",
        "thanks",
        "i can help with that",
    }


def is_terminal_utterance(text: str) -> bool:
    normalized = normalize_final(text)
    return normalized in {
        "goodbye",
        "bye",
        "have a good day",
        "have a great day",
        "thank you goodbye",
        "thanks goodbye",
    }


def is_obviously_incomplete(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(("...", "…")) or bool(_TRAILING_CONNECTOR.search(stripped))

#: Persona card for ad-hoc "does the loop hold a conversation" calls.
BASIC_PERSONA_PROMPT = """\
PERSONA — stay in character:
You are Jordan Hale, 34 years old, date of birth April 12 1992, a new
patient calling this clinic for the first time. Friendly, direct, polite.

OBJECTIVE: Schedule a new-patient general checkup between one and two weeks
from today; mornings work best. Answer the agent's questions briefly and
naturally. If the agent offers a morning slot that works, accept it and
confirm the details once. After confirming, thank them and say goodbye.

SCHEDULING RULES — these override anything the agent assumes:
- Accept nothing earlier than one week from today. A bare weekday means
  nothing to you: pin down the actual date first ("Thursday the 27th?"),
  then judge it against the window.
- If a slot falls inside the refused window ("tomorrow", "this week"),
  decline it politely and repeat the window: not before a week out,
  mornings preferred."""


def date_context(today: date | None = None) -> str:
    """The calendar facts the brain needs to judge offered slots.

    A named weekday is meaningless without today's date — in a live test the
    brain refused "tomorrow", asked for the following fortnight, then
    accepted "Thursday", which was tomorrow.
    """
    today = today or date.today()
    week_out = date.fromordinal(today.toordinal() + 7)
    return (
        f"CALL CONTEXT — today is {today:%A}, {today:%B} {today.day}, {today.year}; "
        f"one week from today is {week_out:%A}, {week_out:%B} {week_out.day}, "
        f"{week_out.year}. "
        "Translate every day or date the agent mentions into an actual date "
        "before accepting or refusing it."
    )

#: If Twilio's ``mark`` echo is lost, assume playback drained once
#: queued_seconds + this grace have passed with no new audio (μ-law plays
#: in real time, so this is a deadline, not a guess).
PLAYBACK_GRACE_S = 1.5

#: Recovery for a stalled line (a live test died here): the agent's last
#: committed turn was gate-refused or answered, then nothing committable
#: arrives — say this, at most ``MAX_NUDGES`` times per call.
NUDGE_TEXT = "Hello? Are you still there?"
NUDGE_AFTER_S = 10.0
MAX_NUDGES = 2

log = logging.getLogger(__name__)


# -- Twilio Media Streams message wrapping (pure, testable) -------------------


def wrap_media(ulaw: bytes, stream_sid: str) -> str:
    """One outbound μ-law chunk as a Media Streams ``media`` message."""
    payload = base64.b64encode(ulaw).decode("ascii")
    return json.dumps(
        {"event": "media", "streamSid": stream_sid, "media": {"payload": payload}},
        separators=(",", ":"),
    )


def wrap_clear(stream_sid: str) -> str:
    """The playback-buffer flush (§5.3 abort sequence's last step)."""
    return json.dumps({"event": "clear", "streamSid": stream_sid}, separators=(",", ":"))


def wrap_mark(stream_sid: str, name: str = END_OF_TURN_MARK) -> str:
    """Queue a mark after the audio; Twilio echoes it when playback drains."""
    return json.dumps(
        {"event": "mark", "streamSid": stream_sid, "mark": {"name": name}},
        separators=(",", ":"),
    )


def media_payload(message: Mapping[str, Any]) -> bytes | None:
    """An inbound ``media`` message → its μ-law payload, else ``None``."""
    if message.get("event") != "media":
        return None
    payload = (message.get("media") or {}).get("payload")
    if not payload:
        return None
    return base64.b64decode(payload)


def _ms_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


# -- the routing state machine ------------------------------------------------


@dataclass
class CallLoop:
    """One call's wiring: Twilio messages and STT events in, audio out.

    Everything runs on the caller's thread except engages, which the
    executor owns; cross-thread calls go through the injected thread-safe
    ``send_json`` port (production wires it to the asyncio sender queue).
    """

    simulator: Any
    session: CallSession
    stt_send: Callable[[bytes], None]
    send_json: Callable[[str], None]
    executor: Executor
    call_later: Callable[[float, Callable[[], None]], None]
    on_turn: Callable[[str, str], None] | None = None
    on_stopped: Callable[[], None] | None = None
    energy: EnergyGate = field(default_factory=EnergyGate)
    opener_text: str | None = DEFAULT_OPENER
    opener_after_s: float = 6.0
    max_call_s: float = 180.0
    nudge_text: str | None = NUDGE_TEXT
    nudge_after_s: float = NUDGE_AFTER_S
    max_nudges: int = MAX_NUDGES
    scripted_barge_ins: tuple[str, ...] = ()
    scripted_barge_after_s: float = 1.0
    scripted_barge_kind: str = "barge_in"
    scripted_barge_voice_id: str | None = None
    scripted_barge_skip_speeches: int = 0
    scripted_response_delays_s: tuple[float, ...] = ()
    director: TurnDirector = field(init=False)

    def __post_init__(self) -> None:
        self.director = TurnDirector(
            self.simulator,
            send_audio=self._port_send_audio,
            clear_playback=self._port_clear,
            energy=self.energy,
            session=self.session,
        )
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self.stopped = False
        self.end_reason: str | None = None
        self._pending_final: tuple[str, float] | None = None
        self._engage_future: Future | None = None
        self._heard_agent = False
        self._opener_fired = False
        self._nudges = 0
        self._idle_arm = 0
        self._queued_bytes = 0
        self._last_send = 0.0
        self._scripted_barge_index = 0
        self._scripted_speech_count = 0
        self._scripted_delay_index = 0
        self._scripted_delay_active = False
        self._speech_arm = 0
        self._recent_finals: dict[str, float] = {}
        self._stop_after_mark = False
        self._incomplete_prefix: tuple[str, float] | None = None
        self._log = self.session.log

    # -- Twilio socket → us ------------------------------------------------

    def on_twilio_message(self, message: Mapping[str, Any]) -> None:
        event = message.get("event")
        if event == "start":
            self._on_start(message)
        elif event == "media":
            self._on_media(message)
        elif event == "mark":
            self._on_mark(message)
        elif event == "stop":
            self.stop("stream_stopped")
        # "connected" and unknown events: nothing to do.

    def _on_start(self, message: Mapping[str, Any]) -> None:
        start = message.get("start") or {}
        self.stream_sid = str(start.get("streamSid") or "")
        self.call_sid = str(start.get("callSid") or "")
        self._log(CALL_CONNECTED, stream_sid=self.stream_sid, call_sid=self.call_sid)
        log.info("stream started (streamSid=%s callSid=%s)", self.stream_sid, self.call_sid)
        self.call_later(self.max_call_s, lambda: self.stop("max_duration"))
        if self.opener_text:
            self.call_later(self.opener_after_s, self._fire_opener_if_needed)
        self._arm_idle_nudge()

    def _on_media(self, message: Mapping[str, Any]) -> None:
        ulaw = media_payload(message)
        if ulaw is None or self.stream_sid is None:
            return
        energy_event = self.director.on_media_frame(ulaw)
        self._handle_scripted_barge_in(energy_event)
        if not self.stopped:
            self.stt_send(ulaw)
            self._watchdog()
            self._maybe_advance()

    def _handle_scripted_barge_in(self, energy_event: str | None) -> None:
        """Arm real patient-over-agent speech for an explicit stress probe.

        The brain normally runs only after STT finalizes an agent turn, which
        is too late to interrupt it. A scripted campaign can opt in to short
        unprompted lines fired while inbound speech is still active.
        """
        if not self.scripted_barge_ins:
            return
        if energy_event == SPEECH_STARTED:
            self._scripted_speech_count += 1
            if self._scripted_speech_count <= self.scripted_barge_skip_speeches:
                return
            self._speech_arm += 1
            arm = self._speech_arm
            self.call_later(
                self.scripted_barge_after_s,
                lambda: self._fire_scripted_barge_in(arm),
            )
        elif energy_event in (BACKCHANNEL, SPEECH_ENDED):
            self._speech_arm += 1

    def _fire_scripted_barge_in(self, arm: int) -> None:
        if (
            arm != self._speech_arm
            or self.stopped
            or not self.energy.in_speech
            or self.director.state is not TurnState.LISTENING
            or self._engage_future is not None
            or self._scripted_barge_index >= len(self.scripted_barge_ins)
        ):
            return
        text = self.scripted_barge_ins[self._scripted_barge_index]
        self._scripted_barge_index += 1
        self.director.interrupt(
            reason="scripted_patient_overtalk", speech_ms=self.energy.speech_ms
        )
        self._engage_future = self._executor_submit(
            self._run_unprompted,
            text,
            self.scripted_barge_kind,
            self.scripted_barge_voice_id,
        )

    def _on_mark(self, message: Mapping[str, Any]) -> None:
        name = (message.get("mark") or {}).get("name")
        if name != END_OF_TURN_MARK:
            return
        self._queued_bytes = 0
        log.debug("mark echo: our turn ended (%s)", self.director.state.value)
        self.director.on_playback_finished()
        if self._stop_after_mark:
            self.stop("patient_goodbye")
            return
        self._maybe_advance()
        self._arm_idle_nudge()

    # -- STT socket → us -----------------------------------------------------

    def on_stt_event(self, event: SttEvent) -> None:
        if not isinstance(event, Final):
            return  # partials never trigger a turn (§5.3)
        self._dispatch_final(event.text)

    def _dispatch_final(self, text: str, *, allow_scripted_delay: bool = True) -> None:
        if self.stopped or not text:
            return
        self._heard_agent = True
        self._arm_idle_nudge()
        now = time.monotonic()
        if is_terminal_utterance(text):
            self._log(STT_FINAL, text=text, gate="terminal")
            self.stop("agent_goodbye")
            return
        if is_noninteractive_utterance(text):
            self._log(STT_FINAL, text=text, gate="noninteractive")
            log.debug("noninteractive agent utterance ignored: %r", text[:60])
            return
        if self._scripted_delay_active:
            self._pending_final = (text, now)
            return
        if self._incomplete_prefix is not None:
            prefix, prefix_at = self._incomplete_prefix
            self._incomplete_prefix = None
            if now - prefix_at <= _FINAL_FRESHNESS_S:
                text = re.sub(r"[\s.\-–—…]+$", "", prefix) + " " + text.lstrip()
                log.debug("joined split agent final: %r", text[:80])
        if is_obviously_incomplete(text):
            self._incomplete_prefix = (text, now)
            self._log(STT_FINAL, text=text, gate="incomplete_heuristic")
            log.debug("obviously incomplete agent utterance ignored: %r", text[:60])
            return
        normalized = normalize_final(text)
        previous = self._recent_finals.get(normalized)
        self._recent_finals[normalized] = now
        if previous is not None and now - previous < _FINAL_FRESHNESS_S:
            self._log(STT_FINAL, text=text, gate="duplicate")
            log.debug("duplicate agent final ignored: %r", text[:60])
            return
        if (
            self._engage_future is None
            and self.director.state is TurnState.LISTENING
            and not self.energy.in_speech
        ):
            if (
                allow_scripted_delay
                and self._scripted_delay_index < len(self.scripted_response_delays_s)
            ):
                delay_s = self.scripted_response_delays_s[self._scripted_delay_index]
                self._scripted_delay_index += 1
                self._scripted_delay_active = True
                self._pending_final = (text, now)
                self._log(
                    BEHAVIOR_FIRED,
                    behavior="long_silence",
                    intentional=True,
                    delay_s=delay_s,
                )
                self.call_later(delay_s, self._release_scripted_delay)
                return
            log.debug("agent final → engage: %r", text[:60])
            self._engage_future = self._executor_submit(self._run_engage, text)
        else:
            # Busy thinking/speaking: keep only the latest — the epoch guard
            # and energy gate own whatever the agent is doing right now.
            log.debug("agent final queued (busy, %s): %r", self.director.state.value, text[:60])
            self._pending_final = (text, now)

    def _release_scripted_delay(self) -> None:
        if not self._scripted_delay_active or self.stopped:
            return
        self._scripted_delay_active = False
        pending, self._pending_final = self._pending_final, None
        if pending is not None:
            text, _ = pending
            self._recent_finals.pop(normalize_final(text), None)
            self._dispatch_final(text, allow_scripted_delay=False)

    def _executor_submit(self, fn: Callable[..., None], *args: Any) -> Future:
        return self.executor.submit(fn, *args)

    def _maybe_advance(self) -> None:
        """Release the turn slot once nothing is in flight, then replay any
        agent final that arrived while we held the floor."""
        if self.stopped:
            return
        if self._engage_future is not None:
            if not self._engage_future.done():
                return
            self._engage_future = None
        if self.director.state is not TurnState.LISTENING:
            return
        if self.energy.in_speech:
            return
        pending, self._pending_final = self._pending_final, None
        if pending:
            text, queued_at = pending
            if time.monotonic() - queued_at <= _FINAL_FRESHNESS_S:
                log.debug("replaying pending agent final: %r", text[:60])
                self._recent_finals.pop(normalize_final(text), None)
                self._dispatch_final(text)
            else:
                self._log(STT_FINAL, text=text, gate="stale")
                log.debug("stale pending agent final dropped: %r", text[:60])

    # -- the two speech paths (worker thread) --------------------------------

    def _run_engage(self, text: str) -> None:
        self._emit("agent", text)
        started = time.monotonic()
        try:
            result = self.director.on_agent_final(text)
        except Exception as exc:  # one bad turn must not kill a live call
            self._log(ERROR, stage="engage", error=repr(exc))
            log.warning("engage failed after %dms: %r", _ms_since(started), exc)
            return
        finally:
            self.call_later(0, self._maybe_advance)
        if result is not None:
            log.info(
                "engage done in %dms → spoke %d chars", _ms_since(started), len(result.text)
            )
            self._emit("patient", result.text)
            if normalize_final(result.text).endswith(("goodbye", "bye")):
                self._stop_after_mark = True
            if self.director.state is TurnState.SPEAKING:
                self.send_json(wrap_mark(self.stream_sid))
        else:
            log.info("engage done in %dms → gate kept us silent", _ms_since(started))
            self._arm_idle_nudge()

    def _fire_opener_if_needed(self) -> None:
        if (
            self.stopped
            or self._opener_fired
            or self._heard_agent
            or self._engage_future is not None
            or self.director.state is not TurnState.LISTENING
        ):
            return
        if self.energy.in_speech:
            self.call_later(0.5, self._fire_opener_if_needed)
            return
        self._opener_fired = True
        log.info("opener fired after %.1fs of silence", self.opener_after_s)
        self._engage_future = self._executor_submit(self._run_opener)

    def _run_opener(self) -> None:
        self._run_unprompted(self.opener_text, "opener")

    # -- the idle nudge (stalled-line recovery) --------------------------------

    def _arm_idle_nudge(self) -> None:
        """Schedule one nudge check; any new agent final or finished turn
        re-arms, so only a genuinely quiet line ever trips it."""
        if self.nudge_text is None:
            return
        self._idle_arm += 1
        arm = self._idle_arm
        self.call_later(self.nudge_after_s, lambda: self._fire_nudge_if_idle(arm))

    def _fire_nudge_if_idle(self, arm: int) -> None:
        if arm != self._idle_arm:  # superseded by newer activity
            return
        if (
            self.stopped
            or self._engage_future is not None
            or self.director.state is not TurnState.LISTENING
            or self.energy.in_speech  # agent is mid-utterance, just uncommitted
        ):
            return
        if self._nudges >= self.max_nudges:
            log.debug("idle again but nudge cap (%d) reached", self.max_nudges)
            return
        self._nudges += 1
        log.info(
            "idle nudge %d/%d after %.1fs without a committable agent turn",
            self._nudges,
            self.max_nudges,
            self.nudge_after_s,
        )
        self._engage_future = self._executor_submit(
            self._run_unprompted, self.nudge_text, "nudge"
        )

    # -- shared unprompted speech (opener + nudge) ------------------------------

    def _run_unprompted(
        self, text: str | None, kind: str, voice_id: str | None = None
    ) -> None:
        assert text is not None
        self.director.state = TurnState.THINKING
        if kind not in ("opener", "nudge"):
            self._log(
                BEHAVIOR_FIRED,
                behavior=kind,
                intentional=True,
                voice_override=voice_id is not None,
                text=text,
            )
        try:
            self.simulator.speak_unprompted(
                text, on_audio=self._port_send_audio, voice_id=voice_id
            )
        except Exception as exc:
            self._log(ERROR, stage=kind, error=repr(exc))
            log.warning("%s failed: %r", kind, exc)
            self.director.state = TurnState.LISTENING
            self.call_later(0, self._maybe_advance)
            return
        self._emit("patient", text)
        self.director.state = TurnState.SPEAKING
        self.send_json(wrap_mark(self.stream_sid))
        self.call_later(0, self._maybe_advance)

    # -- outbound ports (called from the engage thread) ----------------------

    def _port_send_audio(self, ulaw: bytes) -> bool:
        # Belt-and-braces beside the §5.4 epoch guard: an interrupted
        # generation may synthesize a few more chunks before its delta loop
        # notices; after interrupt() the director is LISTENING, so drop them.
        if self.stopped or self.director.state is TurnState.LISTENING:
            return False
        self._queued_bytes += len(ulaw)
        self._last_send = time.monotonic()
        self.send_json(wrap_media(ulaw, self.stream_sid))
        return True

    def _port_clear(self) -> None:
        self._queued_bytes = 0
        self.send_json(wrap_clear(self.stream_sid))

    def _watchdog(self) -> None:
        """Backstop for a lost ``mark`` echo: μ-law drains in real time, so
        if everything queued has long since played and we are still SPEAKING,
        the turn is over whether Twilio told us or not."""
        if self.director.state is not TurnState.SPEAKING or not self._queued_bytes:
            return
        drained_at = self._last_send + self._queued_bytes / 8000 + PLAYBACK_GRACE_S
        if time.monotonic() >= drained_at:
            log.info("playback watchdog released the turn (mark echo lost)")
            self._log(NOTE, text="playback watchdog released the turn (mark echo lost)")
            self._queued_bytes = 0
            self.director.on_playback_finished()
            self._maybe_advance()
            self._arm_idle_nudge()

    # -- lifecycle -------------------------------------------------------------

    def stop(self, reason: str = "completed") -> None:
        if self.stopped:
            return
        self.stopped = True
        self.end_reason = reason
        log.info("call loop stopping: %s", reason)
        self.director.interrupt(reason="call_stopped")  # epoch bump: abandon in-flight
        if self.on_stopped is not None:
            self.on_stopped()

    # -- small helpers ----------------------------------------------------------

    def _emit(self, role: str, text: str) -> None:
        if self.on_turn is not None:
            self.on_turn(role, text)


# -- the asyncio glue ----------------------------------------------------------


@dataclass(frozen=True)
class SttWiring:
    """Provider-agnostic STT socket facts (URL, auth header, parser, encoder)."""

    url: str
    headers: dict[str, str]
    verify_echo: Callable[[Mapping[str, Any]], list[str]] | None
    parse: Callable[[Mapping[str, Any]], SttEvent | None]
    #: μ-law chunk → what goes on the wire. Deepgram takes raw binary frames;
    #: Scribe only accepts its JSON ``input_audio_chunk`` protocol message.
    encode: Callable[[bytes], str | bytes]


def stt_wiring(settings: DeepgramSettings | ScribeSettings) -> SttWiring:
    """Map the secrets-selected STT settings to socket wiring (§3.4)."""
    if isinstance(settings, DeepgramSettings):
        return SttWiring(
            url=build_deepgram_url(settings),
            headers={"Authorization": f"Token {settings.api_key}"},
            verify_echo=None,
            parse=parse_deepgram_message,
            encode=lambda chunk: chunk,
        )
    return SttWiring(
        url=build_scribe_url(settings),
        headers={"xi-api-key": settings.api_key},
        verify_echo=lambda first: verify_scribe_session(first, settings),
        parse=parse_scribe_message,
        encode=wrap_scribe_audio,
    )


async def run_call(
    *,
    to_number: str | None = None,
    stream_url: str,
    persona_prompt: str = BASIC_PERSONA_PROMPT,
    voice_id: str = DEFAULT_VOICE_ID,
    opener_text: str | None = DEFAULT_OPENER,
    max_call_s: float = DEFAULT_CALL_SECONDS,
    ring_timeout_s: float = 45.0,
    port: int = 8080,
    call_id: str = "live-test",
    calls_root: Path | None = None,
    manifest: dict[str, Any] | None = None,
    scripted_barge_ins: tuple[str, ...] = (),
    scripted_barge_kind: str = "barge_in",
    scripted_barge_voice_id: str | None = None,
    scripted_barge_skip_speeches: int = 0,
    scripted_response_delays_s: tuple[float, ...] = (),
    outbound_audio_transform: Callable[[bytes], bytes] | None = None,
    fixed_responder: Callable[[str], str | None] | None = None,
    on_turn: Callable[[str, str], None] | None = None,
    auto_analyze: bool = True,
    analysis_judge: bool = True,
) -> Path:
    """Dial ``to_number`` and run one conversation to its end.

    ``to_number=None`` dials the allowlisted live-test number. ``manifest`` is
    the session's provenance record (persona + objective + destination); the
    campaign runner passes the manifest entry so every recording carries its
    scenario. Returns the session folder (recording, transcript, timings, and
    automatic post-call findings — DESIGN.md §12).
    """
    max_call_s = validate_call_duration(max_call_s)
    root = Path(calls_root or "calls")
    root.mkdir(parents=True, exist_ok=True)
    manifest = manifest or {
        "persona": {"name": "Jordan Hale", "age": 34, "language": "English"},
        "objective": {"type": "smoke_test", "goal": "basic conversation: book a checkup"},
        "to": to_number,
    }
    validate_voice_assignment(voice_id, manifest)
    session = CallSession.start(root, call_id, manifest=manifest)
    log_handler = attach_call_log(session.directory)
    log.info(
        "call session %s → %s (stream %s)", session.directory.name, to_number, stream_url
    )
    try:
        return await _run_connected_call(
            session=session,
            to_number=to_number,
            stream_url=stream_url,
            persona_prompt=persona_prompt,
            voice_id=voice_id,
            opener_text=opener_text,
            max_call_s=max_call_s,
            ring_timeout_s=ring_timeout_s,
            port=port,
            scripted_barge_ins=scripted_barge_ins,
            scripted_barge_kind=scripted_barge_kind,
            scripted_barge_voice_id=scripted_barge_voice_id,
            scripted_barge_skip_speeches=scripted_barge_skip_speeches,
            scripted_response_delays_s=scripted_response_delays_s,
            outbound_audio_transform=outbound_audio_transform,
            fixed_responder=fixed_responder,
            on_turn=on_turn,
        )
    finally:
        if not session.closed:
            session.close("setup_failed")
        log.info("call session %s closed", session.directory.name)
        detach_call_log(log_handler)
        if auto_analyze:
            try:
                from patientqa.analyze import postprocess_session

                await asyncio.to_thread(
                    postprocess_session,
                    session.directory,
                    judge=analysis_judge,
                )
                log.info("post-call analysis written for %s", session.directory.name)
            except Exception as exc:
                # Reporting must not rewrite the call outcome or mask a dial/STT error.
                log.warning(
                    "post-call analysis failed for %s: %r",
                    session.directory.name,
                    exc,
                )


async def _run_connected_call(
    *,
    session: CallSession,
    to_number: str,
    stream_url: str,
    persona_prompt: str,
    voice_id: str,
    opener_text: str | None,
    max_call_s: float,
    ring_timeout_s: float,
    port: int,
    scripted_barge_ins: tuple[str, ...],
    scripted_barge_kind: str,
    scripted_barge_voice_id: str | None,
    scripted_barge_skip_speeches: int,
    scripted_response_delays_s: tuple[float, ...],
    outbound_audio_transform: Callable[[bytes], bytes] | None,
    fixed_responder: Callable[[str], str | None] | None,
    on_turn: Callable[[str, str], None] | None,
) -> Path:
    """The dial-and-converse half of :func:`run_call`; logging is attached."""
    loop = asyncio.get_running_loop()
    twilio = build_twilio_client()
    stt = stt_wiring(build_stt_settings())
    audio_transform = outbound_audio_transform
    if audio_transform is not None:
        transform_fired = False

        def logged_audio_transform(audio: bytes) -> bytes:
            nonlocal transform_fired
            if not transform_fired:
                transform_fired = True
                session.log(
                    BEHAVIOR_FIRED,
                    behavior="degraded_audio_digits",
                    intentional=True,
                    audio_bytes=len(audio),
                )
            return audio_transform(audio)

    else:
        logged_audio_transform = None
    simulator = PatientSimulator(
        brain=PatientBrain(CerebrasSettings.from_secrets()),
        voice=PatientVoice(ElevenLabsSettings.from_secrets()),
        voice_id=voice_id,
        system_prompt=(
            DEFAULT_SYSTEM_PROMPT + "\n\n" + persona_prompt + "\n\n" + date_context()
        ),
        session=session,
        fixed_responder=fixed_responder,
        audio_transform=logged_audio_transform,
    )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engage")
    twilio_out: asyncio.Queue[str] = asyncio.Queue()
    stt_out: asyncio.Queue[bytes] = asyncio.Queue()
    connected = asyncio.Event()
    finished = asyncio.Event()
    call_loop: list[CallLoop] = []

    server = await ws_serve(
        _twilio_handler(
            simulator=simulator,
            session=session,
            executor=executor,
            twilio_out=twilio_out,
            stt_out=stt_out,
            connected=connected,
            finished=finished,
            call_loop=call_loop,
            stt=stt,
            on_turn=on_turn,
            opener_text=opener_text,
            max_call_s=max_call_s,
            scripted_barge_ins=scripted_barge_ins,
            scripted_barge_kind=scripted_barge_kind,
            scripted_barge_voice_id=scripted_barge_voice_id,
            scripted_barge_skip_speeches=scripted_barge_skip_speeches,
            scripted_response_delays_s=scripted_response_delays_s,
        ),
        "127.0.0.1",
        port,
    )

    try:
        call_sid = await loop.run_in_executor(
            None,
            lambda: twilio.place_call(
                stream_url, to=to_number, max_duration_s=max_call_s
            ),
        )
    except Exception as exc:
        session.close(f"dial_failed: {exc!r}")
        server.close()
        raise
    log.info("placed call %s via %s", call_sid, stream_url)
    session.log(NOTE, text=f"placed call {call_sid} via {stream_url}")

    reason = "unknown"
    try:
        try:
            await asyncio.wait_for(connected.wait(), timeout=ring_timeout_s)
        except asyncio.TimeoutError:
            reason = "ring_timeout"
        else:
            try:
                # CallLoop's own max-duration watchdog owns the deadline;
                # this outer wait is a pure backstop for a wedged loop.
                await asyncio.wait_for(finished.wait(), timeout=max_call_s + 60.0)
            except asyncio.TimeoutError:
                await asyncio.sleep(0.5)
            if call_loop:
                reason = call_loop[0].end_reason or "completed"
    finally:
        if call_loop:
            call_loop[0].stop(reason)
        twilio.hang_up(call_sid)  # idempotent belt-and-braces (§9)
        server.close()
        await server.wait_closed()
        executor.shutdown(wait=True)
        session.close(reason, call_sid=call_sid)
    log.info("call ended: %s", reason)
    return session.directory


def _twilio_handler(
    *,
    simulator: PatientSimulator,
    session: CallSession,
    executor: Executor,
    twilio_out: "asyncio.Queue[str]",
    stt_out: "asyncio.Queue[bytes]",
    connected: asyncio.Event,
    finished: asyncio.Event,
    call_loop: list[CallLoop],
    stt: SttWiring,
    on_turn: Callable[[str, str], None] | None,
    opener_text: str | None,
    max_call_s: float,
    scripted_barge_ins: tuple[str, ...],
    scripted_barge_kind: str,
    scripted_barge_voice_id: str | None,
    scripted_barge_skip_speeches: int,
    scripted_response_delays_s: tuple[float, ...],
) -> Callable[[Any], Any]:
    """Build the Media Streams connection handler (wires CallLoop's ports)."""

    async def handler(connection: Any) -> None:
        loop = asyncio.get_running_loop()
        log.info("twilio media stream connected")
        call = CallLoop(
            simulator,
            session,
            stt_send=stt_out.put_nowait,
            send_json=lambda msg: loop.call_soon_threadsafe(twilio_out.put_nowait, msg),
            executor=executor,
            call_later=lambda seconds, fn: loop.call_later(seconds, fn),
            on_turn=on_turn,
            on_stopped=finished.set,
            opener_text=opener_text,
            max_call_s=max_call_s,
            scripted_barge_ins=scripted_barge_ins,
            scripted_barge_kind=scripted_barge_kind,
            scripted_barge_voice_id=scripted_barge_voice_id,
            scripted_barge_skip_speeches=scripted_barge_skip_speeches,
            scripted_response_delays_s=scripted_response_delays_s,
        )
        call_loop.append(call)

        async def stt_session() -> None:
            try:
                async with ws_connect(stt.url, additional_headers=stt.headers) as stt_ws:
                    first = json.loads(await stt_ws.recv())
                    log.info("stt session open (%s)", _url_host(stt.url))
                    if stt.verify_echo is not None:
                        problems = stt.verify_echo(first)
                        if problems:
                            session.log(ERROR, stage="stt.config", problems=problems)
                            log.warning("stt config echo problems: %s", "; ".join(problems))
                    await asyncio.gather(
                        _stt_pump(stt_ws, stt_out, stt.encode),
                        _stt_reader(stt_ws, stt.parse, call),
                    )
            except Exception as exc:
                session.log(ERROR, stage="stt.session", error=repr(exc))
                log.warning("stt session failed: %r", exc)
                call.stop("stt_failed")

        async def recv() -> None:
            async for raw in connection:
                call.on_twilio_message(json.loads(raw))

        recv_task = asyncio.create_task(recv())
        send_task = asyncio.create_task(sender_loop(connection, twilio_out))
        stt_task = asyncio.create_task(stt_session())
        connected.set()
        try:
            await recv_task
        finally:
            log.info("twilio media stream closed")
            call.stop("stream_closed")
            for task in (send_task, stt_task):
                task.cancel()
            await asyncio.gather(send_task, stt_task, return_exceptions=True)
            finished.set()

    return handler


def _url_host(url: str) -> str:
    from urllib.parse import urlparse

    return urlparse(url).netloc


async def sender_loop(connection: Any, out: "asyncio.Queue[str]") -> None:
    """The single writer — every outbound message keeps FIFO order."""
    while True:
        message = await out.get()
        await connection.send(message)


async def _stt_pump(
    stt_ws: Any, out: "asyncio.Queue[bytes]", encode: Callable[[bytes], str | bytes]
) -> None:
    while True:
        chunk = await out.get()
        await stt_ws.send(encode(chunk))


async def _stt_reader(
    stt_ws: Any,
    parse: Callable[[Mapping[str, Any]], SttEvent | None],
    call: CallLoop,
) -> None:
    async for raw in stt_ws:
        event = parse(json.loads(raw))
        if event is not None:
            call.on_stt_event(event)
