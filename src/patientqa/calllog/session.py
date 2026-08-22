"""Per-call session log + recording (DESIGN.md §2 "Recorder", §4 telemetry).

Every call gets one folder under ``calls/`` named ``{call_id}_{UTC-stamp}``
(retries never collide). Inside::

    meta.json        written once at start — call id, manifest entry, versions
    session.jsonl    append-only event log — the source of truth
    debug.log        stdlib-logging capture of the machinery (logsetup.attach_call_log)
    audio/inbound.ulaw    agent leg, byte-exact 8 kHz μ-law exactly as received
    audio/outbound.ulaw   patient leg, byte-exact exactly as sent
    recording.wav    on close(): mono PCM16 8 kHz — both legs placed at their
                     true positions on the session timeline (from the
                     audio.inbound anchor and audio.played events) and summed,
                     so each side is audible in both ears and in sync
    transcript.json  on close(): the dialogue (turn.agent / turn.patient events)
    call.json        on close(): outcome + latency stats

Event lines are ``{"seq", "t_ms", "wall", "type", "data"}`` where ``t_ms`` is
milliseconds since session start on a monotonic clock. μ-law is exactly
8,000 bytes/s and the finalized WAV is padded to the session duration, so an
event's ``t_ms`` maps 1:1 onto position in ``recording.wav`` — the viewer uses
that to sync transcript, timeline and audio. If the process dies mid-call,
everything appended so far survives: that is the whole point of the format.
"""

from __future__ import annotations

import io
import json
import re
import time
import wave
from array import array
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from patientqa import __version__
from patientqa.calllog.ulaw import ulaw_bytes_to_pcm

FORMAT_VERSION = 1
SAMPLE_RATE = 8000  # μ-law bytes per second, and PCM samples per second
BYTES_PER_MS = SAMPLE_RATE // 1000

# Event vocabulary. turn.* events form the transcript; the rest is telemetry.
CALL_STARTED = "call.started"
CALL_CONNECTED = "call.connected"
CALL_ENDED = "call.ended"
TURN_AGENT = "turn.agent"  # {text} — agent utterance (STT final)
TURN_PATIENT = "turn.patient"  # {text, respond_ms} — spoken patient reply
BRAIN_REPLY = "brain.reply"  # {say, latency_ms} — Cerebras output
TTS_DONE = "tts.done"  # {chars, audio_bytes, latency_ms} — ElevenLabs output
STT_FINAL = "stt.final"
AUDIO_INBOUND = "audio.inbound"  # {} — first agent-leg frame; anchors that leg
AUDIO_PLAYED = "audio.played"  # {audio_bytes} — one patient-leg run, in order
BARGE_IN = "barge_in"
BEHAVIOR_FIRED = "behavior.fired"
ERROR = "error"
NOTE = "note"

TRANSCRIPT_EVENTS = (TURN_AGENT, TURN_PATIENT)

_sanitizer = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_call_id(call_id: str) -> str:
    """Make a manifest call_id safe (and boring) as a folder-name fragment."""
    name = _sanitizer.sub("-", call_id).strip("-.")
    return name or "call"


def _utc_stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _event_fields(event: Any) -> tuple[str, int, dict[str, Any]]:
    """Event objects (close) and plain dicts (read_jsonl) → (type, t_ms, data)."""
    if isinstance(event, Mapping):
        return str(event.get("type")), int(event.get("t_ms", 0)), dict(event.get("data") or {})
    return event.event_type, event.t_ms, dict(event.data)


def leg_placements(events: Any) -> tuple[int, list[tuple[int, int]]]:
    """Where each leg's audio sits on the session timeline, from the event log.

    Returns ``(inbound_offset_ms, [(t_ms, bytes), …])``. The agent leg is one
    continuous run from its first frame — Twilio Media Streams deliver frames
    in order, silence included, so byte position maps linearly from that anchor
    (:const:`AUDIO_INBOUND`; sessions predate it fall back to the
    ``call.connected`` event, which precedes the first frame by ~one frame).
    The patient leg maps 1:1 onto :const:`AUDIO_PLAYED` events: the
    orchestrator logs one per ``append_outbound``, on the single engage
    thread, immediately after the bytes land.
    """
    exact: int | None = None
    connected: int | None = None
    segments: list[tuple[int, int]] = []
    for event in events:
        kind, t_ms, data = _event_fields(event)
        if kind == AUDIO_INBOUND and exact is None:
            exact = t_ms
        elif kind == CALL_CONNECTED and connected is None:
            connected = t_ms
        elif kind == AUDIO_PLAYED:
            segments.append((t_ms, int(data.get("audio_bytes", 0))))
    offset = exact if exact is not None else (connected if connected is not None else 0)
    return offset, segments


def mixdown_wav(
    inbound: bytes | None,
    outbound: bytes | None,
    duration_ms: int,
    *,
    inbound_offset_ms: int = 0,
    outbound_segments: list[tuple[int, int]] | None = None,
) -> bytes:
    """Mux both μ-law legs into one mono PCM16 8 kHz WAV, placed in time.

    The agent leg starts at ``inbound_offset_ms``; the patient leg's runs
    (``outbound_segments`` = :func:`leg_placements`) start at their event
    times but never before the previous run has drained in real time — the
    single playback queue serializes them. Both legs are summed into the one
    channel (audible in both ears, in conversation order); genuine cross-talk
    (barge-ins) sums with saturation. ``outbound_segments=None`` packs the
    patient leg at position 0 (callers with no timing). The result is padded
    to ``duration_ms``, so an event's ``t_ms`` still maps 1:1 onto WAV
    position.
    """
    placements: list[tuple[int, array]] = []  # (start sample, run) on the timeline
    if inbound:
        placements.append((inbound_offset_ms * BYTES_PER_MS, ulaw_bytes_to_pcm(inbound)))
    frames = duration_ms * BYTES_PER_MS
    if outbound:
        out_pcm = ulaw_bytes_to_pcm(outbound)
        if outbound_segments is None:
            placements.append((0, out_pcm))
        else:
            # The patient leg plays through one FIFO queue, so a run cannot
            # start before its predecessor drains: synthesis outruns real
            # time (the next chunk's event fires mid-playback of the last),
            # and placing runs at raw event times would stack two responses
            # on top of each other. Pack sequentially, floor at event time.
            cursor = 0
            played_until = 0
            for t_ms, size in outbound_segments:
                if size <= 0 or cursor >= len(out_pcm):
                    continue
                start = max(t_ms * BYTES_PER_MS, played_until)
                placements.append((start, out_pcm[cursor : cursor + size]))
                cursor += size
                played_until = start + size
            if cursor < len(out_pcm):
                # bytes beyond the last event (crash mid-append): keep them
                # contiguous with the last placed run instead of dropping them
                placements.append((played_until, out_pcm[cursor:]))
    for start, pcm in placements:
        frames = max(frames, start + len(pcm))

    mixed = array("h", bytes(2 * frames))
    for start, pcm in placements:
        for i, sample in enumerate(pcm):
            value = mixed[start + i] + sample
            if value > 32767:
                value = 32767
            elif value < -32768:
                value = -32768
            mixed[start + i] = value

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(mixed.tobytes())
    return buffer.getvalue()


@dataclass(frozen=True)
class Event:
    """One logged moment: ``seq`` order, ``t_ms`` timeline position, payload."""

    seq: int
    t_ms: int
    wall: str
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "t_ms": self.t_ms,
                "wall": self.wall,
                "type": self.event_type,
                "data": self.data,
            },
            separators=(",", ":"),
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a session.jsonl into event dicts, skipping unparsable lines."""
    events: list[dict[str, Any]] = []
    if not path.is_file():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


class CallSession:
    """One call's on-disk record; safe against a crash at any instant.

    Use :meth:`start` to create a fresh session folder, :meth:`log` for every
    notable moment, :meth:`append_inbound`/:meth:`append_outbound` for audio
    frames as they arrive/leave, and :meth:`close` to finalize the recording,
    transcript and summary. Also a context manager — an un-closed session is
    finalized with reason ``"abandoned"``.
    """

    def __init__(
        self,
        directory: Path,
        *,
        call_id: str | None = None,
        perf_counter: Callable[[], float] = time.perf_counter,
        utc_now: Callable[[], datetime] | None = None,
    ) -> None:
        self._dir = Path(directory)
        self._call_id = call_id
        self._mono = perf_counter or time.perf_counter
        self._t0 = self._mono()
        self._now = utc_now or (lambda: datetime.now(timezone.utc))
        self._seq = 0
        self._events: list[Event] = []
        self._in_bytes = 0
        self._out_bytes = 0
        self._inbound_anchored = False
        self._closed = False

    # -- construction ----------------------------------------------------

    @classmethod
    def start(
        cls,
        calls_root: Path,
        call_id: str,
        manifest: dict[str, Any] | None = None,
        *,
        utc_now: Callable[[], datetime] | None = None,
        perf_counter: Callable[[], float] | None = None,
    ) -> CallSession:
        """Create ``calls_root/{call_id}_{stamp}/`` with meta + empty legs."""
        now = (utc_now or (lambda: datetime.now(timezone.utc)))()
        base = f"{sanitize_call_id(call_id)}_{_utc_stamp(now)}"
        folder = Path(calls_root) / base
        suffix = 2
        while folder.exists():
            folder = Path(calls_root) / f"{base}-{suffix}"
            suffix += 1
        (folder / "audio").mkdir(parents=True)
        session = cls(folder, call_id=call_id, utc_now=utc_now, perf_counter=perf_counter)
        meta = {
            "format_version": FORMAT_VERSION,
            "call_id": call_id,
            "created_at": session._now().isoformat(timespec="milliseconds"),
            "patientqa_version": __version__,
            "test_intent": (manifest or {}).get("test_intent"),
            "manifest": manifest or {},
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        session.session_path.write_text("", encoding="utf-8")
        session.inbound_path.write_bytes(b"")
        session.outbound_path.write_bytes(b"")
        session.log(CALL_STARTED, manifest=manifest or {})
        return session

    # -- paths -----------------------------------------------------------

    @property
    def directory(self) -> Path:
        return self._dir

    @property
    def call_id(self) -> str:
        if self._call_id is not None:
            return self._call_id
        return self._dir.name

    @property
    def session_path(self) -> Path:
        return self._dir / "session.jsonl"

    @property
    def inbound_path(self) -> Path:
        return self._dir / "audio" / "inbound.ulaw"

    @property
    def outbound_path(self) -> Path:
        return self._dir / "audio" / "outbound.ulaw"

    @property
    def recording_path(self) -> Path:
        return self._dir / "recording.wav"

    @property
    def transcript_path(self) -> Path:
        return self._dir / "transcript.json"

    @property
    def summary_path(self) -> Path:
        return self._dir / "call.json"

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def duration_ms(self) -> int:
        return self._events[-1].t_ms if self._events else 0

    # -- logging ---------------------------------------------------------

    def log(self, event_type: str, **data: Any) -> Event:
        """Append one event; ``data`` fields become the event's payload."""
        event = Event(
            seq=self._seq,
            t_ms=round((self._mono() - self._t0) * 1000),
            wall=self._now().isoformat(timespec="milliseconds"),
            event_type=event_type,
            data=data,
        )
        self._seq += 1
        with self.session_path.open("a", encoding="utf-8") as fh:
            fh.write(event.to_json_line() + "\n")
        self._events.append(event)
        return event

    def note(self, text: str) -> Event:
        """A human-readable breadcrumb — useful during debugging."""
        return self.log(NOTE, text=text)

    # -- audio capture ---------------------------------------------------

    def append_inbound(self, ulaw: bytes) -> int:
        """Agent-leg frames exactly as Twilio delivered them; returns total bytes."""
        if not self._inbound_anchored:
            self._inbound_anchored = True
            self.log(AUDIO_INBOUND)  # t_ms anchors this leg on the timeline
        with self.inbound_path.open("ab") as fh:
            fh.write(ulaw)
        self._in_bytes += len(ulaw)
        return self._in_bytes

    def append_outbound(self, ulaw: bytes) -> int:
        """Patient-leg frames exactly as we sent them; returns total bytes."""
        with self.outbound_path.open("ab") as fh:
            fh.write(ulaw)
        self._out_bytes += len(ulaw)
        return self._out_bytes

    # -- finalization ----------------------------------------------------

    def close(self, reason: str = "completed", **detail: Any) -> Path:
        """Log ``call.ended`` and write recording.wav, transcript.json, call.json.

        Idempotent: the first call wins, later calls are no-ops.
        """
        if self._closed:
            return self._dir
        end = self.log(CALL_ENDED, reason=reason, **detail)
        self._closed = True
        duration_ms = max(
            end.t_ms, self._in_bytes // BYTES_PER_MS, self._out_bytes // BYTES_PER_MS
        )
        inbound_offset_ms, outbound_segments = leg_placements(self._events)
        self.recording_path.write_bytes(
            mixdown_wav(
                self.inbound_path.read_bytes() if self.inbound_path.is_file() else b"",
                self.outbound_path.read_bytes() if self.outbound_path.is_file() else b"",
                duration_ms,
                inbound_offset_ms=inbound_offset_ms,
                outbound_segments=outbound_segments,
            )
        )
        self._write_transcript(duration_ms, reason)
        self._write_summary(duration_ms, reason)
        return self._dir

    def _turn_rows(self) -> list[dict[str, Any]]:
        rows = []
        for event in self._events:
            if event.event_type in TRANSCRIPT_EVENTS:
                row: dict[str, Any] = {
                    "seq": event.seq,
                    "t_ms": event.t_ms,
                    "role": "agent" if event.event_type == TURN_AGENT else "patient",
                    "text": event.data.get("text", ""),
                }
                if event.event_type == TURN_PATIENT and "respond_ms" in event.data:
                    row["respond_ms"] = event.data["respond_ms"]
                rows.append(row)
        return rows

    def _write_transcript(self, duration_ms: int, reason: str) -> None:
        document = {
            "call_id": self.call_id,
            "created_at": self._events[0].wall if self._events else None,
            "duration_ms": duration_ms,
            "end_reason": reason,
            "turns": self._turn_rows(),
        }
        self.transcript_path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _write_summary(self, duration_ms: int, reason: str) -> None:
        replies = [
            e.data["respond_ms"]
            for e in self._events
            if e.event_type == TURN_PATIENT and "respond_ms" in e.data
        ]
        manifest: dict[str, Any] = {}
        for event in self._events:
            if event.event_type == CALL_STARTED and event.data.get("manifest"):
                manifest = event.data["manifest"]
                break
        meta_path = self._dir / "meta.json"
        if meta_path.is_file():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                manifest = loaded.get("manifest") or manifest
            except json.JSONDecodeError:
                pass
        summary = {
            "call_id": self.call_id,
            "directory": self._dir.name,
            "created_at": self._events[0].wall if self._events else None,
            "ended_at": self._events[-1].wall if self._events else None,
            "duration_ms": duration_ms,
            "end_reason": reason,
            "test_intent": manifest.get("test_intent"),
            "stats": {
                "events": len(self._events),
                "agent_turns": sum(
                    1 for e in self._events if e.event_type == TURN_AGENT
                ),
                "patient_turns": sum(
                    1 for e in self._events if e.event_type == TURN_PATIENT
                ),
                "respond_ms_avg": round(sum(replies) / len(replies)) if replies else None,
                "respond_ms_max": max(replies) if replies else None,
                "inbound_bytes": self._in_bytes,
                "outbound_bytes": self._out_bytes,
            },
            "manifest": manifest,
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # -- context manager -------------------------------------------------

    def __enter__(self) -> CallSession:
        return self

    def __exit__(self, *exc_info: object) -> None:
        reason = "completed" if exc_info[0] is None else "abandoned"
        self.close(reason)
