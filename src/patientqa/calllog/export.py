"""Export session folders as submission deliverables (challenge requirement).

Each exported call becomes one folder under the output root:

    deliverables/calls/call-014/
    ├── recording.mp3      (or .ogg when a system ffmpeg exists)
    └── transcript.txt     both sides, [m:ss] stamps that match the audio

plus a root ``INDEX.md`` table of every exported call (persona, objective,
duration, turns, outcome — the reviewer's map). The challenge requires OGG or
MP3 audio next to the transcript; MP3 comes from :mod:`lameenc` (no external
binary), OGG from ffmpeg when it is on PATH — whichever exists, MP3 always.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from patientqa.calllog.session import SAMPLE_RATE

_MP3_KBPS = 64  # plenty for 8 kHz mono speech
_CAMPAIGN_ID = re.compile(r"^call-\d{3}$")


@dataclass
class ExportedCall:
    call_id: str
    directory: Path
    audio_name: str
    transcript_name: str

    @property
    def label(self) -> str:
        return self.directory.name


def format_timestamp(t_ms: int) -> str:
    """Session-relative position in the recording, as ``m:ss``."""
    return f"{t_ms // 60000}:{t_ms // 1000 % 60:02d}"


def render_transcript(transcript: dict) -> str:
    """The readable both-sides transcript, stamped to match the audio."""
    header = [
        f"Call: {transcript.get('call_id', '?')}",
        f"Duration: {transcript.get('duration_ms', 0) / 1000:.0f}s · "
        f"Ended: {transcript.get('end_reason', '?')}",
        "",
    ]
    lines: list[str] = []
    for turn in transcript.get("turns", []):
        stamp = format_timestamp(int(turn.get("t_ms", 0)))
        role = "AGENT" if turn.get("role") == "agent" else "PATIENT"
        text = str(turn.get("text", "")).strip()
        lines.append(f"[{stamp}] {role}: {text}")
    return "\n".join(header + lines) + "\n"


def encode_mp3(wav_bytes: bytes) -> bytes:
    """PCM16 WAV bytes → MP3 frames (mono, 64 kbps — speech-grade)."""
    import io

    import lameenc

    with wave.open(io.BytesIO(wav_bytes)) as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"expected 16-bit PCM, got {width * 8}-bit")
    if channels == 2:  # average down to mono
        mono = bytearray()
        for i in range(0, len(frames), 4):
            left = int.from_bytes(frames[i : i + 2], "little", signed=True)
            right = int.from_bytes(frames[i + 2 : i + 4], "little", signed=True)
            mono += int((left + right) / 2).to_bytes(2, "little", signed=True)
        frames = bytes(mono)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(_MP3_KBPS * 1000)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    return bytes(encoder.encode(frames)) + bytes(encoder.flush())


def encode_ogg(wav_path: Path, ffmpeg: str) -> bytes:
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(wav_path),
            "-c:a",
            "libopus",
            "-f",
            "ogg",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return result.stdout


def export_session(session_dir: Path, out_root: Path, *, ffmpeg: str | None = None) -> ExportedCall:
    """One session folder → one deliverable folder (audio + transcript)."""
    transcript_doc = json.loads(
        (session_dir / "transcript.json").read_text(encoding="utf-8")
    )
    call_id = str(transcript_doc.get("call_id") or session_dir.name)
    target = out_root / call_id
    target.mkdir(parents=True, exist_ok=True)

    audio_name = "recording.mp3"
    if ffmpeg is not None:
        audio_name = "recording.ogg"
        (target / audio_name).write_bytes(
            encode_ogg(session_dir / "recording.wav", ffmpeg)
        )
    else:
        (target / audio_name).write_bytes(
            encode_mp3((session_dir / "recording.wav").read_bytes())
        )
    transcript_name = "transcript.txt"
    (target / transcript_name).write_text(
        render_transcript(transcript_doc), encoding="utf-8"
    )
    return ExportedCall(
        call_id=call_id,
        directory=target,
        audio_name=audio_name,
        transcript_name=transcript_name,
    )


def render_index(sessions: list[tuple[Path, ExportedCall]]) -> str:
    """The reviewer's map: one row per exported call."""
    rows = [
        "| Call | Persona | Objective | Adversarial techniques | Duration | "
        "Turns (agent/patient) | Ended | Transcript | Audio |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for session_dir, exported in sessions:
        summary = json.loads((session_dir / "call.json").read_text(encoding="utf-8"))
        manifest = summary.get("manifest", {})
        persona = manifest.get("persona", {})
        objective = manifest.get("objective", {})
        techniques = ", ".join(
            objective.get("adversarial", {}).get("techniques", [])
        ) or "unclassified"
        stats = summary.get("stats", {})
        name = persona.get("name", "?")
        if persona.get("age"):
            name = f"{name} ({persona['age']})"
        duration = f"{summary.get('duration_ms', 0) / 1000:.0f}s"
        turns = f"{stats.get('agent_turns', '?')}/{stats.get('patient_turns', '?')}"
        ended = summary.get("end_reason", "?")
        transcript = (
            f"[{exported.transcript_name}]({exported.label}/{exported.transcript_name})"
        )
        audio = f"[{exported.audio_name}]({exported.label}/{exported.audio_name})"
        rows.append(
            f"| {exported.call_id} | {name} | {objective.get('type', '?')} "
            f"| {techniques} | {duration} | {turns} | {ended} | {transcript} | {audio} |"
        )
    return "\n".join(rows) + "\n"


def export_sessions(session_dirs: list[Path], out_root: Path) -> list[ExportedCall]:
    """Export each session and write ``INDEX.md``; returns the exports in order."""
    ffmpeg = shutil.which("ffmpeg")
    out_root.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, ExportedCall]] = []
    exported: list[ExportedCall] = []
    for session_dir in session_dirs:
        one = export_session(session_dir, out_root, ffmpeg=ffmpeg)
        pairs.append((session_dir, one))
        exported.append(one)
    (out_root / "INDEX.md").write_text(render_index(pairs), encoding="utf-8")
    return exported


def select_campaign_sessions(calls_root: Path) -> list[Path]:
    """Latest submission-quality session per ``call-NNN`` under a calls root."""
    from patientqa.campaign import CallOutcome

    selected: dict[str, Path] = {}
    if not calls_root.is_dir():
        return []
    for folder in sorted(calls_root.iterdir()):
        if not folder.is_dir() or not (folder / "transcript.json").is_file():
            continue
        call_id = folder.name.rsplit("_", 1)[0]
        if not _CAMPAIGN_ID.fullmatch(call_id):
            continue
        try:
            summary = json.loads((folder / "call.json").read_text(encoding="utf-8"))
            outcome = CallOutcome.from_session(call_id, folder, voice="")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if summary.get("manifest", {}).get("objective") and outcome.good:
            selected[call_id] = folder
    return [selected[call_id] for call_id in sorted(selected)]
