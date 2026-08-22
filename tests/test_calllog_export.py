"""Submission export tests: readable transcript plus required compressed audio."""

import json
import wave
from pathlib import Path

from patientqa.calllog.export import (
    export_sessions,
    format_timestamp,
    render_transcript,
    select_campaign_sessions,
)


def _session(root: Path) -> Path:
    folder = root / "call-001_20260820T010203Z"
    folder.mkdir(parents=True)
    transcript = {
        "call_id": "call-001",
        "duration_ms": 3000,
        "end_reason": "stream_stopped",
        "turns": [
            {"role": "agent", "t_ms": 0, "text": "How can I help?"},
            {"role": "patient", "t_ms": 1500, "text": "I need an appointment."},
        ],
    }
    (folder / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    summary = {
        "duration_ms": 3000,
        "end_reason": "stream_stopped",
        "stats": {"agent_turns": 1, "patient_turns": 1},
        "manifest": {
            "persona": {"name": "Marta Reyes", "age": 71},
            "objective": {"type": "schedule_new"},
        },
    }
    (folder / "call.json").write_text(json.dumps(summary), encoding="utf-8")
    with wave.open(str(folder / "recording.wav"), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\0\0\0\0" * 8000)
    return folder


def test_timestamp_and_readable_transcript() -> None:
    assert format_timestamp(83000) == "1:23"
    text = render_transcript(
        {
            "call_id": "call-9",
            "turns": [{"role": "agent", "t_ms": 83000, "text": "Hello"}],
        }
    )
    assert "[1:23] AGENT: Hello" in text


def test_export_writes_mp3_transcript_and_index(tmp_path: Path, monkeypatch) -> None:
    session = _session(tmp_path / "calls")
    monkeypatch.setattr("patientqa.calllog.export.shutil.which", lambda _: None)

    exported = export_sessions([session], tmp_path / "deliverables" / "calls")

    assert exported[0].audio_name == "recording.mp3"
    assert (exported[0].directory / "recording.mp3").stat().st_size > 0
    assert "AGENT: How can I help?" in (
        exported[0].directory / "transcript.txt"
    ).read_text(encoding="utf-8")
    index = (
        tmp_path / "deliverables" / "calls" / "INDEX.md"
    ).read_text(encoding="utf-8")
    assert "Marta Reyes (71)" in index
    assert "[transcript.txt](call-001/transcript.txt)" in index


def test_campaign_selection_excludes_rehearsals_and_short_calls(tmp_path: Path) -> None:
    good = _session(tmp_path)
    summary_path = good / "call.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["duration_ms"] = 90000
    summary["stats"] = {"agent_turns": 5, "patient_turns": 5}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    rehearsal = tmp_path / "basic-test_20260820T010204Z"
    rehearsal.mkdir()
    (rehearsal / "transcript.json").write_text("{}", encoding="utf-8")
    (rehearsal / "call.json").write_text(json.dumps(summary), encoding="utf-8")

    assert select_campaign_sessions(tmp_path) == [good]
