import json
from datetime import datetime, timezone
from pathlib import Path

from patientqa.calllog.__main__ import main as calllog_main
from patientqa.calllog.session import TURN_AGENT, TURN_PATIENT, CallSession
from patientqa.calllog.viewer import build_report, write_viewer


def make_closed_session(tmp_path: Path) -> Path:
    now = datetime(2026, 8, 17, 15, 30, tzinfo=timezone.utc)
    session = CallSession.start(
        tmp_path,
        "call-042",
        {"persona": {"name": "Marta Reyes"}},
        utc_now=lambda: now,
    )
    session.log(TURN_AGENT, text="How can I help?")
    session.log(TURN_PATIENT, text="I need an appointment", respond_ms=812)
    session.append_inbound(b"\xff\x80" * 400)
    session.append_outbound(b"\x80\xff" * 400)
    session.close("objective_achieved")
    return session.directory


def test_write_viewer_copies_template(tmp_path: Path) -> None:
    path = write_viewer(tmp_path / "nested")
    html = path.read_text(encoding="utf-8")
    assert path.name == "viewer.html"
    assert "PatientQA" in html
    assert "<!--PQA_REPORT_DATA-->" in html  # marker intact for report baking


def test_build_report_embeds_session_and_audio(tmp_path: Path) -> None:
    directory = make_closed_session(tmp_path)
    report = build_report([directory])
    html = report.read_text(encoding="utf-8")

    assert report.name == "report.html"
    assert "window.__PQA_REPORT__" in html
    assert "data:audio/wav;base64," in html
    assert "Marta Reyes" in html
    assert "objective_achieved" in html
    assert "<!--PQA_REPORT_DATA-->" not in html  # marker consumed
    assert "setupLoaders" in html  # the full viewer ships inside


def test_build_report_mixes_legs_when_recording_missing(tmp_path: Path) -> None:
    directory = make_closed_session(tmp_path)
    (directory / "recording.wav").unlink()  # simulate a crashed call
    html = build_report([directory]).read_text(encoding="utf-8")

    assert "data:audio/wav;base64," in html


def test_build_report_escapes_script_closing_tags(tmp_path: Path) -> None:
    session = CallSession.start(tmp_path, "call-e", {})
    session.log("note", text="</script><b>injected</b>")
    session.close("completed")
    html = build_report([session.directory]).read_text(encoding="utf-8")

    assert "<\\/script>" in html
    injection_start = html.index("window.__PQA_REPORT__")
    injection_end = html.index("</script>", injection_start)
    assert "</script><b>" not in html[injection_start:injection_end]


def test_cli_viewer_and_report_round_trip(tmp_path: Path) -> None:
    assert calllog_main(["viewer", str(tmp_path), "--no-open"]) == 0
    assert (tmp_path / "viewer.html").is_file()

    directory = make_closed_session(tmp_path / "calls")
    assert calllog_main(["report", str(directory)]) == 0
    assert json.loads(
        (directory / "call.json").read_text(encoding="utf-8")
    )["call_id"] == "call-042"
    assert (directory / "report.html").is_file()


def test_cli_report_rejects_non_session_dirs(tmp_path: Path) -> None:
    assert calllog_main(["report", str(tmp_path)]) == 1


def test_cli_demo_generates_playable_session(tmp_path: Path) -> None:
    assert calllog_main(["demo", "--out", str(tmp_path)]) == 0
    (directory,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert (directory / "session.jsonl").is_file()
    assert (directory / "recording.wav").stat().st_size > 44  # real audio payload
    summary = json.loads((directory / "call.json").read_text(encoding="utf-8"))
    assert summary["end_reason"] == "objective_achieved"
    assert summary["stats"]["patient_turns"] > 0
