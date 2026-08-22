"""Viewer + report tooling around the static call viewer (``viewer.html``).

The viewer is one dependency-free dark-mode HTML file that runs from
``file://`` and loads call folders via drag-and-drop or a folder picker —
everything is parsed locally. Two helpers ship it:

- :func:`write_viewer` copies the template next to the sessions (drop the
  ``calls/`` folder onto it and browse every call).
- :func:`build_report` bakes sessions into a *self-contained* ``report.html``
  (events + recording embedded as a data URL) that renders identically with
  zero local files — the shareable/submission artifact.
"""

from __future__ import annotations

import base64
import json
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any

from patientqa.calllog.session import leg_placements, mixdown_wav, read_jsonl

VIEWER_FILENAME = "viewer.html"
# replaced by the embedded-data <script> when baking a report
REPORT_MARKER = "<!--PQA_REPORT_DATA-->"


def viewer_html() -> str:
    """The viewer template shipped inside the package."""
    return resource_files("patientqa.calllog").joinpath(VIEWER_FILENAME).read_text(
        encoding="utf-8"
    )


def write_viewer(dest_dir: Path) -> Path:
    """Copy the viewer to ``dest_dir/viewer.html``; returns its path."""
    dest = Path(dest_dir) / VIEWER_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(viewer_html(), encoding="utf-8")
    return dest


def load_session_payload(session_dir: Path) -> dict[str, Any]:
    """One session folder → the viewer's embedded-report payload.

    Audio preference: finalized ``recording.wav``, else a mixdown of the raw
    μ-law legs (crashed/in-progress sessions still get playable audio).
    """
    session_dir = Path(session_dir)
    meta: dict[str, Any] = {}
    meta_path = session_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}

    audio: str | None = None
    wav_path = session_dir / "recording.wav"
    legs = [session_dir / "audio" / f"{leg}.ulaw" for leg in ("inbound", "outbound")]
    events = read_jsonl(session_dir / "session.jsonl")
    if wav_path.is_file():
        audio = wav_data_url(wav_path.read_bytes())
    elif any(leg.is_file() for leg in legs):
        duration_ms = max((e.get("t_ms", 0) for e in events), default=0)
        inbound_offset_ms, outbound_segments = leg_placements(events)
        wav = mixdown_wav(
            legs[0].read_bytes() if legs[0].is_file() else b"",
            legs[1].read_bytes() if legs[1].is_file() else b"",
            duration_ms,
            inbound_offset_ms=inbound_offset_ms,
            outbound_segments=outbound_segments,
        )
        audio = wav_data_url(wav)

    return {
        "name": session_dir.name,
        "meta": meta,
        "events": events,
        "audio_data_url": audio,
    }


def wav_data_url(wav: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")


def build_report(session_dirs: list[Path], out: Path | None = None) -> Path:
    """Bake one or more session folders into a single self-contained HTML report."""
    if not session_dirs:
        raise ValueError("build_report needs at least one session directory")
    payloads = [load_session_payload(Path(d)) for d in session_dirs]
    payload_json = json.dumps({"sessions": payloads}, ensure_ascii=False)
    # keep any "</script>" inside strings from terminating the injected tag
    safe_json = payload_json.replace("</", "<\\/")
    html = viewer_html().replace(
        REPORT_MARKER, f"<script>window.__PQA_REPORT__ = {safe_json};</script>"
    )
    out = Path(out) if out is not None else Path(session_dirs[0]) / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
