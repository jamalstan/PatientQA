"""Synthetic demo session — a full call record with zero network and zero cost.

Useful to try the viewer before the first real call is placed:

    uv run python -m patientqa.calllog demo --out calls
    uv run python -m patientqa.calllog viewer calls

The dialogue is the DESIGN.md §6.3 walkthrough persona; audio is tone-burst
μ-law synthesized per turn (agent 220 Hz, patient 410 Hz) so the viewer's
waveform, transcript sync and timeline all have something real to show.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from patientqa.calllog.session import (
    AUDIO_PLAYED,
    BARGE_IN,
    BRAIN_REPLY,
    CALL_CONNECTED,
    SAMPLE_RATE,
    TTS_DONE,
    TURN_AGENT,
    TURN_PATIENT,
    CallSession,
)
from patientqa.calllog.ulaw import pcm16_to_ulaw_bytes

MANIFEST: dict[str, Any] = {
    "call_id": "call-demo",
    "persona": {
        "name": "Marta Reyes",
        "age": 71,
        "gender": "female",
        "language": "English w/ Spanish code-switching",
        "voice": {
            "provider": "elevenlabs",
            "design_prompt": (
                "elderly Cuban-American woman, warm, slightly deaf, talks around the point"
            ),
            "voice_id": None,
        },
        "background": (
            "Type 2 diabetes + hypertension, sees Dr. Ortiz quarterly, "
            "prefers Tuesday mornings, daughter usually drives her"
        ),
        "speaking_style": "meanders, asks agent to repeat numbers, "
        "occasionally switches to Spanish for drug names",
        "medications": ["metformin", "lisinopril"],
    },
    "objective": {
        "type": "reschedule_edge",
        "goal": "Move next week's endocrinology visit; does not remember exact date",
        "hidden_context": "Will only mention she 'saw the doctor recently' if asked directly",
        "curveballs": [
            {"at": "after agent proposes slot", "action": "counter with a Sunday request"},
            {"at": "confirmation stage", "action": "ask them to repeat the time twice"},
        ],
        "success_criteria": [
            "no Sunday confirmation",
            "existing appointment correctly located",
        ],
        "termination": "goal achieved OR agent fails twice OR 180s",
    },
}

_DIALOGUE = [
    # (agent utterance, patient reply, brain_ms, tts_ms)
    (
        "Good morning, Pretty Good AI Medical, this is Dana — how can I help you today?",
        "Ay, hello, yes — I need to move my doctor appointment, the one next week.",
        214,
        132,
    ),
    (
        "Of course. Can I have your name and date of birth, please?",
        "Marta Reyes. Eh… the birthday is March fourteenth, nineteen fifty-four.",
        187,
        141,
    ),
    (
        "Thank you, Marta. I see an endocrinology visit with Dr. Ortiz on Tuesday "
        "at 10:30 AM. Would you like a different day?",
        "Yes, yes — Tuesday is no good, my daughter works. Do you have Sunday?",
        231,
        158,
    ),
    (
        "Our clinic is closed on Sundays, I'm sorry. The nearest options are "
        "Thursday at 9:00 AM or Friday at 2:15 PM.",
        "Thursday… nine in the morning, that is good. Wait, nine or nine thirty? "
        "Say it again for me.",
        203,
        149,
    ),
    (
        "Thursday at nine in the morning — nine zero zero — with Dr. Ortiz.",
        "Perfect, Thursday at nine. Gracias, mija.",
        176,
        121,
    ),
    (
        "You're all set, Marta. Anything else I can help with?",
        "No, that is all. You were very patient with me. Adiós.",
        165,
        108,
    ),
]


def _tone_ulaw(
    seconds: float, freq: float, volume: float = 0.28, flutter: float = 0.3
) -> bytes:
    """A speech-ish tone burst: syllable-rate amplitude flutter keeps it visible."""
    samples = []
    for n in range(int(seconds * SAMPLE_RATE)):
        t = n / SAMPLE_RATE
        envelope = 0.55 + 0.45 * math.sin(2 * math.pi * 3.1 * t) ** 2
        wobble = 1.0 + flutter * 0.02 * math.sin(2 * math.pi * 0.7 * t)
        samples.append(
            int(volume * envelope * 32767 * math.sin(2 * math.pi * freq * wobble * t))
        )
    # 30 ms in/out ramps so bursts don't click
    ramp = int(0.03 * SAMPLE_RATE)
    for i in range(ramp):
        samples[i] = int(samples[i] * i / ramp)
        samples[-1 - i] = int(samples[-1 - i] * i / ramp)
    return pcm16_to_ulaw_bytes(samples)


def _fake_clocks() -> tuple[Callable[[], float], Callable[[], datetime], Callable[[float], None]]:
    """Explicitly-driven clocks so the demo timeline is deterministic."""
    mono = [1717.0]
    wall = [datetime(2026, 8, 17, 15, 30, 1, tzinfo=timezone.utc)]

    def perf_counter() -> float:
        return mono[0]

    def utc_now() -> datetime:
        return wall[0]

    def advance(seconds: float) -> None:
        mono[0] += seconds
        wall[0] += timedelta(seconds=seconds)

    return perf_counter, utc_now, advance


def generate_demo_session(calls_root: Path) -> Path:
    """Create and finalize one demo call folder; returns its directory."""
    perf_counter, utc_now, advance = _fake_clocks()
    session = CallSession.start(
        calls_root, "call-demo", MANIFEST, utc_now=utc_now, perf_counter=perf_counter
    )
    advance(2.1)  # Twilio dial + ring

    session.log(CALL_CONNECTED, call_sid="CA" + "0" * 32, streams_sid="MS" + "0" * 32)
    session.note("demo session — synthesized dialogue and tone audio, no live call")

    for i, (agent_text, reply, brain_ms, tts_ms) in enumerate(_DIALOGUE):
        advance(1.2)  # thinking pause before the agent speaks
        session.append_inbound(_tone_ulaw(1.9, 220.0))
        session.log(TURN_AGENT, text=agent_text)
        advance(1.9)

        if i == 3:
            session.log(BARGE_IN, note="patient started before agent finished")
            advance(0.15)

        session.log(BRAIN_REPLY, say=reply, latency_ms=brain_ms)
        advance(brain_ms / 1000)
        reply_audio = _tone_ulaw(1.5 + (i % 3) * 0.2, 410.0 + i * 12)
        session.append_outbound(reply_audio)
        session.log(AUDIO_PLAYED, audio_bytes=len(reply_audio))
        session.log(
            TTS_DONE, chars=len(reply), audio_bytes=len(reply_audio), latency_ms=tts_ms
        )
        session.log(TURN_PATIENT, text=reply, respond_ms=brain_ms + tts_ms)
        advance(1.5 + (i % 3) * 0.2)

    advance(1.0)
    session.close("objective_achieved")
    return session.directory
