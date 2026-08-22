"""Post-call checks and evidence-guarded judge tests."""

import json
from pathlib import Path

from patientqa.analyze import heuristics, llm_judge, postprocess_session


def test_weekend_confirmation_tripwire_is_detected() -> None:
    transcript = {
        "call_id": "call-1",
        "turns": [
            {
                "role": "agent",
                "t_ms": 83000,
                "text": "You're booked for Sunday at ten. See you then.",
            }
        ],
    }
    issues, ours = heuristics(
        transcript,
        {"end_reason": "stream_stopped"},
        {"objective": {"type": "sunday_request"}},
    )
    assert ours == []
    assert len(issues) == 1
    assert issues[0].severity == "high"
    assert issues[0].at_ms == 83000


class _FakeCompletions:
    def __init__(self, payload):
        self.payload = payload

    def create(self, **kwargs):
        message = type("Message", (), {"content": json.dumps(self.payload)})()
        return type("Response", (), {"choices": [type("Choice", (), {"message": message})()]})()


class _FakeClient:
    def __init__(self, payload):
        completions = _FakeCompletions(payload)
        self.chat = type("Chat", (), {"completions": completions})()


def test_judge_drops_issue_without_verbatim_evidence() -> None:
    transcript = {
        "call_id": "call-2",
        "turns": [{"role": "agent", "t_ms": 1000, "text": "Tuesday is available."}],
    }
    payload = {
        "issues": [
            {
                "severity": "high",
                "title": "Invented evidence",
                "evidence": "I booked Sunday.",
                "at": "0:01",
                "why": "bad",
            }
        ]
    }
    assert llm_judge(transcript, {}, client=_FakeClient(payload)) == []


def test_judge_anchors_quote_to_turn_instead_of_trusting_model_time() -> None:
    transcript = {
        "call_id": "call-3",
        "turns": [
            {
                "seq": 17,
                "role": "agent",
                "t_ms": 12345,
                "text": "Your date of birth is July 4th, 2000 for demo purposes.",
            }
        ],
    }
    payload = {
        "issues": [
            {
                "severity": "high",
                "title": "Agent invented a date of birth",
                "evidence": "Your date of birth is July 4th, 2000 for demo purposes.",
                "at": "9:59",
                "why": "The caller supplied a different date.",
            }
        ]
    }

    issues = llm_judge(transcript, {}, client=_FakeClient(payload))

    assert len(issues) == 1
    assert issues[0].at_ms == 12345
    assert issues[0].turn_seq == 17
    assert issues[0].role == "agent"


def test_judge_drops_patient_quote_even_when_it_is_verbatim() -> None:
    transcript = {
        "call_id": "call-4",
        "turns": [
            {"seq": 3, "role": "patient", "t_ms": 5000, "text": "I want Sunday."}
        ],
    }
    payload = {
        "issues": [
            {
                "severity": "high",
                "title": "Wrongly attributes patient behavior",
                "evidence": "I want Sunday.",
                "at": "0:05",
                "why": "This is not an agent issue.",
            }
        ]
    }

    assert llm_judge(transcript, {}, client=_FakeClient(payload)) == []


def test_judge_drops_same_day_claim_contradicted_by_quoted_date() -> None:
    evidence = "I can offer Monday, August 24th at 10:30 a.m."
    transcript = {
        "call_id": "call-4b",
        "created_at": "2026-08-21T00:44:00+00:00",
        "turns": [{"seq": 8, "role": "agent", "t_ms": 9000, "text": evidence}],
    }
    payload = {
        "issues": [
            {
                "severity": "medium",
                "title": "Offered a same-day slot instead of next week",
                "evidence": evidence,
                "at": "0:09",
                "why": "The offered slot is the same day.",
            }
        ]
    }

    issues = llm_judge(
        transcript,
        {"generated_at": "2026-08-20"},
        client=_FakeClient(payload),
    )

    assert issues == []


def test_judge_drops_misinterpretation_claim_immediately_accepted_by_patient() -> None:
    evidence = "Would you like me to reschedule it to Thursday at 4:30 p.m.?"
    transcript = {
        "call_id": "call-4c",
        "turns": [
            {"seq": 8, "role": "agent", "t_ms": 9000, "text": evidence},
            {
                "seq": 9,
                "role": "patient",
                "t_ms": 10000,
                "text": "Yes, please reschedule it to Thursday at four-thirty.",
            },
        ],
    }
    payload = {
        "issues": [
            {
                "severity": "medium",
                "title": "Agent misinterpreted the request",
                "evidence": evidence,
                "at": "0:09",
                "why": "The agent incorrectly assumed the patient wanted a reschedule.",
            }
        ]
    }

    assert llm_judge(transcript, {}, client=_FakeClient(payload)) == []


def test_judge_drops_provider_claim_anchored_to_corrected_name() -> None:
    evidence = "Your appointment is with Dr. Zbigniew Lukowski on Monday."
    transcript = {
        "call_id": "call-4d",
        "turns": [
            {"seq": 8, "role": "agent", "t_ms": 9000, "text": evidence},
            {
                "seq": 9,
                "role": "patient",
                "t_ms": 10000,
                "text": "No, Dr. Zbigniew Lukowski on Monday.",
            },
        ],
    }
    payload = {
        "issues": [
            {
                "severity": "high",
                "title": "Provider name misrecognition",
                "evidence": evidence,
                "at": "0:09",
                "why": "The agent later misspelled the provider name.",
            }
        ]
    }

    assert llm_judge(transcript, {}, client=_FakeClient(payload)) == []


def test_judge_drops_normal_demo_profile_onboarding() -> None:
    evidence = "Would you like to create a demo patient profile?"
    transcript = {
        "call_id": "call-4e",
        "turns": [{"seq": 8, "role": "agent", "t_ms": 9000, "text": evidence}],
    }
    payload = {
        "issues": [
            {
                "severity": "medium",
                "title": "Unnecessary demo patient creation prompt",
                "evidence": evidence,
                "at": "0:09",
                "why": "Creating a demo patient profile may confuse the caller.",
            }
        ]
    }

    assert llm_judge(transcript, {}, client=_FakeClient(payload)) == []


def _write_finalized_session(session_dir: Path) -> None:
    session_dir.mkdir(parents=True)
    manifest = {
        "test_intent": {
            "intentional": True,
            "behavior": "sunday_request",
            "isolation": "single_behavior",
        },
        "persona": {"name": "Marta Reyes"},
        "objective": {
            "type": "sunday_request",
            "adversarial": {"techniques": ["weekend boundary"]},
        },
    }
    transcript = {
        "call_id": "call-5",
        "duration_ms": 9000,
        "end_reason": "stream_closed",
        "turns": [
            {
                "seq": 4,
                "role": "agent",
                "t_ms": 5000,
                "text": "You're booked for Sunday at ten. See you then.",
            }
        ],
    }
    summary = {
        "call_id": "call-5",
        "duration_ms": 9000,
        "end_reason": "stream_closed",
        "test_intent": manifest["test_intent"],
        "manifest": manifest,
    }
    events = [
        {
            "seq": 2,
            "t_ms": 3500,
            "type": "behavior.fired",
            "data": {"behavior": "sunday_request", "intentional": True},
        },
        {"seq": 4, "t_ms": 5000, "type": "turn.agent", "data": transcript["turns"][0]},
        {"seq": 6, "t_ms": 9000, "type": "call.ended", "data": {}},
    ]
    (session_dir / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    (session_dir / "call.json").write_text(json.dumps(summary), encoding="utf-8")
    (session_dir / "session.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (session_dir / "recording.wav").write_bytes(b"RIFF")


def test_postprocess_writes_per_call_and_aggregate_reports(tmp_path: Path) -> None:
    session_dir = tmp_path / "calls" / "call-5_20260820T000000Z"
    _write_finalized_session(session_dir)

    analysis = postprocess_session(session_dir, judge=False)

    payload = json.loads((session_dir / "analysis.json").read_text(encoding="utf-8"))
    report = (session_dir / "analysis.md").read_text(encoding="utf-8")
    aggregate = (session_dir.parent / "ISSUES.md").read_text(encoding="utf-8")
    assert analysis.issues[0].at_ms == 5000
    assert payload["issues"][0]["at_ms"] == 5000
    assert payload["intentional_test_behavior"]["intentional"] is True
    assert payload["intentional_behavior_events"][0]["at_ms"] == 3500
    assert payload["detectors"]["llm_judge"] == "not_requested"
    assert "recording.wav#t=5.000" in report
    assert "intentional: `true`" in report
    assert "call-5_20260820T000000Z/recording.wav#t=5.000" in aggregate


def test_infrastructure_failure_uses_logged_error_moment() -> None:
    transcript = {"call_id": "call-6", "turns": []}
    summary = {"call_id": "call-6", "end_reason": "stt_failed"}
    events = [
        {
            "seq": 3,
            "t_ms": 3210,
            "type": "error",
            "data": {"stage": "stt.session", "error": "insufficient_funds"},
        },
        {"seq": 4, "t_ms": 4000, "type": "call.ended", "data": {}},
    ]

    issues, ours = heuristics(transcript, summary, {}, events)

    assert issues == []
    assert ours[0].at_ms == 3210


def test_dead_air_starts_after_patient_audio_finishes() -> None:
    transcript = {
        "call_id": "call-7",
        "turns": [
            {"seq": 3, "role": "patient", "t_ms": 1000, "text": "A long request."},
            {"seq": 6, "role": "agent", "t_ms": 10000, "text": "I can help."},
        ],
    }
    events = [
        {
            "seq": 2,
            "t_ms": 900,
            "type": "audio.played",
            "data": {"audio_bytes": 48000},
        },
        {"seq": 3, "t_ms": 1000, "type": "turn.patient", "data": {}},
    ]

    issues, _ = heuristics(transcript, {"end_reason": "completed"}, {}, events)

    assert issues == []  # 9s between turn events, but only 3.1s of actual silence


def test_conflicting_agent_date_of_birth_is_deterministic() -> None:
    transcript = {
        "call_id": "call-8",
        "turns": [
            {
                "seq": 9,
                "role": "agent",
                "t_ms": 21000,
                "text": "Your date of birth is July 4th, 2000 for demo purposes.",
            }
        ],
    }
    manifest = {"identity": {"date_of_birth": "1965-09-12"}}

    issues, _ = heuristics(transcript, {"end_reason": "completed"}, manifest)

    assert issues[0].severity == "high"
    assert issues[0].at_ms == 21000
    assert "1965-09-12" in issues[0].why


def test_provider_correction_anchors_only_conflicting_agent_turns() -> None:
    transcript = {
        "call_id": "call-9",
        "turns": [
            {
                "seq": 1,
                "role": "agent",
                "t_ms": 1000,
                "text": "Your visit is with Dr. Zbigniew Lukowski.",
            },
            {
                "seq": 2,
                "role": "agent",
                "t_ms": 2000,
                "text": "Cancel Dr. Zidnu Lukoski on Monday?",
            },
            {
                "seq": 3,
                "role": "patient",
                "t_ms": 3000,
                "text": "No, Dr. Zbigniew Lukowski on Monday.",
            },
            {"seq": 4, "role": "agent", "t_ms": 4000, "text": "Is that correct?"},
            {
                "seq": 5,
                "role": "agent",
                "t_ms": 5000,
                "text": "I will cancel Dr. Zigmew Lukowski.",
            },
        ],
    }

    issues, _ = heuristics(transcript, {"end_reason": "completed"}, {})

    assert [issue.at_ms for issue in issues] == [2000, 5000]
    assert all(issue.severity == "high" for issue in issues)
