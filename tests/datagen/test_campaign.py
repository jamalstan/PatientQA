"""Manifest-driven campaign planning, resume, and safe dial wiring."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from factories import valid_entry, valid_starter_set

from patientqa.campaign import completed_call_ids, load_plans, run_campaign


def _write_jsonl(path: Path, models: list[object]) -> None:
    path.write_text(
        "".join(model.model_dump_json() + "\n" for model in models),
        encoding="utf-8",
    )


def test_load_plans_joins_starters_and_is_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    starters = tmp_path / "manifest.starters.jsonl"
    _write_jsonl(manifest, [valid_entry()])
    _write_jsonl(starters, [valid_starter_set()])

    first = load_plans(manifest, starters)
    second = load_plans(manifest, starters)

    options = {starter["text"] for starter in valid_starter_set().model_dump()["starters"]}
    assert first[0].opener in options
    assert first[0].opener == second[0].opener
    assert "Marta Reyes" in first[0].persona_prompt


def test_completed_call_ids_preserves_numeric_suffix(tmp_path: Path) -> None:
    complete = tmp_path / "call-001_20260820T010203Z"
    complete.mkdir()
    (complete / "transcript.json").write_text("{}", encoding="utf-8")
    (complete / "call.json").write_text(
        json.dumps(
            {
                "duration_ms": 120000,
                "end_reason": "stream_stopped",
                "stats": {"agent_turns": 6, "patient_turns": 6},
            }
        ),
        encoding="utf-8",
    )
    incomplete = tmp_path / "call-002_20260820T010204Z"
    incomplete.mkdir()

    assert completed_call_ids(tmp_path) == {"call-001"}


def test_short_call_does_not_satisfy_resume_gate(tmp_path: Path) -> None:
    folder = tmp_path / "call-001_20260820T010203Z"
    folder.mkdir()
    (folder / "transcript.json").write_text("{}", encoding="utf-8")
    (folder / "call.json").write_text(
        json.dumps(
            {
                "duration_ms": 20000,
                "end_reason": "stream_stopped",
                "stats": {"agent_turns": 1, "patient_turns": 1},
            }
        ),
        encoding="utf-8",
    )
    assert completed_call_ids(tmp_path) == set()


def test_resume_requires_the_same_manifest_seed(tmp_path: Path) -> None:
    folder = tmp_path / "call-001_20260820T010203Z"
    folder.mkdir()
    (folder / "transcript.json").write_text("{}", encoding="utf-8")
    (folder / "call.json").write_text(
        json.dumps(
            {
                "duration_ms": 120000,
                "end_reason": "stream_stopped",
                "stats": {"agent_turns": 6, "patient_turns": 6},
                "manifest": {"seed": 42},
            }
        ),
        encoding="utf-8",
    )
    assert completed_call_ids(tmp_path, expected_seeds={"call-001": 42}) == {"call-001"}
    assert completed_call_ids(tmp_path, expected_seeds={"call-001": 43}) == set()


def test_campaign_dry_run_never_dials(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [valid_entry()])
    plans = load_plans(manifest)

    async def forbidden(**kwargs):
        raise AssertionError("dry run attempted a dial")

    monkeypatch.setattr("patientqa.callloop.run_call", forbidden)
    report = asyncio.run(
        run_campaign(
            plans,
            manifest_path=manifest,
            calls_root=tmp_path / "calls",
            stream_url="wss://example.invalid",
            dry_run=True,
        )
    )
    assert report.outcomes == []


def test_campaign_rejects_more_than_five_minutes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [valid_entry()])
    with pytest.raises(ValueError, match="300"):
        asyncio.run(
            run_campaign(
                load_plans(manifest),
                manifest_path=manifest,
                dry_run=True,
                max_call_s=300.1,
            )
        )


def test_campaign_dials_default_target_and_records_outcome(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.jsonl"
    entry = valid_entry(objective={"type": "barge_in"})
    _write_jsonl(manifest, [entry])
    plans = load_plans(manifest)
    seen: dict = {}

    async def fake_run_call(**kwargs):
        seen.update(kwargs)
        folder = kwargs["calls_root"] / "call-014_20260820T010203Z"
        folder.mkdir(parents=True)
        summary = {
            "duration_ms": 179000,
            "end_reason": "stream_stopped",
            "stats": {"agent_turns": 9, "patient_turns": 9},
        }
        (folder / "call.json").write_text(json.dumps(summary), encoding="utf-8")
        (folder / "transcript.json").write_text("{}", encoding="utf-8")
        return folder

    monkeypatch.setattr("patientqa.callloop.run_call", fake_run_call)
    report = asyncio.run(
        run_campaign(
            plans,
            manifest_path=manifest,
            calls_root=tmp_path / "calls",
            stream_url="wss://example.invalid",
            pause_s=0,
        )
    )

    assert seen["to_number"] is None
    assert len(seen["scripted_barge_ins"]) == 2
    assert seen["manifest"]["objective"]["type"] == "barge_in"
    assert seen["manifest"]["test_intent"]["intentional"] is True
    assert seen["manifest"]["test_intent"]["behavior"] == "barge_in"
    assert seen["manifest"]["test_intent"]["isolation"] == "single_behavior"
    assert seen["manifest"]["test_intent"]["runtime"]["delivery"] == "call_loop"
    assert seen["voice_id"] == "EXAVITQu4vr4xnSDxMaL"
    assert seen["manifest"]["voice"]["persona_gender"] == "female"
    assert seen["manifest"]["voice"]["voice_gender"] == "female"
    assert seen["manifest"]["voice"]["gender_match"] is True
    assert seen["manifest"]["voice"]["origin"] == "gender_default"
    assert report.outcomes[0].duration_s == 179.0


def test_campaign_stops_after_stt_provider_failure(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(
        manifest,
        [
            valid_entry(call_id="call-001", seed=1),
            valid_entry(call_id="call-002", seed=2),
        ],
    )
    calls: list[str] = []

    async def fake_run_call(**kwargs):
        call_id = kwargs["call_id"]
        calls.append(call_id)
        folder = kwargs["calls_root"] / f"{call_id}_20260820T010203Z"
        folder.mkdir(parents=True)
        summary = {
            "duration_ms": 15000,
            "end_reason": "stt_failed",
            "stats": {"agent_turns": 0, "patient_turns": 0},
        }
        (folder / "call.json").write_text(json.dumps(summary), encoding="utf-8")
        (folder / "transcript.json").write_text("{}", encoding="utf-8")
        return folder

    monkeypatch.setattr("patientqa.callloop.run_call", fake_run_call)
    report = asyncio.run(
        run_campaign(
            load_plans(manifest),
            manifest_path=manifest,
            calls_root=tmp_path / "calls",
            stream_url="wss://example.invalid",
            pause_s=0,
            max_attempts=1,
        )
    )

    assert calls == ["call-001"]
    assert report.failed[0]["call_id"] == "call-001"
