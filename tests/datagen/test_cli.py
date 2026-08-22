"""CLI smoke tests — run through the real argument parser, offline."""

from __future__ import annotations

from pathlib import Path

from patientqa.datagen.cli import main


def test_generate_and_validate_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)  # keep cwd-relative defaults out of the repo
    out = tmp_path / "manifest.jsonl"

    code = main(
        [
            "generate",
            "--count",
            "12",
            "--seed",
            "7",
            "--elaboration",
            "template",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert out.is_file()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 12
    assert out.with_suffix(".report.json").is_file()

    assert main(["validate", str(out)]) == 0


def test_validate_reports_bad_manifest(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"call_id": "call-001"}\n', encoding="utf-8")
    assert main(["validate", str(bad)]) == 1
    assert "invalid manifest line" in capsys.readouterr().err


def test_taxonomy_lists_all_classes(capsys) -> None:
    assert main(["taxonomy"]) == 0
    out = capsys.readouterr().out
    for class_id in (
        "happy_path",
        "faq_questions",
        "office_hours",
        "temporal_edge",
        "hallucination_bait",
        "identity_phi",
        "conversational_stress",
        "affect",
        "multilingual",
    ):
        assert class_id in out
