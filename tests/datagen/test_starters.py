"""Conversation-starter tests — template mode offline; LLM mode against a fake client."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from factories import valid_entry, valid_starter_set

from patientqa.datagen.cli import main
from patientqa.datagen.schemas import ManifestEntry, parse_starters_line
from patientqa.datagen.starters import (
    LlmStarterGenerator,
    StarterError,
    StartersConfig,
    TemplateStarterGenerator,
    generate_starters,
    resolve_generator,
)
from patientqa.datagen.taxonomy import TEMPLATES, ObjectiveTemplate
from patientqa.datagen.validate import validate_starter_set

TODAY = date(2026, 8, 17)

_PLACEHOLDERS = {
    "specialty": "endocrinology",
    "medication": "sertraline",
    "condition": "type 2 diabetes",
}


def _entry_for(tpl: ObjectiveTemplate) -> ManifestEntry:
    return valid_entry(
        objective={
            "type": tpl.type,
            "goal": tpl.goal.format(**_PLACEHOLDERS),
            "hidden_context": tpl.hidden_context.format(**_PLACEHOLDERS),
            "curveballs": [c.model_dump() for c in tpl.curveballs],
        }
    )


# ---- template path --------------------------------------------------------------


@pytest.mark.parametrize("tpl", TEMPLATES, ids=lambda tpl: tpl.type)
def test_template_starters_pass_validation_for_every_objective_type(tpl) -> None:
    entry = _entry_for(tpl)
    starter_set = TemplateStarterGenerator().generate(entry, today=TODAY)
    assert validate_starter_set(entry, starter_set) == []


def test_template_starters_offer_distinct_angles() -> None:
    starter_set = TemplateStarterGenerator().generate(valid_entry(), today=TODAY)
    angles = [starter.angle for starter in starter_set.starters]
    texts = [starter.text for starter in starter_set.starters]
    assert len(angles) == len(set(angles)) == 3
    assert len(texts) == len(set(texts))
    assert starter_set.elaboration == "template"


def test_template_intent_surgery_strips_stage_directions() -> None:
    # goal carries a directorial clause after the ';' that must not be spoken
    starter_set = TemplateStarterGenerator().generate(valid_entry(), today=TODAY)
    for starter in starter_set.starters:
        assert "move next week's endocrinology visit" in starter.text
        assert "does not remember" not in starter.text


def test_template_starters_honor_requested_count_without_duplicates() -> None:
    starter_set = TemplateStarterGenerator().generate(valid_entry(), today=TODAY, count=2)
    assert len(starter_set.starters) == 2
    texts = [starter.text for starter in starter_set.starters]
    assert len(texts) == len(set(texts))


# ---- fake LLM plumbing ----------------------------------------------------------


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [type("Choice", (), {"message": _FakeMessage(content)})()]


class _FakeCompletions:
    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)

    def create(self, **kwargs) -> _FakeResponse:
        content = self._contents.pop(0) if self._contents else "{}"
        return _FakeResponse(content)


class _FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(contents)


_GOOD_LLM_JSON = json.dumps(
    {
        "starters": [
            {
                "angle": "direct",
                "text": "Hello, I need to move my endocrinology appointment to sometime next week.",
            },
            {
                "angle": "confused",
                "text": (
                    "Hi, sorry — I had a visit coming up with my endocrinologist, "
                    "and I think I need a different day next week?"
                ),
            },
            {
                "angle": "small talk",
                "text": (
                    "Hi, good morning! Listen, my daughter drives me on Tuesdays, so "
                    "I was hoping to move my doctor visit to another day next week."
                ),
            },
        ]
    }
)


def test_llm_starters_parse_and_validate() -> None:
    entry = valid_entry()
    generator = LlmStarterGenerator("test-key", client=_FakeClient([_GOOD_LLM_JSON]))
    starter_set = generator.generate(entry, today=TODAY)
    assert starter_set.elaboration == "llm"
    assert starter_set.model == "gpt-oss-120b"
    assert starter_set.starters[0].angle == "direct"
    assert validate_starter_set(entry, starter_set) == []


def test_llm_starters_reject_garbage() -> None:
    generator = LlmStarterGenerator("test-key", client=_FakeClient(["utter nonsense"]))
    with pytest.raises(StarterError):
        generator.generate(valid_entry(), today=TODAY)


def test_llm_starters_reject_single_candidate() -> None:
    single = json.dumps({"starters": json.loads(_GOOD_LLM_JSON)["starters"][:1]})
    generator = LlmStarterGenerator("test-key", client=_FakeClient([single]))
    with pytest.raises(StarterError):
        generator.generate(valid_entry(), today=TODAY)


# ---- orchestration --------------------------------------------------------------


def _entries() -> list[ManifestEntry]:
    first = valid_entry()
    second = valid_entry(call_id="call-015", persona={"name": "Jonah Pike"})
    return [first, second]


def _config(tmp_path: Path, **overrides) -> StartersConfig:
    defaults = dict(
        count=3,
        elaboration="llm",  # generator is always injected in these tests
        out=tmp_path / "starters.jsonl",
        retries=1,
    )
    defaults.update(overrides)
    return StartersConfig(**defaults)


def test_generate_starters_writes_sets_and_report(tmp_path: Path) -> None:
    config = _config(tmp_path)
    report = generate_starters(
        _entries(), config, generator=TemplateStarterGenerator(), today=TODAY
    )
    assert report.generated == 2
    assert report.fallbacks == []
    lines = config.out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    starter_sets = [parse_starters_line(line) for line in lines]
    assert [s.call_id for s in starter_sets] == ["call-014", "call-015"]
    assert config.out.with_suffix(".report.json").is_file()


def test_generate_starters_falls_back_to_template_on_llm_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    generator = LlmStarterGenerator("test-key", client=_FakeClient(["utter nonsense"]))
    report = generate_starters(_entries(), config, generator=generator, today=TODAY)

    assert report.generated == 2
    assert [fallback.call_id for fallback in report.fallbacks] == ["call-014", "call-015"]
    starter_sets = [parse_starters_line(line) for line in
                    config.out.read_text(encoding="utf-8").splitlines()]
    assert all(s.elaboration == "template" for s in starter_sets)


def test_generate_starters_falls_back_when_llm_output_violates_rules(tmp_path: Path) -> None:
    leaking = json.loads(_GOOD_LLM_JSON)
    leaking["starters"][1]["text"] = (
        "Hi, I will only mention she 'saw the doctor recently' if asked directly."
    )
    generator = LlmStarterGenerator(
        "test-key", client=_FakeClient([json.dumps(leaking), "utter nonsense"])
    )
    report = generate_starters(_entries()[:1], _config(tmp_path), generator=generator, today=TODAY)

    assert report.elaboration_counts == {"template": 1}
    assert "starter_no_hidden_context" in report.fallbacks[0].reason


def test_resolve_generator_modes(monkeypatch) -> None:
    assert isinstance(resolve_generator("template", model="gpt-oss-120b"), TemplateStarterGenerator)

    def _no_key(*args, **kwargs):
        raise KeyError("cerebras.api_key")

    monkeypatch.setattr("patientqa.config.get_secret", _no_key)
    assert isinstance(resolve_generator("auto", model="gpt-oss-120b"), TemplateStarterGenerator)
    with pytest.raises(KeyError):
        resolve_generator("llm", model="gpt-oss-120b")
    with pytest.raises(ValueError, match="unknown elaboration mode"):
        resolve_generator("nonsense", model="gpt-oss-120b")


# ---- validation rules -----------------------------------------------------------


def test_valid_factory_set_passes_validation() -> None:
    assert validate_starter_set(valid_entry(), valid_starter_set()) == []


def test_validation_flags_rule_breaking_sets() -> None:
    entry = valid_entry()

    mismatched = valid_starter_set(call_id="call-999")
    assert any("starter_call_id_match" in v for v in validate_starter_set(entry, mismatched))

    lonely = valid_starter_set(starters=[valid_starter_set().starters[0]])
    assert any("starters_plentiful" in v for v in validate_starter_set(entry, lonely))

    thin = valid_starter_set(
        starters=[
            {"angle": "direct", "text": "Book."},
            {"angle": "chatty", "text": "Hi, I'd like to move my visit to next week."},
        ]
    )
    assert any("starter_not_thin" in v for v in validate_starter_set(entry, thin))

    duplicated = valid_starter_set(starters=valid_starter_set().starters[:1] * 2)
    assert any("starters_distinct" in v for v in validate_starter_set(entry, duplicated))

    absolute = valid_starter_set(
        starters=[
            {"angle": "direct", "text": "Hi, I'd like to come in on 2026-08-24 if possible."},
            {"angle": "chatty", "text": "Hi there! I was hoping to move my visit to next week."},
        ]
    )
    assert any("starter_relative_dates" in v for v in validate_starter_set(entry, absolute))

    leaky = valid_starter_set(
        starters=[
            {"angle": "direct", "text": "Hello, I'd like to move my visit to sometime next week."},
            {"angle": "vague", "text": "Hi — counter with a Sunday request, please."},
        ]
    )
    assert any("starter_no_hidden_context" in v for v in validate_starter_set(entry, leaky))


def test_validation_flags_spanish_openers() -> None:
    entry = valid_entry()
    spanish = valid_starter_set(
        starters=[
            {
                "angle": "direct",
                "text": "Hola, soy Carmen. Necesito agendar una visita con el médico.",
            },
            {"angle": "chatty", "text": "Hi there! I'd like to move my visit to next week."},
        ]
    )
    assert any("starters_in_english" in v for v in validate_starter_set(entry, spanish))


def test_single_spanish_greeting_word_is_allowed() -> None:
    entry = valid_entry()
    flavored = valid_starter_set(
        starters=[
            {
                "angle": "direct",
                "text": "Hola, this is Marta — I'd like to move my visit to next week.",
            },
            {
                "angle": "chatty",
                "text": "Hi there! I was hoping to move my appointment to next week.",
            },
        ]
    )
    assert validate_starter_set(entry, flavored) == []


# ---- CLI ------------------------------------------------------------------------


def test_cli_starters_round_trip(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)  # keep cwd-relative defaults out of the repo
    manifest = tmp_path / "manifest.jsonl"
    assert (
        main(
            [
                "generate",
                "--count",
                "6",
                "--seed",
                "7",
                "--elaboration",
                "template",
                "--out",
                str(manifest),
            ]
        )
        == 0
    )

    code = main(["starters", str(manifest), "--elaboration", "template", "--limit", "4"])
    assert code == 0
    out = tmp_path / "manifest.starters.jsonl"
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4
    starter_sets = [parse_starters_line(line) for line in lines]
    assert starter_sets[0].call_id == "call-001"
    assert out.with_suffix(".report.json").is_file()
    assert "first set (call-001)" in capsys.readouterr().out


def test_cli_starters_rejects_bad_manifest(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"call_id": "call-001"}\n', encoding="utf-8")
    assert main(["starters", str(bad), "--elaboration", "template"]) == 1
    assert "invalid manifest line" in capsys.readouterr().err
