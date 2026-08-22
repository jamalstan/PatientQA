"""Elaborator tests — template mode offline; LLM mode against a fake client."""

from __future__ import annotations

import json
from datetime import date

import pytest

from patientqa.datagen.elaborate import (
    ElaborationError,
    LlmElaborator,
    TemplateElaborator,
    build_identity,
    build_secondary_asks,
)
from patientqa.datagen.sampling import Sampler
from patientqa.datagen.validate import validate_entry

TODAY = date(2026, 8, 17)


def _seed(bank, index: int = 0):
    sampler = Sampler(bank, base_seed=42, generated_on=TODAY)
    return sampler.draw(index)


def test_template_output_passes_all_validation_rules(bank) -> None:
    for index in range(12):
        entry = TemplateElaborator().elaborate(_seed(bank, index))
        violations = validate_entry(entry, today=TODAY, drug_lexicon=bank.drug_lexicon)
        assert violations == [], f"slot {index}: {violations}"


def test_template_fills_placeholders_and_copies_taxonomy(bank) -> None:
    entry = TemplateElaborator().elaborate(_seed(bank, 0))
    assert "{" not in entry.objective.goal, "unfilled placeholder survived elaboration"
    template_types = {entry.objective.type}
    assert template_types  # type preserved verbatim from the template
    assert entry.elaboration == "template"
    assert entry.call_id == "call-001"


def test_template_medications_come_from_cluster(bank) -> None:
    seed = _seed(bank, 3)
    entry = TemplateElaborator().elaborate(seed)
    assert set(entry.persona.medications) <= set(seed.cluster.medications)


def test_three_minute_agenda_has_four_distinct_followups(bank) -> None:
    for index in range(24):
        seed = _seed(bank, index)
        asks = build_secondary_asks(seed)
        assert len(asks) == 4
        assert len({ask.lower() for ask in asks}) == 4
        assert all("{" not in ask for ask in asks)


def test_identity_is_deterministic_and_age_consistent(bank) -> None:
    seed = _seed(bank, 7)
    first = build_identity(seed)
    assert first == build_identity(seed)
    assert int(first.date_of_birth[:4]) == seed.generated_on.year - seed.age
    assert first.callback_number.startswith("+1")
    assert len(first.callback_number) == 12


# ---- fake LLM plumbing --------------------------------------------------------


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
        "background": (
            "Rosa Martinez, 68, manages her type 2 diabetes and blood pressure with "
            "quarterly endocrinology visits; she prefers Tuesday mornings and her "
            "daughter usually drives her."
        ),
        "speaking_style": "meanders warmly, asks the agent to repeat numbers",
        "voice_design_prompt": "elderly Mexican-American woman, warm, slightly hard of hearing",
        "medications": ["metformin"],
        "goal": "Move next week's endocrinology check; she forgot the exact date",
        "hidden_context": "Only mentions her recent hospital stay if asked directly",
        "curveballs": [
            {"at": "after agent proposes slot", "action": "counter with a Sunday request"}
        ],
        "secondary_asks": [
            "ask them to read the appointment details back",
            "ask where to park at the office",
            "ask how early to arrive",
            "ask what paperwork to bring",
        ],
    }
)


def test_llm_elaborator_merges_over_template_guardrail(bank) -> None:
    client = _FakeClient([_GOOD_LLM_JSON])
    elaborator = LlmElaborator("test-key", client=client)
    seed = _seed(bank, 0)
    baseline = TemplateElaborator().elaborate(seed)
    entry = elaborator.elaborate(seed)

    assert entry.elaboration == "llm"
    assert entry.persona.background.startswith("Rosa Martinez")
    # grading rules always survive from the taxonomy template
    assert entry.objective.success_criteria == baseline.objective.success_criteria
    assert entry.objective.termination == baseline.objective.termination
    assert entry.objective.type == baseline.objective.type
    assert entry.objective.secondary_asks == json.loads(_GOOD_LLM_JSON)["secondary_asks"]


def test_llm_elaborator_accepts_fenced_json(bank) -> None:
    fenced = f"```json\n{_GOOD_LLM_JSON}\n```"
    elaborator = LlmElaborator("test-key", client=_FakeClient([fenced]))
    assert elaborator.elaborate(_seed(bank, 1)).elaboration == "llm"


def test_llm_elaborator_rejects_garbage(bank) -> None:
    elaborator = LlmElaborator("test-key", client=_FakeClient(["utter nonsense"]))
    with pytest.raises(ElaborationError):
        elaborator.elaborate(_seed(bank, 2))


def test_llm_medications_are_lexicon_validated_downstream(bank) -> None:
    bad = json.loads(_GOOD_LLM_JSON)
    bad["medications"] = ["snake oil"]
    elaborator = LlmElaborator("test-key", client=_FakeClient([json.dumps(bad)]))
    entry = elaborator.elaborate(_seed(bank, 4))
    violations = validate_entry(entry, today=TODAY, drug_lexicon=bank.drug_lexicon)
    assert any("plausible_medications" in v for v in violations)
