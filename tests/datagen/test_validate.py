"""Post-validation rule tests — each rule fires on a crafted bad entry."""

from __future__ import annotations

from datetime import date

from factories import valid_entry

from patientqa.datagen.schemas import VoiceProfile
from patientqa.datagen.validate import validate_entry, validate_manifest

TODAY = date(2026, 8, 17)
LEXICON = frozenset({"metformin", "lisinopril", "apixaban"})


def _violations(entry):
    return validate_entry(entry, today=TODAY, drug_lexicon=LEXICON)


def test_valid_entry_is_clean() -> None:
    assert _violations(valid_entry()) == []


def test_unknown_objective_type_flagged() -> None:
    assert any("objective_type_registered" in v for v in _violations(valid_entry(
        objective={"type": "definitely_not_a_template"},
    )))


def test_past_date_flagged_future_date_clean() -> None:
    past = valid_entry(objective={"goal": "Book something on 2020-01-01 please"})
    assert any("dates_in_future" in v for v in _violations(past))

    future = valid_entry(objective={"goal": "Book something on 2026-12-01 please"})
    assert not any("dates_in_future" in v for v in _violations(future))

    today_ok = valid_entry(objective={"goal": "Book something on 2026-08-17 please"})
    assert not any("dates_in_future" in v for v in _violations(today_ok))


def test_impossible_date_flagged() -> None:
    bad = valid_entry(objective={"hidden_context": "Says she booked 2026-02-30"})
    assert any("dates_in_future: invalid" in v for v in _violations(bad))


def test_unknown_drug_flagged() -> None:
    bad = valid_entry(persona={"medications": ["metformin", "boneribbit"]})
    assert any("plausible_medications" in v for v in _violations(bad))


def test_known_drug_case_insensitive() -> None:
    ok = valid_entry(persona={"medications": ["Metformin"]})
    assert not any("plausible_medications" in v for v in _violations(ok))


def test_thin_design_prompt_flagged() -> None:
    thin = valid_entry()
    thin.persona.voice = VoiceProfile(design_prompt="old woman")
    assert any("voice_mapping_present" in v for v in _violations(thin))


def test_voice_id_satisfies_mapping_rule() -> None:
    entry = valid_entry()
    entry.persona.voice = VoiceProfile(voice_id="designed-voice-123")
    assert not any("voice_mapping_present" in v for v in _violations(entry))


def test_thin_persona_flagged() -> None:
    thin = valid_entry(persona={"background": "has diabetes"})
    assert any("persona_not_thin" in v for v in _violations(thin))


def test_dates_checked_across_all_objective_text() -> None:
    bad = valid_entry()
    bad.objective.curveballs[0].action = "insist the appointment was on 2019-05-04"
    assert any("dates_in_future" in v for v in _violations(bad))


def test_agenda_requires_exactly_four_speakable_followups() -> None:
    short = valid_entry(objective={"secondary_asks": ["ask about parking"]})
    assert any("agenda_substantial" in v for v in _violations(short))

    placeholder = valid_entry()
    placeholder.objective.secondary_asks[0] = "ask about the {specialty} paperwork"
    assert any("unfilled placeholder" in v for v in _violations(placeholder))


def test_identity_dob_and_phone_are_consistent() -> None:
    future_dob = valid_entry(persona={"identity": {
        "date_of_birth": "2030-01-01",
        "callback_number": "+14155550137",
        "insurance_plan": "Blue Shield PPO",
    }})
    assert any("DOB" in v for v in _violations(future_dob))

    bad_phone = valid_entry()
    bad_phone.persona.identity.callback_number = "555-0137"
    assert any("US E.164" in v for v in _violations(bad_phone))


def test_duplicate_names_and_ids_flagged_at_manifest_level() -> None:
    first = valid_entry()
    second = valid_entry()
    second.persona.name = first.persona.name
    third = valid_entry()
    third.call_id = first.call_id
    report = validate_manifest([first, second, third], today=TODAY, drug_lexicon=LEXICON)
    assert set(report) == {first.call_id, second.call_id, third.call_id}
    joined = "; ".join(report[second.call_id])
    assert "no_duplicate_names" in joined
    assert "call_id_unique" in "; ".join(report[third.call_id])
