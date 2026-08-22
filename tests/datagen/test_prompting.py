"""Persona-card tests: manifest facts become executable conversation direction."""

from datetime import date

from factories import valid_entry

from patientqa.prompting import (
    build_identity_responder,
    build_persona_prompt,
    spoken_date_of_birth,
    spoken_us_phone,
    spoken_year,
)


def test_prompt_contains_consistent_identity_and_full_agenda() -> None:
    entry = valid_entry()
    prompt = build_persona_prompt(entry, today=date(2026, 8, 19))

    assert "March twelfth, nineteen fifty-five" in prompt
    assert "four one five, five five five, zero one three seven" in prompt
    assert "Blue Shield PPO" in prompt
    for ask in entry.objective.secondary_asks:
        assert ask in prompt


def test_prompt_directs_natural_three_minute_pacing() -> None:
    prompt = build_persona_prompt(valid_entry(), today=date(2026, 8, 19))

    assert "one new detail per turn" in prompt.lower()
    assert "at least eight patient turns" in prompt.lower()
    assert "actually, one more thing" in prompt.lower()
    assert "do not pad" in prompt.lower()


def test_prompt_supplies_unambiguous_calendar_context() -> None:
    prompt = build_persona_prompt(valid_entry(), today=date(2026, 8, 19))

    assert "Wednesday, August 19, 2026" in prompt
    assert "Wednesday, August 26" in prompt


def test_spoken_identity_formats_are_unambiguous() -> None:
    assert spoken_year(1945) == "nineteen forty-five"
    assert spoken_year(2000) == "two thousand"
    assert spoken_date_of_birth("1945-05-10") == "May tenth, nineteen forty-five"
    assert spoken_us_phone("+18135553736") == (
        "eight one three, five five five, three seven three six"
    )


def test_identity_responder_never_says_yes_while_correcting_dob() -> None:
    respond = build_identity_responder(valid_entry())
    answer = respond("I have your date of birth as March 12th, 1905. Is that correct?")
    assert answer == "My date of birth is March twelfth, nineteen fifty-five."
    assert "yes" not in answer.lower()
    assert respond("Your patient profile is set up, and your date of birth is July 4th, 2000.") == (
        "My date of birth is March twelfth, nineteen fifty-five."
    )
    assert respond("I have your date of birth.") is None
    assert respond("Your callback number is updated.") is None


def test_identity_responder_honors_objective_overrides() -> None:
    bad_phone = build_identity_responder(valid_entry(objective={"type": "bad_callback_number"}))
    assert bad_phone("What is your callback number?") is None

    refuse = build_identity_responder(valid_entry(objective={"type": "refuse_dob"}))
    assert refuse("What is your date of birth?") is None
