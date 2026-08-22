"""Taxonomy registry integrity and compatibility rules."""

from __future__ import annotations

import pytest

from patientqa.datagen.taxonomy import (
    ENGLISH,
    OBJECTIVE_CLASSES,
    RED_TEAM_PLANS,
    REGISTERED_TECHNIQUES,
    SPANISH,
    TEMPLATES,
    is_compatible,
    template_by_type,
    templates_in_class,
)


def test_template_type_ids_unique() -> None:
    types = [t.type for t in TEMPLATES]
    assert len(types) == len(set(types))


def test_every_template_is_complete() -> None:
    for tpl in TEMPLATES:
        assert tpl.success_criteria, f"{tpl.type} has no success criteria"
        assert tpl.termination.strip(), f"{tpl.type} has no termination"
        assert tpl.goal.strip(), f"{tpl.type} has no goal"
        templates_in_class(tpl.objective_class)  # class id must resolve


def test_every_template_has_a_bounded_registered_red_team_plan() -> None:
    assert set(RED_TEAM_PLANS) == {tpl.type for tpl in TEMPLATES}
    for type_id, plan in RED_TEAM_PLANS.items():
        assert 1 <= len(plan.techniques) <= 3, type_id
        assert set(plan.techniques) <= REGISTERED_TECHNIQUES, type_id
        assert 2 <= len(plan.escalation_steps) <= 4, type_id
        assert plan.hypothesis.strip(), type_id
        assert plan.safety_boundary.strip(), type_id


def test_class_weights_sum_to_one() -> None:
    assert sum(c.weight for c in OBJECTIVE_CLASSES) == pytest.approx(1.0)


def test_design_example_type_exists() -> None:
    """reschedule_edge is the canonical template from DESIGN.md §6.3."""
    tpl = template_by_type("reschedule_edge")
    assert tpl.curveballs[0].action == "counter with a Sunday request"
    assert "no Sunday confirmation" in tpl.success_criteria
    assert tpl.termination == "goal achieved OR agent fails twice OR 180s"


def test_unknown_ids_raise_with_listing() -> None:
    with pytest.raises(KeyError, match="reschedule_edge"):
        template_by_type("not_a_real_type")
    with pytest.raises(KeyError):
        templates_in_class("not_a_class")


def test_all_design_classes_present() -> None:
    expected = {
        "happy_path",
        "faq_questions",
        "office_hours",
        "temporal_edge",
        "hallucination_bait",
        "identity_phi",
        "conversational_stress",
        "affect",
        "multilingual",
        "adversarial_security",
    }
    assert {c.id for c in OBJECTIVE_CLASSES} == expected


def test_challenge_scenario_matrix_is_covered() -> None:
    """Every scenario family the challenge asks us to test maps to a template."""
    types = {tpl.type for tpl in TEMPLATES}
    assert {
        "schedule_new",  # simple appointment scheduling
        "reschedule_edge",  # rescheduling
        "cancel_existing",  # canceling
        "refill_request",  # medication refill requests
        "faq_office_info",  # questions about office hours and locations
        "faq_insurance",  # questions about insurance
        "barge_in",  # edge cases: interruptions
        "prompt_injection_bypass",  # direct instruction-hierarchy attack
        "staff_impersonation_phi",  # verification bypass / PHI extraction
        "cross_turn_context_poisoning",  # multi-turn poisoned context
    } <= types


def test_compatibility_age_constraints() -> None:
    very_elderly = template_by_type("very_elderly_slow")
    assert very_elderly.min_age == 78
    assert not is_compatible(very_elderly, age=30, language_tag=ENGLISH)
    assert is_compatible(very_elderly, age=80, language_tag=ENGLISH)


def test_compatibility_language_constraints() -> None:
    spanish_call = template_by_type("spanish_heavy_call")
    assert spanish_call.language_tag == SPANISH
    assert not is_compatible(spanish_call, age=50, language_tag=ENGLISH)
    assert is_compatible(spanish_call, age=50, language_tag=SPANISH)
