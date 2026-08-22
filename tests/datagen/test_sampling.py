"""Sampler determinism and dependency-constraint tests."""

from __future__ import annotations

from datetime import date

from patientqa.datagen.pipeline import allocate_classes
from patientqa.datagen.sampling import AGE_BANDS, Sampler
from patientqa.datagen.taxonomy import SPANISH, is_compatible

FIXED_DAY = date(2026, 8, 17)


def _sampler(bank, count: int = 24, seed: int = 42) -> Sampler:
    return Sampler(
        bank,
        base_seed=seed,
        class_allocation=allocate_classes(count),
        generated_on=FIXED_DAY,
    )


def test_same_inputs_give_identical_personas(bank) -> None:
    left = _sampler(bank).draw(7)
    right = _sampler(bank).draw(7)
    assert left == right


def test_attempt_shift_changes_the_persona(bank) -> None:
    sampler = _sampler(bank)
    assert sampler.draw(3).seed != sampler.draw(3, attempt=1).seed


def test_age_matches_its_band(bank) -> None:
    sampler = _sampler(bank, count=60, seed=7)
    for index in range(60):
        seed = sampler.draw(index)
        low, high = next((b[1], b[2]) for b in AGE_BANDS if b[0] == seed.age_band)
        assert low <= seed.age <= high


def test_cluster_supports_demographics(bank) -> None:
    sampler = _sampler(bank, count=60, seed=11)
    for index in range(60):
        seed = sampler.draw(index)
        assert seed.cluster.supports(seed.age, seed.gender)


def test_template_compatible_with_demographics(bank) -> None:
    """The sampler must never pair, say, a 30-year-old with very_elderly_slow."""
    sampler = _sampler(bank, count=60, seed=13)
    for index in range(60):
        seed = sampler.draw(index)
        assert is_compatible(seed.template, age=seed.age, language_tag=seed.language_tag), (
            f"slot {index}: {seed.template.type} incompatible with "
            f"age {seed.age}/{seed.language_tag}"
        )


def test_multilingual_slots_get_spanish_personas(bank) -> None:
    sampler = _sampler(bank, count=40, seed=5)
    for index in range(40):
        seed = sampler.draw(index)
        if seed.objective_class == "multilingual":
            assert seed.language_tag == SPANISH, f"slot {index} multilingual but not Spanish"
            assert seed.template.language_tag == SPANISH


def test_very_elderly_template_implies_78_plus(bank) -> None:
    sampler = _sampler(bank, count=60, seed=17)
    seen = False
    for index in range(60):
        seed = sampler.draw(index)
        if seed.template.type == "very_elderly_slow":
            seen = True
            assert seed.age >= 78
    assert seen, "60 draws never produced the very_elderly_slow template"


def test_names_unique_within_a_batch(bank) -> None:
    sampler = _sampler(bank, count=60, seed=19)
    names: set[str] = set()
    for index in range(60):
        seed = sampler.draw(index, taken_names=frozenset(names))
        assert seed.name.lower() not in names
        names.add(seed.name.lower())


def test_spanish_personas_get_hispanic_names(bank) -> None:
    sampler = _sampler(bank, count=60, seed=23)
    for index in range(60):
        seed = sampler.draw(index)
        if seed.language_tag == SPANISH:
            assert seed.heritage, f"slot {index} Spanish-flavored but no heritage"
            assert seed.name.split()[-1] in set(bank.names["last_hispanic"])


def test_generated_on_is_pinned(bank) -> None:
    assert _sampler(bank).draw(0).generated_on == date(2026, 8, 17)
