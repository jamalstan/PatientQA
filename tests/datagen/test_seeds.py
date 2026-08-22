"""Bundled seed data integrity."""

from __future__ import annotations

from patientqa.datagen.seeds import SeedBank


def test_bundled_bank_loads(bank: SeedBank) -> None:
    assert len(bank.clusters) >= 15
    assert len(bank.phrasings) >= 20
    assert len(bank.drug_lexicon) >= 80


def test_cluster_medications_are_in_drug_lexicon(bank: SeedBank) -> None:
    """The invariant that makes template-elaborated personas always pass the
    'plausible drugs' validation rule."""
    for cluster in bank.clusters:
        for med in cluster.medications:
            assert bank.lexicon_contains(med), f"{cluster.key}: {med} missing from drugs.json"


def test_lexicon_is_lowercase_canonical(bank: SeedBank) -> None:
    assert all(drug == drug.lower() for drug in bank.drug_lexicon)
    assert bank.lexicon_contains("MetFORMin")  # case-insensitive lookup


def test_phrasing_has_both_language_families(bank: SeedBank) -> None:
    tags = {p.language_tag for p in bank.phrasings}
    assert {"english", "spanish"} <= tags


def test_name_pools_cover_all_genders_and_hispanic(bank: SeedBank) -> None:
    for key in (
        "first_female",
        "first_male",
        "first_nonbinary",
        "first_hispanic_female",
        "first_hispanic_male",
        "last_general",
        "last_hispanic",
        "heritage",
    ):
        assert bank.names.get(key), f"names.json missing pool {key}"


def test_clusters_cover_young_and_old(bank: SeedBank) -> None:
    def supports(age: int, gender: str = "female") -> list:
        return [c for c in bank.clusters if c.supports(age, gender)]

    assert supports(22), "no cluster available for a 22-year-old"
    assert supports(85), "no cluster available for an 85-year-old"
    assert supports(40, "male"), "no cluster available for a 40-year-old male"


def test_gendered_clusters_exclude_other_genders(bank: SeedBank) -> None:
    female_only = [c for c in bank.clusters if c.gender == "female"]
    assert female_only
    for cluster in female_only:
        assert not cluster.supports(cluster.min_age, "male")
