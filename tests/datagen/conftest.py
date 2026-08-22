"""Shared fixtures for the datagen tests."""

from __future__ import annotations

from datetime import date

import pytest

from patientqa.datagen.seeds import SeedBank, load_bundled_seedbank

TODAY = date(2026, 8, 17)


@pytest.fixture(scope="session")
def bank() -> SeedBank:
    return load_bundled_seedbank()


@pytest.fixture()
def today() -> date:
    return TODAY
