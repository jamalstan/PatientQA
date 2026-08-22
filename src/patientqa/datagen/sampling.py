"""Deterministic sampling layer (DESIGN §6.3, the 'seeded RNG' box).

Column order mirrors the design's dependency chain — demographics first, then
everything downstream conditions on them:

    demographics (age band, gender, language, disposition)   <- diversity
    condition cluster (age/gender appropriate)               <- medical realism
    phrasing style (language-matched)                        <- patient voice
    objective template (class per slot; demographics are
    rejection-sampled against the drawn template)            <- test intent

Every draw is a pure function of ``(base_seed, index, attempt)`` via
``derive_seed`` (blake2b), so a manifest is byte-for-byte reproducible and any
single persona can be regenerated in isolation.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from datetime import date

from patientqa.datagen.seeds import ConditionCluster, PhrasingSnippet, SeedBank
from patientqa.datagen.taxonomy import (
    ENGLISH,
    OBJECTIVE_CLASSES,
    SPANISH,
    ObjectiveTemplate,
    is_compatible,
    templates_in_class,
)

# (name, low, high, weight) — callers at a medical practice skew older
AGE_BANDS: tuple[tuple[str, int, int, float], ...] = (
    ("young_adult", 18, 29, 0.05),
    ("adult", 30, 44, 0.20),
    ("middle_aged", 45, 59, 0.25),
    ("senior", 60, 74, 0.30),
    ("elderly", 75, 90, 0.20),
)

# (language label, language_tag, weight) — tags drive template compatibility
# and phrasing/matching; SPANISH covers any Spanish-influenced variant.
LANGUAGES: tuple[tuple[str, str, float], ...] = (
    ("English", ENGLISH, 0.80),
    ("English with occasional Spanish phrases", SPANISH, 0.10),
    ("English w/ Spanish code-switching", SPANISH, 0.08),
    ("Spanish-heavy with English drug names", SPANISH, 0.02),
)

GENDERS: tuple[tuple[str, float], ...] = (("female", 0.52), ("male", 0.45), ("nonbinary", 0.03))

DISPOSITIONS: tuple[tuple[str, float], ...] = (
    ("warm and chatty", 0.25),
    ("brisk and businesslike", 0.20),
    ("anxious and detail-seeking", 0.15),
    ("sweet but scattered", 0.10),
    ("gruff and impatient", 0.10),
    ("guarded and formal", 0.10),
    ("confused but determined", 0.10),
)

_DEMOGRAPHIC_RETRIES = 12
_NAME_RETRIES = 24


def derive_seed(*parts: object) -> int:
    """Stable, portable RNG seed from arbitrary parts (no PYTHONHASHSEED games)."""
    joined = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(joined, digest_size=8).digest(), "big")


@dataclass(frozen=True)
class PersonaSeed:
    """Everything the elaborators need to write one manifest entry."""

    index: int
    seed: int  # per-persona seed; elaborate further rng from this
    name: str
    age: int
    age_band: str
    gender: str
    language: str
    language_tag: str
    heritage: str | None  # e.g. "Cuban-American" for Spanish-influenced personas
    disposition: str
    cluster: ConditionCluster
    phrasing: PhrasingSnippet
    template: ObjectiveTemplate
    objective_class: str
    generated_on: date


class Sampler:
    """Draws persona seeds deterministically from a :class:`SeedBank`."""

    def __init__(
        self,
        seedbank: SeedBank,
        *,
        base_seed: int,
        class_allocation: list[str] | None = None,
        generated_on: date | None = None,
    ) -> None:
        self.bank = seedbank
        self.base_seed = base_seed
        self.class_allocation = class_allocation
        self.generated_on = generated_on or date.today()

    def draw(
        self, index: int, *, attempt: int = 0, taken_names: frozenset[str] = frozenset()
    ) -> PersonaSeed:
        """Draw the persona for slot ``index``.

        ``attempt`` shifts the derived seed so the pipeline can redraw a persona
        that failed validation without disturbing other slots.
        """
        rng = random.Random(derive_seed(self.base_seed, "persona", index, attempt))
        seed = derive_seed(self.base_seed, index, attempt)

        class_id = self._pick_class(rng, index)
        template = self._pick_template(rng, class_id, index)
        age, age_band, gender, language, language_tag = self._demographics_for(rng, template)

        clusters = [c for c in self.bank.clusters if c.supports(age, gender)]
        if not clusters:  # gendered pools can't fully cover every draw; relax to age only
            clusters = [c for c in self.bank.clusters if c.min_age <= age <= c.max_age]
        cluster = _weighted_choice(rng, clusters, [c.weight for c in clusters])

        phrasing = self._draw_phrasing(rng, language_tag)
        name, heritage = self._draw_name(rng, gender, language_tag, taken_names)
        disposition = _weighted_choice(rng, *zip(*DISPOSITIONS, strict=True))

        return PersonaSeed(
            index=index,
            seed=seed,
            name=name,
            age=age,
            age_band=age_band,
            gender=gender,
            language=language,
            language_tag=language_tag,
            heritage=heritage,
            disposition=disposition,
            cluster=cluster,
            phrasing=phrasing,
            template=template,
            objective_class=class_id,
            generated_on=self.generated_on,
        )

    # ---- stages -------------------------------------------------------------

    def _pick_class(self, rng: random.Random, index: int) -> str:
        if self.class_allocation is not None:
            class_id = self.class_allocation[index % len(self.class_allocation)]
            templates_in_class(class_id)  # validates; raises KeyError with the known ids
            return class_id
        ids = [c.id for c in OBJECTIVE_CLASSES]
        weights = [c.weight for c in OBJECTIVE_CLASSES]
        return _weighted_choice(rng, ids, weights)

    def _pick_template(
        self, rng: random.Random, class_id: str, index: int
    ) -> ObjectiveTemplate:
        templates = templates_in_class(class_id)
        if class_id == "adversarial_security" and self.class_allocation is not None:
            slot = index % len(self.class_allocation)
            occurrence = sum(
                1 for value in self.class_allocation[:slot] if value == class_id
            )
            return templates[occurrence % len(templates)]
        return rng.choice(templates)

    def _demographics_for(
        self, rng: random.Random, template: ObjectiveTemplate
    ) -> tuple[int, str, str, str, str]:
        """Demographics first in the draw order, but rejection-sampled against
        the *specific* template so constrained probes (Spanish for multilingual,
        78+ for very-elderly affect) are not starved by class-level fallbacks."""
        for _ in range(_DEMOGRAPHIC_RETRIES):
            demographics = self._draw_demographics(rng)
            if is_compatible(
                template, age=demographics[0], language_tag=demographics[4]
            ):
                return demographics
        return self._forced_demographics(rng, template)

    def _draw_demographics(self, rng: random.Random) -> tuple[int, str, str, str, str]:
        band = _weighted_choice(rng, [b[0] for b in AGE_BANDS], [b[3] for b in AGE_BANDS])
        bounds = next(b for b in AGE_BANDS if b[0] == band)
        age = rng.randint(bounds[1], bounds[2])
        gender = _weighted_choice(rng, *zip(*GENDERS, strict=True))
        language = _weighted_choice(
            rng, [item[0] for item in LANGUAGES], [item[2] for item in LANGUAGES]
        )
        language_tag = next(item[1] for item in LANGUAGES if item[0] == language)
        return age, band, gender, language, language_tag

    def _forced_demographics(
        self, rng: random.Random, template: ObjectiveTemplate
    ) -> tuple[int, str, str, str, str]:
        """Fallback after rejection fails: construct demographics the template allows."""
        if template.language_tag == SPANISH:
            spanish = [item for item in LANGUAGES if item[1] == SPANISH]
            language = _weighted_choice(
                rng, [item[0] for item in spanish], [item[2] for item in spanish]
            )
            language_tag = SPANISH
        else:
            language, language_tag = LANGUAGES[0][0], ENGLISH
        age = rng.randint(template.min_age, template.max_age)
        gender = _weighted_choice(rng, *zip(*GENDERS, strict=True))
        band = next((b[0] for b in AGE_BANDS if b[1] <= age <= b[2]), AGE_BANDS[-1][0])
        return age, band, gender, language, language_tag

    def _draw_phrasing(self, rng: random.Random, language_tag: str) -> PhrasingSnippet:
        pool = self.bank.phrasings
        if language_tag == SPANISH and rng.random() < 0.7:
            spanish = [p for p in pool if p.language_tag == SPANISH]
            if spanish:
                return rng.choice(spanish)
        english = [p for p in pool if p.language_tag == ENGLISH]
        return rng.choice(english or list(pool))

    def _draw_name(
        self, rng: random.Random, gender: str, language_tag: str, taken: frozenset[str]
    ) -> tuple[str, str | None]:
        hispanic = language_tag == SPANISH
        names = self.bank.names
        for _ in range(_NAME_RETRIES):
            if hispanic and gender in ("male", "female"):
                first = rng.choice(names[f"first_hispanic_{gender}"])
            else:
                first = rng.choice(names[f"first_{gender}"])
            last_pool_key = "last_hispanic" if hispanic else "last_general"
            last = rng.choice(names[last_pool_key])
            name = f"{first} {last}"
            if name.lower() not in taken:
                heritage = rng.choice(names["heritage"]) if hispanic else None
                return name, heritage
        # Pools are large enough that this is unreachable in practice; keep a
        # deterministic escape hatch rather than looping forever.
        first = rng.choice(names[f"first_{gender}"])
        last = f"{rng.choice(names['last_general'])} {rng.choice(names['last_general'])}"
        return f"{first} {last}", None


def _weighted_choice(
    rng: random.Random, items: tuple | list, weights: tuple | list
) -> object:
    return rng.choices(list(items), weights=list(weights), k=1)[0]
