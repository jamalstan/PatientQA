"""Pydantic schemas for personas, objectives, and manifest entries.

Shapes follow the manifest example in DESIGN.md §6.3 exactly; the two added
fields (``Persona.medications``, ``ManifestEntry.seed``/``generated_at``/
``elaboration``) carry provenance and machine-checkable facts that the
post-validation rules (``validate.py``) need.

Structural validation lives here (field types, required pairs); quality rules
(future dates, plausible drugs, duplicate names, ...) live in ``validate.py``
so a manifest that parses is distinct from a manifest that is *good*.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Gender = Literal["male", "female", "nonbinary"]

ElaborationMode = Literal["llm", "template"]


class VoiceProfile(BaseModel):
    """How the persona is voiced.

    ElevenLabs Voice Design creates a bespoke voice from ``design_prompt``
    (DESIGN §3.3); ``voice_id`` is filled in later, once the designed voice
    exists, so the orchestrator can pass it straight to TTS. Exactly one of
    the two must be present — that is the "voice-ID mapping present" rule.
    """

    provider: Literal["elevenlabs"] = "elevenlabs"
    design_prompt: str | None = None
    voice_id: str | None = None

    @model_validator(mode="after")
    def _require_prompt_or_id(self) -> VoiceProfile:
        if not (self.design_prompt or self.voice_id):
            raise ValueError("voice needs a design_prompt or a voice_id")
        return self


class Curveball(BaseModel):
    """A scripted mid-call twist, fired by the orchestrator at stage ``at``."""

    at: str = Field(description="conversation stage, e.g. 'after agent proposes slot'")
    action: str = Field(description="what the patient does, e.g. 'counter with a Sunday request'")

    @field_validator("at", "action")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("curveball fields must not be empty")
        return value.strip()


class Identity(BaseModel):
    """The patient's verification facts — what the agent will actually collect.

    A scheduling agent asks for name, DOB, callback number, insurance; the
    brain needs one consistent set of answers or every verification loop
    derails. Deterministically derived from the persona seed (never
    LLM-written — the facts must be reproducible), and deliberately NOT part
    of the relative-date validation sweep: a date of birth is an absolute
    *past* date by definition.
    """

    date_of_birth: str = Field(description="ISO date, e.g. '1954-03-12'")
    callback_number: str = Field(description="E.164, e.g. '+14155550137'")
    insurance_plan: str = Field(description="e.g. 'Blue Shield PPO'")

    @field_validator("date_of_birth", "callback_number", "insurance_plan")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identity fields must not be empty")
        return value.strip()


class AdversarialPlan(BaseModel):
    """A bounded, measurable red-team experiment embedded in one call.

    The plan is deterministic taxonomy data, not LLM-authored prose.  It says
    what failure is being tested and how the synthetic patient should escalate
    over several turns while staying inside the authorized test boundary.
    """

    techniques: list[str] = Field(min_length=1, max_length=3)
    hypothesis: str
    escalation_steps: list[str] = Field(min_length=2, max_length=4)
    safety_boundary: str

    @field_validator("hypothesis", "safety_boundary")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("adversarial plan fields must not be empty")
        return value.strip()

    @field_validator("techniques", "escalation_steps")
    @classmethod
    def _non_empty_items(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("adversarial plan list items must not be empty")
        return cleaned


class TestIntent(BaseModel):
    """Machine-readable declaration that a call's stress behavior is deliberate.

    A call has exactly one behavior under test: its objective type.  Technique
    labels may map that behavior to several risk taxonomies, but they are not
    additional caller behaviors.  This block is promoted into the session's
    ``meta.json`` and ``call.json`` so a reviewer never mistakes the behavior
    for an accidental simulator failure.
    """

    intentional: Literal[True] = True
    behavior: str = Field(description="the one deliberately introduced behavior")
    isolation: Literal["single_behavior"] = "single_behavior"
    hypothesis: str
    protocol: Literal["introduce_only_this_behavior"] = "introduce_only_this_behavior"

    @field_validator("behavior", "hypothesis")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("test intent fields must not be empty")
        return value.strip()


class Objective(BaseModel):
    """The hidden test intent for one call (DESIGN §7 stress-test layer)."""

    type: str = Field(description="taxonomy template id, e.g. 'reschedule_edge'")
    goal: str
    hidden_context: str = ""
    curveballs: list[Curveball] = Field(default_factory=list)
    secondary_asks: list[str] = Field(
        default_factory=list,
        description=(
            "the rest of the patient's agenda, surfaced one item at a time after "
            "the primary goal — four compact follow-ups make a natural 3-minute conversation "
            "instead of one question and a hang-up"
        ),
    )
    success_criteria: list[str] = Field(min_length=1, description="machine-checkable outcomes")
    adversarial: AdversarialPlan = Field(
        description="typed multi-turn red-team hypothesis, escalation, and safety boundary"
    )
    termination: str = Field(description="e.g. 'goal achieved OR agent fails twice OR 180s'")

    @field_validator("type", "goal", "termination")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("objective fields must not be empty")
        return value.strip()


class Persona(BaseModel):
    """A fully-specified synthetic patient. Fictional by construction."""

    name: str
    age: int = Field(ge=18, le=100)
    gender: Gender
    language: str = Field(description="e.g. 'English w/ Spanish code-switching'")
    voice: VoiceProfile
    background: str = Field(description="medical background a scheduler would plausibly hear")
    speaking_style: str = Field(description="dialogue-style guidance for the patient LLM")
    medications: list[str] = Field(
        default_factory=list,
        description="structured drug list, validated against the bundled lexicon",
    )
    identity: Identity | None = Field(
        default=None,
        description="DOB / callback number / insurance — deterministic, seed-derived",
    )

    @field_validator("name", "background", "speaking_style")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("persona fields must not be empty")
        return value.strip()


class ManifestEntry(BaseModel):
    """One line of ``manifest.jsonl`` — one planned call (DESIGN §6.3)."""

    call_id: str = Field(description="e.g. 'call-014'; unique within a manifest")
    persona: Persona
    objective: Objective
    seed: int = Field(description="per-persona derived RNG seed, for reproducibility")
    generated_at: str = Field(description="ISO date the entry was generated")
    elaboration: ElaborationMode = Field(
        default="template", description="who wrote the prose: 'llm' or 'template'"
    )
    test_intent: TestIntent | None = Field(
        default=None,
        description="explicit single-behavior provenance copied into call metadata",
    )

    @field_validator("call_id")
    @classmethod
    def _call_id_shape(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("call_id must not be empty")
        return value

    @model_validator(mode="after")
    def _derive_and_verify_test_intent(self) -> ManifestEntry:
        expected = TestIntent(
            behavior=self.objective.type,
            hypothesis=self.objective.adversarial.hypothesis,
        )
        if self.test_intent is None:
            self.test_intent = expected
        elif self.test_intent != expected:
            raise ValueError(
                "test_intent must declare the objective type and adversarial hypothesis "
                "as one intentional behavior"
            )
        return self


class Starter(BaseModel):
    """One candidate opening line — what the patient says first on the call."""

    angle: str = Field(description="opening strategy in 1-3 words, e.g. 'direct', 'chatty'")
    text: str = Field(description="the opening utterance; first person, spoken register")

    @field_validator("angle", "text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("starter fields must not be empty")
        return value.strip()


class StarterSet(BaseModel):
    """Conversation starters for one manifest entry, joined by ``call_id``."""

    call_id: str = Field(description="must match the manifest entry's call_id")
    starters: list[Starter] = Field(min_length=1)
    elaboration: ElaborationMode = Field(default="template")
    model: str = Field(default="", description="Cerebras model id when elaboration == 'llm'")
    generated_at: str = Field(default="", description="ISO date the set was generated")

    @field_validator("call_id")
    @classmethod
    def _call_id_shape(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("call_id must not be empty")
        return value


def parse_manifest_line(line: str) -> ManifestEntry:
    """Parse one JSONL line into a ManifestEntry, with an actionable error."""
    import json

    try:
        return ManifestEntry.model_validate(json.loads(line))
    except ValueError as exc:  # includes json.JSONDecodeError and ValidationError
        snippet = line.strip()[:80]
        raise ValueError(f"invalid manifest line ({snippet!r}...): {exc}") from exc


def parse_starters_line(line: str) -> StarterSet:
    """Parse one JSONL line of a starters file, with an actionable error."""
    import json

    try:
        return StarterSet.model_validate(json.loads(line))
    except ValueError as exc:  # includes json.JSONDecodeError and ValidationError
        snippet = line.strip()[:80]
        raise ValueError(f"invalid starters line ({snippet!r}...): {exc}") from exc
