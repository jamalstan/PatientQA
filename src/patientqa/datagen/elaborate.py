"""Persona elaboration (DESIGN §6.3): fuse a sampled seed into a full manifest entry.

Two interchangeable elaborators:

- :class:`TemplateElaborator` — deterministic, offline prose composition from
  the seed. The default; used for tests, cheap regeneration, and as the
  guardrail/fallback for the LLM path.
- :class:`LlmElaborator` — one small gpt-oss-120b completion on Cerebras
  (DESIGN §3.2) that rewrites the template prose in-character. Objective
  ``type``, ``success_criteria``, and ``termination`` are ALWAYS copied
  verbatim from the taxonomy template — the LLM never invents grading rules.
"""

from __future__ import annotations

import json
import random
import re
from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError

from patientqa.datagen.sampling import PersonaSeed, derive_seed
from patientqa.datagen.schemas import (
    AdversarialPlan,
    Curveball,
    Identity,
    ManifestEntry,
    Objective,
    Persona,
    VoiceProfile,
)
from patientqa.datagen.taxonomy import eligible_asks, red_team_plan

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
DEFAULT_MODEL = "gpt-oss-120b"

#: Four compact follow-ups plus verification and the primary task reliably
#: produce an 8–12 turn conversation without padding or repetition. This is
#: the natural three-minute shape the assessment asks for.
_TARGET_TOTAL_ASKS = 4

#: Callback numbers are fictional (555/556 exchanges); the E.164 shape is
#: what the agent reads back, so it must look like a real mobile number.
_PHONE_EXCHANGES = ("555", "556", "557")

_INSURANCE_PLANS = (
    "Blue Shield PPO",
    "Aetna Medicare Advantage",
    "UnitedHealthcare Choice Plus",
    "Kaiser Permanente HMO",
    "Cigna Open Access",
    "original Medicare",
    "Anthem Blue Cross EPO",
)

# Affect templates override the sampled disposition so the persona's tone
# matches the stress intent (design: "persona facts pinned" coherence rule).
_AFFECT_DISPOSITIONS = {
    "anxious_patient": "anxious and detail-seeking",
    "angry_patient": "gruff and impatient",
    "very_elderly_slow": "sweet but scattered",
}

_STYLE_VOICE = {
    "hedging": "hedges and downplays symptoms",
    "minimizing": "downplays everything",
    "vague_timing": "vague about dates and timelines",
    "precise": "gives exact medication and appointment details",
    "asks_to_repeat": "keeps asking the agent to repeat numbers",
    "self_diagnosis": "cites things they read online",
    "deferential": "defers to whatever the office suggests",
    "scattered": "jumps between topics and loses the thread",
    "insistent": "repeats fixed requirements over and over",
    "laconic": "answers in as few words as possible",
    "polite_chatty": "makes small talk before getting to the point",
    "confused_terms": "mixes up doctor names and specialties",
    "spanish_interleave": "sprinkles Spanish into English, especially for drug names",
    "spanish_heavy": "speaks mostly Spanish, using English mainly for drug names",
    "hf_mined": "phrases things the way real patients do",
}

_VOICE_TRAIT = {
    "warm and chatty": "talks around the point",
    "brisk and businesslike": "crisp, no wasted words",
    "anxious and detail-seeking": "worried-sounding",
    "sweet but scattered": "gently rambling",
    "gruff and impatient": "gruff",
    "guarded and formal": "measured and formal",
    "confused but determined": "determined despite being muddled",
}

_AGE_ADJECTIVE = {
    "young_adult": "young adult",
    "adult": "adult",
    "middle_aged": "middle-aged",
    "senior": "senior",
    "elderly": "elderly",
}

_SCHEDULING_PREFERENCES = (
    "prefers Tuesday mornings",
    "can only do afternoons",
    "needs the first appointment of the day",
    "avoids Mondays",
    "needs late-morning slots",
    "is flexible most days",
    "can only do Thursdays",
    "has no preference on provider",
)

_ELDERLY_SUPPORT = (
    "daughter usually drives {}",
    "son handles {} appointments",
    "relies on the senior shuttle",
    "walks over from the retirement community nearby",
)

_PRONOUN = {"female": "her", "male": "him", "nonbinary": "them"}


class ElaborationError(Exception):
    """The elaborator could not produce a valid entry (LLM output unusable)."""


class Elaborator(Protocol):
    def elaborate(self, seed: PersonaSeed) -> ManifestEntry: ...


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return f"{{{key}}}"  # leave unknown placeholders untouched instead of raising


def _fill(template: str, seed: PersonaSeed) -> str:
    meds = seed.cluster.medications
    fields = _SafeFormatDict(
        specialty=seed.cluster.specialty,
        condition=seed.cluster.conditions[0] if seed.cluster.conditions else "the condition",
        medication=meds[0] if meds else "the medication",
    )
    return template.format_map(fields)


def build_identity(seed: PersonaSeed) -> Identity:
    """Deterministic verification facts: DOB consistent with the sampled age,
    a fictional-but-shaped callback number, a plausible plan.

    The year is derived from ``generated_on`` (not today at call time), so a
    manifest stays internally consistent whenever it is regenerated.
    """
    rng = random.Random(derive_seed(seed.seed, "identity"))
    year = seed.generated_on.year - seed.age
    dob = date(year, rng.randint(1, 12), rng.randint(1, 28))
    number = f"+1{rng.randint(200, 989)}{rng.choice(_PHONE_EXCHANGES)}{rng.randint(0, 9999):04d}"
    return Identity(
        date_of_birth=dob.isoformat(),
        callback_number=number,
        insurance_plan=rng.choice(_INSURANCE_PLANS),
    )


def build_secondary_asks(seed: PersonaSeed) -> list[str]:
    """The call's agenda: every template ask, plus shared-pool asks drawn
    deterministically from the eligible set. The persona's hidden objective
    stays primary — asks are the 'one more thing…' layer that stretches a
    call into a real multi-minute conversation."""
    rng = random.Random(derive_seed(seed.seed, "agenda"))
    asks = [_fill(ask, seed) for ask in seed.template.secondary_asks]
    pool = list(eligible_asks(seed.template, age=seed.age, has_meds=bool(seed.cluster.medications)))
    want = _TARGET_TOTAL_ASKS - len(asks)
    if pool and want > 0:
        rng.shuffle(pool)
        asks.extend(_fill(ask.text, seed) for ask in pool[:want])
    return asks[:_TARGET_TOTAL_ASKS]


class TemplateElaborator:
    """Deterministic offline elaboration — reproducible without any API key."""

    def elaborate(self, seed: PersonaSeed) -> ManifestEntry:
        rng = random.Random(derive_seed(seed.seed, "elaborate"))
        cluster = seed.cluster
        meds = list(cluster.medications)

        background = ", ".join(
            [
                " and ".join(cluster.conditions),
                f"{cluster.cadence} with {cluster.specialty}",
                f"takes {' and '.join(meds)}" if meds else "takes no regular medication",
                rng.choice(_SCHEDULING_PREFERENCES),
            ]
        )
        if seed.age >= 70 and rng.random() < 0.6:
            support = rng.choice(_ELDERLY_SUPPORT)
            try:
                support = support.format(_PRONOUN[seed.gender])
            except (KeyError, IndexError):
                pass
            background += f", {support}"

        disposition = _AFFECT_DISPOSITIONS.get(seed.template.type, seed.disposition)
        style = _STYLE_VOICE.get(seed.phrasing.style, seed.phrasing.style.replace("_", " "))
        speaking_style = f"{disposition}; {style}"
        if seed.age_band == "elderly":
            speaking_style += ", speaks at a slower pace"
        if seed.language_tag != "english":
            speaking_style += ", comfortable in Spanish and English"

        noun = {"female": "woman", "male": "man", "nonbinary": "person"}[seed.gender]
        origin = f"{seed.heritage} " if seed.heritage else ""
        trait = _VOICE_TRAIT.get(disposition, "natural and unforced")
        voice_bits = [disposition]
        if seed.age_band == "elderly" and rng.random() < 0.4:
            voice_bits.append("slightly hard of hearing")
        voice_bits.append(trait)
        design_prompt = (
            f"{_AGE_ADJECTIVE.get(seed.age_band, 'adult')} {origin}{noun}, " + ", ".join(voice_bits)
        )

        red_team = red_team_plan(seed.template.type)
        return ManifestEntry(
            call_id=f"call-{seed.index + 1:03d}",
            persona=Persona(
                name=seed.name,
                age=seed.age,
                gender=seed.gender,  # type: ignore[arg-type]  (Literal; sampler emits valid ids)
                language=seed.language,
                voice=VoiceProfile(design_prompt=design_prompt),
                background=background,
                speaking_style=speaking_style,
                medications=meds,
                identity=build_identity(seed),
            ),
            objective=Objective(
                type=seed.template.type,
                goal=_fill(seed.template.goal, seed),
                hidden_context=_fill(seed.template.hidden_context, seed),
                curveballs=[c.model_copy() for c in seed.template.curveballs],
                secondary_asks=build_secondary_asks(seed),
                success_criteria=list(seed.template.success_criteria),
                adversarial=AdversarialPlan(
                    techniques=list(red_team.techniques),
                    hypothesis=_fill(red_team.hypothesis, seed),
                    escalation_steps=[_fill(step, seed) for step in red_team.escalation_steps],
                    safety_boundary=_fill(red_team.safety_boundary, seed),
                ),
                termination=seed.template.termination,
            ),
            seed=seed.seed,
            generated_at=seed.generated_on.isoformat(),
            elaboration="template",
        )


class _ElaboratedFields(BaseModel):
    """What the LLM is allowed to write. Everything else comes from the template."""

    background: str = ""
    speaking_style: str = ""
    voice_design_prompt: str = ""
    goal: str = ""
    hidden_context: str = ""
    curveballs: list[Curveball] = Field(default_factory=list)
    secondary_asks: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)


_SYSTEM_PROMPT = (
    "You write fictional patient personas for load-testing a medical scheduling "
    "voice agent. Every character is synthetic; never reference real people. "
    "Reply with a single JSON object and nothing else."
)


def _user_prompt(seed: PersonaSeed) -> str:
    template = seed.template
    payload = {
        "persona_facts": {
            "name": seed.name,
            "age": seed.age,
            "gender": seed.gender,
            "language": seed.language,
            "heritage": seed.heritage,
            "disposition": _AFFECT_DISPOSITIONS.get(template.type, seed.disposition),
        },
        "condition_cluster": {
            "conditions": list(seed.cluster.conditions),
            "allowed_medications": list(seed.cluster.medications),
            "specialty": seed.cluster.specialty,
            "visit_cadence": seed.cluster.cadence,
        },
        "phrasing_example": seed.phrasing.text,
        "phrasing_style": seed.phrasing.style,
        "objective_template": {
            "type": template.type,
            "goal_hint": _fill(template.goal, seed),
            "hidden_context_hint": _fill(template.hidden_context, seed),
            "curveballs": [c.model_dump() for c in template.curveballs],
        },
        "secondary_asks_hint": build_secondary_asks(seed),
    }
    rules = [
        "Keep every medical fact from persona_facts/condition_cluster. "
        "Use medications ONLY from allowed_medications, exactly as spelled.",
        "background: 1-2 sentences in a lay patient's voice; include the "
        "conditions, the visit cadence, and one scheduling preference.",
        "speaking_style: one concrete sentence a voice director could act on.",
        "goal: same intent as goal_hint, personalized; relative dates only "
        "('next week'), never absolute dates like 2026-08-20.",
        "curveballs: keep the template's curveballs, same 'at' values; you may "
        "reword 'action' to fit the persona.",
        "secondary_asks: rewrite each item of secondary_asks_hint in this "
        "patient's voice as a directive ('before hanging up, ask …'). Keep "
        "EVERY item — same count, same intent — the call needs the full agenda.",
        "voice_design_prompt: one phrase — age, gender, accent/background, "
        "disposition, speech texture. Example: 'elderly Cuban-American woman, "
        "warm, slightly deaf, talks around the point'.",
        "JSON keys: background, speaking_style, voice_design_prompt, "
        "medications, goal, hidden_context, curveballs, secondary_asks.",
    ]
    return "Seed data:\n" + json.dumps(payload, indent=2) + "\n\nRules:\n" + "\n".join(
        f"- {rule}" for rule in rules
    )


class LlmElaborator:
    """Rewrites the template prose in-character via one Cerebras completion.

    A fresh :class:`TemplateElaborator` entry is computed first and serves as
    both the merge baseline and the guardrail: ``type``, ``success_criteria``
    and ``termination`` always survive verbatim.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = CEREBRAS_BASE_URL,
        client: Any = None,
        temperature: float = 0.8,
        baseline: Elaborator | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self.model = model
        self.temperature = temperature
        self.baseline: Elaborator = baseline or TemplateElaborator()

    def elaborate(self, seed: PersonaSeed) -> ManifestEntry:
        entry = self.baseline.elaborate(seed)
        try:
            fields = _ElaboratedFields.model_validate(self._complete(seed))
        except (ValidationError, ValueError, KeyError, IndexError, AttributeError) as exc:
            raise ElaborationError(f"LLM elaboration failed for {seed.name}: {exc}") from exc

        entry.persona.background = fields.background or entry.persona.background
        entry.persona.speaking_style = fields.speaking_style or entry.persona.speaking_style
        entry.persona.voice = VoiceProfile(
            design_prompt=fields.voice_design_prompt or entry.persona.voice.design_prompt,
            voice_id=entry.persona.voice.voice_id,
        )
        if fields.medications:
            entry.persona.medications = fields.medications
        if fields.goal:
            entry.objective.goal = fields.goal
        if fields.hidden_context:
            entry.objective.hidden_context = fields.hidden_context
        if fields.curveballs:
            entry.objective.curveballs = fields.curveballs
        if len(fields.secondary_asks) >= len(entry.objective.secondary_asks):
            entry.objective.secondary_asks = fields.secondary_asks
        entry.elaboration = "llm"
        return entry

    def _complete(self, seed: PersonaSeed) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(seed)},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
            max_completion_tokens=900,
            extra_body={"reasoning_effort": "low"},  # DESIGN §3.2: pin effort to low
        )
        content = response.choices[0].message.content or ""
        return json.loads(_strip_code_fence(content))


def _strip_code_fence(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    return fenced.group(1) if fenced else text
