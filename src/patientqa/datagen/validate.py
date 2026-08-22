"""Post-validation rules (DESIGN §6.3) — the gate between elaboration and the manifest.

Distinct from the Pydantic schemas: a line that parses may still be *bad*.
Rules (all machine-checkable, all reported with the offending ``call_id``):

- ``objective_type_registered`` — objective.type must exist in the taxonomy.
- ``objective_not_empty`` — goal, success criteria, and termination present.
- ``identity_consistent`` — DOB parses, is in the past, and its year matches
  the persona's age (±1); callback number is a plausible US E.164 mobile;
  insurance plan named. Identity is optional (older manifests) but never
  relative-date-scanned — a DOB is an absolute past date by definition.
- ``agenda_substantial`` — exactly 4 secondary asks, each a speakable directive
  (10-200 chars), no leftover ``{placeholders}``, no absolute dates.
- ``dates_in_future`` — any explicit ISO date (YYYY-MM-DD) must not be in the
  past. Personas are generated with *relative* dates by design ("next week");
  an absolute date slipping in past-ward is a generation bug.
- ``plausible_medications`` — every listed drug must be in the bundled lexicon.
- ``no_duplicate_names`` — one persona name may appear at most once per manifest.
- ``voice_mapping_present`` — provider set and a usable design prompt (>= 20
  chars) or a concrete voice_id.
- ``adversarial_plan_complete`` — the test hypothesis, registered techniques,
  2–4 escalation steps, and authorization boundary are present and concrete.
- ``persona_not_thin`` — background/speaking style long enough to act on
  (rejects empty/contradictory-feeling personas, DESIGN §9).
- ``call_id_unique`` — call ids unique within the manifest.

Starter-artifact rules (``validate_starter_set``, keyed to a manifest entry):

- ``starter_call_id_match`` — the set belongs to the entry it is checked against.
- ``starters_plentiful`` — at least 2 candidates to actually choose from.
- ``starter_not_thin`` — each opener is 12-300 chars (a speakable phone line).
- ``starters_distinct`` — no duplicate opener texts.
- ``starter_relative_dates`` — openers use relative dates only, like the rest
  of the manifest; any absolute ISO date is a generation bug.
- ``starter_no_hidden_context`` — an opener must not quote the objective's
  hidden context or curveball actions; those surface mid-call, if at all.
- ``starters_in_english`` — openers are English (the test line is graded on
  coherent English conversation). Heuristic guardrail, not a translator: >= 2
  unambiguous Spanish marker words flags the opener; a single greeting word
  of persona flavor ("Hola, ...") passes.
- ``starter_no_identity_facts`` — no digit runs (phone/DOB-shaped): identity
  facts belong to the brain's persona card, not the opener.
- ``starter_no_tactic_leak`` — none of the giveaway words ("transposed",
  "garbled", "testing you", ...) that announce a probe's trick in advance.
"""

from __future__ import annotations

import re
from datetime import date

from patientqa.datagen.schemas import ManifestEntry, StarterSet

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_WORD = re.compile(r"[a-záéíóúñü]+")
_LEFTOVER_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
_US_PHONE = re.compile(r"^\+1\d{10}$")
_DIGIT_RUN = re.compile(r"\d{7,}")
_TACTIC_WORDS = (
    "transposed",
    "transposing",
    "garbled",
    "mispronounce",
    "mispronounced",
    "made-up name",
    "made up name",
    "doesn't exist",
    "does not exist",
    "not real doctor",
    "fictional",
    "testing you",
    "test you",
    "this is a test",
    "skip verification",
    "without the usual verification",
    "without verification",
    "quality reviewer",
    "quality assurance",
    "clinic it",
    "authorized qa",
)

_MIN_DESIGN_PROMPT = 20
_MIN_BACKGROUND = 40
_MIN_SPEAKING_STYLE = 15
_MIN_STARTER = 12
_MAX_STARTER = 300
_MIN_LEAK_PHRASE = 10  # shorter hidden phrases are too generic to police by containment
_SPANISH_MARKER_LIMIT = 2  # one greeting word of flavor is fine; a Spanish sentence is not
_MIN_SECONDARY_ASK = 10
_MAX_SECONDARY_ASK = 200

# Unambiguous Spanish tokens only — every word here is also spelled the same
# as some English word is excluded (no "mi", "si", "no", "doctor", ...).
_SPANISH_MARKERS = frozenset(
    """
    hola buenos buenas dias días tardes noches soy necesito necesitaria
    necesitaría quiero gustaria gustaría preferiria preferiría podria podría
    pueden podrian podrían gracias por para pero porque cuando cómo qué dónde
    quién muy tambien también ahora aqui aquí mañana semana semanas proximo
    próximo proxima próxima cita citas medico médico medicina salud receta
    recetas llamada llamar llame habla hablar estoy esta está tengo tiene
    visita visitas temprano lunes martes miercoles miércoles jueves viernes
    sabado sábado domingo agendar programar confirmar ayudar ayuda turno
    farmacia seguro aceptan acepta antes despues después entonces
    """.split()
)


def validate_entry(
    entry: ManifestEntry, *, today: date, drug_lexicon: frozenset[str] | set[str]
) -> list[str]:
    """Return human-readable violations for one entry ([] means clean)."""
    violations: list[str] = []

    if not _type_registered(entry.objective.type):
        violations.append(f"objective_type_registered: unknown type {entry.objective.type!r}")

    if not entry.objective.goal.strip() or not entry.objective.success_criteria:
        violations.append("objective_not_empty: goal or success_criteria is empty")
    if not entry.objective.termination.strip():
        violations.append("objective_not_empty: termination is empty")

    for year, month, day in _ISO_DATE.findall(_entry_text(entry)):
        try:
            if date(int(year), int(month), int(day)) < today:
                violations.append(f"dates_in_future: past date {year}-{month}-{day}")
        except ValueError:
            violations.append(f"dates_in_future: invalid date {year}-{month}-{day}")

    violations.extend(_identity_violations(entry, today))
    violations.extend(_agenda_violations(entry))
    violations.extend(_adversarial_violations(entry))

    for drug in entry.persona.medications:
        if drug.strip().lower() not in drug_lexicon:
            violations.append(f"plausible_medications: {drug!r} not in drug lexicon")

    if not entry.persona.voice.design_prompt and not entry.persona.voice.voice_id:
        violations.append("voice_mapping_present: neither design_prompt nor voice_id set")
    elif not entry.persona.voice.voice_id and len(entry.persona.voice.design_prompt or "") < (
        _MIN_DESIGN_PROMPT
    ):
        violations.append(
            f"voice_mapping_present: design_prompt shorter than {_MIN_DESIGN_PROMPT} chars"
        )

    if len(entry.persona.background.strip()) < _MIN_BACKGROUND:
        violations.append(f"persona_not_thin: background shorter than {_MIN_BACKGROUND} chars")
    if len(entry.persona.speaking_style.strip()) < _MIN_SPEAKING_STYLE:
        violations.append("persona_not_thin: speaking_style too short to act on")

    return violations


def validate_manifest(
    entries: list[ManifestEntry], *, today: date, drug_lexicon: frozenset[str] | set[str]
) -> dict[str, list[str]]:
    """Validate every entry plus the manifest-level rules (duplicate names/ids)."""
    report: dict[str, list[str]] = {}

    seen_names: dict[str, str] = {}
    seen_ids: set[str] = set()
    for entry in entries:
        violations = validate_entry(entry, today=today, drug_lexicon=drug_lexicon)

        name_key = entry.persona.name.lower()
        if name_key in seen_names:
            violations.append(
                f"no_duplicate_names: {entry.persona.name!r} already used by {seen_names[name_key]}"
            )
        else:
            seen_names[name_key] = entry.call_id

        if entry.call_id in seen_ids:
            violations.append(f"call_id_unique: {entry.call_id!r} appears more than once")
        seen_ids.add(entry.call_id)

        if violations:
            report[entry.call_id] = violations
    return report


def validate_starter_set(entry: ManifestEntry, starter_set: StarterSet) -> list[str]:
    """Return human-readable violations for one entry's starter set ([] = clean)."""
    violations: list[str] = []

    if starter_set.call_id != entry.call_id:
        violations.append(
            f"starter_call_id_match: set is for {starter_set.call_id!r}, "
            f"entry is {entry.call_id!r}"
        )

    texts = [starter.text for starter in starter_set.starters]
    if len(texts) < 2:
        violations.append("starters_plentiful: fewer than 2 candidates to choose from")

    for text in texts:
        if not _MIN_STARTER <= len(text) <= _MAX_STARTER:
            violations.append(
                f"starter_not_thin: opener of {len(text)} chars outside "
                f"{_MIN_STARTER}-{_MAX_STARTER}"
            )
    lowered = [text.lower() for text in texts]
    if len(set(lowered)) != len(lowered):
        violations.append("starters_distinct: duplicate opener text")

    for text in texts:
        if _ISO_DATE.search(text):
            violations.append(f"starter_relative_dates: absolute date in opener {text[:60]!r}")

    for text in texts:
        if _spanish_marker_count(text) >= _SPANISH_MARKER_LIMIT:
            violations.append(f"starters_in_english: opener is not English: {text[:60]!r}")

    for text in texts:
        if _DIGIT_RUN.search(text):
            violations.append(
                f"starter_no_identity_facts: digit run in opener {text[:60]!r}"
            )
        lowered_text = text.lower()
        for word in _TACTIC_WORDS:
            if word in lowered_text:
                violations.append(
                    f"starter_no_tactic_leak: opener announces the trick "
                    f"({word!r}): {text[:60]!r}"
                )
                break

    for text in lowered:
        for secret in _starter_secrets(entry):
            if secret in text:
                violations.append(f"starter_no_hidden_context: opener reveals {secret[:60]!r}")

    return violations


def _identity_violations(entry: ManifestEntry, today: date) -> list[str]:
    """Verification-fact checks; identity is optional for legacy manifests."""
    identity = entry.persona.identity
    if identity is None:
        return []
    violations: list[str] = []
    try:
        dob = date.fromisoformat(identity.date_of_birth)
    except ValueError:
        return [f"identity_consistent: DOB {identity.date_of_birth!r} is not an ISO date"]
    if dob >= today:
        violations.append(f"identity_consistent: DOB {dob} is not in the past")
    if abs((today.year - dob.year) - entry.persona.age) > 1:
        violations.append(
            f"identity_consistent: DOB year {dob.year} does not match age {entry.persona.age}"
        )
    if not _US_PHONE.match(identity.callback_number):
        violations.append(
            f"identity_consistent: callback number {identity.callback_number!r} is not US E.164"
        )
    return violations


def _agenda_violations(entry: ManifestEntry) -> list[str]:
    """The secondary-ask agenda must be substantial and speakable."""
    asks = entry.objective.secondary_asks
    if len(asks) != 4:
        return [
            "agenda_substantial: expected exactly 4 secondary asks, "
            f"found {len(asks)} — calls would not fill their 2-3 minute budget"
        ]
    violations: list[str] = []
    for ask in asks:
        if not _MIN_SECONDARY_ASK <= len(ask) <= _MAX_SECONDARY_ASK:
            violations.append(
                f"agenda_substantial: ask of {len(ask)} chars outside "
                f"{_MIN_SECONDARY_ASK}-{_MAX_SECONDARY_ASK}: {ask[:60]!r}"
            )
        if _LEFTOVER_PLACEHOLDER.search(ask):
            violations.append(f"agenda_substantial: unfilled placeholder in {ask[:60]!r}")
        if _ISO_DATE.search(ask):
            violations.append(f"agenda_substantial: absolute date in {ask[:60]!r}")
    return violations


def _adversarial_violations(entry: ManifestEntry) -> list[str]:
    """The red-team plan must be registered, concrete, and bounded."""
    from patientqa.datagen.taxonomy import REGISTERED_TECHNIQUES

    plan = entry.objective.adversarial
    violations: list[str] = []
    unknown = sorted(set(plan.techniques) - REGISTERED_TECHNIQUES)
    if unknown:
        violations.append(
            "adversarial_plan_complete: unknown technique(s): " + ", ".join(unknown)
        )
    texts = [plan.hypothesis, plan.safety_boundary, *plan.escalation_steps]
    for text in texts:
        if _LEFTOVER_PLACEHOLDER.search(text):
            violations.append(
                f"adversarial_plan_complete: unfilled placeholder in {text[:60]!r}"
            )
        if _ISO_DATE.search(text):
            violations.append(
                f"adversarial_plan_complete: absolute date in {text[:60]!r}"
            )
    return violations


def _spanish_marker_count(text: str) -> int:
    """How many unambiguous Spanish words the text contains (lowercased)."""
    return sum(1 for token in _WORD.findall(text.lower()) if token in _SPANISH_MARKERS)


def _starter_secrets(entry: ManifestEntry) -> list[str]:
    """Phrases (lowercased) an opener must not contain: hidden context + curveballs."""
    secrets = []
    hidden = entry.objective.hidden_context.strip()
    if len(hidden) >= _MIN_LEAK_PHRASE:
        secrets.append(hidden.lower())
    secrets.extend(
        curveball.action.strip().lower()
        for curveball in entry.objective.curveballs
        if len(curveball.action.strip()) >= _MIN_LEAK_PHRASE
    )
    return secrets


def _entry_text(entry: ManifestEntry) -> str:
    adversarial = entry.objective.adversarial
    parts = [
        entry.persona.background,
        entry.persona.speaking_style,
        entry.objective.goal,
        entry.objective.hidden_context,
        *[f"{c.at} {c.action}" for c in entry.objective.curveballs],
        adversarial.hypothesis,
        adversarial.safety_boundary,
        *adversarial.escalation_steps,
    ]
    return " ".join(parts)


def _type_registered(type_id: str) -> bool:
    from patientqa.datagen.taxonomy import TEMPLATES

    return any(t.type == type_id for t in TEMPLATES)
