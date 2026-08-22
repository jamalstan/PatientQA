# Synthetic Data Generation (`patientqa.datagen`)

Implements **DESIGN.md §6** (persona & scenario pipeline, §6.5 conversation
starters) and **§7** (objective taxonomy). Everything for this stage lives in
this subpackage:

```
seeded RNG sampling  →  elaboration (template or LLM)  →  post-validation
                     →  manifest.jsonl  →  starters.jsonl  →  (later) the call orchestrator
```

```
patientqa/datagen/
├── schemas.py     Pydantic models for Persona / Objective / ManifestEntry / Starter
├── taxonomy.py    §7 stress-test classes + 24 objective templates (incl. FAQ)
├── sampling.py    deterministic demographics → cluster → phrasing → template
├── seeds.py       seed loading: bundled JSON (default) or Hugging Face (optional)
├── seed_data/     conditions.json, phrasing.jsonl, drugs.json, names.json
├── elaborate.py   TemplateElaborator (offline) + LlmElaborator (Cerebras)
├── starters.py    conversation starters per entry (template + LLM, DESIGN §6.5)
├── validate.py    post-validation rules (dates, drugs, names, voice, starters, ...)
├── pipeline.py    orchestration, retry/drop policy, atomic manifest writing
└── cli.py         python -m patientqa.datagen {generate,starters,validate,taxonomy}
```

## Usage

```bash
# full 60-persona campaign manifest (LLM prose if a Cerebras key is configured,
# deterministic template prose otherwise)
uv run python -m patientqa.datagen generate --count 60 --seed 2026 --out manifest.jsonl

# fully offline, byte-for-byte reproducible
uv run python -m patientqa.datagen generate --count 60 --elaboration template

# conversation starters: persona background + patient query (goal) as LLM
# context → candidate opening lines per manifest entry
uv run python -m patientqa.datagen starters manifest.jsonl --limit 8

# check an existing manifest against all post-validation rules
uv run python -m patientqa.datagen validate manifest.jsonl

# inspect the stress-test taxonomy
uv run python -m patientqa.datagen taxonomy

# curated one-factor-at-a-time campaign from the 2026 voice-agent research matrix
uv run python -m patientqa.datagen research10 --out manifests/research10.jsonl
```

`research10` writes the manifest, template starters, and coverage reports for ten fixed probes:
clean scheduling, one self-correction, two five-second pauses, barge-in, backchannels, a distinct
second speaker, deterministic low road noise around identity digits, Spanish/English switching,
cancel/reschedule rollback, and verification pressure. Transport behaviors execute in the live
call loop rather than relying on the patient LLM, and emit `behavior.fired` timeline events.

Each run also writes a `<artifact>.report.json` sidecar (class, intentional-behavior, and
adversarial-technique coverage, elaboration mode counts, template fallbacks, drop reasons).

## Manifest format

One JSON object per line; the shape follows the DESIGN.md §6.3 example, plus
provenance fields (`seed`, `generated_at`, `elaboration`) and a structured
`persona.medications` list used by the plausible-drugs rule:

```json
{
  "call_id": "call-014",
  "persona": {
    "name": "Marta Reyes", "age": 71, "gender": "female",
    "language": "English w/ Spanish code-switching",
    "voice": {"provider": "elevenlabs", "design_prompt": "elderly Cuban-American woman, warm, ..."},
    "background": "type 2 diabetes and high blood pressure, quarterly check-ins with endocrinology, ...",
    "speaking_style": "warm and chatty; sprinkles Spanish into English, especially for drug names",
    "medications": ["metformin", "lisinopril"]
  },
  "objective": {
    "type": "reschedule_edge",
    "goal": "Move next week's endocrinology visit; does not remember exact date",
    "hidden_context": "...",
    "curveballs": [{"at": "after agent proposes slot", "action": "counter with a Sunday request"}],
    "secondary_asks": ["ask what to bring", "ask about parking", "ask how early to arrive", "request a final read-back"],
    "success_criteria": ["no Sunday confirmation", "existing appointment correctly located"],
    "adversarial": {
      "techniques": ["temporal_ambiguity", "state_confusion"],
      "hypothesis": "Date ambiguity plus a counteroffer may corrupt appointment state.",
      "escalation_steps": [
        "Ask to move next week's visit without naming its current date.",
        "Counter the proposed slot, then require a concrete final read-back."
      ],
      "safety_boundary": "Use only fictional assessment data; never seek real records or claim an emergency."
    },
    "termination": "goal achieved OR agent fails twice OR 180s"
  },
  "seed": 842397..., "generated_at": "2026-08-17", "elaboration": "template",
  "test_intent": {
    "intentional": true,
    "behavior": "reschedule_edge",
    "isolation": "single_behavior",
    "hypothesis": "Date ambiguity plus a counteroffer may corrupt appointment state.",
    "protocol": "introduce_only_this_behavior"
  }
}
```

`test_intent` is deterministic and never LLM-authored. There is exactly one behavior under test
per call (`objective.type`); `adversarial.techniques` are risk-taxonomy mappings, not additional
caller behaviors. The campaign runner promotes this block to the top level of both `meta.json`
and `call.json`, making it explicit that the behavior was deliberate.

`voice.voice_id` starts empty — the Voice Design step (§3.3) fills it in later;
the validator requires at least one of `design_prompt` / `voice_id`.

## Conversation starters

The manifest's two prose fields double as generation context for the call's
first patient utterance: `persona.background` (who is calling) +
`objective.goal` (the patient query). `starters` turns that pair into N
candidate opening lines — a separate JSONL artifact joined to the manifest by
`call_id`, regenerable without touching the manifest:

```json
{
  "call_id": "call-014",
  "starters": [
    {"angle": "direct", "text": "Hi, I need to move my endocrinology appointment to next week."},
    {"angle": "confused", "text": "Hi, sorry — I had a visit coming up, and I think I need a different day?"},
    {"angle": "small talk", "text": "Hi, good morning! My daughter drives me Tuesdays, so I hoped to move my visit."}
  ],
  "elaboration": "llm", "model": "gpt-oss-120b", "generated_at": "2026-08-17"
}
```

Same two-tier guardrail as persona elaboration: `TemplateStarterGenerator`
frames the goal text deterministically (offline fallback, goal surgery strips
directorial clauses after `;`), and `LlmStarterGenerator` writes in-character
candidates via one Cerebras completion per entry (background + query +
speaking style as context; grading rules never enter the prompt, and the LLM
is told not to reveal `hidden_context` or fire curveballs in the opener).
Every set is validated (`starter_*` rules: ≥2 distinct candidates, speakable
length, relative dates only, **English only** — one Spanish greeting word of
persona flavor is tolerated, a Spanish sentence is rejected — and no
hidden-context/curveball leaks); any LLM or validation failure falls back to
template starters per entry, recorded in the
`<manifest>.starters.report.json` sidecar.

## How the layers fit together

- **Determinism**: every persona is a pure function of `(base_seed, index,
  attempt)` via blake2b-derived seeds (`sampling.derive_seed`). Same seed →
  byte-identical manifest (tested). `--seed` is the only knob you need to
  regenerate a fresh-but-reproducible campaign.
- **Diversity → realism → intent**: age band/gender/language/disposition are
  sampled first; the condition cluster (age/gender appropriate, real drugs) and
  phrasing style condition on them; the objective template comes from the §7
  taxonomy with per-template constraints (e.g. `very_elderly_slow` forces 78+,
  `spanish_heavy_call` forces a Spanish-influenced persona).
- **Coverage**: class allocation across slots is stratified by §7 class weight
  (every class guaranteed ≥ 1 slot, no two adjacent calls in the same class).
  The sidecar separately counts every adversarial technique exercised.
- **Elaboration**: `TemplateElaborator` composes prose offline. If a Cerebras
  key is present, `LlmElaborator` rewrites it in-character with one small
  gpt-oss-120b call (`reasoning_effort: low`, JSON mode) — but `type`,
  `success_criteria`, `adversarial`, and `termination` are always copied verbatim from the
  taxonomy, so the LLM can never invent grading rules. LLM failures fall back
  to template prose; validation failures redraw the persona (bounded retries),
  then drop the slot with a recorded reason — a bad entry never kills the batch.
- **Validation** (`validate.py`): unknown objective types, past/impossible ISO
  dates, medications outside the bundled drug lexicon, duplicate persona names,
  missing voice mapping, thin personas, duplicate call ids, inconsistent
  identity facts, an agenda with anything other than four follow-ups, or an
  incomplete/unregistered adversarial plan.

## Seed data & licensing

The bundled files in `seed_data/` are **original synthetic content** written
for this project, in the style of the surveyed HF datasets (DESIGN §6.2):
condition clusters à la PMC-Patients, patient phrasing à la MedDialog. Nothing
is vendored from those datasets, so there is no license exposure.

To enrich from the real datasets at runtime (nothing is committed):

```bash
uv add datasets   # optional extra, not a default dependency
uv run python -m simulator.datagen generate --seed-source hf ...
```

This streams a few hundred rows (`zhengyun21/PMC-Patients`,
`UCSD26/medical_dialog`) and only mines them for runtime texture; the loader is
experimental and guarded. All generated personas are fictional characters
regardless of source.

## Extending

- **New probe**: add an `ObjectiveTemplate` and matching `RED_TEAM_PLANS` entry in `taxonomy.py` (type, goal with
  `{specialty}`/`{medication}`/`{condition}` placeholders, success criteria,
  termination, adversarial hypothesis/techniques/escalation/boundary, optional age/language constraints). Registry-integrity tests
  will keep it honest.
- **New patient population**: add a cluster to `seed_data/conditions.json` —
  medications must exist in `drugs.json` (enforced by tests).
- **New drug names**: extend `seed_data/drugs.json`; the lexicon is the single
  source of truth for the plausible-drugs rule.
