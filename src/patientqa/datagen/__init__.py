"""Synthetic data pipeline (DESIGN.md §6-§7): personas + stress-test objectives.

Offline, deterministic, validated:

    seeded RNG sampling  ->  elaboration (template or LLM)  ->  post-validation
    ->  manifest.jsonl  ->  (later) the call orchestrator

Quick start::

    uv run python -m patientqa.datagen generate --count 60 --out manifest.jsonl
    uv run python -m patientqa.datagen validate manifest.jsonl

See ``src/patientqa/datagen/README.md`` for the full guide.
"""

from patientqa.datagen.elaborate import ElaborationError, LlmElaborator, TemplateElaborator
from patientqa.datagen.pipeline import (
    DropRecord,
    GenerationReport,
    PipelineConfig,
    allocate_classes,
    generate_manifest,
)
from patientqa.datagen.sampling import PersonaSeed, Sampler, derive_seed
from patientqa.datagen.schemas import (
    AdversarialPlan,
    Curveball,
    ManifestEntry,
    Objective,
    Persona,
    Starter,
    StarterSet,
    VoiceProfile,
    parse_manifest_line,
    parse_starters_line,
)
from patientqa.datagen.seeds import ConditionCluster, PhrasingSnippet, SeedBank
from patientqa.datagen.starters import (
    StarterError,
    StartersConfig,
    StartersReport,
    TemplateStarterGenerator,
    generate_starters,
)
from patientqa.datagen.taxonomy import (
    OBJECTIVE_CLASSES,
    RED_TEAM_PLANS,
    REGISTERED_TECHNIQUES,
    TEMPLATES,
    ObjectiveClass,
    ObjectiveTemplate,
    RedTeamPlan,
    red_team_plan,
    template_by_type,
)
from patientqa.datagen.validate import (
    validate_entry,
    validate_manifest,
    validate_starter_set,
)

__all__ = [
    "AdversarialPlan",
    "ConditionCluster",
    "Curveball",
    "DropRecord",
    "ElaborationError",
    "GenerationReport",
    "LlmElaborator",
    "ManifestEntry",
    "OBJECTIVE_CLASSES",
    "RED_TEAM_PLANS",
    "REGISTERED_TECHNIQUES",
    "Objective",
    "ObjectiveClass",
    "ObjectiveTemplate",
    "Persona",
    "PersonaSeed",
    "PhrasingSnippet",
    "PipelineConfig",
    "RedTeamPlan",
    "Sampler",
    "SeedBank",
    "Starter",
    "StarterError",
    "StarterSet",
    "StartersConfig",
    "StartersReport",
    "TEMPLATES",
    "TemplateElaborator",
    "TemplateStarterGenerator",
    "VoiceProfile",
    "allocate_classes",
    "derive_seed",
    "generate_manifest",
    "generate_starters",
    "parse_manifest_line",
    "parse_starters_line",
    "red_team_plan",
    "template_by_type",
    "validate_entry",
    "validate_manifest",
    "validate_starter_set",
]
