"""Seed data loading (DESIGN §6.2 — HF datasets as seeds, not scripts).

Two sources:

- ``bundled`` (default): small JSON/JSONL files shipped inside the package.
  They are original synthetic content written for this project, in the style
  of the surveyed HF datasets (PMC-Patients for condition clusters, MedDialog
  for patient phrasing). No license exposure, works offline, deterministic.
- ``hf`` (experimental, needs network + ``uv add datasets``): streams a few
  hundred rows from the surveyed HF datasets to enrich the bundled pools —
  condition clusters from PubMed case summaries (``zhengyun21/PMC-Patients``)
  and phrasing from real patient utterances (``UCSD26/medical_dialog``).
  Downloaded rows are used at runtime only; nothing is vendored into the repo
  (the license caveat in DESIGN §6.2).

All personas built on top of any source are explicitly fictional characters.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SEED_DATA_DIR = Path(__file__).parent / "seed_data"


@dataclass(frozen=True)
class ConditionCluster:
    """A realistic condition bundle: the 'who' and the 'medical facts'."""

    key: str
    conditions: tuple[str, ...]
    medications: tuple[str, ...]
    specialty: str
    cadence: str
    min_age: int = 18
    max_age: int = 100
    gender: str | None = None
    weight: float = 1.0

    def supports(self, age: int, gender: str) -> bool:
        return (
            self.min_age <= age <= self.max_age and (self.gender is None or self.gender == gender)
        )


@dataclass(frozen=True)
class PhrasingSnippet:
    """A short sample of how real-ish patients phrase things (MedDialog role)."""

    style: str
    text: str
    language_tag: str = "english"


@dataclass(frozen=True)
class SeedBank:
    """Everything the sampler draws from."""

    clusters: tuple[ConditionCluster, ...]
    phrasings: tuple[PhrasingSnippet, ...]
    drug_lexicon: frozenset[str]
    names: dict[str, tuple[str, ...]]

    def lexicon_contains(self, drug: str) -> bool:
        return drug.strip().lower() in self.drug_lexicon


def load_bundled_seedbank(directory: Path = SEED_DATA_DIR) -> SeedBank:
    """Load the packaged seed files. Offline, deterministic, license-clean."""
    conditions = json.loads((directory / "conditions.json").read_text(encoding="utf-8"))
    clusters = tuple(
        ConditionCluster(
            key=row["key"],
            conditions=tuple(row["conditions"]),
            medications=tuple(row["medications"]),
            specialty=row["specialty"],
            cadence=row["cadence"],
            min_age=row.get("min_age", 18),
            max_age=row.get("max_age", 100),
            gender=row.get("gender"),
            weight=row.get("weight", 1.0),
        )
        for row in conditions["clusters"]
    )

    phrasings = tuple(
        PhrasingSnippet(
            style=line["style"],
            text=line["text"],
            language_tag=line.get("language_tag", "english"),
        )
        for line in _read_jsonl(directory / "phrasing.jsonl")
    )

    drugs = json.loads((directory / "drugs.json").read_text(encoding="utf-8"))["drugs"]

    raw_names = json.loads((directory / "names.json").read_text(encoding="utf-8"))
    names = {key: tuple(value) for key, value in raw_names.items() if key != "$comment"}

    return SeedBank(
        clusters=clusters,
        phrasings=phrasings,
        drug_lexicon=frozenset(d.lower() for d in drugs),
        names=names,
    )


def load_hf_seedbank(rows_per_dataset: int = 200, base: SeedBank | None = None) -> SeedBank:
    """Enrich the bundled seed bank with streamed HF rows (experimental).

    Requires the ``datasets`` package (``uv add datasets``) and network access.
    Rows are streamed (never fully downloaded) and only mined for texture:
    PMC-Patients rows that mention bundled conditions boost those clusters'
    weights; MedDialog patient-side utterances become extra phrasing snippets.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "seed source 'hf' needs the datasets package: uv add datasets "
            "(or use the default bundled source)"
        ) from exc

    bank = base or load_bundled_seedbank()

    weighted = {c.key: c.weight for c in bank.clusters}
    for row in _stream_rows(load_dataset, "zhengyun21/PMC-Patients", rows_per_dataset):
        text = _longest_text_column(row)
        for cluster in bank.clusters:
            if any(condition.lower() in text.lower() for condition in cluster.conditions):
                weighted[cluster.key] = cluster.weight + 0.05

    clusters = tuple(
        ConditionCluster(**{**cluster.__dict__, "weight": weighted[cluster.key]})
        for cluster in bank.clusters
    )

    extra_phrasings: list[PhrasingSnippet] = []
    seen: set[str] = set()
    for row in _stream_rows(load_dataset, "UCSD26/medical_dialog", rows_per_dataset):
        utterance = _patient_utterance(row)
        if utterance and utterance not in seen:
            seen.add(utterance)
            extra_phrasings.append(
                PhrasingSnippet(style="hf_mined", text=utterance, language_tag="english")
            )

    return SeedBank(
        clusters=clusters,
        phrasings=bank.phrasings + tuple(extra_phrasings),
        drug_lexicon=bank.drug_lexicon,
        names=bank.names,
    )


# ---- internals --------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip() and not line.startswith("{ \"$\"")]


def _stream_rows(load_dataset: Any, dataset_id: str, limit: int) -> list[dict[str, Any]]:
    """Stream ``limit`` rows without downloading the full dataset."""
    import itertools

    dataset = load_dataset(dataset_id, split="train", streaming=True)
    return [dict(row) for row in itertools.islice(iter(dataset), limit)]


def _longest_text_column(row: dict[str, Any]) -> str:
    values = [v for v in row.values() if isinstance(v, str)]
    return max(values, key=len, default="")


_DOCTOR_PREFIX = re.compile(r"^\s*(doctor|dr\.?|physician)\s*[:,-]?\s*", re.IGNORECASE)


def _patient_utterance(row: dict[str, Any]) -> str | None:
    """Best-effort extraction of a patient-side sentence from a MedDialog row.

    The dataset's exact column layout has changed across revisions, so we look
    for the longest text column and, if it is turn-formatted, keep the first
    patient-labelled turn; otherwise keep the first 1-2 sentences.
    """
    text = _longest_text_column(row).strip()
    if len(text) < 40:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith(("patient", "p:", "q:")):
            cleaned = _DOCTOR_PREFIX.sub("", line)
            cleaned = re.sub(r"^\s*(patient|p|q)\s*[:,.-]\s*", "", cleaned, flags=re.IGNORECASE)
            if len(cleaned) >= 40:
                return cleaned[:280]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    joined = " ".join(sentences[:2])
    return joined[:280] if len(joined) >= 40 else None
