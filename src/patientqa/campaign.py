"""The campaign runner — manifest.jsonl → a sequence of live calls (DESIGN §10).

One command places every pending manifest entry as a real call to the
allowlisted test number:

    uv run python -m patientqa --campaign manifest.jsonl

For each entry it builds the persona prompt (``patientqa.prompting``), casts a
voice (``patientqa.voicing``), picks the opening line from the starters
artifact (deterministically — the entry's seed decides, so a rerun casts the
same opener), and runs one call through the loop. Calls whose session folder
already has a transcript are skipped, so an interrupted campaign resumes
where it stopped. A cloudflared quick tunnel (``patientqa.tunnel``) is
started once and shared by every call unless ``--stream-url`` points at an
existing bridge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from patientqa.datagen.schemas import ManifestEntry, StarterSet, parse_manifest_line
from patientqa.datagen.starters import TemplateStarterGenerator
from patientqa.prompting import build_identity_responder, build_persona_prompt
from patientqa.tunnel import Tunnel, start_tunnel
from patientqa.voicing import VoiceCaster, fallback_voice

log = logging.getLogger(__name__)

MIN_GOOD_DURATION_S = 60.0
MIN_GOOD_TURNS_PER_SIDE = 4
BAD_END_REASONS = {"dial_failed", "ring_timeout", "stt_failed", "unknown"}

_BARGE_IN_LINES = (
    "Sorry to interrupt — was that a morning appointment?",
    "Wait, before you continue, could you repeat the date?",
)
_BACKCHANNEL_LINES = ("Mm-hmm.", "Right.")
_THIRD_PARTY_LINES = ("Friday afternoon would work better for us.",)
_THIRD_PARTY_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # account voice Sarah; distinct speaker


@dataclass(frozen=True)
class BehaviorRuntime:
    """Transport controls that make an intentional behavior real on the wire."""

    delivery: str = "patient_brain"
    barge_lines: tuple[str, ...] = ()
    barge_kind: str = "barge_in"
    barge_voice_id: str | None = None
    barge_skip_speeches: int = 0
    response_delays_s: tuple[float, ...] = ()
    audio_transform: Callable[[bytes], bytes] | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "delivery": self.delivery,
            "scripted_utterances": len(self.barge_lines),
            "response_delays_s": list(self.response_delays_s),
            "voice_override": self.barge_voice_id is not None,
            "skip_agent_speech_segments": self.barge_skip_speeches,
            "audio_transform": self.audio_transform is not None,
        }


def behavior_runtime(entry: ManifestEntry) -> BehaviorRuntime:
    """Resolve the one declared behavior to its execution layer."""

    behavior = entry.test_intent.behavior if entry.test_intent else entry.objective.type
    if behavior == "barge_in":
        return BehaviorRuntime(delivery="call_loop", barge_lines=_BARGE_IN_LINES)
    if behavior == "backchannel_during_readback":
        return BehaviorRuntime(
            delivery="call_loop",
            barge_lines=_BACKCHANNEL_LINES,
            barge_kind="backchannel_during_readback",
            barge_skip_speeches=2,
        )
    if behavior == "third_party_interruption":
        return BehaviorRuntime(
            delivery="call_loop_second_voice",
            barge_lines=_THIRD_PARTY_LINES,
            barge_kind="third_party_interruption",
            barge_voice_id=_THIRD_PARTY_VOICE_ID,
            barge_skip_speeches=2,
        )
    if behavior == "long_silence":
        return BehaviorRuntime(delivery="call_loop", response_delays_s=(5.0, 5.0))
    if behavior == "degraded_audio_digits":
        from patientqa.audiofx import make_road_noise

        return BehaviorRuntime(
            delivery="audio_transform",
            audio_transform=make_road_noise(entry.seed),
        )
    return BehaviorRuntime()


@dataclass
class CallPlan:
    """Everything one dial needs, resolved offline before any socket opens."""

    entry: ManifestEntry
    opener: str

    @property
    def persona_prompt(self) -> str:
        return build_persona_prompt(self.entry)


def load_plans(manifest_path: Path, starters_path: Path | None = None) -> list[CallPlan]:
    """Manifest (+ optional starters artifact) → call plans, in file order."""
    entries = [
        parse_manifest_line(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    starters: dict[str, StarterSet] = {}
    if starters_path is not None and starters_path.is_file():
        from patientqa.datagen.schemas import parse_starters_line

        for line in starters_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                starter_set = parse_starters_line(line)
                starters[starter_set.call_id] = starter_set
    default_starters = TemplateStarterGenerator()
    plans: list[CallPlan] = []
    for entry in entries:
        starter_set = starters.get(entry.call_id)
        if starter_set is None:
            starter_set = default_starters.generate(entry, today=_entry_date(entry))
        rng = random.Random(entry.seed)
        opener = starter_set.starters[rng.randrange(len(starter_set.starters))].text
        plans.append(CallPlan(entry=entry, opener=opener))
    return plans


def _entry_date(entry: ManifestEntry):
    try:
        return datetime.strptime(entry.generated_at, "%Y-%m-%d").date()
    except ValueError:
        return datetime.now(timezone.utc).date()


def completed_call_ids(
    calls_root: Path, *, expected_seeds: dict[str, int] | None = None
) -> set[str]:
    """Call ids with a quality session for the same generated scenario.

    When expected seeds are supplied, old recordings with the same ``call-NNN``
    label do not accidentally satisfy resume after a manifest regeneration.
    """
    done: set[str] = set()
    if not calls_root.is_dir():
        return done
    for folder in calls_root.iterdir():
        if folder.is_dir() and (folder / "transcript.json").is_file():
            # CallSession names folders ``<call_id>_<UTC timestamp>``.  Keep
            # the complete call id (including its numeric suffix); stripping
            # ``-001`` here would turn every campaign id into plain ``call``
            # and silently defeat resume.
            try:
                if expected_seeds is not None:
                    summary = json.loads((folder / "call.json").read_text(encoding="utf-8"))
                    call_id = folder.name.rsplit("_", 1)[0]
                    if summary.get("manifest", {}).get("seed") != expected_seeds.get(call_id):
                        continue
                outcome = CallOutcome.from_session(
                    folder.name.rsplit("_", 1)[0], folder, voice=""
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if outcome.good:
                done.add(outcome.call_id)
    return done


@dataclass
class CallOutcome:
    call_id: str
    directory: str
    duration_s: float
    agent_turns: int
    patient_turns: int
    end_reason: str
    voice: str

    @property
    def quality_issues(self) -> list[str]:
        issues: list[str] = []
        if self.duration_s < MIN_GOOD_DURATION_S:
            issues.append(f"only {self.duration_s:.0f}s (need at least 60s)")
        if self.agent_turns < MIN_GOOD_TURNS_PER_SIDE:
            issues.append(f"only {self.agent_turns} agent turns (need at least 4)")
        if self.patient_turns < MIN_GOOD_TURNS_PER_SIDE:
            issues.append(f"only {self.patient_turns} patient turns (need at least 4)")
        if self.end_reason.split(":", 1)[0] in BAD_END_REASONS:
            issues.append(f"bad end reason: {self.end_reason}")
        return issues

    @property
    def good(self) -> bool:
        return not self.quality_issues

    @classmethod
    def from_session(cls, call_id: str, folder: Path, voice: str) -> CallOutcome:
        summary = json.loads((folder / "call.json").read_text(encoding="utf-8"))
        stats = summary["stats"]
        return cls(
            call_id=call_id,
            directory=folder.name,
            duration_s=round(summary["duration_ms"] / 1000, 1),
            agent_turns=stats["agent_turns"],
            patient_turns=stats["patient_turns"],
            end_reason=summary["end_reason"],
            voice=voice,
        )


@dataclass
class CampaignReport:
    started_at: str
    manifest: str
    outcomes: list[CallOutcome] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)

    def write(self, calls_root: Path) -> Path:
        stamp = self.started_at.replace(":", "").replace("-", "")[:15]
        path = calls_root / f"campaign-{stamp}.json"
        path.write_text(
            json.dumps(
                {
                    "started_at": self.started_at,
                    "manifest": self.manifest,
                    "outcomes": [o.__dict__ for o in self.outcomes],
                    "skipped": self.skipped,
                    "failed": self.failed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


async def run_campaign(
    plans: list[CallPlan],
    *,
    manifest_path: Path,
    calls_root: Path = Path("calls"),
    stream_url: str | None = None,
    max_call_s: float = 180.0,
    pause_s: float = 15.0,
    port: int = 8080,
    resume: bool = True,
    dry_run: bool = False,
    voice_caster: VoiceCaster | None = None,
    max_attempts: int = 2,
) -> CampaignReport:
    """Dial every pending plan in order; one bad call never stops the batch."""
    from patientqa.callloop import (  # late: pulls the provider SDKs
        run_call,
        validate_call_duration,
    )

    max_call_s = validate_call_duration(max_call_s)
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    report = CampaignReport(
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        manifest=str(manifest_path),
    )
    caster = voice_caster  # dry runs never build one (no client needed)

    expected_seeds = {plan.entry.call_id: plan.entry.seed for plan in plans}
    done = (
        completed_call_ids(calls_root, expected_seeds=expected_seeds) if resume else set()
    )
    pending = [p for p in plans if p.entry.call_id not in done]
    report.skipped = [p.entry.call_id for p in plans if p.entry.call_id in done]

    if dry_run:
        for plan in pending:
            intent = plan.entry.test_intent
            runtime = behavior_runtime(plan.entry)
            print(
                f"[dry-run] {plan.entry.call_id}: {plan.entry.persona.name} "
                f"({plan.entry.persona.age}) · {plan.entry.objective.type} · "
                f"intentional={intent.intentional if intent else True} · "
                f"isolation={intent.isolation if intent else 'single_behavior'} · "
                f"delivery={runtime.delivery} · "
                f"{len(plan.entry.objective.secondary_asks)} agenda items · "
                f'opener: "{plan.opener}"'
            )
        return report

    tunnel: Tunnel | None = None
    if stream_url is None:
        tunnel = await start_tunnel(port)
        stream_url = tunnel.wss_url
        log.info("campaign tunnel: %s", stream_url)

    try:
        for position, plan in enumerate(pending):
            entry = plan.entry
            cast = caster.cast(entry) if caster is not None else fallback_voice(entry)
            if cast.gender != entry.persona.gender:
                log.error(
                    "voice cast gender mismatch for %s (%s != %s); using safe fallback",
                    entry.call_id,
                    cast.gender or "unverified",
                    entry.persona.gender,
                )
                cast = fallback_voice(entry)
            voice_id = cast.voice_id
            log.info(
                "call %s (%d/%d): %s — %s — voice %s (%s/%s)",
                entry.call_id,
                position + 1,
                len(pending),
                entry.persona.name,
                entry.objective.type,
                cast.name or cast.voice_id,
                cast.gender,
                cast.origin,
            )
            outcome: CallOutcome | None = None
            last_problem = ""
            fatal_provider_failure = False
            for attempt in range(1, max_attempts + 1):
                runtime = behavior_runtime(entry)
                test_intent = entry.test_intent.model_dump() if entry.test_intent else {}
                test_intent["runtime"] = runtime.metadata()
                try:
                    folder = await run_call(
                        to_number=None,
                        stream_url=stream_url,
                        persona_prompt=plan.persona_prompt,
                        voice_id=voice_id,
                        opener_text=plan.opener,
                        max_call_s=max_call_s,
                        port=port,
                        call_id=entry.call_id,
                        calls_root=calls_root,
                        manifest={
                            "seed": entry.seed,
                            "generated_at": entry.generated_at,
                            "test_intent": test_intent,
                            "persona": entry.persona.model_dump(exclude={"identity"}),
                            "objective": entry.objective.model_dump(),
                            "identity": entry.persona.identity.model_dump()
                            if entry.persona.identity
                            else None,
                            "voice": {
                                "voice_id": voice_id,
                                "name": cast.name,
                                "origin": cast.origin,
                                "persona_gender": entry.persona.gender,
                                "voice_gender": cast.gender,
                                "gender_match": cast.gender == entry.persona.gender,
                            },
                        },
                        scripted_barge_ins=runtime.barge_lines,
                        scripted_barge_kind=runtime.barge_kind,
                        scripted_barge_voice_id=runtime.barge_voice_id,
                        scripted_barge_skip_speeches=runtime.barge_skip_speeches,
                        scripted_response_delays_s=runtime.response_delays_s,
                        outbound_audio_transform=runtime.audio_transform,
                        fixed_responder=build_identity_responder(entry),
                    )
                    candidate = CallOutcome.from_session(
                        entry.call_id, folder, cast.name or cast.voice_id
                    )
                except Exception as exc:  # a failed dial never stops the batch
                    last_problem = repr(exc)
                    log.error(
                        "call %s attempt %d/%d failed: %r",
                        entry.call_id,
                        attempt,
                        max_attempts,
                        exc,
                    )
                else:
                    if candidate.good:
                        outcome = candidate
                        break
                    last_problem = "; ".join(candidate.quality_issues)
                    fatal_provider_failure = (
                        candidate.end_reason.split(":", 1)[0] == "stt_failed"
                    )
                    log.warning(
                        "call %s attempt %d/%d not submission quality: %s",
                        entry.call_id,
                        attempt,
                        max_attempts,
                        last_problem,
                    )
                if attempt < max_attempts and pause_s:
                    await asyncio.sleep(pause_s)
            if outcome is None:
                report.failed.append({"call_id": entry.call_id, "error": last_problem})
                if fatal_provider_failure:
                    log.error("aborting campaign: STT provider failed; later calls cannot run")
                    break
                continue
            report.outcomes.append(outcome)
            print(
                f"  {outcome.call_id}: {outcome.duration_s}s · "
                f"{outcome.agent_turns} agent turns · {outcome.end_reason}"
            )
            if position < len(pending) - 1 and pause_s:
                await asyncio.sleep(pause_s)
    finally:
        if tunnel is not None:
            tunnel.stop()
    report.write(calls_root)
    return report
