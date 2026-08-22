"""Entry point: ``uv run python -m patientqa`` (or just ``patientqa``).

No arguments → smoke-check credentials (key NAMES only, never values).
``--campaign manifest.jsonl`` → run every pending manifest entry as a live
call to the allowlisted test number (a cloudflared quick tunnel is started
automatically unless ``--stream-url`` says otherwise).
``--call +1…`` → one ad-hoc live call through the same loop; requires
``--stream-url`` (the public wss:// tunnel Twilio dials back on).
"""

import argparse
import asyncio
import json
import sys

from patientqa.config import load_secrets
from patientqa.stt import PROVIDER_DEEPGRAM, PROVIDER_SCRIBE, stt_provider

REQUIRED_SECTIONS = ("elevenlabs", "cerebras", "twilio", "github")


def smoke_check() -> None:
    secrets = load_secrets()
    for section in REQUIRED_SECTIONS:
        keys = ", ".join(sorted(secrets.get(section, {}))) or "MISSING"
        print(f"{section:>12}: {keys}")
    try:
        provider = stt_provider()
    except ValueError as exc:
        print(f"         stt: INVALID — {exc}")
        return
    # Scribe rides the ElevenLabs key; Deepgram needs its own section.
    needed = "elevenlabs" if provider == PROVIDER_SCRIBE else PROVIDER_DEEPGRAM
    keys = ", ".join(sorted(secrets.get(needed, {}))) or "MISSING"
    print(f"         stt: {provider} (via {needed}: {keys})")


def _print_turn(role: str, text: str) -> None:
    prefix = "  AGENT ▸" if role == "agent" else "PATIENT ▸"
    print(f"{prefix} {text}", flush=True)


def _run_live_call(args: argparse.Namespace) -> int:
    from patientqa.callloop import run_call

    if not (args.call.startswith("+") and args.call[1:].isdigit()):
        print(f"--call must be E.164 like +15551234567, got {args.call!r}", file=sys.stderr)
        return 2
    print(f"dialing {args.call} (stream {args.stream_url}) …", flush=True)
    try:
        folder = asyncio.run(
            run_call(
                to_number=args.call,
                stream_url=args.stream_url,
                voice_id=args.voice_id,
                opener_text=args.opener,
                max_call_s=args.max_seconds,
                port=args.port,
                call_id=args.call_id,
                on_turn=_print_turn,
            )
        )
    except KeyboardInterrupt:
        print("\ninterrupted — call torn down", flush=True)
        return 130
    summary = json.loads((folder / "call.json").read_text(encoding="utf-8"))
    stats = summary["stats"]
    ended = summary["end_reason"]
    print(f"\ncall ended: {ended} ({summary['duration_ms'] / 1000:.1f}s)", flush=True)
    print(
        "turns: {a} agent / {p} patient · response avg {avg} ms, max {mx} ms".format(
            a=stats["agent_turns"],
            p=stats["patient_turns"],
            avg=stats["respond_ms_avg"],
            mx=stats["respond_ms_max"],
        ),
        flush=True,
    )
    print(f"session: {folder}", flush=True)
    return 0


def _run_campaign(args: argparse.Namespace) -> int:
    from pathlib import Path

    from patientqa.campaign import load_plans, run_campaign

    manifest_path = Path(args.campaign)
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    starters_path = Path(args.starters) if args.starters else manifest_path.with_suffix(
        ".starters.jsonl"
    )
    plans = load_plans(manifest_path, starters_path if starters_path.is_file() else None)
    if args.limit:
        plans = plans[: args.limit]
    if not plans:
        print("no call plans in manifest", file=sys.stderr)
        return 2
    print(
        f"{len(plans)} call plans from {manifest_path}"
        + (f" (starters: {starters_path.name})" if starters_path.is_file() else ""),
        flush=True,
    )

    caster = None
    if not args.dry_run:
        from patientqa.voicing import VoiceCaster

        caster = VoiceCaster(design=args.design_voices)

    report = asyncio.run(
        run_campaign(
            plans,
            manifest_path=manifest_path,
            stream_url=args.stream_url,
            max_call_s=args.max_seconds,
            pause_s=args.pause,
            port=args.port,
            resume=not args.no_resume,
            dry_run=args.dry_run,
            voice_caster=caster,
            max_attempts=args.attempts,
        )
    )
    if args.dry_run:
        return 0
    skipped = f" · {len(report.skipped)} skipped (already run)" if report.skipped else ""
    failed = f" · {len(report.failed)} FAILED" if report.failed else ""
    total = sum(o.duration_s for o in report.outcomes)
    print(
        f"\ncampaign done: {len(report.outcomes)} calls · {total:.0f}s total"
        f"{skipped}{failed}",
        flush=True,
    )
    for failure in report.failed:
        print(f"  failed {failure['call_id']}: {failure['error']}", flush=True)
    return 1 if report.failed else 0


def main(argv: list[str] | None = None) -> None:
    # Windows commonly exposes a cp1252 console. LLM-authored manifest prose
    # legitimately contains curly quotes and non-breaking hyphens, so make
    # CLI output deterministic instead of crashing halfway through a dry run.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="patientqa",
        description="Synthetic-patient voice bot for Pretty Good AI's scheduling line",
    )
    parser.add_argument(
        "--call", metavar="E164", help="place one live call to this allowlisted number"
    )
    parser.add_argument(
        "--campaign", metavar="MANIFEST", help="run every pending manifest entry as a call"
    )
    parser.add_argument(
        "--starters", metavar="JSONL", help="starters artifact (default: <manifest>.starters.jsonl)"
    )
    parser.add_argument("--limit", type=int, default=0, help="only the first N plans")
    parser.add_argument(
        "--pause", type=float, default=15.0, help="seconds between calls (default 15)"
    )
    parser.add_argument(
        "--attempts", type=int, default=2, help="attempts per call until quality gate passes"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the plan for each call; dial nothing"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun selected plans even when a prior call passed the quality gate",
    )
    parser.add_argument(
        "--design-voices",
        action="store_true",
        help="Voice Design per persona (may cost credits); default: cast from the shared library",
    )
    parser.add_argument(
        "--stream-url",
        metavar="WSS",
        help="public wss:// base URL of this machine (tunnel) for Twilio's media stream",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=180.0, help="per-call cap (default 180)"
    )
    parser.add_argument(
        "--opener", default=None, help="opening line if the agent stays silent (default: built-in)"
    )
    parser.add_argument(
        "--no-opener", action="store_true", help="never speak first; wait for the agent"
    )
    parser.add_argument(
        "--voice-id", default=None, help="ElevenLabs voice (default: public Rachel)"
    )
    parser.add_argument(
        "--call-id", default="live-test", help="session/call id (default live-test)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="local media-streams port (default 8080)"
    )
    args = parser.parse_args(argv)

    if not 0 < args.max_seconds <= 300:
        parser.error("--max-seconds must be greater than 0 and no more than 300")

    if args.campaign:
        raise SystemExit(_run_campaign(args))
    if not args.call:
        smoke_check()
        return
    if not args.stream_url:
        parser.error("--stream-url is required with --call (Twilio must reach us over wss://)")
    # Late import: the live-call path pulls the provider SDKs; the smoke
    # check should stay runnable on a bare checkout.
    from patientqa import callloop

    if args.voice_id is None:
        args.voice_id = callloop.DEFAULT_VOICE_ID
    if args.no_opener:
        args.opener = None
    elif args.opener is None:
        args.opener = callloop.DEFAULT_OPENER
    raise SystemExit(_run_live_call(args))


if __name__ == "__main__":
    main()
