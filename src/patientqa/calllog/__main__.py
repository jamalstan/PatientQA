"""``python -m patientqa.calllog`` — session viewer, reports and demo data.

    uv run python -m patientqa.calllog viewer [calls]   # viewer.html beside the sessions
    uv run python -m patientqa.calllog report DIR...    # self-contained report.html
    uv run python -m patientqa.calllog export DIR...    # submission audio + transcript + INDEX.md
    uv run python -m patientqa.calllog demo [--out calls]
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        prog="python -m patientqa.calllog",
        description="Call session logs, recordings and the static viewer",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    viewer = subparsers.add_parser(
        "viewer", help="write the static dark-mode viewer and open it in a browser"
    )
    viewer.add_argument(
        "path", nargs="?", type=Path, default=Path("calls"),
        help="calls root to place the viewer in (default: ./calls)",
    )
    viewer.add_argument(
        "--no-open", action="store_true", help="write the file but do not open a browser"
    )

    report = subparsers.add_parser(
        "report", help="bake session folder(s) into a self-contained report.html"
    )
    report.add_argument("dirs", nargs="+", type=Path, help="per-call session folder(s)")
    report.add_argument(
        "--out", type=Path, default=None,
        help="output path (default: report.html in the first session folder; "
        "only valid with a single DIR)",
    )

    demo = subparsers.add_parser(
        "demo", help="generate a synthetic demo session (no network, no cost)"
    )
    demo.add_argument("--out", type=Path, default=Path("calls"), help="calls root")

    export = subparsers.add_parser(
        "export", help="export session folder(s) as submission deliverables"
    )
    export.add_argument("dirs", nargs="+", type=Path, help="per-call session folder(s)")
    export.add_argument(
        "--out", type=Path, default=Path("deliverables") / "calls",
        help="output root (default: deliverables/calls)",
    )
    export.add_argument(
        "--campaign-only",
        action="store_true",
        help="from a calls root, select latest quality-passing call-NNN sessions",
    )

    args = parser.parse_args(argv)
    if args.command == "viewer":
        return _viewer(args)
    if args.command == "report":
        return _report(args)
    if args.command == "export":
        return _export(args)
    return _demo(args)


def _viewer(args: argparse.Namespace) -> int:
    from patientqa.calllog.viewer import write_viewer

    path = write_viewer(args.path)
    print(f"wrote {path}")
    print("open it in a browser, then drop the call folders onto the page")
    if not args.no_open:
        webbrowser.open(path.resolve().as_uri())
    return 0


def _report(args: argparse.Namespace) -> int:
    from patientqa.calllog.viewer import build_report

    if args.out is not None and len(args.dirs) > 1:
        print("--out only works with a single session directory", file=sys.stderr)
        return 2
    for directory in args.dirs:
        if not (directory / "session.jsonl").is_file():
            print(f"{directory}: no session.jsonl — not a session folder", file=sys.stderr)
            return 1
    out = build_report(args.dirs, args.out)
    print(f"wrote {out}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    from patientqa.calllog.demo import generate_demo_session

    directory = generate_demo_session(args.out)
    print(f"wrote demo session {directory}")
    print(f"try: uv run python -m patientqa.calllog report {directory}")
    return 0


def _export(args: argparse.Namespace) -> int:
    from patientqa.calllog.export import export_sessions, select_campaign_sessions

    directories: list[Path] = []
    for path in args.dirs:
        if args.campaign_only and path.is_dir() and not (path / "transcript.json").is_file():
            directories.extend(select_campaign_sessions(path))
            continue
        if (path / "transcript.json").is_file():
            directories.append(path)
        elif path.is_dir():
            directories.extend(
                child
                for child in sorted(path.iterdir())
                if child.is_dir() and (child / "transcript.json").is_file()
            )
        else:
            print(
                f"{path}: no transcript.json and not a calls root",
                file=sys.stderr,
            )
            return 1
    if not directories:
        print("no finalized sessions found", file=sys.stderr)
        return 1
    exported = export_sessions(directories, args.out)
    for one in exported:
        print(f"exported {one.call_id} → {one.directory}")
    print(f"index: {args.out / 'INDEX.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
