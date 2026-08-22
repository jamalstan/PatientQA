"""Standard-library logging for live calls, persisted per call folder.

``session.jsonl`` is the structured source of truth for *domain* events
(turns, gate verdicts, latencies). This module tells the *machinery's* story
— socket lifecycles, state transitions, engage timings, idle nudges — as
plain timestamped lines, so a post-mortem never needs a re-run:

    <call folder>/debug.log    DEBUG, every module under ``patientqa.``
    stderr                     INFO, the same stream minus the chatter

Attach with :func:`attach_call_log` (wired in
:func:`patientqa.callloop.run_call`) and always pair with
:func:`detach_call_log` — one call, one handler, no leaks across calls.
"""

import logging
import sys
from pathlib import Path

LOGGER_NAME = "patientqa"

_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def attach_call_log(call_dir: Path) -> logging.Handler:
    """Log the ``patientqa`` tree to ``call_dir/debug.log`` (and stderr at INFO).

    Returns the file handler; pass it to :func:`detach_call_log` on teardown.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(Path(call_dir) / "debug.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    logger.addHandler(file_handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
    logger.addHandler(console)
    logger.propagate = False  # root stays quiet; we own the output
    return file_handler


def detach_call_log(file_handler: logging.Handler) -> None:
    """Undo :func:`attach_call_log` (both handlers; idempotent-ish by design:
    called exactly once per attach from ``run_call``'s ``finally``)."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if handler is file_handler or (
            isinstance(handler, logging.StreamHandler)
            and handler.stream is sys.stderr
        ):
            logger.removeHandler(handler)
            handler.close()
