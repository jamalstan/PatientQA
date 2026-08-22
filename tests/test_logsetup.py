"""Per-call stdlib logging (logsetup) — file capture, console pair, cleanup."""

import logging

from patientqa.logsetup import LOGGER_NAME, attach_call_log, detach_call_log


def test_attach_writes_debug_lines_to_the_call_folder(tmp_path):
    handler = attach_call_log(tmp_path)
    try:
        logging.getLogger(f"{LOGGER_NAME}.callloop").debug("turn machinery: %dms", 42)
        logging.getLogger(f"{LOGGER_NAME}.turns").info("state: listening → thinking")
    finally:
        detach_call_log(handler)

    lines = (tmp_path / "debug.log").read_text(encoding="utf-8").splitlines()
    assert any("DEBUG" in line and "patientqa.callloop" in line and "42ms" in line
               for line in lines)
    assert any("INFO" in line and "patientqa.turns" in line and "thinking" in line
               for line in lines)
    # timestamps lead every line: YYYY-MM-DD HH:MM:SS.mmm
    assert all(len(line.split()[0]) == 10 and ":" in line.split()[1] for line in lines)


def test_detach_removes_handlers_so_nothing_leaks(tmp_path):
    logger = logging.getLogger(LOGGER_NAME)
    before = set(logger.handlers)  # pytest may inject its own capture handlers
    handler = attach_call_log(tmp_path)
    assert set(logger.handlers) - before  # we attached our pair
    detach_call_log(handler)
    assert set(logger.handlers) == before  # exactly our pair is gone

    logging.getLogger(LOGGER_NAME).debug("post-detach must not reach the file")
    assert "post-detach" not in (tmp_path / "debug.log").read_text(encoding="utf-8")


def test_unrelated_loggers_stay_unconfigured(tmp_path):
    handler = attach_call_log(tmp_path)
    try:
        logging.getLogger("websockets").debug("library chatter")
    finally:
        detach_call_log(handler)
    assert "library chatter" not in (tmp_path / "debug.log").read_text(encoding="utf-8")
