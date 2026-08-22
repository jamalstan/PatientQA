"""Twilio Programmable Voice integration (DESIGN.md §3.1).

Places the outbound call and answers it with TwiML that bridges the call into
our Media Streams WebSocket. Audio is 8 kHz μ-law in both directions and is
never transcoded (DESIGN.md §2 step 2).

Dialable destinations are an allowlist in secrets.toml (``[twilio]
allowed_numbers``) so real phone numbers never appear in committed code. The
top-level ``twilio`` package is Twilio's SDK; this subpackage is our thin
wrapper around it. Python 3 absolute imports keep the two from colliding.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr

from twilio.rest import Client as TwilioRestClient

from patientqa.config import get_secret

_E164 = re.compile(r"^\+[1-9]\d{6,14}$")

#: Sent on the Media Streams WebSocket to flush queued outbound audio
#: immediately — the last step of the §5.3 abort sequence.
MEDIA_CLEAR = '{"event":"clear"}'


def media_clear_message() -> str:
    """Twilio's playback-buffer clear message (§5.3 delta 2 abort step 3)."""
    return MEDIA_CLEAR


def parse_allowed_numbers(raw: str | list[str]) -> tuple[str, ...]:
    """Normalize ``allowed_numbers`` — a list from TOML, comma-separated from the env.

    Every entry must be E.164 (``+`` then 8–15 digits). A malformed or empty
    allowlist fails loudly at load time, long before anything can dial.
    """
    items = raw.split(",") if isinstance(raw, str) else raw
    numbers = tuple(str(item).strip() for item in items if str(item).strip())
    invalid = [number for number in numbers if not _E164.match(number)]
    if invalid:
        raise ValueError(
            "secrets.toml [twilio] allowed_numbers: not E.164: " + ", ".join(invalid)
        )
    if not numbers:
        raise ValueError(
            "secrets.toml [twilio] allowed_numbers is empty; the bot may dial nothing."
        )
    return numbers


@dataclass(frozen=True)
class TwilioSettings:
    """Everything needed to sign Twilio requests and place a call."""

    api_key_sid: str
    api_key_secret: str
    account_sid: str
    from_number: str
    allowed_numbers: tuple[str, ...]
    live_test_number: str

    @classmethod
    def from_secrets(cls, path: Path | None = None) -> "TwilioSettings":
        allowed_numbers = parse_allowed_numbers(
            get_secret("twilio", "allowed_numbers", path=path)
        )
        live_test_number = get_secret("twilio", "live_test_number", path=path)
        if live_test_number not in allowed_numbers:
            raise ValueError(
                f"secrets.toml [twilio] live_test_number {live_test_number!r} must be "
                "one of [twilio] allowed_numbers."
            )
        return cls(
            api_key_sid=get_secret("twilio", "api_key_sid", path=path),
            api_key_secret=get_secret("twilio", "api_key_secret", path=path),
            account_sid=get_secret("twilio", "account_sid", path=path),
            from_number=get_secret("twilio", "from_number", path=path),
            allowed_numbers=allowed_numbers,
            live_test_number=live_test_number,
        )


def build_stream_twiml(stream_url: str) -> str:
    """TwiML that connects the answered call to our media stream.

    ``<Connect><Stream>`` keeps the call in-band and delivers both audio
    tracks to ``stream_url`` as 8 kHz μ-law frames.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Connect><Stream url={quoteattr(stream_url)} /></Connect></Response>"
    )


class TwilioClient:
    """Dials an allowed destination and hands the call's audio to our orchestrator."""

    def __init__(
        self, settings: TwilioSettings, rest_client: TwilioRestClient | None = None
    ) -> None:
        self._settings = settings
        self._rest = rest_client or TwilioRestClient(
            settings.api_key_sid, settings.api_key_secret, settings.account_sid
        )

    def place_call(
        self, stream_url: str, to: str | None = None, *, max_duration_s: float = 300.0
    ) -> str:
        """Call ``to`` and bridge it to ``stream_url``; returns the Call SID.

        ``to`` defaults to the designated live-test number. Any destination
        outside the allowlist is refused, so a typo in a manifest can never
        dial a stranger (DESIGN.md §9). The allowlist itself lives in
        secrets.toml, never in committed code.
        """
        destination = to or self._settings.live_test_number
        if destination not in self._settings.allowed_numbers:
            raise ValueError(
                f"Refusing to call {destination!r}: destination is not in the allowlist "
                "([twilio] allowed_numbers in secrets.toml, DESIGN.md §9)."
            )
        if not 0 < max_duration_s <= 300:
            raise ValueError("max_duration_s must be between 1 and 300 seconds")
        duration = math.ceil(max_duration_s)
        call = self._rest.calls.create(
            to=destination,
            from_=self._settings.from_number,
            twiml=build_stream_twiml(stream_url),
            time_limit=duration,
        )
        return call.sid

    def hang_up(self, call_sid: str) -> None:
        """Complete the call (§9 termination). Safe on an already-ended call
        — Twilio rejects the update and we treat that as success."""
        try:
            self._rest.calls(call_sid).update(status="completed")
        except Exception:
            pass  # already completed/canceled — the line is down either way
