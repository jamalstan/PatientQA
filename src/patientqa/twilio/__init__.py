"""Twilio integration: outbound calls + Media Streams transport (DESIGN.md §3.1)."""

from patientqa.twilio.client import (
    TwilioClient,
    TwilioSettings,
    build_stream_twiml,
    parse_allowed_numbers,
)

__all__ = [
    "TwilioClient",
    "TwilioSettings",
    "build_stream_twiml",
    "parse_allowed_numbers",
]
