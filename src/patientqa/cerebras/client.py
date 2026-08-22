"""Cerebras patient brain (DESIGN.md §3.2, §5.3 delta 1).

gpt-oss-120b served over Cerebras' OpenAI-compatible API. One short JSON
reply per turn that answers *two* questions at once (§5.3: the semantic
end-of-turn gate is folded into the brain call, so the verdict costs zero
extra stages):

1. Is the agent's turn over? — ``agent_turn``: ``finished | backchannel |
   incomplete``; anything but ``finished`` and the simulator stays silent.
2. If it is over, what does the patient say? — ``say``.

The verdict is emitted *before* ``say`` in the contract so a streaming
response can start speaking after a handful of tokens: :class:`SayStream`
parses the verdict as soon as it closes, then streams the ``say`` value
incrementally to the speakable-chunk pipeline (DESIGN.md §5.3 delta 3).

Dialogue history is rendered as terse ``AGENT:``/``PATIENT:`` lines rather
than multi-role messages (DESIGN.md §4), which keeps context small and TTFT
flat through the call.
"""

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from patientqa.config import get_secret
from patientqa.turns import (
    AGENT_TURN_FINISHED,
    VERDICTS,
)

DEFAULT_BASE_URL = "https://api.cerebras.ai/v1"
DEFAULT_MODEL = "gpt-oss-120b"
REASONING_EFFORT = "low"

AGENT = "agent"
PATIENT = "patient"

# Fallback prompt used until the persona pipeline (DESIGN.md §6) starts
# prepending per-call persona cards + objectives to these rules. The
# telephone register and the one-thought-per-turn rule are the voice-friendly
# constraints from the turn-taking review (§5.3).
DEFAULT_SYSTEM_PROMPT = """\
You are a synthetic patient calling a medical clinic's scheduling line.
Stay in character and pursue your objective no matter what the agent says.
You are speaking on a telephone: short natural clauses, one thought per
turn, no lists, no URLs. For identity facts, use the persona card's exact
spoken wording; clear words and digit groups matter more than brevity.

Reply with ONE JSON object and nothing else:
{"agent_turn": "<finished|backchannel|incomplete>", "say": "<patient utterance>"}

"agent_turn" judges the agent's last utterance: "finished" = a complete
turn addressed to you; "backchannel" = a listener noise ("mm-hm", "sure",
"I see"); "incomplete" = they trailed off or are still mid-sentence.
When the agent's turn is NOT finished, "say" must be "" (you stay silent).
When it is finished, "say" is ONE short patient utterance, at most two
sentences. No role labels, no narration, no explanations."""


@dataclass(frozen=True)
class CerebrasSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    @classmethod
    def from_secrets(cls, path: Path | None = None) -> "CerebrasSettings":
        return cls(api_key=get_secret("cerebras", "api_key", path=path))


@dataclass(frozen=True)
class Turn:
    """One dialogue turn; ``role`` is ``AGENT`` (their receptionist) or ``PATIENT`` (us)."""

    role: str
    content: str


def render_history(history: Sequence[Turn]) -> str:
    """Render the dialogue so far as terse role-labelled lines (DESIGN.md §4)."""
    return "\n".join(f"{turn.role.upper()}: {turn.content}" for turn in history)


@dataclass(frozen=True)
class BrainReply:
    """The folded gate + reply contract (§5.3 delta 1).

    ``agent_turn`` defaults to ``finished`` whenever the model omits it or
    replies malformed text — a broken turn must never kill a live call, and
    the historical contract (``{"say": ...}`` only) stays valid.
    """

    say: str
    agent_turn: str = AGENT_TURN_FINISHED


def extract_say(raw: str) -> str:
    """Pull the patient utterance out of a model reply (tolerant of plain text)."""
    text = raw.strip()
    try:
        say = str(json.loads(text)["say"]).strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return text
    return say or text


def extract_reply(raw: str) -> BrainReply:
    """Parse the full reply: verdict + utterance, with plain-text tolerance.

    An empty ``say`` is valid and meaningful (the model explicitly chose to
    stay silent); the raw-text fallback only applies when the reply is not
    parseable JSON at all.
    """
    text = raw.strip()
    try:
        parsed = json.loads(text)
        say = str(parsed["say"]).strip()
        verdict = str(parsed.get("agent_turn", AGENT_TURN_FINISHED)).strip()
        if verdict not in VERDICTS:
            verdict = AGENT_TURN_FINISHED
        return BrainReply(say=say, agent_turn=verdict)
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
        return BrainReply(say=text, agent_turn=AGENT_TURN_FINISHED)


_VERDICT_HEAD = re.compile(
    r'^\s*\{\s*"agent_turn"\s*:\s*"(' + "|".join(VERDICTS) + r')"\s*,?'
)
_SAY_OPEN = re.compile(r'"say"\s*:\s*"')
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "n": "\n", "t": "\t", "r": "\r"}


class SayStream:
    """Incremental parser for ``{"agent_turn": ..., "say": ...}`` streams.

    ``feed`` takes each streamed delta and returns the ``say`` text that
    became newly available (often ``""`` — e.g. while the verdict is still
    incomplete). The verdict is exposed the moment its closing quote lands;
    a non-``finished`` verdict blocks the stream so the caller can discard
    the rest without waiting for it. Replies that never look like the JSON
    contract fall through to plain-text mode (same tolerance as
    :func:`extract_reply`). Escapes inside ``say`` are decoded as they are
    completed; a trailing lone backslash is held back until more arrives.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._verdict: str | None = None
        self._blocked = False
        self._plain = False
        self._say_start: int | None = None
        self._emitted = 0  # index into the raw (still-escaped) say payload
        self._say_closed = False
        self._say_text = ""

    @property
    def verdict(self) -> str | None:
        """The gate verdict once parsed, else ``None`` (still pending)."""
        return self._verdict

    @property
    def blocked(self) -> bool:
        """True once a non-finished verdict is known — discard the rest."""
        return self._blocked

    def feed(self, delta: str) -> str:
        self._buffer += delta
        if self._blocked:
            return ""
        if self._plain:
            return delta
        if self._verdict is None:
            match = _VERDICT_HEAD.match(self._buffer)
            if match is None:
                stripped = self._buffer.lstrip()
                if stripped and not stripped.startswith("{"):
                    # Plain text (no JSON): everything is the utterance.
                    self._plain = True
                    self._verdict = AGENT_TURN_FINISHED
                    return self._buffer
                return ""
            self._verdict = match.group(1)
            if self._verdict != AGENT_TURN_FINISHED:
                self._blocked = True
                return ""
        if self._say_start is None:
            match = _SAY_OPEN.search(self._buffer)
            if match is None:
                return ""
            self._say_start = match.end()

        newly = self._decode_available()
        self._say_text += newly
        return newly

    def _decode_available(self) -> str:
        """Decode raw ``say`` payload from ``_emitted`` to a safe stop point."""
        if self._say_closed:
            return ""
        raw = self._buffer[self._say_start :]
        out: list[str] = []
        index = self._emitted
        while index < len(raw):
            char = raw[index]
            if char == "\\":
                if index + 1 >= len(raw):
                    break  # escape incomplete — wait for the next delta
                escape = raw[index + 1]
                out.append(_ESCAPES.get(escape, "\\" + escape))
                index += 2
                continue
            if char == '"':
                self._say_closed = True  # the closing quote — say is complete
                break
            out.append(char)
            index += 1
        self._emitted = index
        return "".join(out)

    def finish(self) -> BrainReply:
        """Final verdict + full utterance once the stream ends."""
        if self._plain:
            return BrainReply(say=self._buffer.strip())
        if self._blocked:
            assert self._verdict is not None
            return BrainReply(say="", agent_turn=self._verdict)
        reply = extract_reply(self._buffer)
        # While streaming we may have held back an incomplete escape; prefer
        # the parsed JSON's say when the two disagree, it saw the whole text.
        return reply if reply.say else BrainReply(say=self._say_text)


class PatientBrain:
    """One gated patient reply per agent turn, straight from gpt-oss-120b."""

    def __init__(self, settings: CerebrasSettings, client: OpenAI | None = None) -> None:
        self._settings = settings
        self._client = client or OpenAI(
            api_key=settings.api_key, base_url=settings.base_url
        )

    def reply(self, system_prompt: str, history: Sequence[Turn]) -> BrainReply:
        """Generate the gated reply for the dialogue so far."""
        completion = self._client.chat.completions.create(
            model=self._settings.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": render_history(history)},
            ],
            response_format={"type": "json_object"},
            reasoning_effort=REASONING_EFFORT,
        )
        raw = completion.choices[0].message.content or ""
        return extract_reply(raw)

    def reply_stream(self, system_prompt: str, history: Sequence[Turn]) -> Iterator[str]:
        """Stream raw reply deltas; parse them with :class:`SayStream`.

        Streaming collapses speech start to TTFT (§4): the verdict closes
        within a few tokens, then ``say`` text flows straight into the
        speakable-chunk buffer.
        """
        stream = self._client.chat.completions.create(
            model=self._settings.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": render_history(history)},
            ],
            response_format={"type": "json_object"},
            reasoning_effort=REASONING_EFFORT,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content:
                yield content
