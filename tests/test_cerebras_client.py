from pathlib import Path
from types import SimpleNamespace
from typing import Any

from patientqa.cerebras.client import (
    AGENT,
    PATIENT,
    BrainReply,
    CerebrasSettings,
    PatientBrain,
    SayStream,
    Turn,
    extract_reply,
    extract_say,
    render_history,
)
from patientqa.turns import (
    AGENT_TURN_BACKCHANNEL,
    AGENT_TURN_FINISHED,
)

HISTORY = [
    Turn(role=AGENT, content="Clinic, how can I help you?"),
    Turn(role=PATIENT, content="I'd like to see Dr. Ortiz."),
    Turn(role=AGENT, content="What day works for you?"),
]


def test_settings_load_from_secrets(secrets_file: Path) -> None:
    settings = CerebrasSettings.from_secrets(secrets_file)
    assert settings.api_key == "test-cerebras"
    assert settings.model == "gpt-oss-120b"
    assert settings.base_url == "https://api.cerebras.ai/v1"


def test_render_history_is_terse_role_lines() -> None:
    rendered = render_history(HISTORY)
    assert rendered.splitlines() == [
        "AGENT: Clinic, how can I help you?",
        "PATIENT: I'd like to see Dr. Ortiz.",
        "AGENT: What day works for you?",
    ]


def test_extract_say_parses_json_reply() -> None:
    assert extract_say('{"say": "Tuesday morning, please."}') == "Tuesday morning, please."


def test_extract_say_tolerates_plain_text() -> None:
    assert extract_say("  Tuesday morning, please.  ") == "Tuesday morning, please."


def test_extract_say_tolerates_empty_say() -> None:
    assert extract_say('{"say": ""}') == '{"say": ""}'


def test_extract_reply_parses_verdict_and_say() -> None:
    reply = extract_reply('{"agent_turn": "backchannel", "say": ""}')
    assert reply == BrainReply(say="", agent_turn=AGENT_TURN_BACKCHANNEL)


def test_extract_reply_defaults_missing_verdict_to_finished() -> None:
    assert extract_reply('{"say": "Tuesday, please."}') == BrainReply(say="Tuesday, please.")


def test_extract_reply_rejects_unknown_verdicts() -> None:
    reply = extract_reply('{"agent_turn": "sort of", "say": "Tuesday"}')
    assert reply.agent_turn == AGENT_TURN_FINISHED


def test_extract_reply_tolerates_plain_text() -> None:
    reply = extract_reply("Sorry, I can only talk on the phone.")
    assert reply.say == "Sorry, I can only talk on the phone."
    assert reply.agent_turn == AGENT_TURN_FINISHED


# -- SayStream: incremental verdict + say extraction ---------------------------


def _stream(deltas: list[str]) -> tuple[SayStream, str]:
    parser = SayStream()
    spoken = ""
    for delta in deltas:
        spoken += parser.feed(delta)
    return parser, spoken


def test_say_stream_exposes_verdict_then_streams_say() -> None:
    parser, spoken = _stream(
        ['{"agent_', 'turn": "fin', 'ished", ', '"say": "Tue', 'sday, please."', "}"]
    )
    assert parser.verdict == AGENT_TURN_FINISHED
    assert spoken == "Tuesday, please."
    assert parser.finish() == BrainReply(say="Tuesday, please.")


def test_say_stream_blocks_on_backchannel_verdict() -> None:
    parser, spoken = _stream(['{"agent_turn": "bac', 'kchannel", "say": "ignored"}'])
    assert parser.verdict == AGENT_TURN_BACKCHANNEL
    assert parser.blocked
    assert spoken == ""
    assert parser.finish() == BrainReply(say="", agent_turn=AGENT_TURN_BACKCHANNEL)


def test_say_stream_decodes_escapes_inside_say() -> None:
    raw = r'{"agent_turn": "finished", "say": "Dr. Ortiz said \"call back\" twice"}'
    parser, spoken = _stream([raw[:25], raw[25:]])
    assert spoken == 'Dr. Ortiz said "call back" twice'
    assert parser.finish().say == spoken


def test_say_stream_plain_text_falls_through() -> None:
    parser, spoken = _stream(["Sorry, ", "can't talk now."])
    assert spoken == "Sorry, can't talk now."
    reply = parser.finish()
    assert reply.say == "Sorry, can't talk now."
    assert reply.agent_turn == AGENT_TURN_FINISHED


def test_say_stream_holds_incomplete_escape() -> None:
    parser = SayStream()
    assert parser.feed('{"agent_turn": "finished", "say": "five ') == "five "
    assert parser.feed("back") == "back"
    assert parser.feed("\\") == ""  # lone backslash held until the escape completes
    assert parser.feed("n") == "\n"


# -- the brain ------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            deltas = [self._content[i : i + 7] for i in range(0, len(self._content), 7)]
            return [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=d))])
                for d in deltas
            ]
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAI:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


def _brain(content: str) -> tuple[PatientBrain, _FakeCompletions]:
    fake = _FakeOpenAI(content)
    brain = PatientBrain(CerebrasSettings(api_key="k"), client=fake)
    return brain, fake.chat.completions


def test_reply_parses_say_and_sends_rendered_history() -> None:
    brain, completions = _brain('{"agent_turn": "finished", "say": "Tuesday morning, please."}')
    reply = brain.reply("persona-card", HISTORY)
    assert reply.say == "Tuesday morning, please."
    assert reply.agent_turn == AGENT_TURN_FINISHED

    (request,) = completions.calls
    assert request["model"] == "gpt-oss-120b"
    assert request["reasoning_effort"] == "low"
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0] == {"role": "system", "content": "persona-card"}
    assert request["messages"][1]["content"] == render_history(HISTORY)


def test_reply_falls_back_to_raw_text_on_malformed_json() -> None:
    brain, _ = _brain("Sorry, I can only talk on the phone.")
    reply = brain.reply("persona-card", HISTORY[:1])
    assert reply.say == "Sorry, I can only talk on the phone."
    assert reply.agent_turn == AGENT_TURN_FINISHED


def test_reply_stream_yields_content_deltas() -> None:
    content = '{"agent_turn": "incomplete", "say": ""}'
    brain, completions = _brain(content)
    deltas = list(brain.reply_stream("persona-card", HISTORY[:1]))

    assert "".join(deltas) == content
    (request,) = completions.calls
    assert request["stream"] is True
