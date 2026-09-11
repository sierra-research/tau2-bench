# Copyright Sierra
"""Unit tests for the audio multipart path through `to_litellm_messages()`.

No network calls: these exercise the pure conversion (`to_litellm_messages`)
and the debug-log formatter (`_format_messages_for_logging`), which must never
write raw base64 audio payloads into log files.
"""

import base64

from tau2.data_model.audio import AudioEncoding, AudioFormat
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.utils.llm_utils import _format_messages_for_logging, to_litellm_messages

# base64 of a WAV-magic payload (b"RIFF....WAVE") — container detection input.
_WAV_B64 = base64.b64encode(b"RIFF\x00\x00\x00\x00WAVE").decode()


def _audio_message(content="listen to this", audio_format=None):
    return UserMessage(
        role="user",
        content=content,
        audio_content=_WAV_B64,
        audio_format=audio_format,
    )


def test_user_message_without_audio_stays_plain_string():
    out = to_litellm_messages([UserMessage(role="user", content="hi")])
    assert out == [{"role": "user", "content": "hi"}]


def test_user_message_with_audio_becomes_multipart():
    out = to_litellm_messages(
        [
            SystemMessage(role="system", content="sys"),
            _audio_message(),
        ]
    )
    assert out[0] == {"role": "system", "content": "sys"}
    content = out[1]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "listen to this"}
    assert content[1] == {
        "type": "input_audio",
        "input_audio": {"data": _WAV_B64, "format": "wav"},
    }


def test_audio_only_message_omits_empty_text_part():
    out = to_litellm_messages([_audio_message(content=None)])
    content = out[0]["content"]
    assert [part["type"] for part in content] == ["input_audio"]


def test_audio_format_falls_back_to_declared_encoding():
    # Non-container payload (no RIFF/ID3 magic): fall back to the message's
    # declared raw encoding rather than mislabeling it as wav.
    raw_b64 = base64.b64encode(b"\x7f" * 32).decode()
    msg = UserMessage(
        role="user",
        content="x",
        audio_content=raw_b64,
        audio_format=AudioFormat(encoding=AudioEncoding.ULAW, sample_rate=8000),
    )
    out = to_litellm_messages([msg])
    assert out[0]["content"][1]["input_audio"]["format"] == "ulaw"


def test_logging_redacts_audio_payload():
    litellm_messages = to_litellm_messages([_audio_message()])
    formatted = _format_messages_for_logging(litellm_messages)
    audio_part = formatted[0]["content"][1]
    assert audio_part["input_audio"]["data"] == f"<audio: {len(_WAV_B64)} b64 chars>"
    assert audio_part["input_audio"]["format"] == "wav"
    # Text part untouched; original message list not mutated.
    assert formatted[0]["content"][0] == {"type": "text", "text": "listen to this"}
    assert litellm_messages[0]["content"][1]["input_audio"]["data"] == _WAV_B64


def test_logging_still_splits_plain_string_content():
    formatted = _format_messages_for_logging(
        [{"role": "user", "content": "line1\nline2"}]
    )
    assert formatted[0]["content"] == ["line1", "line2"]
