"""FormatSFT node — reuses the harness Qwen3 codec.

Most assertions need the parent `tau2` package + `transformers` + the Qwen3
tokenizer (install via `uv sync --extra harness`). They skip gracefully when the
codec/tokenizer isn't available so the base suite stays dependency-light.
"""
import pytest

from curation import FormatSFT, Pipeline, build_node, registered_nodes


def test_format_sft_is_registered():
    assert "format_sft" in registered_nodes()
    assert isinstance(build_node({"type": "format_sft"}), FormatSFT)


def test_format_sft_helpful_error_without_codec():
    """If the harness codec isn't importable, we raise a clear, actionable error."""
    pytest.importorskip  # noqa
    try:
        import tau2.utils.qwen3_codec  # noqa: F401
        import transformers  # noqa: F401
        has_codec = True
    except Exception:
        has_codec = False
    if has_codec:
        pytest.skip("codec available; error path not exercised")
    rec = {"conversations": [{"role": "user", "content": "hi"}], "tools": []}
    with pytest.raises(ImportError, match="harness"):
        FormatSFT().transform(rec)


# --- Real codec path (skipped unless tau2 + transformers + tokenizer present) --- #

@pytest.fixture(scope="module")
def codec_available():
    codec = pytest.importorskip("tau2.utils.qwen3_codec")
    pytest.importorskip("transformers")
    try:
        codec._get_tokenizer(codec.DEFAULT_TOKENIZER_ID)
    except Exception as e:  # tokenizer not downloadable offline
        pytest.skip(f"Qwen3 tokenizer unavailable: {e}")
    return codec


def _toolmind_record():
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {
                        "type": "object",
                        "properties": {"tz": {"type": "string"}},
                        "required": ["tz"],
                    },
                },
            }
        ],
        "conversations": [
            {"role": "user", "content": "time in NY?"},
            {
                "role": "assistant",
                "content": "<think>need the tz</think>",
                "tool_calls": [{"function": {"name": "get_time", "arguments": {"tz": "America/New_York"}}}],
            },
        ],
    }


def test_format_sft_matches_harness_codec(codec_available):
    rec = _toolmind_record()
    out = FormatSFT(out_field="sft").transform(rec)
    expected = codec_available.format_training_example(rec)
    assert out["sft"] == expected
    # assistant-only loss mask: some tokens supervised, prefix masked
    labels = out["sft"]["labels"]
    assert any(t != -100 for t in labels) and labels[0] == -100
    assert len(labels) == len(out["sft"]["input_ids"])


def test_format_sft_text_only(codec_available):
    rec = _toolmind_record()
    out = FormatSFT(text_only=True).transform(rec)
    assert isinstance(out["sft"], str)
    assert out["sft"].startswith("<|im_start|>")


def test_format_sft_in_pipeline(codec_available):
    from curation import ValidateToolCalls, FilterByField, DropFields

    rec = _toolmind_record()
    pipe = Pipeline(
        [
            ValidateToolCalls(),
            FilterByField(field="tool_calls_valid", is_true=True),
            DropFields(fields=["tool_calls_valid", "tool_call_errors"]),
            FormatSFT(text_only=True),
        ]
    )
    out = list(pipe([rec]))
    assert len(out) == 1 and out[0]["sft"].startswith("<|im_start|>")
