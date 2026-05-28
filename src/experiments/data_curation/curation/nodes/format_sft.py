"""FormatSFT — render records into train-ready examples using the EVAL harness's
Qwen3 codec (``tau2.utils.qwen3_codec``).

This is the node that closes the train/eval loop: it reuses the exact same
``format_training_example`` the harness uses, so curated training data is
formatted (and loss-masked) identically to what the model sees at eval time.

Requires the parent ``tau2`` package (+ ``transformers`` and the Qwen3 tokenizer).
Install via the optional extra:  ``uv sync --extra harness``.
"""
from __future__ import annotations

from typing import Optional

from ..core import MapNode, Record, register

_IMPORT_HINT = (
    "FormatSFT needs the harness codec. Install the parent package with "
    "`uv sync --extra harness` (provides tau2 + transformers)."
)


@register("format_sft")
class FormatSFT(MapNode):
    """Annotate each record with a Qwen3 SFT example via the harness codec.

    Writes ``record[out_field]``:
        * if ``text_only``: the rendered prompt string, else
        * ``{"text", "input_ids", "labels"}`` with an assistant-only loss mask.

    Args:
        out_field: where to write the result (default "sft").
        tokenizer_id: HF tokenizer id (default Qwen/Qwen3-8B; must match serving).
        enable_thinking: Qwen3 thinking switch, kept consistent with eval.
        text_only: store just the rendered text instead of tokenized example.
    """

    def __init__(
        self,
        out_field: str = "sft",
        tokenizer_id: str = "Qwen/Qwen3-8B",
        enable_thinking: bool = True,
        text_only: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.out_field = out_field
        self.tokenizer_id = tokenizer_id
        self.enable_thinking = enable_thinking
        self.text_only = text_only

    def _codec(self):
        try:
            from tau2.utils.qwen3_codec import format_training_example, render_chat_dicts
        except ImportError as e:  # pragma: no cover - env-dependent
            raise ImportError(_IMPORT_HINT) from e
        return format_training_example, render_chat_dicts

    def transform(self, record: Record) -> Record:
        format_training_example, render_chat_dicts = self._codec()
        out = dict(record)
        if self.text_only:
            out[self.out_field] = render_chat_dicts(
                record.get("conversations", []),
                tools=record.get("tools"),
                add_generation_prompt=False,
                enable_thinking=self.enable_thinking,
                tokenizer_id=self.tokenizer_id,
            )
        else:
            out[self.out_field] = format_training_example(
                record,
                tokenizer_id=self.tokenizer_id,
                enable_thinking=self.enable_thinking,
            )
        return out
