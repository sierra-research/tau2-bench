# Copyright Sierra
"""Run-level lexical-diversity statistics over agent speech.

Machine-translationese research (Vanmassenhove et al., EACL 2021) shows that
generated language measurably collapses in lexical richness relative to native
speech. This module computes that signal deterministically from stored
transcripts: MATTR (moving-average type-token ratio) over the pooled agent-side
tokens of one run.

Two honest uses, both within a language: comparing providers/arms on the same
task set, and tracking a pack over time. Raw values are NOT comparable across
languages — morphology and script drive the baseline, and space-delimited and
character-tokenized languages are tokenized differently.

Per-call transcripts are too short for stable type-token statistics, so the
unit is the RUN: tokens are pooled across simulations in stored order.
"""

import re
from typing import Optional

from pydantic import BaseModel, Field

from tau2.data_model.simulation import Results
from tau2.metrics.interaction_quality import extract_spoken_turns

LEXICAL_DIVERSITY_VERSION = "lexdiv-v1"
DEFAULT_MATTR_WINDOW = 100

# Languages without whitespace-delimited words: fall back to character tokens.
_CHAR_TOKEN_LANGUAGES = frozenset({"zh"})

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def tokenize(text: str, language: Optional[str]) -> list[str]:
    """Tokens for diversity statistics: words, or characters for zh."""
    if (language or "").lower() in _CHAR_TOKEN_LANGUAGES:
        return [ch for ch in text if not ch.isspace() and ch.isalnum()]
    return _WORD_RE.findall(text.casefold())


def mattr(tokens: list[str], window: int) -> Optional[float]:
    """Moving-average type-token ratio; plain TTR when shorter than one window.

    None for an empty token stream (nothing to measure).
    """
    if not tokens:
        return None
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts: dict[str, int] = {}
    for token in tokens[:window]:
        counts[token] = counts.get(token, 0) + 1
    total = len(counts)
    windows = 1
    for i in range(window, len(tokens)):
        out_token, in_token = tokens[i - window], tokens[i]
        if out_token != in_token:
            if counts[out_token] == 1:
                del counts[out_token]
            else:
                counts[out_token] -= 1
            counts[in_token] = counts.get(in_token, 0) + 1
        total += len(counts)
        windows += 1
    return total / (windows * window)


class LexicalDiversityReport(BaseModel):
    """Run-level lexical-diversity statistics for agent speech."""

    version: str = Field(default=LEXICAL_DIVERSITY_VERSION)
    language: Optional[str] = Field(
        default=None, description="Resolved run language (None for English runs)."
    )
    tokenization: str = Field(description="'word' or 'char' (zh/ja).")
    window: int = Field(description="MATTR window size in tokens.")
    num_sims: int = Field(description="Simulations contributing agent turns.")
    token_count: int = Field(description="Pooled agent-side token count.")
    type_count: int = Field(description="Distinct token count over the pool.")
    mattr: Optional[float] = Field(
        default=None,
        description="Moving-average type-token ratio over the pooled stream "
        "(plain TTR when the pool is shorter than one window); None when no "
        "agent tokens exist.",
    )


def compute_lexical_diversity(
    results: Results, *, window: int = DEFAULT_MATTR_WINDOW
) -> LexicalDiversityReport:
    """Pool agent tokens across a run's simulations and compute MATTR."""
    # Lazy import: the evaluator package is heavy and pulls in domain wiring.
    from tau2.evaluator.evaluator import get_simulation_language_info

    language: Optional[str] = None
    tokens: list[str] = []
    contributing = 0
    for sim in results.simulations:
        if language is None:
            sim_language, _script = get_simulation_language_info(sim)
            language = (sim_language or "").lower() or None
        sim_tokens: list[str] = []
        for turn in extract_spoken_turns(sim):
            if turn.speaker == "agent":
                sim_tokens.extend(tokenize(turn.text, language))
        if sim_tokens:
            contributing += 1
            tokens.extend(sim_tokens)
    return LexicalDiversityReport(
        language=language,
        tokenization="char" if language in _CHAR_TOKEN_LANGUAGES else "word",
        window=window,
        num_sims=contributing,
        token_count=len(tokens),
        type_count=len(set(tokens)),
        mattr=mattr(tokens, window),
    )
