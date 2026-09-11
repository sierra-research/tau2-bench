# Copyright Sierra
"""Deterministic lexical-diversity statistics (tokenization + MATTR)."""

import pytest

from tau2.metrics.lexical_diversity import mattr, tokenize

# --- tokenize ----------------------------------------------------------------


def test_tokenize_words_casefolds_and_strips_punctuation():
    assert tokenize("Quería CONSULTAR mi reserva, ¿vale?", "es") == [
        "quería",
        "consultar",
        "mi",
        "reserva",
        "vale",
    ]


def test_tokenize_chinese_uses_characters():
    assert tokenize("我想改签 一张机票", "zh") == list("我想改签一张机票")


def test_tokenize_empty():
    assert tokenize("", "es") == []
    assert tokenize("   ", "zh") == []


# --- mattr --------------------------------------------------------------------


def test_mattr_empty_is_none():
    assert mattr([], 100) is None


def test_mattr_short_stream_is_plain_ttr():
    # 4 tokens, 3 distinct, window larger than the stream.
    assert mattr(["a", "b", "a", "c"], 100) == pytest.approx(3 / 4)


def test_mattr_all_distinct_is_one():
    tokens = [f"t{i}" for i in range(50)]
    assert mattr(tokens, 10) == pytest.approx(1.0)


def test_mattr_all_identical_is_one_over_window():
    tokens = ["same"] * 50
    assert mattr(tokens, 10) == pytest.approx(1 / 10)


def test_mattr_repetitive_stream_scores_below_diverse_stream():
    diverse = [f"w{i % 40}" for i in range(200)]
    collapsed = [f"w{i % 5}" for i in range(200)]
    assert mattr(collapsed, 25) < mattr(diverse, 25)


def test_mattr_matches_bruteforce_on_random_stream():
    import random

    rng = random.Random(7)
    tokens = [f"w{rng.randint(0, 12)}" for _ in range(120)]
    window = 30
    brute = sum(
        len(set(tokens[i : i + window])) / window
        for i in range(len(tokens) - window + 1)
    ) / (len(tokens) - window + 1)
    assert mattr(tokens, window) == pytest.approx(brute)
