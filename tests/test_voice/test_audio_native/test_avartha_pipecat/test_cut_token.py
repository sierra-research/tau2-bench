"""The unit model and the unit counters for a started token (cut_token.py)."""

import pytest

from tau2.voice.audio_native.avartha_pipecat.cut_token import (
    DEFAULT_UNIT_MS,
    UNIT_SAFETY,
    TokenKind,
    split_token,
    units_heard,
)
from tau2.voice.audio_native.avartha_pipecat.discrete_time_adapter import _next_word


# Started words from recorded calls' cut lines (en, es, pt, hi, zh, ko).
@pytest.mark.parametrize(
    "word, kind, units",
    [
        (" 6941833.", TokenKind.CODE, ["6", "9", "4", "1", "8", "3", "3"]),
        (" 59XX6W", TokenKind.CODE, ["5", "9", "X", "X", "6", "W"]),
        (" — HAT110", TokenKind.CODE, ["H", "A", "T", "1", "1", "0"]),
        (" i3668", TokenKind.CODE, ["i", "3", "6", "6", "8"]),
        (" garcia5901", TokenKind.CODE, ["garcia", "5", "9", "0", "1"]),
        (" 8-8-9-0-0.", TokenKind.CODE, ["8-", "8-", "9-", "0-", "0"]),
        ("7091239、", TokenKind.CODE, ["7", "0", "9", "1", "2", "3", "9"]),
        (" HAT138", TokenKind.CODE, ["H", "A", "T", "1", "3", "8"]),
        (" 1777", TokenKind.NUMBER_WORDS, ["1777"]),
        (" 1177", TokenKind.NUMBER_WORDS, ["1177"]),
        (" 930", TokenKind.NUMBER_WORDS, ["930"]),
        (" 69418", TokenKind.CODE, ["6", "9", "4", "1", "8"]),
        (" H9ZU", TokenKind.ORDINARY, ["H9ZU"]),  # under 5 units
        (" HSE", TokenKind.ORDINARY, ["HSE"]),  # capitals under 5 units
        (" IFOYYZ,", TokenKind.CODE, ["I", "F", "O", "Y", "Y", "Z"]),
        (" HATHAT", TokenKind.CODE, ["H", "A", "T", "H", "A", "T"]),
        (" GARCIA", TokenKind.CODE, ["G", "A", "R", "C", "I", "A"]),  # errs on fewer
        (" Garcia", TokenKind.ORDINARY, ["Garcia"]),
        (" confirmação", TokenKind.LONG_WORD, ["confirmação"]),
        (" 확인해드리겠습니다", TokenKind.ORDINARY, ["확인해드리겠습니다"]),
        (
            " ivan_rossi_8555,",
            TokenKind.IDENTIFIER,
            ["ivan", "_", "rossi", "_", "8", "5", "5", "5"],
        ),
        (
            " JIHUN_GA_1177",
            TokenKind.IDENTIFIER,
            ["JIHUN", "_", "GA", "_", "1", "1", "7", "7"],
        ),
        (" $ 1,387", TokenKind.NUMBER_WORDS, ["1,387"]),
        (" 2.420", TokenKind.NUMBER_WORDS, ["2.420"]),
        (" 5,27,520", TokenKind.NUMBER_WORDS, ["5,27,520"]),
        (" 282,000", TokenKind.NUMBER_WORDS, ["282,000"]),
        ("R$137,30", TokenKind.NUMBER_WORDS, ["R$137,30"]),
        (" R$247,55,", TokenKind.NUMBER_WORDS, ["R$247,55"]),
        (" 02:30", TokenKind.ORDINARY, ["02:30"]),  # a short time, shown whole
        # Never shown whole: _ or @, a dot between letters or digits, or a
        # digit among 5+ letters and digits (zh hao.h18836 in a recorded call).
        (
            " hao.h18836,",
            TokenKind.IDENTIFIER,
            ["hao", ".", "h", "1", "8", "8", "3", "6"],
        ),
        (" li.wei1234", TokenKind.IDENTIFIER, ["li", ".", "wei", "1", "2", "3", "4"]),
        (" josé_12345", TokenKind.IDENTIFIER, ["josé", "_", "1", "2", "3", "4", "5"]),
        (" Ana_Díaz12", TokenKind.IDENTIFIER, ["Ana", "_", "Díaz", "1", "2"]),
        (" HAT138's", TokenKind.UNPARSED, ["HAT138's"]),
        (" HAT138का", TokenKind.UNPARSED, ["HAT138का"]),
        # The envelope, as intended: read by units (some as words, so fewer
        # are shown), or short and shown whole.
        (" 12000", TokenKind.CODE, ["1", "2", "0", "0", "0"]),
        (" iPhone15", TokenKind.CODE, ["i", "P", "hone", "1", "5"]),
        (" 2026-11-18", TokenKind.CODE, ["2", "0", "2", "6-", "1", "1-", "1", "8"]),
        (" +14155550123", TokenKind.CODE, list("14155550123")),
        (" 50%", TokenKind.ORDINARY, ["50"]),
        (" 10:30", TokenKind.ORDINARY, ["10:30"]),
        (" पहली", TokenKind.ORDINARY, ["पहली"]),  # a vowel sign is in the word
        # Dotted abbreviations are words, shown whole; a clock
        # time with letters is read by units but not parsed: nothing.
        (" p.m.", TokenKind.ORDINARY, ["p.m"]),
        (" e.g.", TokenKind.ORDINARY, ["e.g"]),
        (" U.S.", TokenKind.ORDINARY, ["U.S"]),
        # Not abbreviations: more than two runs, or no closing dot.
        (
            " A.B.C.D.E",
            TokenKind.IDENTIFIER,
            ["A", ".", "B", ".", "C", ".", "D", ".", "E"],
        ),
        (" j.li", TokenKind.IDENTIFIER, ["j", ".", "li"]),
        # Full-width capitals are spelled one by one, like ASCII ones.
        (" ＡＢＣ１２３４", TokenKind.CODE, ["Ａ", "Ｂ", "Ｃ", "１", "２", "３", "４"]),
        (" ＩＦＯＹＹＺ", TokenKind.CODE, ["Ｉ", "Ｆ", "Ｏ", "Ｙ", "Ｙ", "Ｚ"]),
        (" 4:30pm", TokenKind.UNPARSED, ["4:30pm"]),
        (" card", TokenKind.ORDINARY, ["card"]),
        (" PayPal", TokenKind.ORDINARY, ["PayPal"]),
        (" 18", TokenKind.ORDINARY, ["18"]),
        (" 26일", TokenKind.ORDINARY, ["26일"]),
        ("您", TokenKind.ORDINARY, ["您"]),
        (" नंबर", TokenKind.ORDINARY, ["नंबर"]),
        (" Manuel-guión-Gas", TokenKind.LONG_WORD, ["Manuel-guión-Gas"]),
    ],
)
def test_split_token_kind_and_units(word, kind, units):
    token = split_token(word)

    assert token.kind is kind
    assert list(token.units) == units
    assert token.prefix + "".join(token.units) + token.suffix == word


@pytest.mark.parametrize(
    "line, word, kind",
    [
        # ko: a particle after a code or number is a script change, so the
        # started word ends before it.
        (" HAT138편은 출발합니다", " HAT138", TokenKind.CODE),
        (" 1777로 바꿨습니다", " 1777", TokenKind.NUMBER_WORDS),
        (" 282,000원에 예약", " 282,000", TokenKind.NUMBER_WORDS),
        # hi: a postposition after a code is a script change too.
        (" HAT138का किराया", " HAT138", TokenKind.CODE),
        # en: an attached possessive is not, and is not parsed: nothing shown.
        (" HAT138's fare", " HAT138's", TokenKind.UNPARSED),
        # pt: a currency sign inside the word.
        (" R$247,55, certo", " R$247,55,", TokenKind.NUMBER_WORDS),
    ],
)
def test_started_words_as_the_adapter_finds_them(line, word, kind):
    assert _next_word(line, 0) == word
    assert split_token(word).kind is kind


@pytest.mark.parametrize(
    "word, heard, shown",
    [
        (" 6941833.", 0, ""),
        (" 6941833.", 2, " 69"),
        (" TAMSDK.", 3, " TAM"),
        (" 6941833.", 7, " 6941833."),
        (" 6941833.", 9, " 6941833."),
        (" ivan_rossi_8555,", 3, " ivan_rossi"),
        (" 8-8-9-0-0.", 2, " 8-8"),
        (" HAT138", 6, " HAT138"),
        ("R$137,30", 3, ""),
        (" HAT138's", 6, ""),
        (" josé_12345", 1, " josé"),
        (" — HAT110", 1, " — H"),
        (" $ 1,387", 1, ""),
        (" card", 0, " card"),
    ],
)
def test_shown_holds_only_the_units_heard(word, heard, shown):
    assert split_token(word).shown(heard) == shown


@pytest.mark.parametrize(
    "played_ms, unit_ms, units",
    [
        (0, 400, 0),
        (919, 400, 0),
        (920, 400, 1),
        (1999, 400, 2),
        (4600, 400, 5),
    ],
)
def test_units_heard_rounds_down_with_the_safety_margin(played_ms, unit_ms, units):
    assert UNIT_SAFETY == 2.3
    assert units_heard(played_ms, unit_ms) == units


def test_a_completed_code_s_time_never_shows_more_than_its_units():
    """Ground truth: a completed 7-digit code at the call's unit (its whole
    sound played) shows fewer units than it has, never more."""

    for unit_ms in (316.0, 384.0, DEFAULT_UNIT_MS, 424.0):
        assert units_heard(7 * unit_ms, unit_ms) == 3


@pytest.mark.parametrize(
    "word, measures",
    [
        (" 6941833.", True),
        (" HAT110", True),
        (" ivan_rossi_8555,", True),
        (" garcia5901", True),
        (" 3RK2T9,", True),
        # Read as words, or not surely by units: they never set the unit.
        (" p.m.", False),
        (" 24-hour", False),
        (" 2-piece", False),
        (" 2GB까지", False),
        (" 750ml", False),
        (" iPhone15", False),
        (" 2026-11-18", False),
        (" hao.h18836", False),
        (" 1,387", False),
        # Read as words: a round number, a number and a short unit, capitals
        # only.
        (" 10000", False),
        (" 1000000", False),
        (" HAT000", False),
        (" AB1000", False),
        (" 128GB", False),
        (" 750ML", False),
        (" 100KM", False),
        (" 1500W", False),
        (" IFOYYZ", False),  # spelled for display, but capitals only
        (" NEVER", False),
        (" GARCIA", False),
        (" SEUNGHYEUN", False),
        (" Garcia5901", False),  # title case fails closed
        # Accepted residual: capitals and digits partly read as a word.
        (" COVID19", True),
        (" RTX4090", True),
        (" H1000X", True),
        # Full-width codes are shown by units, but never measure.
        (" ＩＦＯＹＹＺ", False),
        (" ＡＢＣ１２３４", False),
        (" card", False),
    ],
)
def test_measures_unit_is_stricter_than_display(word, measures):
    assert split_token(word).measures_unit is measures
