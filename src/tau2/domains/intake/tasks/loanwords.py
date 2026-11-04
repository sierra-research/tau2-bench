# Copyright Sierra
"""Validated source-language labels for foreign-name pronunciation metadata."""

from enum import Enum


class LoanLanguage(str, Enum):
    """Source-language labels carried by reviewed bank pronunciations."""

    CZECH = "ces"
    WELSH = "cym"
    GERMAN = "deu"
    FRENCH = "fra"
    SCOTTISH_GAELIC = "gla"
    IRISH = "gle"
    HUNGARIAN = "hun"
    ITALIAN = "ita"
    DUTCH = "nld"
    NORWEGIAN = "nob"
    POLISH = "pol"
    PORTUGUESE = "por"
    ROMANIAN = "ron"
    SPANISH = "spa"
    SWEDISH = "swe"
    TURKISH = "tur"
