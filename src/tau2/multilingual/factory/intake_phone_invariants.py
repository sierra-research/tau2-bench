"""Source-relative phone error shapes, independent of national numbering lengths."""

import re
from typing import Annotated

from pydantic import Field, model_validator

from tau2.multilingual.localization import LocalizationModel


class PhoneDigitShape(LocalizationModel):
    """Dialing digits and extension digits counted separately from formatting."""

    main: Annotated[
        int,
        Field(
            ge=1,
            description="ASCII dialing digits, including any country code; formatting is ignored.",
        ),
    ]
    extension: Annotated[
        int,
        Field(
            ge=0,
            description="ASCII extension digits after the existing protocol or localized extension marker.",
        ),
    ]


def phone_digit_shape(value: str) -> PhoneDigitShape:
    """Count ASCII dialing digits, recognizing existing localized extension notation."""
    match = re.fullmatch(
        r"([+()0-9 .-]+?)(?:\s*(?:ext\.?|x|#|ramal|내선|naeseon)\s*([0-9]+)(?:번|beon)?)?",
        value.strip(),
        re.I,
    )
    if match is None:
        raise ValueError(
            "Phone error-shape checks require ASCII dialing syntax and a recognized extension marker"
        )
    return PhoneDigitShape(
        main=len(re.findall(r"[0-9]", match[1])), extension=len(match[2] or "")
    )


class PhoneDecoyLengthEvidence(LocalizationModel):
    """Actual primary/decoy strings and reproducible source-relative error evidence."""

    source_primary: Annotated[
        str,
        Field(
            description="Exact source primary from the same selected written occurrence."
        ),
    ]
    source_value: Annotated[
        str, Field(description="Exact source decoy at its original array position.")
    ]
    current_primary: Annotated[
        str,
        Field(
            description="Authoritative completed localized primary in this representation."
        ),
    ]
    current_value: Annotated[
        str, Field(description="Current proposed decoy in this representation.")
    ]
    source_primary_shape: Annotated[
        PhoneDigitShape, Field(description="Parsed source primary digit counts.")
    ]
    source_decoy_shape: Annotated[
        PhoneDigitShape, Field(description="Parsed source decoy digit counts.")
    ]
    current_primary_shape: Annotated[
        PhoneDigitShape, Field(description="Parsed completed primary digit counts.")
    ]
    current_decoy_shape: Annotated[
        PhoneDigitShape, Field(description="Parsed proposed decoy digit counts.")
    ]

    @model_validator(mode="after")
    def verify_counts(self):
        for value, shape in (
            (self.source_primary, self.source_primary_shape),
            (self.source_value, self.source_decoy_shape),
            (self.current_primary, self.current_primary_shape),
            (self.current_value, self.current_decoy_shape),
        ):
            if phone_digit_shape(value) != shape:
                raise ValueError(
                    "Phone digit-count evidence differs from its exact written value"
                )
        return self

    @property
    def preserves_source_error_shape(self) -> bool:
        """Keep main-number and extension errors separate; allow locale length changes."""
        source_main = self.source_decoy_shape.main - self.source_primary_shape.main
        current_main = self.current_decoy_shape.main - self.current_primary_shape.main
        # A wholly omitted/added extension stays omitted/added, even when the
        # primary's ordinary local extension length changes. Otherwise retain
        # the extension's own extra/missing-digit relation.
        source_pair = (
            self.source_primary_shape.extension,
            self.source_decoy_shape.extension,
        )
        current_pair = (
            self.current_primary_shape.extension,
            self.current_decoy_shape.extension,
        )
        if bool(source_pair[0]) != bool(source_pair[1]):
            extension_ok = tuple(map(bool, source_pair)) == tuple(
                map(bool, current_pair)
            )
        else:
            extension_ok = (
                current_pair[1] - current_pair[0] == source_pair[1] - source_pair[0]
            )
        return source_main == current_main and extension_ok


def phone_decoy_length_evidence(
    source_primary: str, source_value: str, current_primary: str, current_value: str
) -> PhoneDecoyLengthEvidence:
    """Compute factual comparisons without asserting subscriber or country validity."""
    return PhoneDecoyLengthEvidence(
        source_primary=source_primary,
        source_value=source_value,
        current_primary=current_primary,
        current_value=current_value,
        source_primary_shape=phone_digit_shape(source_primary),
        source_decoy_shape=phone_digit_shape(source_value),
        current_primary_shape=phone_digit_shape(current_primary),
        current_decoy_shape=phone_digit_shape(current_value),
    )
