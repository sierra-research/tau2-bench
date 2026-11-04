"""Reviewed airport city labels bound to one regional airline map."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from tau2.multilingual.localization import GenerationProvenance, LocalizationModel

CityLabel = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
IataCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class AirportCity(LocalizationModel):
    """City served by one mapped airport, with its source correspondence."""

    source_iata: IataCode = Field(description="Canonical airport being replaced.")
    iata: IataCode = Field(description="Exact reviewed regional airport code.")
    native_city: CityLabel = Field(
        description="City name written in the call language."
    )
    romanized_city: CityLabel = Field(
        description="Latin-script form of that city name."
    )
    english_city: CityLabel = Field(description="English city label for review.")
    notes: str = Field(description="English explanation of the served-city choice.")


class AirportCityReply(LocalizationModel):
    """Complete model reply; duplicate graph nodes are never silently collapsed."""

    airports: list[AirportCity] = Field(
        min_length=1, description="One row per mapped airport."
    )

    @model_validator(mode="after")
    def distinct_airports(self):
        """Several airports may serve one city, but every airport appears once."""
        for field in ("source_iata", "iata"):
            if len({getattr(row, field) for row in self.airports}) != len(
                self.airports
            ):
                raise ValueError(f"airport catalog has duplicate {field} values")
        return self


class AirportCatalog(AirportCityReply):
    """City-label candidate; publication requires its separate ReviewApproval."""

    schema_version: Literal["regional-airport-catalog-v1"] = (
        "regional-airport-catalog-v1"
    )
    revision: str = Field(min_length=1, description="Immutable city-catalog revision.")
    language: str = Field(min_length=1, description="Regional map's spoken language.")
    country_code: str = Field(
        pattern=r"^[A-Z]{2}$", description="Regional map's country."
    )
    map_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Semantic digest of the exact reviewed compact airline map.",
    )
    provenance: GenerationProvenance = Field(
        description="Fixed prompt and model settings that produced these labels."
    )
