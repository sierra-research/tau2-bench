"""Regional values used by airline tools, separate from mutable booking data."""

from pydantic import Field, NaiveDatetime, field_validator

from tau2.domains.airline.data_model import AirportCode
from tau2.environment.regional import DomainRegionalSettings


class AirlineRegionalSettings(DomainRegionalSettings):
    """Reviewed airport catalog, fee units and unchanged scenario wall clock."""

    airports: list[AirportCode] = Field(
        min_length=1,
        description="Available airports in the same regional flight graph.",
    )
    current_datetime: NaiveDatetime = Field(
        description="Fixed scenario wall clock, interpreted in time_zone without conversion."
    )

    @field_validator("airports")
    @classmethod
    def distinct_airports(cls, value: list[AirportCode]) -> list[AirportCode]:
        """Every available airport retains a distinct three-letter IATA code."""
        codes = [airport.iata for airport in value]
        if len(set(codes)) != len(codes) or any(
            len(code) != 3
            or not code.isascii()
            or not code.isalpha()
            or not code.isupper()
            for code in codes
        ):
            raise ValueError(
                "Regional airports require distinct three-letter IATA codes"
            )
        if any(not airport.city.strip() for airport in value):
            raise ValueError("Regional airport city labels must be nonblank")
        return value
