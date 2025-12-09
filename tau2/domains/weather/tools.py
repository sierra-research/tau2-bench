
from __future__ import annotations

import logging
from copy import deepcopy
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.domains.weather.data_model import *


# ------------------------------------------------------------------------------
# Minimal tool framework (following the formats used in airline tools)
# ------------------------------------------------------------------------------

# class ToolType(Enum):
#     READ = "read"
#     WRITE = "write"
#     GENERIC = "generic"


# def is_tool(tool_type: ToolType):
#     def decorator(func):
#         setattr(func, "_tool_type", tool_type.value)
#         return func
#     return decorator


# logger = logging.getLogger(__name__)





# ------------------------------------------------------------------------------
# Weather Tools
# ------------------------------------------------------------------------------

class WeatherTools(ToolKitBase):  # Tools
    """All the tools for the weather domain."""

    db: WeatherDB

    def __init__(self, db: WeatherDB) -> None:
        super().__init__(db)

    # -----------------------------
    # Internal helper methods
    # -----------------------------
    def _get_location(self, location_id: str) -> Location:
        if location_id not in self.db.locations:
            raise ValueError(f"Location {location_id} not found")
        return self.db.locations[location_id]

    def _get_forecast(self, forecast_id: str) -> Forecast:
        if forecast_id not in self.db.forecasts:
            raise ValueError(f"Forecast {forecast_id} not found")
        return self.db.forecasts[forecast_id]

    def _get_observation(self, observation_id: str) -> Observation:
        if observation_id not in self.db.observations:
            raise ValueError(f"Observation {observartion_id} not found")
        return self.db.observations[observation_id]

    def _parse_utc(self, ts: str) -> datetime:
        # Accept ISO 8601 with optional 'Z'
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(ts)
        except Exception as e:
            raise ValueError(f"Invalid UTC datetime format: {ts}") from e

    def _time_ranges_overlap(self, a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
        a0 = self._parse_utc(a_start)
        a1 = self._parse_utc(a_end)
        b0 = self._parse_utc(b_start)
        b1 = self._parse_utc(b_end)
        return max(a0, b0) <= min(a1, b1)

    def _now_iso(self) -> str:
        # Return a fixed or current timestamp; align with airline tools style if needed
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # -----------------------------
    # Read tools
    # -----------------------------
    @is_tool(ToolType.READ)
    def list_all_locations(self) -> List[LocationCode]:
        """
        Returns a list of all available locations with a simple label.
        """
        results: List[LocationCode] = []
        for loc in self.db.locations.values():
            label = f"{loc.name.city}, {loc.name.state}, {loc.name.country}"
            results.append(LocationCode(location_id=loc.location_id, label=label))
        return results

    @is_tool(ToolType.READ)
    def get_location_details(self, location_id: str) -> Location:
        """
        Get the details of a location.

        Args:
            location_id: The location ID.

        Returns:
            The Location object.
        """
        return self._get_location(location_id)

    @is_tool(ToolType.READ)
    def get_forecast_details(self, forecast_id: str) -> Forecast:
        """
        Get the details of a forecast by its ID.

        Args:
            forecast_id: The forecast ID.

        Returns:
            The Forecast object.
        """
        return self._get_forecast(forecast_id)

    @is_tool(ToolType.READ)
    def search_forecasts(
        self,
        location_id: str,
        valid_from_utc: Optional[str] = None,
        valid_to_utc: Optional[str] = None,
        source_model: Optional[str] = None,
    ) -> List[Forecast]:
        """
        Search for forecasts for a specific location, optionally filtered by validity window and model.

        Args:
            location_id: The location ID.
            valid_from_utc: Start of desired valid window (inclusive).
            valid_to_utc: End of desired valid window (inclusive).
            source_model: Optional source model filter, e.g., "GFS", "ECMWF".

        Returns:
            A list of Forecast objects.
        """
        self._get_location(location_id)  # validate exists

        results: List[Forecast] = []
        for f in self.db.forecasts.values():
            if f.location_id != location_id:
                continue
            if source_model and f.source_model != source_model:
                continue
            if valid_from_utc and valid_to_utc:
                if not self._time_ranges_overlap(
                    f.valid_from_utc, f.valid_to_utc, valid_from_utc, valid_to_utc
                ):
                    continue
            results.append(f)
        # Sort by issued_at descending (most recent first)
        results.sort(key=lambda x: self._parse_utc(x.issued_at_utc), reverse=True)
        return results

    @is_tool(ToolType.READ)
    def get_hourly_forecast_window(
        self,
        location_id: str,
        start_utc: str,
        end_utc: str,
        source_model: Optional[str] = None,
    ) -> List[HourlyForecastEntry]:
        """
        Return all hourly forecast entries for a location within a time window.
        If multiple forecasts overlap, entries are merged; more recently issued forecasts override earlier duplicates.

        Args:
            location_id: The location ID.
            start_utc: Inclusive start time (UTC ISO 8601).
            end_utc: Inclusive end time (UTC ISO 8601).
            source_model: Optional model filter.

        Returns:
            A list of HourlyForecastEntry.
        """
        self._get_location(location_id)  # validate exists
        start_dt = self._parse_utc(start_utc)
        end_dt = self._parse_utc(end_utc)
        if start_dt > end_dt:
            raise ValueError("start_utc must be <= end_utc")

        # Collect candidate forecasts
        candidates = self.search_forecasts(
            location_id=location_id,
            valid_from_utc=start_utc,
            valid_to_utc=end_utc,
            source_model=source_model,
        )

        # Merge hourly entries; prefer latest issued forecast for the same hour
        slot_map: Dict[str, Tuple[datetime, HourlyForecastEntry]] = {}
        for f in candidates:
            issued_dt = self._parse_utc(f.issued_at_utc)
            for h in f.hourly:
                t = self._parse_utc(h.time_utc)
                if t < start_dt or t > end_dt:
                    continue
                # replace if not present or this forecast is more recent
                cur = slot_map.get(h.time_utc)
                if cur is None or issued_dt > cur[0]:
                    slot_map[h.time_utc] = (issued_dt, h)

        merged = [v[1] for v in sorted(slot_map.values(), key=lambda x: self._parse_utc(x[1].time_utc))]
        return merged

    @is_tool(ToolType.READ)
    def get_daily_forecast_range(
        self,
        location_id: str,
        start_date: str,
        end_date: str,
        source_model: Optional[str] = None,
    ) -> List[DailyForecastEntry]:
        """
        Return daily forecasts for a location between two dates (YYYY-MM-DD inclusive).
        If multiple forecasts provide the same date, the most recent issued forecast is used.

        Args:
            location_id: The location ID.
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            source_model: Optional model filter.

        Returns:
            List of DailyForecastEntry for the requested dates.
        """
        self._get_location(location_id)
        if start_date > end_date:
            raise ValueError("start_date must be <= end_date")

        candidates = self.search_forecasts(
            location_id=location_id,
            source_model=source_model,
        )
        # Map date -> (issued_at, entry)
        day_map: Dict[str, Tuple[datetime, DailyForecastEntry]] = {}
        for f in candidates:
            issued_dt = self._parse_utc(f.issued_at_utc)
            for d in f.daily:
                if d.date < start_date or d.date > end_date:
                    continue
                cur = day_map.get(d.date)
                if cur is None or issued_dt > cur[0]:
                    day_map[d.date] = (issued_dt, d)

        return [day_map[d][1] for d in sorted(day_map.keys())]

    @is_tool(ToolType.READ)
    def get_current_conditions(self, location_id: str) -> Observation:
        """
        Get the most recent observation for a location.

        Args:
            location_id: The location ID.

        Returns:
            The latest Observation object for the location.
        """
        self._get_location(location_id)
        cands = [o for o in self.db.observations.values() if o.location_id == location_id]
        if not cands:
            raise ValueError(f"No observations found for location {location_id}")
        cands.sort(key=lambda o: self._parse_utc(o.timestamp_utc), reverse=True)
        return cands[0]

    @is_tool(ToolType.READ)
    def get_observations(
        self,
        location_id: str,
        start_utc: str,
        end_utc: str,
        qc_filter: Optional[str] = None,  # "passed", "suspect", "failed"
    ) -> List[Observation]:
        """
        Get observations in a time range for a location.

        Args:
            location_id: The location ID.
            start_utc: Inclusive start time.
            end_utc: Inclusive end time.
            qc_filter: Optional QC flag filter.

        Returns:
            A list of observations ordered by time ascending.
        """
        self._get_location(location_id)
        start_dt = self._parse_utc(start_utc)
        end_dt = self._parse_utc(end_utc)
        if start_dt > end_dt:
            raise ValueError("start_utc must be <= end_utc")

        out: List[Observation] = []
        for o in self.db.observations.values():
            if o.location_id != location_id:
                continue
            t = self._parse_utc(o.timestamp_utc)
            if t < start_dt or t > end_dt:
                continue
            if qc_filter and o.quality_control.qc_flag != qc_filter:
                continue
            out.append(o)
        out.sort(key=lambda o: self._parse_utc(o.timestamp_utc))
        return out

    @is_tool(ToolType.WRITE)
    def verify_forecast_for_date(
        self,
        forecast_id: str,
        date: str,
        actual_high_c: float,
        actual_low_c: float,
        actual_precip_mm: float,
        notes: str = "",
        status: str = "verified",
    ) -> Forecast:
        """
        Add or update verification record for a forecast on a specific date.

        Args:
            forecast_id: The forecast ID.
            date: The date to verify (YYYY-MM-DD).
            actual_high_c: Observed high temperature.
            actual_low_c: Observed low temperature.
            actual_precip_mm: Observed precipitation total in mm.
            notes: Optional notes.
            status: Verification status: "pending", "verified", or "revised".

        Returns:
            The updated Forecast object.
        """
        if status not in {"pending", "verified", "revised"}:
            raise ValueError("Invalid status")
        fc = self._get_forecast(forecast_id)
        fc.verification_by_date[date] = VerificationEntry(
            status=status,
            actual_high_c=actual_high_c,
            actual_low_c=actual_low_c,
            actual_precip_mm=actual_precip_mm,
            notes=notes,
        )
        return fc

    # -----------------------------
    # Generic tools
    # -----------------------------
    @is_tool(ToolType.GENERIC)
    def calculate(self, expression: str) -> str:
        """
        Calculate the result of a mathematical expression.

        Args:
            expression: The mathematical expression to calculate.

        Returns:
            The result of the mathematical expression (string).
        """
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         - the user explicitly asks for a human agent
         - given the policy and the available tools, you cannot solve the user's issue.
        """
        return "Transfer successful"

