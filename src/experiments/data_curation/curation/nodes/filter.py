"""Generic annotation-consuming filters.

`FilterByField` keeps/drops records based on a field that an upstream node wrote
(a category, a validity bool, a numeric score, ...). This is how classify->act
composes. `DropFields` removes annotation fields before writing the final output.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from ..core import FilterNode, MapNode, Record, register

_UNSET = object()


@register("filter_by_field")
class FilterByField(FilterNode):
    """Keep records whose `field` satisfies the given condition(s).

    Conditions (combine freely; all must hold):
        is_true: bool  — truthiness of the value must equal this
        equals:  any   — exact equality
        in_:     list  — membership
        min/max: num   — inclusive numeric bounds
    `missing` controls records lacking the field: "drop" (default) or "keep".
    """

    def __init__(
        self,
        field: str,
        is_true: Optional[bool] = None,
        equals: Any = _UNSET,
        in_: Optional[list] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        missing: str = "drop",
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        if missing not in ("drop", "keep"):
            raise ValueError("missing must be 'drop' or 'keep'")
        if (
            is_true is None
            and equals is _UNSET
            and in_ is None
            and min is None
            and max is None
        ):
            raise ValueError("FilterByField needs at least one condition")
        self.field = field
        self.is_true = is_true
        self.equals = equals
        self.in_ = in_
        self.min = min
        self.max = max
        self.missing = missing

    def keep(self, record: Record) -> bool:
        if self.field not in record:
            return self.missing == "keep"
        v = record[self.field]
        if self.is_true is not None and bool(v) != self.is_true:
            return False
        if self.equals is not _UNSET and v != self.equals:
            return False
        if self.in_ is not None and v not in self.in_:
            return False
        if self.min is not None and (v is None or v < self.min):
            return False
        if self.max is not None and (v is None or v > self.max):
            return False
        return True


@register("drop_fields")
class DropFields(MapNode):
    """Strip annotation fields from each record (e.g. before writing output)."""

    def __init__(self, fields: Iterable[str], name: Optional[str] = None):
        super().__init__(name=name)
        self.fields = set(fields)

    def transform(self, record: Record) -> Record:
        return {k: v for k, v in record.items() if k not in self.fields}
