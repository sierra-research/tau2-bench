"""Tool-call validity — an *annotating* node (classify, don't transform).

For each assistant `tool_calls` entry it checks:
  * the called function name exists in the record's `tools`, and
  * the `arguments` validate against that tool's JSON-schema (`parameters`).

It does NOT drop anything itself. Instead it writes a classification onto the
record so a downstream node (e.g. `filter_by_field`) can act on it:

    record[field]        -> bool   (default field: "tool_calls_valid")
    record[errors_field] -> list[str] of human-readable problems

This separation is the point: a classifier emits a signal; later nodes consume it.

Compose:
    Pipeline([
        ValidateToolCalls(),                              # annotate
        FilterByField(field="tool_calls_valid", is_true=True),  # act on it
        DropFields(fields=["tool_calls_valid", "tool_call_errors"]),  # clean up
    ])
"""
from __future__ import annotations

import json
from typing import Any, Optional

from ..core import MapNode, Record, register

_MESSAGE_KEYS = ("conversations", "messages", "conversation", "turns")

# Many tool-use datasets (ToolMind included) declare parameter schemas with
# Python-style type names rather than JSON-Schema ones. Normalize so jsonschema
# can actually validate the arguments instead of erroring on "Unknown type 'dict'".
_TYPE_ALIASES = {
    "dict": "object", "object": "object",
    "list": "array", "array": "array", "tuple": "array",
    "float": "number", "double": "number", "number": "number",
    "int": "integer", "integer": "integer", "long": "integer",
    "str": "string", "string": "string", "text": "string",
    "bool": "boolean", "boolean": "boolean",
    "none": "null", "null": "null",
}


def _normalize_schema(schema):
    """Recursively map Python-style type names to JSON-Schema types."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = _TYPE_ALIASES.get(v.lower(), v)
            elif k == "type" and isinstance(v, list):
                out[k] = [_TYPE_ALIASES.get(x.lower(), x) if isinstance(x, str) else x for x in v]
            elif isinstance(v, (dict, list)):
                out[k] = _normalize_schema(v)
            else:
                out[k] = v
        return out
    if isinstance(schema, list):
        return [_normalize_schema(x) for x in schema]
    return schema


def _messages(record: Record) -> list[dict]:
    for k in _MESSAGE_KEYS:
        v = record.get(k)
        if v:
            return v
    return []


def _tool_schemas(record: Record) -> dict[str, Any]:
    """name -> parameters-schema (or None if the tool declares no params)."""
    schemas: dict[str, Any] = {}
    for t in record.get("tools") or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        name = fn.get("name")
        if name:
            schemas[name] = fn.get("parameters")
    return schemas


def _iter_tool_calls(message: dict):
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", tc) if isinstance(tc, dict) else {}
        yield fn.get("name"), fn.get("arguments")


@register("validate_tool_calls")
class ValidateToolCalls(MapNode):
    """Annotate each record with tool-call validity (bool) + error list.

    Args:
        field: where to write the boolean verdict.
        errors_field: where to write the list of error strings.
        require_tool_call: if True, a record with no tool calls at all is marked
            invalid (useful when you only want trajectories that exercise tools).
    """

    def __init__(
        self,
        field: str = "tool_calls_valid",
        errors_field: str = "tool_call_errors",
        require_tool_call: bool = False,
        normalize_schema: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.field = field
        self.errors_field = errors_field
        self.require_tool_call = require_tool_call
        self.normalize_schema = normalize_schema
        self._n_invalid = 0

    def _validator(self, schema: Any):
        # Lazy import keeps the base framework dependency-free.
        from jsonschema import Draft7Validator
        if self.normalize_schema:
            schema = _normalize_schema(schema)
        return Draft7Validator(schema)

    def transform(self, record: Record) -> Record:
        schemas = _tool_schemas(record)
        errors: list[str] = []
        n_calls = 0

        for msg in _messages(record):
            for name, args in _iter_tool_calls(msg):
                n_calls += 1
                if name not in schemas:
                    errors.append(f"unknown tool: {name!r}")
                    continue
                schema = schemas[name]
                if not schema:  # None / {} -> no declared constraints
                    continue
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        errors.append(f"{name}: arguments are not valid JSON")
                        continue
                if args is None:
                    args = {}
                try:
                    v = self._validator(schema)
                    for e in v.iter_errors(args):
                        loc = "/".join(str(p) for p in e.absolute_path) or "<root>"
                        errors.append(f"{name}: {loc}: {e.message}")
                except Exception as e:  # malformed schema, etc.
                    errors.append(f"{name}: schema error: {e}")

        if self.require_tool_call and n_calls == 0:
            errors.append("no tool calls present")

        valid = not errors
        if not valid:
            self._n_invalid += 1
            self.stats.extra["invalid"] = self._n_invalid
        out = dict(record)
        out[self.field] = valid
        out[self.errors_field] = errors
        return out
