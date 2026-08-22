"""Focused tests for the disclosed subset shell-order oracle scope."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_oracle_module():
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "reproduction" / "tau3_banking" / "compare_shell_oracle.py"
    spec = importlib.util.spec_from_file_location("tau3_compare_shell_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


oracle = _load_oracle_module()


@pytest.mark.parametrize(
    "output",
    [
        "./doc.md\n",
        "banner\n./doc.md:matching text\n",
        "./one.md\n./two.md:value\n",
    ],
)
def test_recursive_filename_line_accepts_bare_and_colon_forms(output):
    assert oracle.has_recursive_filename_line(output)


@pytest.mark.parametrize(
    "output",
    [
        "prefix ./doc.md:value\n",
        "./nested/doc.md:value\n",
        "./not-markdown.txt:value\n",
        "./doc.md-context\n",
        "(no output)",
    ],
)
def test_recursive_filename_line_rejects_non_fixture_forms(output):
    assert not oracle.has_recursive_filename_line(output)


def test_unique_expected_by_command_rejects_conflicts_before_filtering():
    fixtures = [
        {"command": "grep", "expected": "./doc.md\n"},
        {"command": "grep", "expected": "different\n"},
    ]

    with pytest.raises(oracle.OracleError, match="conflicts"):
        oracle.unique_expected_by_command(fixtures)


def test_recursive_scope_selects_exactly_59_expected_outputs():
    expected = {
        f"command-{index}": f"banner\n./doc-{index}.md:value\n" for index in range(59)
    }
    expected["not-recursive"] = "ordinary output"

    selected = oracle.select_oracle_scope(
        expected, mode="subset", scope="recursive-filename-lines"
    )

    assert len(selected) == 59
    assert "not-recursive" not in selected


def test_recursive_scope_rejects_wrong_mode_and_cardinality():
    with pytest.raises(oracle.OracleError, match="not defined"):
        oracle.select_oracle_scope({}, mode="full", scope="recursive-filename-lines")
    with pytest.raises(oracle.OracleError, match="expected 59"):
        oracle.select_oracle_scope(
            {"one": "./one.md\n"},
            mode="subset",
            scope="recursive-filename-lines",
        )
