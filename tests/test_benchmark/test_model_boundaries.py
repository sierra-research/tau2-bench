"""Shared benchmark contracts can be read without execution or scoring imports."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", ["benchmark", "simulation"])
def test_benchmark_models_build_in_either_import_order_without_runner(first):
    code = """
import importlib
import importlib.util
import sys
import types

# Isolate schema imports from tau2's pre-existing eager public runner exports.
spec = importlib.util.find_spec("tau2")
package = types.ModuleType("tau2")
package.__path__ = list(spec.submodule_search_locations)
sys.modules["tau2"] = package
importlib.import_module("tau2.data_model." + sys.argv[1])
from tau2.data_model.benchmark import BenchmarkPlan, SavedBenchmark, SavedBenchmarkCell

for model in (BenchmarkPlan, SavedBenchmark, SavedBenchmarkCell):
    assert model.__module__ == "tau2.data_model.benchmark"
    model.model_json_schema()
for name in sys.modules:
    assert name.split(".")[:2] not in [["tau2", "runner"], ["tau2", "metrics"]], name
"""
    result = subprocess.run(
        [sys.executable, "-c", code, first],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
