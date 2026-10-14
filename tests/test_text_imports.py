import subprocess
import sys
from pathlib import Path


def test_text_runner_imports_without_optional_dependencies():
    project_root = Path(__file__).resolve().parents[1]
    script = """
import builtins

real_import = builtins.__import__


def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'rank_bm25', 'scipy'}:
        raise ModuleNotFoundError(f'blocked optional dependency: {name}')
    return real_import(name, *args, **kwargs)


builtins.__import__ = guarded_import
from tau2.runner import run_simulation

assert run_simulation.__name__ == 'run_simulation'
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        check=True,
    )
