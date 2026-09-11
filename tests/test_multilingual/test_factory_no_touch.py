# Copyright Sierra
"""Boundary guards for the (parallel-PR) Factory package.

Mirrors ``test_no_touch_language_pack.py``: the Factory is authoring tooling,
so it must never reach into the runtime simulation stack. Statically checked,
so the rules bind from the very first factory module merged:

- no module under ``src/tau2/multilingual/factory/`` imports the user
  simulator, the evaluators, or the voice provider implementations (the
  factory talks to LLMs only through ``tau2.utils.llm_utils``);
- ``data/tau2/multilingual/_factory/`` (factory scratch space) never contains
  a file named exactly ``pack.yaml`` — draft packs must not be discoverable
  by the loader's ``*/pack.yaml`` glob.

Both checks pass trivially while the factory package does not exist yet.
"""

import ast
from pathlib import Path

from tau2.utils import DATA_DIR

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
FACTORY_SRC_DIR = REPO_SRC / "tau2" / "multilingual" / "factory"
FACTORY_DATA_DIR = DATA_DIR / "tau2" / "multilingual" / "_factory"

# Import prefixes that would couple the factory to the runtime simulation
# stack. The factory's only LLM gateway is tau2.utils.llm_utils.
FORBIDDEN_IMPORT_PREFIXES = (
    "tau2.user.user_simulator",  # runtime user simulator(s)
    "tau2.evaluator",  # reward evaluators
    "tau2.voice.synthesis",  # TTS providers
    "tau2.voice.transcription",  # STT providers
    "tau2.voice.audio_native",  # audio-native providers
    "litellm",  # raw provider client; use tau2.utils.llm_utils
)


def _imported_modules(source: str, filename: str) -> set[str]:
    """Every module name imported by a Python source file (static)."""
    imported: set[str] = set()
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: stays inside the factory
                continue
            module = node.module or ""
            imported.add(module)
            # `from tau2.user import user_simulator` must also be caught.
            imported.update(f"{module}.{alias.name}" for alias in node.names)
    return imported


def test_factory_modules_do_not_import_runtime_stack():
    if not FACTORY_SRC_DIR.is_dir():
        return  # trivially passes until the parallel PRs land
    violations: list[str] = []
    py_files = sorted(FACTORY_SRC_DIR.rglob("*.py"))
    for py_file in py_files:
        for module in sorted(_imported_modules(py_file.read_text(), str(py_file))):
            for prefix in FORBIDDEN_IMPORT_PREFIXES:
                if module == prefix or module.startswith(prefix + "."):
                    violations.append(
                        f"{py_file.relative_to(REPO_SRC)} imports '{module}' "
                        f"(forbidden prefix '{prefix}')"
                    )
    assert not violations, (
        "factory modules must not import the runtime simulation stack:\n"
        + "\n".join(violations)
    )


def test_factory_scratch_space_has_no_discoverable_pack_yaml():
    if not FACTORY_DATA_DIR.is_dir():
        return  # trivially passes until the factory writes scratch data
    offenders = sorted(FACTORY_DATA_DIR.rglob("pack.yaml"))
    assert not offenders, (
        "draft packs in _factory/ must not be named pack.yaml (the loader "
        f"glob would discover them): {[str(p) for p in offenders]}"
    )
