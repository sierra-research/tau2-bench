# Copyright Sierra
"""Every bed a voice task names must exist on disk.

These references are plain strings in checked-in JSON, resolved only when a
simulation actually reaches that task, so a renamed or retired bed leaves the
task set silently broken until a run dies on it. That is what happened to the
mock domain: it named ``traffic.wav``, ``baby.wav`` and ``drilling.wav`` from
the original release while the verified bed inventory was replaced underneath
it, and ``--domain mock --audio-native`` failed on 6 of 18 configs from then
until 2026-07-28.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tau2.utils import DATA_DIR
from tau2.voice.utils.utils import BURST_NOISE_FILES, CONTINUOUS_NOISE_FILES

DOMAINS_DIR = DATA_DIR / "tau2" / "domains"


def _named_beds(node: object, out: list[tuple[str, str]]) -> None:
    """Collect every (kind, filename) bed reference under ``node``."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "background_noise_file" and value:
                out.append(("continuous", value))
            elif key == "burst_noise_files" and value:
                out.extend(("burst", name) for name in value)
            else:
                _named_beds(value, out)
    elif isinstance(node, list):
        for value in node:
            _named_beds(value, out)


def _voice_task_files() -> list[Path]:
    return sorted(DOMAINS_DIR.glob("*/tasks_voice.json"))


def test_there_are_voice_task_files_to_check():
    # Guards the parametrization below: a glob that matches nothing would make
    # every check vacuously pass.
    assert _voice_task_files()


@pytest.mark.parametrize("path", _voice_task_files(), ids=lambda p: p.parent.name)
def test_every_named_bed_exists(path: Path):
    available = {
        "continuous": {p.name for p in CONTINUOUS_NOISE_FILES},
        "burst": {p.name for p in BURST_NOISE_FILES},
    }
    named: list[tuple[str, str]] = []
    _named_beds(json.loads(path.read_text()), named)

    missing = sorted(
        {(kind, name) for kind, name in named if name not in available[kind]}
    )
    assert not missing, (
        f"{path.relative_to(DATA_DIR)} names beds that do not exist: {missing}. "
        "Repoint them at a verified bed or drop the reference — a run reaching "
        "one of these tasks dies on FileNotFoundError."
    )
