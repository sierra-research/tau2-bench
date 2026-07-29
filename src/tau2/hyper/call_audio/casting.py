# Copyright Sierra
"""Deterministic voice casting for phone-call transcript rendering.

Every case gets a stable (agent voice, customer voice, seed) assignment
derived from its path, so re-renders are reproducible. Assignments can be
exported to a JSON manifest, hand-edited (e.g. to match a voice to the
customer's name or described demographics), and passed back to the renderer.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from pydantic import BaseModel

from tau2.hyper.call_audio.personas import (
    AGENT_VOICES,
    CUSTOMER_VOICES,
    get_call_voice,
)


class CaseCasting(BaseModel):
    case_path: str  # path relative to the casting root
    agent_voice: str
    customer_voice: str
    seed: int
    notes: str = ""


def _stable_seed(rel_path: str) -> int:
    return int(hashlib.md5(rel_path.encode()).hexdigest()[:8], 16)


def _relative_case_path(case_path: Path, root: Path) -> str:
    # Resolve both sides so mixed absolute/relative inputs (e.g. `cast` run
    # with a relative dir, `render` with an absolute file) still match.
    return case_path.resolve().relative_to(root.resolve()).as_posix()


def default_casting(case_path: Path, root: Path) -> CaseCasting:
    """Deterministic casting for a case, keyed on its path relative to root."""
    rel_path = _relative_case_path(case_path, root)
    seed = _stable_seed(rel_path)
    rng = random.Random(seed)
    agent_voice = rng.choice(AGENT_VOICES)
    customer_voice = rng.choice(
        [voice for voice in CUSTOMER_VOICES if voice.name != agent_voice.name]
    )
    return CaseCasting(
        case_path=rel_path,
        agent_voice=agent_voice.name,
        customer_voice=customer_voice.name,
        seed=seed,
    )


class CastingManifest(BaseModel):
    root: str
    cases: list[CaseCasting]

    def lookup(self, case_path: Path) -> CaseCasting | None:
        rel_path = _relative_case_path(case_path, Path(self.root))
        for case in self.cases:
            if case.case_path == rel_path:
                return case
        return None

    def validate_entries(self) -> None:
        for case in self.cases:
            get_call_voice(case.agent_voice)
            get_call_voice(case.customer_voice)
            path = Path(case.case_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    f"manifest case_path must be relative to root without "
                    f"'..' segments: {case.case_path!r}"
                )

    @classmethod
    def build(cls, case_paths: list[Path], root: Path) -> "CastingManifest":
        return cls(
            root=str(root),
            cases=[default_casting(path, root) for path in case_paths],
        )

    @classmethod
    def load(cls, path: Path) -> "CastingManifest":
        manifest = cls.model_validate(json.loads(path.read_text()))
        manifest.validate_entries()
        return manifest

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")
