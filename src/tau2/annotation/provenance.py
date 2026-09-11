# Copyright Sierra
"""Typed provenance records for annotation packet builds.

``ArtifactManifest.provenance`` is a free-form JSON blob (each packet family
records different pipeline knobs there), but the part that says WHAT WAS READ
is a closed contract shared by two modules — the builder writes it and
``source_audit`` resolves it against the run tree. That part is modeled here
and validated at every boundary; the surrounding blob stays a short-lived dict
that only ever touches JSON.

A build's sources are :class:`RecordedRun`s: a path AND the identity of the
run that path held at build time. The identity is what makes the record
durable — see ``tau2.data_model.run_identity``.
"""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

from tau2.data_model.run_identity import RunIdentity


class RecordedRun(BaseModel):
    """One results source a build read: where it was, and WHICH run it was.

    ``path`` points at a run dir OR at its ``results.json`` (retired matrix
    builds recorded the file) — a repair must preserve whichever shape was
    recorded.
    """

    model_config = ConfigDict(extra="forbid")

    path: Annotated[str, Field(description="The path as written at build time.")]
    identity: Annotated[
        RunIdentity,
        Field(description="Identity of the run that path held at build time."),
    ]


class BuildFilters(BaseModel):
    """The selection knobs one packet build invocation ran with."""

    model_config = ConfigDict(extra="forbid")

    reward: Annotated[
        Optional[str], Field(description="Reward filter expression, e.g. '< 1'.")
    ] = None
    tasks: Annotated[
        Optional[list[str]], Field(description="Task-id allowlist, if any.")
    ] = None
    shuffle: Annotated[bool, Field(description="Whether selection was shuffled.")] = (
        False
    )
    seed: Annotated[
        Optional[int], Field(description="Shuffle seed (None when not shuffled).")
    ] = None
    max_items: Annotated[
        Optional[int], Field(description="Cap on exported simulations.")
    ] = None


class BuildRecord(BaseModel):
    """One packet build invocation. ``--append`` adds one; none is discarded."""

    model_config = ConfigDict(extra="forbid")

    results: Annotated[
        list[RecordedRun], Field(description="Results sources this build read.")
    ]
    filters: Annotated[BuildFilters, Field(description="Selection knobs used.")]
    timestamp: Annotated[str, Field(description="Build wall-clock time (UTC).")]
    seed_evidence: Annotated[
        Optional[bool],
        Field(description="VE packets: whether judge seeds were rendered."),
    ] = None
    seeds: Annotated[
        Optional[int], Field(description="VE packets: seeds rendered.")
    ] = None
    seeds_unanchored: Annotated[
        Optional[int],
        Field(
            description="VE packets: seeds the quote search could not place on "
            "a tick span (a spike means quote/transcript drift)."
        ),
    ] = None


def build_records(provenance: dict) -> list[BuildRecord]:
    """Validate the ``builds`` list out of a manifest's provenance blob.

    Raises ``pydantic.ValidationError`` on anything that is not the current
    contract — packets are regenerable, so a manifest that predates it is a
    loud rebuild, never a silently-tolerated second shape.
    """
    return [
        BuildRecord.model_validate(build) for build in (provenance.get("builds") or [])
    ]
