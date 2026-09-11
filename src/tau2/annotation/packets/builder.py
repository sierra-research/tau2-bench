# Copyright Sierra
"""Packet build orchestration: results dirs → standalone HTML annotation packet.

A packet is a directory of per-simulation pages (``task_<id>_sim_<id>/index.html``
plus copied audio), an index page, and a ``manifest.json`` at the packet root
(``ArtifactManifest`` with typed per-conversation ``entries``). The manifest is
the packet's source of truth:

- ``batch_id`` is CONTENT-DERIVED from the initial entry set (idempotent:
  same inputs + code version reproduce it) and PRESERVED across ``--append``
  (the id namespaces the annotators' browser localStorage — it must not move
  under saved drafts).
- ``provenance`` is an append-preserving ``builds`` list of typed
  ``BuildRecord``s: one entry per build invocation (results sources, filters,
  timestamp); ``--append`` merges the prior manifest's entries in front of the
  new one, so no invocation's provenance is ever discarded. Each source is a
  ``RecordedRun`` — a path AND the identity of the run that path held — so a
  later audit can tell a moved run from a regenerated one
  (``tau2 annotate source-audit``).
- ``--append`` dedupes new sims by ``entry.sim_id`` (never directory-name
  parsing) AND within the invocation (overlapping inputs export a sim once,
  first occurrence wins), exports only new sims, rewrites the merged manifest
  and regenerates the index from entries — prior entries keep their real
  experiment/domain labels. Appending a different form OR a different
  ``--batch-name`` to an existing batch is an error (the batch identity must
  stay stable).

Memory: sims are streamed twice rather than materialized — a cheap first pass
collects only sim ids (the content-derived batch id needs the full new-id set
before any page renders), then a second pass renders the selected sims one at
a time, so large audio-bearing runs never sit in memory whole.
"""

import hashlib
import random
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
from pydantic import BaseModel, Field, ValidationError, model_validator

from tau2.annotation.artifacts import (
    ArtifactManifest,
    PacketEntry,
    derive_batch_id,
    git_sha,
    write_json_artifact,
)
from tau2.annotation.loading import (
    LoadedSim,
    iter_loaded_sims,
    load_policy,
    load_task,
)
from tau2.annotation.packets.forms import (
    FormType,
    packet_kind_for,
    rubric_json_for_provenance,
)
from tau2.annotation.packets.page import (
    IndexRowVM,
    PacketIdentity,
    PageNavVM,
    build_nav_vms,
    render_index_page,
    render_nav_fragment,
    render_sim_page,
)
from tau2.annotation.provenance import (
    BuildFilters,
    BuildRecord,
    RecordedRun,
    build_records,
)
from tau2.data_model.run_identity import compute_run_identity
from tau2.data_model.simulation import SimulationRun

PACKET_MANIFEST_NAME = "manifest.json"
DEFAULT_OUTPUT_ROOT = Path("data/annotations")
AUDIO_FILENAME = "audio.wav"


class PacketBuildOptions(BaseModel):
    """Everything one ``tau2 annotate packets`` invocation needs."""

    form: Annotated[FormType, Field(description="Which annotation form to render.")]
    batch_name: Annotated[
        str, Field(description="Human batch name (default output dir name).")
    ]
    results: Annotated[
        list[Path],
        Field(
            description="Results dirs and/or results.json files. Empty for VE "
            "packets — the sample family's slice rows carry the results paths."
        ),
    ] = []
    out_dir: Annotated[
        Optional[Path],
        Field(description="Packet directory (default data/annotations/<batch>)."),
    ] = None
    append: Annotated[
        bool, Field(description="Add new sims to an existing packet.")
    ] = False
    shuffle: Annotated[
        bool, Field(description="Shuffle before --max-items (seeded sampling).")
    ] = False
    seed: Annotated[int, Field(description="Seed for --shuffle.")] = 42
    max_items: Annotated[
        Optional[int], Field(description="Cap on exported simulations.")
    ] = None
    filter_reward: Annotated[
        Optional[str], Field(description="Reward filter, e.g. '< 1'.")
    ] = None
    filter_tasks: Annotated[
        Optional[list[str]], Field(description="Keep only these task ids.")
    ] = None
    emit_zip: Annotated[
        bool,
        Field(
            description="Also write <packet>.zip, the form the packet actually "
            "ships in. Default on; --no-zip builds only the tree."
        ),
    ] = True

    @model_validator(mode="after")
    def _check_shape(self) -> "PacketBuildOptions":
        if not self.results:
            raise ValueError("at least one RESULTS path is required")
        return self

    @property
    def effective_batch_name(self) -> str:
        return self.batch_name

    @property
    def packet_dir(self) -> Path:
        return self.out_dir or (DEFAULT_OUTPUT_ROOT / self.effective_batch_name)


def sync_packet_zip(packet_dir: Path, *, emit: bool) -> Optional[Path]:
    """Bring ``<packet_dir>.zip`` in line with the packet just written; return
    the archive, or None when the packet ships as a tree.

    Packets ship over Google Drive, and a packet is a folder TREE — 336
    directories and 1000 files for a five-rater preference wave. Drive syncs
    that shape one cloud-create round-trip at a time, and it is pathological:
    the same bytes that move in minutes as five archives sat for twenty
    minutes as a tree without creating a single folder. Nothing is lost by
    archiving, because Drive cannot serve HTML anyway — a rater downloads the
    folder, which Drive zips server-side on the way out. So the archive IS the
    shipping artifact and the tree is the local working copy.

    Deflate, never a lossy audio codec: § ④ asks raters to score intonation
    and pronunciation, so the bytes they hear have to be the bytes that were
    recorded. On PCM call audio deflate still halves the packet (measured
    52%), and level 6 gets all of it — level 9 bought 1.4% for far more time.

    Not emitting DELETES an archive from an earlier build rather than leaving
    it: a zip beside a packet reads as the shipment, so a rebuild or --append
    that stops emitting one would otherwise keep handing raters stale bytes.
    """
    zip_path = packet_dir.parent / f"{packet_dir.name}.zip"
    if not emit:
        if zip_path.exists():
            logger.info(f"packet ships as a tree; removed stale {zip_path}")
        zip_path.unlink(missing_ok=True)
        return None
    # Staged and renamed: an interrupted write must not leave a truncated
    # archive at the final name, where it is indistinguishable from a whole one.
    staging = packet_dir.parent / f".{packet_dir.name}.zip.tmp"
    staging.unlink(missing_ok=True)
    # Sorted so two builds of the same packet produce the same archive.
    files = sorted(p for p in packet_dir.rglob("*") if p.is_file())
    root = Path(packet_dir.name)
    try:
        with zipfile.ZipFile(
            staging, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in files:
                archive.write(path, root / path.relative_to(packet_dir))
        staging.replace(zip_path)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    raw = sum(p.stat().st_size for p in files)
    packed = zip_path.stat().st_size
    logger.info(
        f"packed {len(files)} file(s), {raw / 1e9:.2f} GB -> "
        f"{packed / 1e9:.2f} GB ({100 - 100 * packed / max(raw, 1):.0f}% smaller) "
        f"-> {zip_path}"
    )
    return zip_path


def packet_dir_name(sim: SimulationRun) -> str:
    """Page directory name for a sim.

    Restricted to [A-Za-z0-9._-]: packets are downloaded and browsed over
    file:// on annotator machines, where characters like ``|``/``[``/``]``
    (present in generated-pool task ids) are illegal in Windows filenames
    and break relative links.
    """
    task_id_str = re.sub(r"[^A-Za-z0-9._-]+", "-", str(sim.task_id)).strip("-")
    if len(task_id_str) > 60:
        task_id_str = task_id_str[:60]
    dir_name = f"task_{task_id_str}_sim_{sim.id}"
    if len(dir_name) > 200:
        dir_name = f"sim_{sim.id}"
    return dir_name


def find_audio(results_dir: Path, sim: SimulationRun) -> Optional[Path]:
    """Locate the sim's combined-channel wav (the judges pillar owns the
    layout knowledge; delivery rejudge-from-disk slices the same file)."""
    from tau2.judges.delivery.disk_audio import find_both_wav

    return find_both_wav(results_dir, sim)


def _initial_batch_id(kind: str, batch_name: str, sim_ids: list[str]) -> str:
    """Content-derived batch id from the initial entry set."""
    content = hashlib.sha256("\n".join(sorted(sim_ids)).encode()).hexdigest()
    return derive_batch_id(kind, batch_name, {"sims": content})


def _read_prior_manifest(
    packet_dir: Path, kind: str, form: FormType, batch_name: str
) -> ArtifactManifest:
    manifest_path = packet_dir / PACKET_MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"--append requires an existing packet manifest at {manifest_path} "
            "(a packet cannot be appended to without its provenance)"
        )
    prior = ArtifactManifest.model_validate_json(manifest_path.read_text())
    if prior.kind != kind:
        raise ValueError(
            f"packet at {packet_dir} was built with form "
            f"'{prior.form or prior.kind}' — appending form '{form.value}' to it "
            "is an error (one form per batch)"
        )
    if prior.batch_name != batch_name:
        raise ValueError(
            f"packet at {packet_dir} is batch '{prior.batch_name}' — appending "
            f"with --batch-name '{batch_name}' would silently rename the batch "
            "(batch identity must stay stable; pass the original name)"
        )
    # --append rewrites the whole builds list, so the prior records must be
    # readable NOW; a packet predating run identities is rebuilt, not patched.
    try:
        build_records(prior.provenance)
    except ValidationError as exc:
        raise ValueError(
            f"packet at {packet_dir} records its source runs in an older "
            "provenance shape (no run identity) — rebuild it from scratch "
            "instead of appending"
        ) from exc
    return prior


def _index_rows(
    entries: list[PacketEntry],
    doubles: Optional[set[str]] = None,
) -> list[IndexRowVM]:
    rows = []
    for entry in sorted(entries, key=lambda e: e.dir_name):
        task_id = entry.task_id
        display = task_id if len(task_id) <= 40 else task_id[:37] + "..."
        domain_label = entry.domain.capitalize()
        rows.append(
            IndexRowVM(
                href=f"{entry.dir_name}/index.html",
                task_id=task_id,
                display_task_id=display,
                domain_prefix=f"{domain_label}: " if domain_label else "",
                sim_id=entry.sim_id,
                experiment=entry.experiment,
                double_annotate=entry.sim_id in (doubles or set()),
            )
        )
    return rows


def _stream_matched_sims(opts: PacketBuildOptions, results: list[Path]):
    return iter_loaded_sims(
        results,
        reward_filter=opts.filter_reward,
        task_ids=opts.filter_tasks,
    )


def _select_new_sim_ids(
    opts: PacketBuildOptions,
    results: list[Path],
    known_ids: set[str],
    allowed_ids: Optional[set[str]] = None,
) -> tuple[list[str], int, dict[str, str]]:
    """Pass 1: stream ONLY sim identities (each sim is dropped immediately)
    and apply dedupe / shuffle / --max-items. Returns (new ids to export,
    matched count, sim id -> page dir name) — the dir map is what lets the
    prev/next navigation be computed over the FULL packet before pass 2
    renders any page.

    Dedupe is both against the prior packet (``known_ids``) and WITHIN the
    invocation: overlapping inputs export a sim once, first occurrence wins.
    For VE builds ``allowed_ids`` restricts to the sample's variant slice, and
    a sampled sim missing from its recorded results path is a LOUD error —
    the sample family would silently thin out otherwise.
    """
    ids: list[str] = []
    dir_by_id: dict[str, str] = {}
    for ls in _stream_matched_sims(opts, results):
        ids.append(ls.sim.id)
        dir_by_id.setdefault(ls.sim.id, packet_dir_name(ls.sim))
    if allowed_ids is not None:
        found = set(ids)
        missing = sorted(allowed_ids - found)
        if missing:
            raise ValueError(
                f"{len(missing)} sampled sim(s) not found in their recorded "
                f"results paths (sample/results drift?): {', '.join(missing[:5])}"
                f"{' …' if len(missing) > 5 else ''}"
            )
        ids = [i for i in ids if i in allowed_ids]
    matched = len(ids)
    selected: list[str] = []
    seen = set(known_ids)
    for sim_id in ids:
        if sim_id in seen:
            logger.info(f"skipping duplicate/already-exported sim {sim_id}")
            continue
        seen.add(sim_id)
        selected.append(sim_id)
    # The cap is on simulations exported, not raw rows encountered. Applying it
    # before dedupe lets prior/overlapping results consume every available slot.
    if opts.shuffle:
        random.Random(opts.seed).shuffle(selected)
        logger.info(f"shuffled {len(selected)} new simulations (seed={opts.seed})")
    if opts.max_items is not None:
        selected = selected[: opts.max_items]
    return selected, matched, dir_by_id


def build_packet(opts: PacketBuildOptions) -> Path:
    """Build (or append to) an annotation packet; returns the manifest path."""
    kind = packet_kind_for(opts.form)
    packet_dir = opts.packet_dir
    batch_name = opts.effective_batch_name

    results = opts.results

    prior: Optional[ArtifactManifest] = None
    if packet_dir.exists() and any(packet_dir.iterdir()):
        if not opts.append:
            raise FileExistsError(
                f"output directory already exists: {packet_dir} — remove it, "
                "choose a different path, or use --append"
            )
        prior = _read_prior_manifest(packet_dir, kind, opts.form, batch_name)

    entries: list[PacketEntry] = list(prior.entries or []) if prior else []
    known_ids = {e.sim_id for e in entries}
    new_ids, matched, dir_by_id = _select_new_sim_ids(
        opts,
        results,
        known_ids,
    )
    logger.info(
        f"{matched} simulations matched; {len(new_ids)} new to export "
        f"({len(known_ids)} already in the packet)"
    )
    if not new_ids and prior is None:
        raise ValueError("no simulations matched the filters — nothing to export")

    # Canonical page order = the index listing's dir-name sort, over the FULL
    # merged packet (prior + new entries); prev/next bars are injected
    # statically from it. On --append, prior pages' bars are refreshed in
    # place below so every page navigates the merged order.
    ordered_dirs = sorted(
        [e.dir_name for e in entries] + [dir_by_id[i] for i in new_ids]
    )
    nav_by_dir = dict(zip(ordered_dirs, build_nav_vms(ordered_dirs)))

    packet_dir.mkdir(parents=True, exist_ok=True)

    if prior is not None:
        batch_id = prior.batch_id
    else:
        batch_id = _initial_batch_id(kind, batch_name, new_ids)
    identity = PacketIdentity(batch_id=batch_id, batch_name=batch_name, form=opts.form)

    # Pass 2: stream again and render only the selected sims, one at a time.
    to_export = set(new_ids)
    policies: dict[str, Optional[str]] = {}
    for ls in _stream_matched_sims(opts, results):
        if not to_export:
            break
        if ls.sim.id not in to_export:
            continue
        to_export.discard(ls.sim.id)
        entries.append(
            _export_sim(ls, packet_dir, identity, policies, opts, nav_by_dir)
        )

    if prior is not None:
        _refresh_prior_nav(packet_dir, prior.entries or [], nav_by_dir)

    index_html = render_index_page(identity=identity, rows=_index_rows(entries, set()))
    (packet_dir / "index.html").write_text(index_html)

    domains = {e.domain for e in entries if e.domain}
    build_record = BuildRecord(
        # A path only LOCATES a run; the identity says which run it was, so a
        # later audit can tell a moved run from a regenerated one.
        results=[
            RecordedRun(path=str(p), identity=compute_run_identity(p)) for p in results
        ],
        filters=BuildFilters(
            reward=opts.filter_reward,
            tasks=opts.filter_tasks,
            shuffle=opts.shuffle,
            seed=opts.seed if opts.shuffle else None,
            max_items=opts.max_items,
        ),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    provenance: dict = {
        # One entry per build invocation — --append preserves prior builds.
        "builds": [
            *(
                record.model_dump(mode="json")
                for record in build_records(prior.provenance if prior else {})
            ),
            build_record.model_dump(mode="json"),
        ],
        **rubric_json_for_provenance(),
    }
    manifest = ArtifactManifest(
        kind=kind,
        batch_id=batch_id,
        batch_name=batch_name,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha(),
        domain=domains.pop() if len(domains) == 1 else None,
        form=opts.form.value,
        provenance=provenance,
        files=[],
        entries=entries,
    )
    manifest_path = write_json_artifact(packet_dir / PACKET_MANIFEST_NAME, manifest)
    logger.info(
        f"packet '{batch_name}' ({batch_id}): {len(entries)} conversations "
        f"-> {packet_dir}"
    )
    # After the manifest, so the archive carries the finished packet.
    sync_packet_zip(packet_dir, emit=opts.emit_zip)
    return manifest_path


# Exactly the block _nav.html.j2's macro renders (dir names are restricted to
# [A-Za-z0-9._-], so no '>' can appear inside the block).
_NAV_BLOCK_RE = re.compile(r'<nav class="packet-nav">.*?</nav>', re.DOTALL)


def _refresh_prior_nav(
    packet_dir: Path,
    prior_entries: list[PacketEntry],
    nav_by_dir: dict[str, PageNavVM],
) -> None:
    """--append renders only NEW pages, but the prev/next bars are derived
    from the FULL packet order (like index.html, which the build regenerates
    whole) — so rewrite each prior page's bars in place with the merged-order
    fragment. Pages predating the nav bars have no block and stay untouched.
    """
    for entry in prior_entries:
        page = packet_dir / entry.dir_name / "index.html"
        if not page.exists():
            continue
        fragment = render_nav_fragment(nav_by_dir[entry.dir_name])
        refreshed, n = _NAV_BLOCK_RE.subn(lambda _: fragment, page.read_text())
        if n:
            page.write_text(refreshed)


def _export_sim(
    ls: LoadedSim,
    packet_dir: Path,
    identity: PacketIdentity,
    policies: dict[str, Optional[str]],
    opts: PacketBuildOptions,
    nav_by_dir: dict[str, PageNavVM],
) -> PacketEntry:
    """Render one sim's page dir (page + audio copy); returns its entry."""
    sim = ls.sim
    dir_name = packet_dir_name(sim)
    page_dir = packet_dir / dir_name
    page_dir.mkdir(parents=True, exist_ok=True)

    audio_src = find_audio(ls.results_dir, sim)
    audio_filename = None
    if audio_src is not None:
        shutil.copy(audio_src, page_dir / AUDIO_FILENAME)
        audio_filename = AUDIO_FILENAME

    # Prefer the run's OWN task (localized for non-English runs); fall back to
    # the domain's base-language tasks.json. Typed end to end: the page view
    # models read Task attributes, never raw dicts.
    task = ls.task if ls.task is not None else load_task(ls.domain, str(sim.task_id))
    if ls.domain not in policies:
        policies[ls.domain] = load_policy(ls.domain)

    html = render_sim_page(
        sim=sim,
        task=task,
        form=identity.form,
        identity=identity,
        experiment_label=ls.experiment_label,
        domain=ls.domain,
        policy=policies[ls.domain],
        audio_filename=audio_filename,
        nav=nav_by_dir[dir_name],
    )
    (page_dir / "index.html").write_text(html)
    logger.info(f"exported task {sim.task_id} / sim {sim.id} -> {page_dir}")

    return PacketEntry(
        task_id=str(sim.task_id),
        sim_id=sim.id,
        trial=sim.trial,
        dir_name=dir_name,
        experiment=ls.experiment_label,
        domain=ls.domain,
        has_audio=audio_filename is not None,
    )
