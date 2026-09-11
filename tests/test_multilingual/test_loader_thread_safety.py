# Copyright Sierra
"""Concurrency regression test for language-pack discovery.

Under concurrent voice runs (``tau2 run --user-persona-id <lang>
--max-concurrency N``), the lazy registry getters are hit from many
ThreadPoolExecutor worker threads at once. ``load_language_packs`` used to set
its ``_loaded`` flag at the *top* of the function, before registering packs in
sorted order, so a second thread could see ``_loaded==True`` and look up a
late-alphabet pack (ko/pt/zh) before it was registered — failing the task's
first attempt with ``Unknown persona``.

This test forces a cold registry and slams the getters from many threads
gated on a barrier, asserting that no getter ever observes a
partially-populated registry. With the pre-fix loader it fails; with the
double-checked-locking loader (flag set last) it passes.
"""

from __future__ import annotations

import threading

import pytest

import tau2.data_model.voice_personas as voice_personas
import tau2.multilingual.loader as ml_loader
import tau2.multilingual.registry as ml_registry
from tau2.multilingual.registry import (
    get_language_pack,
    get_multilingual_persona,
    list_language_packs,
    resolve_run_language,
)

# Real late-alphabet packs (registered last in sorted order) — the ones the
# race used to drop. A persona id from the very last pack (zh) is the most
# sensitive probe.
ZH_PERSONA_ID = "jianguo_zh_v1"
LATE_LANGS = ("ko", "pt", "zh")
N_THREADS = 32


@pytest.fixture
def cold_real_registry():
    """Force a cold registry of the REAL shipped packs; restore state after.

    Mirrors the snapshot/restore dance in
    ``factory_testing/conftest_helpers.py`` but does NOT swap in a tmp
    DATA_DIR — we want the real packs so ``zh`` et al. actually exist. Voice
    personas must be cleared too: ``register_voice_persona`` raises on
    duplicates, so a cold reload requires an empty persona registry.
    """
    saved_loaded = ml_loader._loaded
    saved_experiments = dict(ml_loader._EXPERIMENTS)
    saved_packs = dict(ml_registry._LANGUAGE_PACKS)
    saved_personas = dict(voice_personas.ALL_PERSONAS)
    saved_persona_names = list(voice_personas.ALL_PERSONA_NAMES)

    ml_loader._reset_for_tests()
    ml_registry._reset_for_tests()
    voice_personas.ALL_PERSONAS.clear()
    voice_personas.ALL_PERSONA_NAMES.clear()

    try:
        yield
    finally:
        ml_loader._reset_for_tests()
        ml_registry._reset_for_tests()
        ml_loader._loaded = saved_loaded
        ml_loader._EXPERIMENTS.update(saved_experiments)
        ml_registry._LANGUAGE_PACKS.update(saved_packs)
        voice_personas.ALL_PERSONAS.clear()
        voice_personas.ALL_PERSONAS.update(saved_personas)
        voice_personas.ALL_PERSONA_NAMES[:] = saved_persona_names


def test_concurrent_getters_never_see_partial_registry(cold_real_registry):
    """N threads hit late-alphabet getters at once on a cold registry."""
    barrier = threading.Barrier(N_THREADS)
    failures: list[str] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        # Line everyone up so the getters fire simultaneously, right when the
        # registry is cold — maximizing the race window.
        barrier.wait()
        try:
            pack = get_language_pack("zh")
            if pack is None:
                raise AssertionError("get_language_pack('zh') returned None")

            lang = resolve_run_language("pt")
            if lang != "pt":
                raise AssertionError(f"resolve_run_language('pt') -> {lang!r}")

            persona = get_multilingual_persona(ZH_PERSONA_ID)
            if persona is None:
                raise AssertionError(
                    f"get_multilingual_persona({ZH_PERSONA_ID!r}) returned None"
                )
        except Exception as exc:  # noqa: BLE001 — capture for cross-thread assert
            with lock:
                failures.append(f"thread {idx}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not failures, "Getters saw a partial registry:\n" + "\n".join(failures)


def test_failed_load_rolls_back_so_retry_is_clean(cold_real_registry, monkeypatch):
    """A mid-load failure must roll back so the next call retries from cold.

    Without rollback, the packs/personas registered before the failure would
    survive while ``_loaded`` stays False, and the retry's
    ``register_language_pack`` would raise "already registered" instead of
    recovering (the Cursor Bugbot "Failed load poisons retry" finding).
    """
    real_load_yaml = ml_loader._load_yaml_pack
    calls = {"n": 0}

    def flaky(pack_path):
        calls["n"] += 1
        # Fail on the 3rd pack, after a couple have already registered.
        if calls["n"] == 3:
            raise RuntimeError("boom: simulated bad pack.yaml")
        return real_load_yaml(pack_path)

    monkeypatch.setattr(ml_loader, "_load_yaml_pack", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        ml_loader.load_language_packs()

    # Rolled back to the clean cold state the fixture set up.
    assert ml_loader._loaded is False
    assert ml_registry._LANGUAGE_PACKS == {}
    assert voice_personas.ALL_PERSONAS == {}
    assert ml_loader._EXPERIMENTS == {}

    # Retry with the failure removed must succeed (no "already registered").
    monkeypatch.setattr(ml_loader, "_load_yaml_pack", real_load_yaml)
    ml_loader.load_language_packs()
    assert ml_loader._loaded is True
    for late in LATE_LANGS:
        assert get_language_pack(late) is not None


def test_registry_complete_after_concurrent_load(cold_real_registry):
    """After the storm, all shipped packs and personas are fully registered."""
    barrier = threading.Barrier(N_THREADS)

    def worker() -> None:
        barrier.wait()
        get_language_pack("zh")

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    codes = list_language_packs()
    for late in LATE_LANGS:
        assert late in codes, f"{late} missing from {codes}"

    # Every pack's personas must be resolvable as voice personas (the symptom
    # of a partial registry is a pack whose personas never got registered).
    for code in codes:
        pack = get_language_pack(code)
        assert pack is not None
        for persona_id in pack.personas:
            assert persona_id in voice_personas.ALL_PERSONAS, (
                f"voice persona '{persona_id}' (pack '{code}') not registered"
            )
