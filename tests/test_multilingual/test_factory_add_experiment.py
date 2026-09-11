# Copyright Sierra
"""`tau2 factory add-experiment` + multi-domain experiments machinery.

The verb appends one domain's deterministic ``experiments`` entry to a
SHIPPED pack.yaml (append-only inside the block, byte-identical elsewhere,
idempotent). The loader stashes one typed spec per (language, domain) and
``run_presets`` generates one preset per spec — a second domain's entry means
a second preset for the same language.
"""

import json

import pytest
import yaml

import tau2.multilingual.loader as ml_loader
import tau2.multilingual.run_presets as ml_run_presets
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.pack_assembly import (
    main_arm_name_for,
    run_add_experiment,
)


class TestMainArmNaming:
    def test_legacy_unsuffixed_domain_keeps_bare_name(self):
        assert main_arm_name_for("hindi", "airline") == "hindi"

    def test_other_domains_are_suffixed(self):
        assert main_arm_name_for("hindi", "telecom") == "hindi_telecom"


class TestAddExperiment:
    def test_appends_entry_and_is_idempotent(self, isolated_pack_env):
        pack_path = isolated_pack_env.pack_dir / "pack.yaml"
        original = pack_path.read_text()

        outcome = run_add_experiment("tl", "telecom", smoke_task_stem="[toy]s1")
        assert outcome.written
        assert outcome.main_arm_name == "toylang_telecom"
        assert outcome.preset_name == "multilingual_v1_toylang_telecom"
        new_text = pack_path.read_text()
        # Append-only inside the experiments block: the original text minus
        # nothing — every original line still present, in order.
        for line in original.splitlines():
            assert line in new_text
        data = yaml.safe_load(new_text)
        assert [e["domain"] for e in data["experiments"]] == ["toyair", "telecom"]
        entry = data["experiments"][1]
        assert entry["task_set_suffix"] == "tl"
        assert entry["smoke_task_stem"] == "[toy]s1"

        # Second run: skip, byte-identical.
        again = run_add_experiment("tl", "telecom", smoke_task_stem="[toy]s1")
        assert not again.written
        assert again.preset_name == "multilingual_v1_toylang_telecom"
        assert pack_path.read_text() == new_text

    def test_smoke_stem_defaults_from_domain_profile(self, isolated_pack_env):
        from tau2.multilingual.domain_profiles import get_domain_profile

        outcome = run_add_experiment("tl", "telecom")
        assert outcome.written
        pack_path = isolated_pack_env.pack_dir / "pack.yaml"
        entry = yaml.safe_load(pack_path.read_text())["experiments"][1]
        assert entry["smoke_task_stem"] == (
            get_domain_profile("telecom").default_smoke_task_stem
        )

    def test_unknown_domain_fails_loud(self, isolated_pack_env):
        with pytest.raises(FactoryDraftError, match="Unknown multilingual domain"):
            run_add_experiment("tl", "no_such_domain", smoke_task_stem="1")

    def test_missing_pack_fails_loud(self, isolated_pack_env):
        with pytest.raises(FactoryDraftError, match="no shipped pack"):
            run_add_experiment("zz", "telecom", smoke_task_stem="1")


class TestCliGuardrailRollback:
    def test_guardrail_failure_restores_the_pack(
        self, isolated_pack_env, monkeypatch, capsys
    ):
        """A written entry that then fails the FULL guardrails must not stay
        on disk — the CLI rolls pack.yaml back to its pre-write text (the real
        case: adding a domain a localized pack has no glossary for yet)."""
        import argparse

        import tau2.multilingual.factory.guardrails as guardrails
        from tau2.multilingual.factory.cli import run_factory_add_experiment

        pack_path = isolated_pack_env.pack_dir / "pack.yaml"
        original = pack_path.read_text()

        class _Report:
            def __init__(self, problems):
                self.problems = problems
                self.notes: list = []
                self.ok = not problems

        def _validate(lang):
            # The problem only exists once the entry is on disk: this is a
            # problem the write INTRODUCED, which is what must roll back.
            if "telecom" in pack_path.read_text():
                return _Report(["experiments[1]: the pack has no 'telecom' glossary"])
            return _Report([])

        monkeypatch.setattr(guardrails, "validate_final_pack", _validate)
        args = argparse.Namespace(
            lang="tl", domain="telecom", smoke_task_stem="[toy]s1"
        )
        with pytest.raises(SystemExit):
            run_factory_add_experiment(args)
        assert pack_path.read_text() == original
        assert "rolled back" in capsys.readouterr().out

    def test_preexisting_problem_does_not_block_an_unrelated_write(
        self, isolated_pack_env, monkeypatch, capsys
    ):
        """A pack can carry a known problem it is waiting on its native
        reviewer for (Italian's pending pure-continuer review). A verb that
        writes an unrelated field must not be held hostage to it — but the
        problem stays visible in the output."""
        import argparse

        import tau2.multilingual.factory.guardrails as guardrails
        from tau2.multilingual.factory.cli import run_factory_add_experiment

        pack_path = isolated_pack_env.pack_dir / "pack.yaml"
        original = pack_path.read_text()
        inherited = "persona 'tessa_tl_v1' carries 8 backchannel_phrases"

        class _Report:
            ok = False
            problems = [inherited]
            notes: list = []

        monkeypatch.setattr(guardrails, "validate_final_pack", lambda lang: _Report())
        args = argparse.Namespace(
            lang="tl", domain="telecom", smoke_task_stem="[toy]s1"
        )
        run_factory_add_experiment(args)

        assert pack_path.read_text() != original  # the write survived
        out = capsys.readouterr().out
        assert "no NEW problems" in out
        assert f"(pre-existing) {inherited}" in out


class TestMultiDomainPresets:
    def _install_telecom_tasks(self, env, task_ids):
        path = env.pack_dir / "telecom_tasks_tl.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": task_id,
                        "user_scenario": {"instructions": {"domain": "telecom"}},
                    }
                    for task_id in task_ids
                ]
            )
        )
        return path

    def test_two_domains_two_presets(self, isolated_pack_env):
        isolated_pack_env.install_localized_tasks()
        self._install_telecom_tasks(isolated_pack_env, ["[toy]s1_tl", "[toy]s2_tl"])
        run_add_experiment("tl", "telecom", smoke_task_stem="[toy]s1")
        ml_loader._reset_for_tests()
        ml_run_presets._reset_presets_cache_for_tests()

        specs = ml_loader.get_language_experiments("tl")
        assert [spec.domain for spec in specs] == ["toyair", "telecom"]
        assert ml_loader.get_language_experiment("tl", "telecom") is specs[1] or (
            ml_loader.get_language_experiment("tl", "telecom").preset_name
            == specs[1].preset_name
        )
        assert ml_loader.get_language_experiment("tl", "airline") is None

        presets = ml_run_presets.get_run_presets()
        assert "multilingual_v1_toylang" in presets
        telecom = presets["multilingual_v1_toylang_telecom"]
        (arm,) = telecom.arms
        assert arm.domain == "telecom"
        assert arm.task_set_name == "telecom_tl"
        assert arm.task_ids == ["[toy]s1_tl", "[toy]s2_tl"]
        assert arm.smoke_task_ids == ["[toy]s1_tl"]
        assert arm.user_persona_id == "tl"

    def test_missing_task_file_skips_only_that_preset(self, isolated_pack_env):
        isolated_pack_env.install_localized_tasks()
        run_add_experiment("tl", "telecom", smoke_task_stem="[toy]s1")
        ml_loader._reset_for_tests()
        ml_run_presets._reset_presets_cache_for_tests()

        presets = ml_run_presets.get_run_presets()
        # No telecom_tasks_tl.json: the telecom preset is skipped with a
        # warning; the same language's toyair preset still generates.
        assert "multilingual_v1_toylang" in presets
        assert "multilingual_v1_toylang_telecom" not in presets

    def test_duplicate_domain_fails_at_load(self, isolated_pack_env):
        pack_path = isolated_pack_env.pack_dir / "pack.yaml"
        data = yaml.safe_load(pack_path.read_text())
        data["experiments"].append(dict(data["experiments"][0]))
        pack_path.write_text(yaml.safe_dump(data, sort_keys=False))
        ml_loader._reset_for_tests()
        with pytest.raises(ValueError, match="duplicate experiments entry"):
            ml_loader.get_language_experiments("tl")


class TestExperimentsKeyLocator:
    """The insertion point must tolerate benign spellings of the key line and
    refuse the rest — a missed match appends a SECOND top-level
    ``experiments:`` key, and YAML last-key-wins silently drops the existing
    entries (and their shipped presets) on every subsequent load."""

    def test_trailing_comment_on_key_still_inserts_into_block(self, isolated_pack_env):
        pack_path = isolated_pack_env.pack_dir / "pack.yaml"
        pack_path.write_text(
            pack_path.read_text().replace(
                "experiments:\n", "experiments:  # one entry per domain\n"
            )
        )

        outcome = run_add_experiment("tl", "telecom", smoke_task_stem="[toy]s1")
        assert outcome.written
        new_text = pack_path.read_text()
        key_lines = [
            line for line in new_text.splitlines() if line.startswith("experiments:")
        ]
        assert len(key_lines) == 1
        data = yaml.safe_load(new_text)
        assert [e["domain"] for e in data["experiments"]] == ["toyair", "telecom"]

    def test_flow_style_experiments_key_is_refused(self, isolated_pack_env):
        pack_path = isolated_pack_env.pack_dir / "pack.yaml"
        text = pack_path.read_text()
        block_start = text.index("experiments:\n")
        flow = (
            "experiments: [{preset_name: multilingual_v1_toylang, "
            "description: toy, domain: toyair, main_arm_name: toylang, "
            "task_set_suffix: tl, smoke_task_stem: '1'}]\n"
        )
        pack_path.write_text(text[:block_start] + flow)

        original = pack_path.read_text()
        with pytest.raises(FactoryDraftError, match="block-style"):
            run_add_experiment("tl", "telecom", smoke_task_stem="[toy]s1")
        assert pack_path.read_text() == original


class TestSmokeStemValidation:
    def test_stem_matching_no_arm_task_is_refused(self, isolated_pack_env):
        path = isolated_pack_env.pack_dir / "telecom_tasks_tl.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": task_id,
                        "user_scenario": {"instructions": {"domain": "telecom"}},
                    }
                    for task_id in ["[toy]s1_tl", "[toy]s2_tl"]
                ]
            )
        )
        pack_path = isolated_pack_env.pack_dir / "pack.yaml"
        original = pack_path.read_text()
        with pytest.raises(FactoryDraftError, match="smoke task"):
            run_add_experiment("tl", "telecom", smoke_task_stem="[toy]nope")
        assert pack_path.read_text() == original

    def test_missing_arm_file_defers_the_check(self, isolated_pack_env):
        # No telecom_tasks_tl.json on disk: preset generation is the loud
        # failure point for a missing set; the stem check must not block.
        outcome = run_add_experiment("tl", "telecom", smoke_task_stem="[toy]s1")
        assert outcome.written


class TestCliValidationRaiseRollback:
    def test_raised_validation_error_restores_the_pack(
        self, isolated_pack_env, monkeypatch, capsys
    ):
        import argparse

        import tau2.multilingual.factory.guardrails as guardrails
        from tau2.multilingual.factory.cli import run_factory_add_experiment

        def _boom(lang):
            raise RuntimeError("validator crashed")

        monkeypatch.setattr(guardrails, "validate_final_pack", _boom)
        pack_path = isolated_pack_env.pack_dir / "pack.yaml"
        original = pack_path.read_text()
        args = argparse.Namespace(
            lang="tl", domain="telecom", smoke_task_stem="[toy]s1"
        )
        with pytest.raises(RuntimeError, match="validator crashed"):
            run_factory_add_experiment(args)
        assert pack_path.read_text() == original
        assert "rolled back" in capsys.readouterr().out
