# Copyright Sierra
"""`tau2 factory arm-tasks` — per-language arm files for a CURATED source set.

The complement of ``seed-tasks``: nothing is materialized (the source IS the
reviewed, checked-in ``tasks.json`` the registry loads), so the verb only
emits ``<domain>_tasks_<lang>.json`` — English source prose, ids suffixed
``_<lang>`` — plus the English caller-gender sidecar. Both verbs go through
the one emitter in ``tau2.multilingual.factory.task_arms``.

The tests run against the toy ``toyair`` curated domain inside the isolated
tmp DATA_DIR, plus SHIPPED-artifact guards over every profiled domain.
"""

import copy
import json

import pytest

from tau2.multilingual.domain_profiles import (
    DOMAIN_PROFILES,
    REGISTRY_TASKS_FILENAME,
    get_domain_profile,
)
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.seed_tasks import run_seed_tasks
from tau2.multilingual.factory.task_arms import run_arm_tasks
from test_multilingual.factory_testing.toy_language import TOY_DOMAIN


@pytest.fixture
def untranslated_toy_domain(isolated_pack_env, monkeypatch):
    """The toy curated domain under the standard English-prose regime.

    The shared fixture registers ``toyair`` with ``tasks_translated=True``
    (it also ships a translated localized set for the translation tests);
    arm emission is the OTHER regime, so flip the profile for these cases.
    """
    monkeypatch.setitem(
        DOMAIN_PROFILES,
        TOY_DOMAIN,
        get_domain_profile(TOY_DOMAIN).model_copy(update={"tasks_translated": False}),
    )
    return isolated_pack_env


def _source_tasks(env) -> list[dict]:
    return json.loads((env.domain_dir / REGISTRY_TASKS_FILENAME).read_text())


class TestRunArmTasks:
    def test_emits_english_prose_arm_and_en_sidecar(self, untranslated_toy_domain):
        env = untranslated_toy_domain
        outcome = run_arm_tasks(TOY_DOMAIN)

        source = _source_tasks(env)
        assert outcome.task_count == len(source)
        assert outcome.source_path == env.domain_dir / REGISTRY_TASKS_FILENAME
        assert sorted(outcome.emitted) == ["en"]

        arm = json.loads(outcome.emitted["en"].read_text())
        assert [t["id"] for t in arm] == [f"{t['id']}_en" for t in source]
        # Prose byte-identical to the source: english-prompt mode reads it.
        for armed, task in zip(arm, source):
            assert armed["user_scenario"] == task["user_scenario"]
            assert armed["evaluation_criteria"] == task["evaluation_criteria"]

        # en arm + en caller-gender sidecar.
        assert len(outcome.written) == 2
        sidecar = json.loads(
            (
                env.multilingual_dir / "en" / f"{TOY_DOMAIN}_caller_gender_en.json"
            ).read_text()
        )
        assert set(sidecar) == {t["id"] for t in arm}
        assert set(sidecar.values()) <= {"male", "female"}

    def test_multiple_language_arms(self, untranslated_toy_domain):
        outcome = run_arm_tasks(TOY_DOMAIN, langs=["en", "tl"])
        assert sorted(outcome.emitted) == ["en", "tl"]
        tl = json.loads(outcome.emitted["tl"].read_text())
        assert all(t["id"].endswith("_tl") for t in tl)
        # Same prose in every arm — only the id suffix differs.
        en = json.loads(outcome.emitted["en"].read_text())
        assert [t["user_scenario"] for t in tl] == [t["user_scenario"] for t in en]

    def test_idempotent(self, untranslated_toy_domain):
        run_arm_tasks(TOY_DOMAIN, langs=["en", "tl"])
        again = run_arm_tasks(TOY_DOMAIN, langs=["en", "tl"])
        assert again.written == []

    def test_generated_pool_domain_is_refused(self, untranslated_toy_domain):
        """One owner per arm file: a domain whose source is materialized by
        seed-tasks must not also be emittable here."""
        with pytest.raises(FactoryDraftError, match="seed-tasks"):
            run_arm_tasks("telecom")

    def test_unknown_domain_fails_loud(self, untranslated_toy_domain):
        with pytest.raises(FactoryDraftError, match="Unknown multilingual domain"):
            run_arm_tasks("no_such_domain")


class TestTranslatedDomainClobberGuard:
    """Once a domain's prose is TRANSLATED its localized arm files belong to
    the retired task-translation loop — emitting English source prose over
    them would silently revert the translation. The ``en`` arm is exempt: it
    IS the source prose under either regime."""

    def test_non_english_arms_refused(self, isolated_pack_env):
        # The shared fixture's toy profile declares tasks_translated=True.
        with pytest.raises(FactoryDraftError, match="retired task-translation loop"):
            run_arm_tasks(TOY_DOMAIN, langs=["en", "tl"])

    def test_english_arm_alone_stays_legitimate(self, isolated_pack_env):
        outcome = run_arm_tasks(TOY_DOMAIN, langs=["en"])
        assert sorted(outcome.emitted) == ["en"]


class TestShippedDomainRegime:
    """The profiled domains' declared regime, pinned.

    ``curated_source`` decides which verb owns a domain's arm files;
    ``tasks_translated`` decides whether those files are English-prose copies
    at all. Both are read by the invariant seams, so a silent flip here is a
    silent change to what the benchmark validates.
    """

    def test_airline_is_curated_and_english_prose(self):
        profile = get_domain_profile("airline")
        assert profile.curated_source
        assert profile.source_tasks_filename == REGISTRY_TASKS_FILENAME
        assert not profile.tasks_translated

    def test_telecom_is_pool_derived_and_english_prose(self):
        profile = get_domain_profile("telecom")
        assert not profile.curated_source
        assert profile.source_tasks_filename == "tasks_multilingual.json"
        assert not profile.tasks_translated

    @pytest.mark.parametrize("domain", sorted(DOMAIN_PROFILES))
    def test_exactly_one_emitter_owns_each_domain(self, domain):
        """The two verbs are exact complements: each refuses the other's
        domains, so no arm file has two owners."""
        profile = get_domain_profile(domain)
        if profile.curated_source:
            with pytest.raises(FactoryDraftError, match="arm-tasks"):
                run_seed_tasks(domain)
        else:
            with pytest.raises(FactoryDraftError, match="seed-tasks"):
                run_arm_tasks(domain)


class TestShippedArmContent:
    """Every committed arm file is its domain's source set modulo the id
    suffix — id-only guards miss upstream CONTENT edits that keep ids, which
    would silently diverge multilingual runs from `tau2 run --domain <d>`.
    This is also what keeps stale translated prose from sneaking back in.
    """

    @pytest.mark.parametrize("domain", sorted(DOMAIN_PROFILES))
    def test_committed_arm_files_match_the_source_set(self, domain):
        from tau2.multilingual.localize_lib import (
            load_domain_tasks,
            multilingual_data_dir,
        )

        assert not get_domain_profile(domain).tasks_translated
        source = load_domain_tasks(domain)
        checked = 0
        for arm_path in sorted(
            multilingual_data_dir().glob(f"*/{domain}_tasks_*.json")
        ):
            lang = arm_path.parent.name
            if arm_path.name != f"{domain}_tasks_{lang}.json":
                continue  # identity variants have their own gate
            expected = []
            for task in source:
                out = copy.deepcopy(task)
                out["id"] = f"{task['id']}_{lang}"
                expected.append(out)
            assert json.loads(arm_path.read_text()) == expected, (
                f"{arm_path.name} diverges from the {domain} source set — "
                f"re-run the domain's arm emitter"
            )
            checked += 1
        assert checked >= 2, f"expected at least two committed {domain} arms"
