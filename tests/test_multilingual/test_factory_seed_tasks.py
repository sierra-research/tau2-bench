# Copyright Sierra
"""`tau2 factory seed-tasks` — materializing a domain's multilingual seed set.

The verb selects a reviewed split out of a generated task pool, writes it as
the domain profile's ``source_tasks_filename`` plus a provenance manifest,
and emits per-language ``<domain>_tasks_<lang>.json`` arm files (ids suffixed,
prose byte-identical English). Everything is deterministic and idempotent.

The tests run against a miniature telecom pool inside the isolated tmp
DATA_DIR — 'telecom' because seed-tasks is gated on the closed domain-profile
catalog (only profiled domains with a dedicated source file are seedable).
"""

import json

import pytest
import yaml

from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.caller_diversity import (
    AuthMode,
    CallerPool,
    caller_pool_path,
    check_seed_customer_patch_integrity,
    diversify_seed_tasks,
    load_caller_pool,
    natural_date,
    task_auth_mode,
)
from tau2.multilingual.factory.seed_tasks import run_seed_tasks

# The toy domain DB the seed fixture installs: the canonical caller plus one
# bystander (mirrors the real telecom shape).
TOY_DB = {
    "customers": [
        {
            "customer_id": "C1001",
            "full_name": "John Smith",
            "date_of_birth": "1985-06-15",
            "phone_number": "555-123-2002",
            "email": "john.smith@example.com",
        },
        {
            "customer_id": "C1002",
            "full_name": "Sarah Johnson",
            "date_of_birth": "1990-11-22",
            "phone_number": "555-123-3001",
            "email": "sarah.j@example.com",
        },
    ]
}

# Reviewed-pool stand-in: given names must be in GIVEN_NAME_GENDERS.
TOY_CALLER_POOL = """\
callers:
  - full_name: Mia Torres
    gender: female
    email: mia.torres@example.com
    date_of_birth: "1990-04-12"
  - full_name: Ethan Brooks
    gender: male
    email: ethan.brooks@example.com
    date_of_birth: "1978-09-03"
"""


def _toy_pool_task(task_id: str, name: str = "John Smith") -> dict:
    return {
        "id": task_id,
        "description": {"purpose": f"Pool task {task_id}"},
        "user_scenario": {
            "instructions": {
                "domain": "telecom",
                "reason_for_call": "Your data is broken.",
                "known_info": f"You are {name} with phone number 555-123-2002.",
                "task_instructions": "Follow the agent's guidance.",
            }
        },
        "initial_state": {
            "initialization_actions": [
                {
                    "env_type": "user",
                    "func_name": "set_user_info",
                    "arguments": {"name": name, "phone_number": "555-123-2002"},
                }
            ]
        },
        "evaluation_criteria": {"reward_basis": ["ENV_ASSERTION"]},
    }


@pytest.fixture
def telecom_pool(isolated_pack_env):
    """A 4-task telecom pool with a 2-task 'base' split in the tmp DATA_DIR."""
    domain_dir = isolated_pack_env.data_dir / "tau2" / "domains" / "telecom"
    domain_dir.mkdir(parents=True)
    pool = [_toy_pool_task(f"[toy]t{i}") for i in range(4)]
    (domain_dir / "tasks.json").write_text(json.dumps(pool))
    # Split lists ids OUT of pool order: materialization must follow the
    # POOL's order (matching the domain's registry loader), not the split's.
    splits = {"base": ["[toy]t2", "[toy]t0"], "small": ["[toy]t1"]}
    (domain_dir / "split_tasks.json").write_text(json.dumps(splits))
    (domain_dir / "db.json").write_text(json.dumps(TOY_DB))
    (domain_dir / "caller_pool.yaml").write_text(TOY_CALLER_POOL)
    return domain_dir


class TestRunSeedTasks:
    def test_materializes_split_manifest_and_en_arm(
        self, isolated_pack_env, telecom_pool
    ):
        outcome = run_seed_tasks("telecom", split="base")

        assert outcome.task_count == 2
        seed = json.loads(outcome.source_path.read_text())
        # Pool order, not split order.
        assert [t["id"] for t in seed] == ["[toy]t0", "[toy]t2"]
        assert outcome.source_path.name == "tasks_multilingual.json"

        manifest = json.loads(outcome.manifest_path.read_text())
        assert manifest["domain"] == "telecom"
        assert manifest["split"] == "base"
        assert manifest["task_ids"] == ["[toy]t0", "[toy]t2"]
        assert manifest["git_sha"]

        arm = json.loads(outcome.emitted["en"].read_text())
        assert [t["id"] for t in arm] == ["[toy]t0_en", "[toy]t2_en"]
        # Prose byte-identical to the source (english-prompt mode reads it).
        assert (
            arm[0]["user_scenario"]["instructions"]
            == seed[0]["user_scenario"]["instructions"]
        )
        # Caller diversification: position 0 -> pool[0] phone-auth, position
        # 1 -> pool[1] name-auth, callers patched via initialization_data.
        assert "Mia Torres" in seed[0]["user_scenario"]["instructions"]["known_info"]
        assert task_auth_mode(seed[0]) is AuthMode.PHONE
        assert (
            "September 3, 1978"
            in (seed[1]["user_scenario"]["instructions"]["known_info"])
        )
        assert task_auth_mode(seed[1]) is AuthMode.NAME_DOB
        patched = seed[1]["initial_state"]["initialization_data"]["agent_data"][
            "customers"
        ]
        assert [c["customer_id"] for c in patched] == ["C1001", "C1002"]
        assert patched[0]["full_name"] == "Ethan Brooks"
        assert patched[1] == TOY_DB["customers"][1]
        # source + manifest + en arm + en caller-gender sidecar
        assert len(outcome.written) == 4
        sidecar_path = outcome.emitted["en"].parent / "telecom_caller_gender_en.json"
        assert json.loads(sidecar_path.read_text()) == {
            "[toy]t0_en": "female",
            "[toy]t2_en": "male",
        }

    def test_idempotent_and_manifest_provenance_kept(
        self, isolated_pack_env, telecom_pool
    ):
        first = run_seed_tasks("telecom", split="base")
        manifest_before = first.manifest_path.read_text()
        again = run_seed_tasks("telecom", split="base")
        assert again.written == []
        # The manifest keeps its original provenance sha on a no-op re-run.
        assert again.manifest_path.read_text() == manifest_before

    def test_multiple_language_arms(self, isolated_pack_env, telecom_pool):
        outcome = run_seed_tasks("telecom", split="base", langs=["en", "tl"])
        assert sorted(outcome.emitted) == ["en", "tl"]
        tl = json.loads(outcome.emitted["tl"].read_text())
        assert all(t["id"].endswith("_tl") for t in tl)

    def test_unknown_split_fails_loud(self, isolated_pack_env, telecom_pool):
        with pytest.raises(FactoryDraftError, match="Unknown split 'huge'"):
            run_seed_tasks("telecom", split="huge")

    def test_split_id_missing_from_pool_fails_loud(
        self, isolated_pack_env, telecom_pool
    ):
        splits_path = telecom_pool / "split_tasks.json"
        splits = json.loads(splits_path.read_text())
        splits["base"].append("[toy]ghost")
        splits_path.write_text(json.dumps(splits))
        with pytest.raises(FactoryDraftError, match="absent from tasks.json"):
            run_seed_tasks("telecom", split="base")

    def test_curated_source_domain_is_not_seedable(self, isolated_pack_env):
        with pytest.raises(FactoryDraftError, match="curated"):
            run_seed_tasks("airline")

    def test_unknown_domain_fails_loud(self, isolated_pack_env):
        with pytest.raises(FactoryDraftError, match="Unknown multilingual domain"):
            run_seed_tasks("no_such_domain")


class TestNaturalDate:
    """The date the user sim speaks is rendered by fixed code, never the
    process locale — a non-English LC_TIME must not change the seed bytes."""

    @pytest.mark.parametrize(
        ("iso", "want"),
        [
            ("1992-03-21", "March 21, 1992"),
            ("1978-09-03", "September 3, 1978"),  # no leading zero on the day
            ("1959-11-09", "November 9, 1959"),
            ("1940-01-01", "January 1, 1940"),
            ("2005-12-31", "December 31, 2005"),
        ],
    )
    def test_renders_english_month_names(self, iso, want):
        assert natural_date(iso) == want


class TestDiversifyCallers:
    """`diversify_seed_tasks` as a pure function of (tasks, pool, db)."""

    def _pool(self) -> CallerPool:
        return CallerPool.model_validate(yaml.safe_load(TOY_CALLER_POOL))

    def _tasks(self, n: int, suffix: str = "") -> list[dict]:
        tasks = [_toy_pool_task(f"[toy]t{i}") for i in range(n)]
        if suffix:
            for task in tasks:
                ins = task["user_scenario"]["instructions"]
                ins["known_info"] += suffix
        return tasks

    def test_round_robin_and_auth_split_deterministic(self):
        pool = self._pool()
        tasks = self._tasks(8)
        out = diversify_seed_tasks(tasks, pool, TOY_DB)
        again = diversify_seed_tasks(tasks, pool, TOY_DB)
        assert out == again
        for i, task in enumerate(out):
            caller = pool.callers[i % 2]
            info = task["initial_state"]["initialization_actions"][0]["arguments"]
            assert info["name"] == caller.full_name
            assert info["phone_number"] == "555-123-2002"
            want = AuthMode.NAME_DOB if i % 2 else AuthMode.PHONE
            assert task_auth_mode(task) is want
        # The inputs are never mutated.
        assert tasks == self._tasks(8)

    @pytest.mark.parametrize(
        "suffix",
        [
            "",
            " You are currently abroad in France.",
            " You are currently at home in the United States.",
        ],
    )
    def test_known_info_remainder_byte_preserved(self, suffix):
        out = diversify_seed_tasks(self._tasks(2, suffix), self._pool(), TOY_DB)
        phone_info = out[0]["user_scenario"]["instructions"]["known_info"]
        assert phone_info == (
            "You are Mia Torres with phone number 555-123-2002." + suffix
        )
        name_info = out[1]["user_scenario"]["instructions"]["known_info"]
        assert name_info == (
            "You are Ethan Brooks and your date of birth is September 3, 1978." + suffix
        )

    def test_no_task_carries_unknown_info(self):
        out = diversify_seed_tasks(self._tasks(2), self._pool(), TOY_DB)
        for task in out:
            assert task["user_scenario"]["instructions"].get("unknown_info") is None

    def test_line_clause_appended_only_in_name_mode(self):
        """Name-auth callers know which line they are calling about and say so
        on request; withholding it entirely leaves the three-line account
        unidentifiable (the arm's original defect). The clause never reaches
        the phone arm, whose known_info already carries the number."""
        out = diversify_seed_tasks(self._tasks(2), self._pool(), TOY_DB)
        phone_instr = out[0]["user_scenario"]["instructions"]["task_instructions"]
        name_instr = out[1]["user_scenario"]["instructions"]["task_instructions"]
        assert "The phone line you are calling about" not in phone_instr
        assert phone_instr == "Follow the agent's guidance."
        assert name_instr.startswith("Follow the agent's guidance.\n")
        assert (
            "The phone line you are calling about is 555-123-2002. If the "
            "agent asks for your phone number in order to look you up or "
            "verify your identity, say you would rather be looked up by your "
            "name, even if the agent insists. Give the number only when the "
            "agent asks which line or which phone number your issue "
            "concerns." in name_instr
        )
        # The refusal offers the name only — naming the DOB would hand the
        # agent the verification step the policy makes it responsible for.
        assert "date of birth" not in name_instr.split("The phone line")[1]

    def test_line_clause_does_not_flip_the_auth_mode(self):
        """The clause carries the phone number, so it must live outside
        known_info — `task_auth_mode` reads phone digits there."""
        out = diversify_seed_tasks(self._tasks(2), self._pool(), TOY_DB)
        assert task_auth_mode(out[1]) is AuthMode.NAME_DOB
        assert "555" not in out[1]["user_scenario"]["instructions"]["known_info"]

    def test_name_mode_without_task_instructions_fails_loud(self):
        tasks = self._tasks(2)
        del tasks[1]["user_scenario"]["instructions"]["task_instructions"]
        with pytest.raises(FactoryDraftError, match="no task_instructions"):
            diversify_seed_tasks(tasks, self._pool(), TOY_DB)

    def test_ticket_renamed_but_keeps_phone(self):
        tasks = self._tasks(2)
        for task in tasks:
            task["ticket"] = "Customer name: John Smith, phone number: 555-123-2002."
        out = diversify_seed_tasks(tasks, self._pool(), TOY_DB)
        for task, name in zip(out, ("Mia Torres", "Ethan Brooks")):
            assert task["ticket"] == (
                f"Customer name: {name}, phone number: 555-123-2002."
            )

    def test_patch_changes_exactly_the_caller_identity_fields(self):
        out = diversify_seed_tasks(self._tasks(2), self._pool(), TOY_DB)
        patched = out[1]["initial_state"]["initialization_data"]["agent_data"][
            "customers"
        ]
        assert [c["customer_id"] for c in patched] == ["C1001", "C1002"]
        caller, bystander = patched
        source = TOY_DB["customers"][0]
        changed = {k for k in caller if caller[k] != source.get(k)}
        assert changed == {"full_name", "email", "date_of_birth"}
        assert caller["full_name"] == "Ethan Brooks"
        assert caller["email"] == "ethan.brooks@example.com"
        assert caller["date_of_birth"] == "1978-09-03"
        assert bystander == TOY_DB["customers"][1]

    def test_non_canonical_known_info_fails_loud(self):
        tasks = self._tasks(1)
        tasks[0]["user_scenario"]["instructions"]["known_info"] = (
            "You are John Smith. Your phone number is 555-123-2002."
        )
        with pytest.raises(FactoryDraftError, match="canonical identity sentence"):
            diversify_seed_tasks(tasks, self._pool(), TOY_DB)

    def test_preexisting_initialization_data_fails_loud(self):
        tasks = self._tasks(1)
        tasks[0]["initial_state"]["initialization_data"] = {"agent_data": {}}
        with pytest.raises(FactoryDraftError, match="initialization_data"):
            diversify_seed_tasks(tasks, self._pool(), TOY_DB)

    def test_seed_gate_passes_on_fresh_output(self):
        pool = self._pool()
        out = diversify_seed_tasks(self._tasks(4), pool, TOY_DB)
        assert check_seed_customer_patch_integrity(out, TOY_DB, pool) == []


class TestShippedCallerPool:
    """The committed telecom caller_pool.yaml is reviewed artifact — guard
    its invariants (validation happens in CallerRecord/CallerPool; this
    pins the shipped file to them and to the db)."""

    def test_committed_pool_validates_and_is_disjoint_from_db(self):
        from collections import Counter

        from tau2.multilingual.localize_lib import load_domain_db

        if not caller_pool_path("telecom").exists():
            pytest.skip("telecom caller pool not committed")
        pool = load_caller_pool("telecom")
        assert len(pool.callers) == 25
        # 13F/12M with 5 F in the first 14 positions makes the 114-task
        # round-robin land exactly 57 male / 57 female at task level.
        assert Counter(c.gender for c in pool.callers) == Counter(
            {"female": 13, "male": 12}
        )
        db = load_domain_db("telecom")
        db_names = {c["full_name"] for c in db["customers"]}
        db_emails = {c.get("email") for c in db["customers"]}
        assert not db_names & {c.full_name for c in pool.callers}
        assert not db_emails & {c.email for c in pool.callers}


class TestShippedSeedArtifacts:
    """The committed telecom seed set must stay in sync with its pool/split."""

    def test_committed_seed_matches_registry_base_split(self):
        from tau2.multilingual.localize_lib import data_dir, load_domain_tasks

        seed_path = (
            data_dir() / "tau2" / "domains" / "telecom" / "tasks_multilingual.json"
        )
        if not seed_path.exists():
            pytest.skip("telecom seed set not materialized")
        from tau2.registry import registry

        seed_ids = [t["id"] for t in load_domain_tasks("telecom")]
        registry_ids = [t.id for t in registry.get_tasks_loader("telecom")()]
        assert seed_ids == registry_ids

        manifest = json.loads(seed_path.with_suffix(".manifest.json").read_text())
        assert manifest["task_ids"] == seed_ids

    def test_committed_seed_diversification_integrity(self):
        """The committed seed passes the seed-level patch gate, and the
        manifest's provenance fields agree with what the single derivation
        helpers reproduce from the artifact itself."""
        from collections import Counter

        from tau2.multilingual.factory.name_genders import source_caller_gender
        from tau2.multilingual.invariants import caller_set_user_info
        from tau2.multilingual.localize_lib import data_dir, load_domain_db

        seed_path = (
            data_dir() / "tau2" / "domains" / "telecom" / "tasks_multilingual.json"
        )
        if not seed_path.exists():
            pytest.skip("telecom seed set not materialized")
        seed = json.loads(seed_path.read_text())
        db = load_domain_db("telecom")
        pool = load_caller_pool("telecom")
        assert check_seed_customer_patch_integrity(seed, db, pool) == []

        manifest = json.loads(seed_path.with_suffix(".manifest.json").read_text())
        derived_name_auth = [
            t["id"] for t in seed if task_auth_mode(t) is AuthMode.NAME_DOB
        ]
        assert manifest["name_auth_task_ids"] == derived_name_auth
        assert len(derived_name_auth) == len(seed) - len(derived_name_auth) == 57

        # Round-robin caller assignment, in seed order.
        assert manifest["caller_assignment"] == {
            task["id"]: pool.callers[i % len(pool.callers)].full_name
            for i, task in enumerate(seed)
        }
        # Task-level caller gender splits exactly 57/57.
        counts = Counter(
            source_caller_gender(caller_set_user_info(t)["name"].split()[0])
            for t in seed
        )
        assert counts == Counter({"male": 57, "female": 57})


class TestTranslatedDomainClobberGuard:
    def test_non_english_arms_refused_once_prose_is_translated(
        self, isolated_pack_env, telecom_pool, monkeypatch
    ):
        """Once a domain's task prose is translated, its per-language arm
        files are owned by the retired task-translation loop — a seed-tasks
        re-run must refuse rather than silently revert them to English source."""
        from tau2.multilingual import domain_profiles

        profile = domain_profiles.get_domain_profile("telecom")
        monkeypatch.setitem(
            domain_profiles.DOMAIN_PROFILES,
            "telecom",
            profile.model_copy(update={"tasks_translated": True}),
        )
        with pytest.raises(FactoryDraftError, match="retired task-translation loop"):
            run_seed_tasks("telecom", langs=["en", "es"])
        # The English arm alone stays legitimate (it IS the source prose).
        outcome = run_seed_tasks("telecom", langs=["en"])
        assert sorted(outcome.emitted) == ["en"]


class TestManifestProvenance:
    def test_manifest_carries_pool_content_hash(self, isolated_pack_env, telecom_pool):
        """pool_sha256 is the manifest's VERIFIABLE provenance — commit shas
        are advisory (a rebase strands them unreachable in every clone)."""
        import hashlib

        outcome = run_seed_tasks("telecom")
        manifest = json.loads(outcome.manifest_path.read_text())
        expected = hashlib.sha256(
            (telecom_pool / "tasks.json").read_bytes()
        ).hexdigest()
        assert manifest["pool_sha256"] == expected
