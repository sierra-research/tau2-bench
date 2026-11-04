"""Small published language fixtures shared by benchmark behavior tests."""

import json
from pathlib import Path

import pytest
import yaml

from tau2.data_model.benchmark import BenchmarkFrame, CellFrame
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.user_data_model import TelecomUserDB
from tau2.multilingual.runtime import load_language
from tau2.runner.presets import binding_revision

REVIEWED_CONTENT = (
    Path(__file__).parents[1]
    / "test_multilingual/fixtures/reviewed_prompt_content.json"
)


@pytest.fixture
def published_frame(tmp_path, monkeypatch):
    import tau2.utils

    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    # These selection-only tasks need no business records, but English binds
    # its complete canonical DB/policy snapshot just like a real run.
    for domain, database in (
        ("airline", FlightDB(flights={}, users={}, reservations={})),
        ("retail", RetailDB(users={}, products={}, orders={})),
        ("telecom", TelecomDB()),
    ):
        directory = tmp_path / "tau2/domains" / domain
        directory.mkdir(parents=True)
        database.dump(
            str(directory / ("db.toml" if domain == "telecom" else "db.json"))
        )
        policies = (
            ("main_policy", "tech_support_manual", "tech_support_workflow")
            if domain == "telecom"
            else ("policy",)
        )
        for policy in policies:
            (directory / f"{policy}.md").write_text("Fixture domain policy.\n")
        if domain == "telecom":
            TelecomUserDB().dump(str(directory / "user_db.toml"))
    cells = []
    for language in ("en", "es", "pt", "hi", "zh", "ko"):
        root = tmp_path / "tau2" / "multilingual" / language
        root.mkdir(parents=True)
        domains = {}
        for domain in ("airline", "retail", "telecom"):
            rows = [
                {
                    "id": f"{domain}-localized-{i}",
                    "user_scenario": {"instructions": "Fixture caller."},
                }
                for i in range(50)
            ]
            (root / f"{domain}.json").write_text(json.dumps(rows))
            (root / f"{domain}-metadata.json").write_text(
                json.dumps(
                    {
                        row["id"]: {
                            "canonical_task_id": str(i),
                            "caller_gender": "female",
                        }
                        for i, row in enumerate(rows)
                    }
                )
            )
            domains[domain] = dict(
                revision="fixture-tasks-v1",
                tasks_file=f"{domain}.json",
                task_metadata_file=f"{domain}-metadata.json",
                canonical_scenario=language == "en",
            )
        reviewed = json.loads(REVIEWED_CONTENT.read_text())["en"]
        speaker = dict(
            gender="female",
            locale="Fixture",
            variety="Fixture",
            gender_guidance="You are a female customer.",
            voice=dict(
                voice_id="fixture-voice",
                backchannel_phrases=["Mm"],
                voice_examples=reviewed["prompt_content"]["voice_examples"],
            ),
        )
        if language != "en":
            speaker["language_directive"] = {"origin": "Fixture region"}
        pack = dict(
            schema_version=3,
            language=language,
            revision="fixture-pack-v1",
            display_name=language,
            agent_language_clause="Fixture language instruction.",
            agent_greeting="Fixture greeting.",
            voice=dict(
                spellout_section=reviewed["spellout_section"],
                speech_insert_events_per_minute=1.0,
                backchannel_level="medium",
            ),
            personas={"speaker": speaker},
            domains=domains,
        )
        (root / "pack.yaml").write_text(yaml.safe_dump(pack))
        for domain in domains:
            cells.append(
                CellFrame(
                    language=language,
                    domain=domain,
                    canonical_task_ids=[str(i) for i in range(50)],
                    binding_revision=binding_revision(
                        load_language(language, domain, mode="voice")
                    ),
                )
            )
    return BenchmarkFrame(
        benchmark_version="voice-v2",
        revision="fixture-frame-v1",
        mode="voice",
        trials=[0],
        cells=cells,
    )


@pytest.fixture
def submitter_file(tmp_path) -> Path:
    """``--metadata-file`` for a 2.0 package named fixture_test_2026-10-02."""
    path = tmp_path / "submitter.json"
    path.write_text(
        json.dumps(
            {
                "model_name": "fixture",
                "model_organization": "test",
                "submitting_organization": "test",
                "submission_date": "2026-10-02",
                "contact_info": {"email": "review@example.org"},
            }
        )
    )
    return path
