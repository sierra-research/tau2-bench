# Copyright Sierra
"""Result-directory claims shared by single and multi-run controllers."""

import pytest

from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_domains
from tau2.runner.run_lock import RUN_LOCK_FILENAME, claim_run_directories


def test_a_claimed_run_directory_rejects_a_second_writer(tmp_path):
    run_dir = tmp_path / "run"
    with claim_run_directories([run_dir]):
        assert (run_dir / RUN_LOCK_FILENAME).read_text().startswith("pid=")
        with pytest.raises(RuntimeError, match="already being written"):
            with claim_run_directories([run_dir]):
                pass


def test_claim_is_released_when_the_writer_exits(tmp_path):
    run_dir = tmp_path / "run"
    with claim_run_directories([run_dir]):
        pass
    with claim_run_directories([run_dir]):
        pass


def test_multi_run_claim_rolls_back_when_one_directory_is_busy(tmp_path):
    free = tmp_path / "a-free"
    busy = tmp_path / "z-busy"
    with claim_run_directories([busy]):
        with pytest.raises(RuntimeError, match="z-busy"):
            with claim_run_directories([free, busy]):
                pass
        with claim_run_directories([free]):
            pass


def test_run_domains_rejects_duplicate_result_directories():
    config = TextRunConfig(domain="mock", save_to="duplicate")
    with pytest.raises(ValueError, match="distinct results dir"):
        run_domains([config, config], workers=1)


def test_run_domains_checks_claims_before_preparing(tmp_path, monkeypatch):
    monkeypatch.setattr("tau2.runner.batch.DATA_DIR", tmp_path)
    config = TextRunConfig(domain="mock", save_to="claimed")
    run_dir = tmp_path / "simulations" / "claimed"
    with claim_run_directories([run_dir]):
        with pytest.raises(RuntimeError, match="already being written"):
            run_domains([config], workers=1)
