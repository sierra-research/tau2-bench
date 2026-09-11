from types import SimpleNamespace

import pytest

from tau2 import cli
from tau2.data_model.simulation import TerminationReason


def test_cli_run_exits_nonzero_for_persisted_infrastructure_error(monkeypatch):
    results = SimpleNamespace(
        simulations=[
            SimpleNamespace(termination_reason=TerminationReason.INFRASTRUCTURE_ERROR)
        ]
    )
    monkeypatch.setattr(cli, "run_domain", lambda config: results)

    with pytest.raises(SystemExit) as excinfo:
        cli._run_domain_cli(SimpleNamespace())

    assert excinfo.value.code == 1


def test_cli_run_returns_successful_results(monkeypatch):
    results = SimpleNamespace(
        simulations=[SimpleNamespace(termination_reason=TerminationReason.USER_STOP)]
    )
    monkeypatch.setattr(cli, "run_domain", lambda config: results)

    assert cli._run_domain_cli(SimpleNamespace()) is results
