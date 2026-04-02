"""Tests for data extraction from Results objects."""


from tau2.data_model.simulation import TerminationReason

from tau2_reliability.extract import _extract_action_sequence, extract_task_trial_data


class TestExtractActionSequence:
    def test_extracts_tool_call_names(self, make_sim):
        sim = make_sim(action_names=["search", "book", "confirm"])
        actions = _extract_action_sequence(sim)
        assert actions == ["search", "book", "confirm"]

    def test_empty_actions(self, make_sim):
        sim = make_sim(action_names=[])
        actions = _extract_action_sequence(sim)
        assert actions == []


class TestExtractTaskTrialData:
    def test_groups_by_task_id(self, make_sim, make_results):
        sims = [
            make_sim(task_id="t1", trial=0),
            make_sim(task_id="t1", trial=1),
            make_sim(task_id="t2", trial=0),
        ]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        assert len(task_data) == 2
        assert task_data[0].task_id == "t1"
        assert task_data[0].num_trials == 2
        assert task_data[1].task_id == "t2"
        assert task_data[1].num_trials == 1

    def test_filters_infrastructure_errors(self, make_sim, make_results):
        sims = [
            make_sim(task_id="t1", trial=0),
            make_sim(
                task_id="t1", trial=1,
                termination=TerminationReason.INFRASTRUCTURE_ERROR,
            ),
        ]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        assert len(task_data) == 1
        assert task_data[0].num_trials == 1

    def test_extracts_outcomes(self, make_sim, make_results):
        sims = [
            make_sim(task_id="t1", trial=0, reward=1.0),
            make_sim(task_id="t1", trial=1, reward=0.0),
            make_sim(task_id="t1", trial=2, reward=1.0),
        ]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        assert task_data[0].outcomes == [True, False, True]

    def test_extracts_costs(self, make_sim, make_results):
        sims = [
            make_sim(task_id="t1", trial=0, cost=0.1),
            make_sim(task_id="t1", trial=1, cost=0.2),
        ]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        assert task_data[0].costs == [0.1, 0.2]

    def test_extracts_action_sequences(self, make_sim, make_results):
        sims = [
            make_sim(task_id="t1", trial=0, action_names=["a", "b"]),
            make_sim(task_id="t1", trial=1, action_names=["a", "c"]),
        ]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        assert task_data[0].action_sequences == [["a", "b"], ["a", "c"]]

    def test_empty_results(self, make_results):
        results = make_results([])
        task_data = extract_task_trial_data(results)
        assert task_data == []

    def test_sorted_by_task_id(self, make_sim, make_results):
        sims = [
            make_sim(task_id="z_task", trial=0),
            make_sim(task_id="a_task", trial=0),
        ]
        results = make_results(sims)
        task_data = extract_task_trial_data(results)
        assert task_data[0].task_id == "a_task"
        assert task_data[1].task_id == "z_task"
