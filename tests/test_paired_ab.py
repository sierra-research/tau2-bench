"""Tests for paired A/B scoring (fix for the T19 asymmetric-exclusion artifact)."""

import pytest

from tau2.data_model.simulation import TerminationReason
from tau2.metrics.paired_ab import episode_status, paired_compare


class FakeActionCheck:
    def __init__(self, match):
        self.action_match = match


class FakeRewardInfo:
    def __init__(self, reward, checks):
        self.reward = reward
        self.action_checks = checks


class FakeSim:
    def __init__(self, task_id, termination, reward=None, checks=None):
        self.task_id = task_id
        self.termination_reason = termination
        self.reward_info = (
            None if reward is None else FakeRewardInfo(reward, checks or [])
        )


class FakeResults:
    def __init__(self, sims):
        self.simulations = sims


def scored(task_id, reward, matched, total):
    checks = [FakeActionCheck(True)] * matched + [FakeActionCheck(False)] * (total - matched)
    return FakeSim(task_id, TerminationReason.AGENT_STOP, reward, checks)


def test_happy_path_paired_metrics():
    a = FakeResults([scored("t1", 1.0, 3, 4), scored("t2", 0.0, 1, 2)])
    b = FakeResults([scored("t1", 1.0, 4, 4), scored("t2", 1.0, 2, 2)])
    r = paired_compare(a, b, "fp4", "fp8")
    assert r.n_pairs == 2
    assert r.arm_a.pass_count == 1 and r.arm_b.pass_count == 2
    assert (r.arm_a.action_matched, r.arm_a.action_total) == (4, 6)
    assert (r.arm_b.action_matched, r.arm_b.action_total) == (6, 6)


def test_infra_error_drops_pair_in_both_arms():
    # t2 died as infra error in arm a: it must not count for arm b either.
    a = FakeResults([scored("t1", 1.0, 2, 2),
                     FakeSim("t2", TerminationReason.INFRASTRUCTURE_ERROR)])
    b = FakeResults([scored("t1", 1.0, 2, 2), scored("t2", 1.0, 5, 5)])
    r = paired_compare(a, b, "fp4", "fp8")
    assert r.paired_task_ids == ["t1"]
    assert r.arm_b.action_total == 2  # t2's 5 checks excluded from the comparison
    assert r.exclusions["fp4"]["infra_error:infrastructure_error"] == 1
    assert r.tasks_only_in_one_arm["fp8"] == ["t2"]


def test_premature_termination_not_scored_as_pair():
    # max_steps loop: reward 0 but no action checks; pair must be excluded, not
    # silently averaged (the T19 artifact).
    a = FakeResults([scored("t1", 1.0, 2, 2),
                     FakeSim("t2", TerminationReason.MAX_STEPS, reward=0.0)])
    b = FakeResults([scored("t1", 1.0, 2, 2), scored("t2", 1.0, 3, 3)])
    r = paired_compare(a, b)
    assert r.paired_task_ids == ["t1"]
    assert any(k.startswith("unscored_premature") for k in r.exclusions["arm_a"])


def test_empty_inputs():
    r = paired_compare(FakeResults([]), FakeResults([]))
    assert r.n_pairs == 0
    assert r.arm_a.action_match_rate is None
    assert "WARNING" in r.render()


def test_small_n_warning_in_render():
    a = FakeResults([scored("t1", 1.0, 1, 1)])
    b = FakeResults([scored("t1", 1.0, 1, 1)])
    assert "not statistically meaningful" in paired_compare(a, b).render()


def test_episode_status_classification():
    assert episode_status(FakeSim("t", TerminationReason.AGENT_STOP, 1.0, [])) == "scored"
    assert episode_status(FakeSim("t", TerminationReason.INFRASTRUCTURE_ERROR)) == "infra_error"
    assert episode_status(FakeSim("t", TerminationReason.MAX_STEPS)) == "unscored_premature"
