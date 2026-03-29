"""Tests for confidence elicitation and parsing."""

import pytest

from tau2_reliability.models import TaskTrialData
from tau2_reliability.runners.confidence import _parse_confidence, attach_confidences


class TestParseConfidence:
    def test_json_format(self):
        text = '{"confidence": 85, "reasoning": "mostly correct"}'
        assert _parse_confidence(text) == pytest.approx(85.0)

    def test_json_in_code_block(self):
        text = '```json\n{"confidence": 72}\n```'
        assert _parse_confidence(text) == pytest.approx(72.0)

    def test_plain_number(self):
        assert _parse_confidence("My confidence is 90.") == pytest.approx(90.0)

    def test_number_zero(self):
        assert _parse_confidence("0") == pytest.approx(0.0)

    def test_number_hundred(self):
        assert _parse_confidence("100") == pytest.approx(100.0)

    def test_no_number(self):
        assert _parse_confidence("I have no idea") is None

    def test_clamps_to_100(self):
        text = '{"confidence": 150}'
        assert _parse_confidence(text) == pytest.approx(100.0)

    def test_handles_float(self):
        text = '{"confidence": 67.5}'
        assert _parse_confidence(text) == pytest.approx(67.5)


class TestAttachConfidences:
    def test_attaches_scores(self):
        td = TaskTrialData(
            task_id="t1", outcomes=[True, False, True],
            action_sequences=[["a"]] * 3,
            costs=[0.1] * 3, durations=[30] * 3, num_actions=[1] * 3,
        )
        confidences = {"t1": [0.9, 0.3, 0.8]}
        result = attach_confidences([td], confidences)
        assert result[0].confidence_scores == [0.9, 0.3, 0.8]

    def test_pads_missing_scores(self):
        td = TaskTrialData(
            task_id="t1", outcomes=[True, False, True],
            action_sequences=[["a"]] * 3,
            costs=[0.1] * 3, durations=[30] * 3, num_actions=[1] * 3,
        )
        confidences = {"t1": [0.9]}  # Only 1 score for 3 trials
        result = attach_confidences([td], confidences)
        assert len(result[0].confidence_scores) == 3
        assert result[0].confidence_scores[0] == 0.9
        assert result[0].confidence_scores[1] == 0.5  # Default

    def test_no_confidence_data(self):
        td = TaskTrialData(
            task_id="t1", outcomes=[True],
            action_sequences=[["a"]],
            costs=[0.1], durations=[30], num_actions=[1],
        )
        result = attach_confidences([td], {})
        assert result[0].confidence_scores is None

    def test_trims_excess_scores(self):
        td = TaskTrialData(
            task_id="t1", outcomes=[True],
            action_sequences=[["a"]],
            costs=[0.1], durations=[30], num_actions=[1],
        )
        confidences = {"t1": [0.9, 0.8, 0.7]}  # 3 scores for 1 trial
        result = attach_confidences([td], confidences)
        assert len(result[0].confidence_scores) == 1
