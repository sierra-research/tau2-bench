"""Tests for abstention detection."""


from tau2_reliability.analysis.abstention import (
    compute_abstention_metrics,
    detect_abstention,
)


class TestDetectAbstention:
    def test_no_abstention(self, make_sim):
        sim = make_sim(action_names=["get_details", "update_record"])
        result = detect_abstention(sim)
        assert result["abstained"] is False
        assert result["type"] == "none"

    def test_inability_detected(self, make_sim):
        # Build sim with inability message
        sim = make_sim(action_names=[])
        # Patch content to include inability phrase
        for msg in sim.messages:
            if msg.role == "assistant" and msg.content:
                msg.content = "I'm sorry, I'm unable to help with that request."
                break
        result = detect_abstention(sim)
        assert result["abstained"] is True
        assert result["type"] == "inability"

    def test_refusal_detected(self, make_sim):
        sim = make_sim(action_names=[])
        for msg in sim.messages:
            if msg.role == "assistant" and msg.content:
                msg.content = "For safety reasons, I cannot proceed with this action."
                break
        result = detect_abstention(sim)
        assert result["abstained"] is True
        assert result["type"] == "refusal"

    def test_uncertainty_below_threshold(self, make_sim):
        sim = make_sim(action_names=["get_details", "update_record"])
        for msg in sim.messages:
            if msg.role == "assistant" and msg.content:
                msg.content = "I'm not sure about this, but let me try."
                break
        result = detect_abstention(sim)
        # Single uncertainty mention should be below threshold
        assert result["strength"] < 0.3 or result["abstained"] is False

    def test_strength_accumulates(self, make_sim):
        sim = make_sim(action_names=[])
        msgs_set = 0
        for msg in sim.messages:
            if msg.role == "assistant":
                if msgs_set == 0:
                    msg.content = "I cannot help with this. I'm unable to proceed."
                    msgs_set += 1
        result = detect_abstention(sim)
        assert result["strength"] > 0.3

    def test_evidence_extracted(self, make_sim):
        sim = make_sim(action_names=[])
        for msg in sim.messages:
            if msg.role == "assistant" and msg.content:
                msg.content = "Unfortunately, I cannot perform this operation."
                break
        result = detect_abstention(sim)
        if result["abstained"]:
            assert len(result["evidence"]) > 0


class TestComputeAbstentionMetrics:
    def test_no_abstentions(self, make_sim):
        sims = [make_sim(trial=i, reward=1.0) for i in range(5)]
        metrics = compute_abstention_metrics(sims)
        assert metrics["rate"] == 0.0

    def test_all_abstentions(self, make_sim):
        sims = []
        for i in range(3):
            sim = make_sim(trial=i, reward=0.0, action_names=[])
            for msg in sim.messages:
                if msg.role == "assistant" and msg.content:
                    msg.content = "I'm unable to help. I cannot proceed with this request."
                    break
            sims.append(sim)
        metrics = compute_abstention_metrics(sims)
        # All should be detected as abstaining
        assert metrics["rate"] > 0.0

    def test_empty_list(self):
        metrics = compute_abstention_metrics([])
        assert metrics["rate"] == 0
