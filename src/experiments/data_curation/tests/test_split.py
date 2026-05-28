"""Tests for the (optional) SplitTrajectory node. Not used for pre-split ToolMind."""
from curation import Pipeline, SplitTrajectory, build_node


def _traj():
    # u1, a1, t1, a2, t2, u2, a3  -> 3 assistant anchors
    return {
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "conversations": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1", "tool_calls": [{"function": {"name": "f", "arguments": {}}}]},
            {"role": "tool", "content": "t1"},
            {"role": "assistant", "content": "a2", "tool_calls": [{"function": {"name": "f", "arguments": {}}}]},
            {"role": "tool", "content": "t2"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a3"},
        ],
    }


def test_one_sample_per_anchor():
    out = list(SplitTrajectory()([_traj()]))
    assert len(out) == 3
    assert [len(s["conversations"]) for s in out] == [2, 4, 7]  # growing prefixes
    for s in out:
        assert s["conversations"][-1]["role"] == "assistant"
        assert s["tools"] == _traj()["tools"]  # carried through


def test_samples_are_prefix_chain():
    a, b, c = (s["conversations"] for s in SplitTrajectory()([_traj()]))
    assert b[: len(a)] == a and c[: len(b)] == b


def test_no_anchor_returns_unchanged():
    rec = {"conversations": [{"role": "user", "content": "hi"}]}
    assert list(SplitTrajectory()([rec])) == [rec]


def test_fans_out_with_stats():
    node = SplitTrajectory()
    out = list(node([_traj(), _traj()]))
    assert len(out) == 6 and node.stats.seen == 2 and node.stats.emitted == 6


def test_registered_and_composes():
    assert isinstance(build_node({"type": "split_trajectory"}), SplitTrajectory)
    # raw-trajectory path is [SplitTrajectory, FormatSFT]; split half verified here
    assert len(list(Pipeline([SplitTrajectory(name="split")])([_traj()]))) == 3
