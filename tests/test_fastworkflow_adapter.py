import os
import queue
import types

import pytest

from tau2.agent.base_agent import validate_message_format
from tau2.agent.fastworkflow_adapter import (
    FastWorkflowAgentAdapter,
    create_fastworkflow_agent,
)
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import as_tool


def _tool_one(x: int) -> str:
    """Test tool one."""
    return str(x)


def _tool_two(y: int) -> str:
    """Test tool two."""
    return str(y)


@pytest.fixture
def tools():
    return [as_tool(_tool_one), as_tool(_tool_two)]


@pytest.fixture
def adapter(monkeypatch, tools):
    monkeypatch.setattr(
        FastWorkflowAgentAdapter,
        "_find_workflow_path",
        lambda self, workflow_type: f"/tmp/{workflow_type}_workflow",
    )
    return FastWorkflowAgentAdapter(
        tools=tools,
        domain_policy="policy",
        workflow_type="retail",
    )


def test_find_workflow_path_success(monkeypatch, tools):
    adapter = FastWorkflowAgentAdapter.__new__(FastWorkflowAgentAdapter)
    adapter.tools = tools
    adapter.domain_policy = "policy"
    adapter.workflow_type = "retail"
    monkeypatch.setattr(os, "getcwd", lambda: "/repo")
    monkeypatch.setattr(
        os.path, "exists", lambda path: path == "/repo/examples/retail_workflow"
    )
    assert adapter._find_workflow_path("retail") == "/repo/examples/retail_workflow"


def test_find_workflow_path_telecom_workflow_variant(monkeypatch, tools):
    """'telecom-workflow' domain maps to the same telecom_workflow folder."""
    adapter = FastWorkflowAgentAdapter.__new__(FastWorkflowAgentAdapter)
    adapter.tools = tools
    adapter.domain_policy = "policy"
    adapter.workflow_type = "telecom-workflow"
    monkeypatch.setattr(os, "getcwd", lambda: "/repo")
    monkeypatch.setattr(
        os.path, "exists", lambda path: path == "/repo/examples/telecom_workflow"
    )
    assert (
        adapter._find_workflow_path("telecom-workflow")
        == "/repo/examples/telecom_workflow"
    )


def test_find_workflow_path_missing(monkeypatch, tools):
    adapter = FastWorkflowAgentAdapter.__new__(FastWorkflowAgentAdapter)
    adapter.tools = tools
    adapter.domain_policy = "policy"
    adapter.workflow_type = "retail"
    monkeypatch.setattr(os, "getcwd", lambda: "/repo")
    monkeypatch.setattr(os.path, "exists", lambda path: False)
    with pytest.raises(FileNotFoundError):
        adapter._find_workflow_path("retail")


def test_find_workflow_path_unsupported_domain(tools):
    """Unsupported domain names raise ValueError, not FileNotFoundError."""
    adapter = FastWorkflowAgentAdapter.__new__(FastWorkflowAgentAdapter)
    adapter.tools = tools
    adapter.domain_policy = "policy"
    adapter.workflow_type = "unsupported"
    with pytest.raises(ValueError, match="Unsupported domain"):
        adapter._find_workflow_path("unsupported")


@pytest.mark.parametrize(
    "workflow_type,model_module,model_name,path_module,path_attr,expected_path",
    [
        (
            "retail",
            "tau2.domains.retail.data_model",
            "RetailDB",
            "tau2.domains.retail.utils",
            "RETAIL_DB_PATH",
            "/retail/db.json",
        ),
        (
            "airline",
            "tau2.domains.airline.data_model",
            "FlightDB",
            "tau2.domains.airline.utils",
            "AIRLINE_DB_PATH",
            "/airline/db.json",
        ),
        (
            "telecom",
            "tau2.domains.telecom.data_model",
            "TelecomDB",
            "tau2.domains.telecom.utils",
            "TELECOM_DB_PATH",
            "/telecom/db.toml",
        ),
        (
            "telecom-workflow",
            "tau2.domains.telecom.data_model",
            "TelecomDB",
            "tau2.domains.telecom.utils",
            "TELECOM_DB_PATH",
            "/telecom/db.toml",
        ),
    ],
)
def test_load_domain_db(
    monkeypatch,
    adapter,
    workflow_type,
    model_module,
    model_name,
    path_module,
    path_attr,
    expected_path,
):
    adapter.workflow_type = workflow_type
    seen = {}

    class FakeDB:
        @classmethod
        def load(cls, path):
            seen["path"] = path
            return {"loaded": path}

    module = __import__(model_module, fromlist=[model_name])
    paths = __import__(path_module, fromlist=[path_attr])
    monkeypatch.setattr(module, model_name, FakeDB)
    monkeypatch.setattr(paths, path_attr, expected_path)
    assert adapter._load_domain_db() == {"loaded": expected_path}
    assert seen["path"] == expected_path


def test_load_domain_db_unsupported(adapter):
    adapter.workflow_type = "unsupported"
    with pytest.raises(KeyError):
        adapter._load_domain_db()


# ---------------------------------------------------------------------------
# env_db deep-copy and _sync_fw_db
# ---------------------------------------------------------------------------


class _FakeDB:
    """Minimal stand-in for a Pydantic DB model with model_copy()."""

    def __init__(self, **data):
        self._data = dict(data)

    def model_copy(self, deep: bool = False):
        import copy

        return copy.deepcopy(self) if deep else copy.copy(self)

    def __eq__(self, other):
        return isinstance(other, _FakeDB) and self._data == other._data


def test_load_domain_db_uses_env_db_deep_copy(monkeypatch, tools):
    """When env_db is provided, _load_domain_db returns a deep copy instead of loading from disk."""
    monkeypatch.setattr(
        FastWorkflowAgentAdapter,
        "_find_workflow_path",
        lambda self, wt: f"/tmp/{wt}_workflow",
    )
    fake_db = _FakeDB(customer="C1001", balance=100)
    adapter = FastWorkflowAgentAdapter(
        tools=tools,
        domain_policy="policy",
        workflow_type="retail",
        env_db=fake_db,
    )
    loaded = adapter._load_domain_db()
    # Should be equal in content
    assert loaded == fake_db
    # But must be a different object (deep copy)
    assert loaded is not fake_db


def test_load_domain_db_env_db_mutation_isolation(monkeypatch, tools):
    """Mutating the deep copy must not affect the original env_db."""
    monkeypatch.setattr(
        FastWorkflowAgentAdapter,
        "_find_workflow_path",
        lambda self, wt: f"/tmp/{wt}_workflow",
    )
    fake_db = _FakeDB(items=["a", "b"])
    adapter = FastWorkflowAgentAdapter(
        tools=tools,
        domain_policy="policy",
        workflow_type="telecom",
        env_db=fake_db,
    )
    loaded = adapter._load_domain_db()
    loaded._data["items"].append("c")
    # Original must be untouched
    assert fake_db._data["items"] == ["a", "b"]


def test_load_domain_db_falls_back_to_disk_when_no_env_db(adapter):
    """Without env_db, _load_domain_db should hit the disk-loading branches."""
    assert adapter._env_db is None
    # The existing parametrized test_load_domain_db already covers the disk path;
    # this just documents that the default adapter fixture has no env_db.


def test_sync_fw_db_updates_workflow_context(monkeypatch, tools):
    """_sync_fw_db replaces workflow.context['db'] with a fresh deep copy of env_db."""
    monkeypatch.setattr(
        FastWorkflowAgentAdapter,
        "_find_workflow_path",
        lambda self, wt: f"/tmp/{wt}_workflow",
    )
    fake_db = _FakeDB(bill_paid=False)
    adapter = FastWorkflowAgentAdapter(
        tools=tools,
        domain_policy="policy",
        workflow_type="retail",
        env_db=fake_db,
    )
    # Simulate initialized state with a workflow whose context has a stale DB
    adapter.is_initialized = True
    stale_db = _FakeDB(bill_paid=False)
    workflow = types.SimpleNamespace(context={"db": stale_db})
    adapter.chat_session = types.SimpleNamespace(get_active_workflow=lambda: workflow)

    # Mutate env_db to simulate orchestrator executing a bill payment
    fake_db._data["bill_paid"] = True

    adapter._sync_fw_db()

    # Workflow's DB should now reflect the updated env_db
    assert workflow.context["db"]._data["bill_paid"] is True
    # But it should be a copy, not the same object
    assert workflow.context["db"] is not fake_db


def test_sync_fw_db_noop_when_no_env_db(adapter):
    """_sync_fw_db is a no-op when env_db is None."""
    adapter.is_initialized = True
    adapter.chat_session = types.SimpleNamespace(
        get_active_workflow=lambda: (_ for _ in ()).throw(
            AssertionError("should not be called")
        )
    )
    # Should not raise — early return because _env_db is None
    adapter._sync_fw_db()


def test_sync_fw_db_noop_when_not_initialized(monkeypatch, tools):
    """_sync_fw_db is a no-op when adapter is not yet initialized."""
    monkeypatch.setattr(
        FastWorkflowAgentAdapter,
        "_find_workflow_path",
        lambda self, wt: f"/tmp/{wt}_workflow",
    )
    adapter = FastWorkflowAgentAdapter(
        tools=tools,
        domain_policy="policy",
        workflow_type="retail",
        env_db=_FakeDB(),
    )
    assert not adapter.is_initialized
    # Should not raise
    adapter._sync_fw_db()


def test_sync_fw_db_noop_when_no_active_workflow(monkeypatch, tools):
    """_sync_fw_db handles None active workflow gracefully."""
    monkeypatch.setattr(
        FastWorkflowAgentAdapter,
        "_find_workflow_path",
        lambda self, wt: f"/tmp/{wt}_workflow",
    )
    adapter = FastWorkflowAgentAdapter(
        tools=tools,
        domain_policy="policy",
        workflow_type="retail",
        env_db=_FakeDB(),
    )
    adapter.is_initialized = True
    adapter.chat_session = types.SimpleNamespace(get_active_workflow=lambda: None)
    # Should not raise
    adapter._sync_fw_db()


def test_sync_fw_db_called_every_turn(monkeypatch, tools):
    """_sync_fw_db is called at the start of every generate_next_message call."""
    monkeypatch.setattr(
        FastWorkflowAgentAdapter,
        "_find_workflow_path",
        lambda self, wt: f"/tmp/{wt}_workflow",
    )
    adapter = FastWorkflowAgentAdapter(
        tools=tools,
        domain_policy="policy",
        workflow_type="retail",
        env_db=_FakeDB(),
    )
    adapter.is_initialized = True

    sync_calls = {"count": 0}
    original_sync = adapter._sync_fw_db

    def counting_sync():
        sync_calls["count"] += 1
        original_sync()

    monkeypatch.setattr(adapter, "_sync_fw_db", counting_sync)
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    monkeypatch.setattr(
        adapter, "_drain_queues_until_idle", lambda **kwargs: ([], ["reply"])
    )

    adapter.generate_next_message(UserMessage(role="user", content="hi"), None)
    assert sync_calls["count"] == 1

    adapter.generate_next_message(UserMessage(role="user", content="again"), None)
    assert sync_calls["count"] == 2


def test_to_plain_kwargs(adapter):
    class HasModelDump:
        def model_dump(self):
            return {"a": 1}

    class HasDict:
        def dict(self):
            return {"b": 2}

    assert adapter._to_plain_kwargs(None) == {}
    assert adapter._to_plain_kwargs({"x": 1}) == {"x": 1}
    assert adapter._to_plain_kwargs(HasModelDump()) == {"a": 1}
    assert adapter._to_plain_kwargs(HasDict()) == {"b": 2}
    assert adapter._to_plain_kwargs([("k", 3)]) == {"k": 3}
    assert adapter._to_plain_kwargs(42) == {}


def test_drain_command_trace_filters_and_extracts(adapter):
    class Dir:
        AGENT_TO_WORKFLOW = "A2W"

    adapter.fastworkflow = types.SimpleNamespace(CommandTraceEventDirection=Dir)
    adapter.is_initialized = True
    adapter.chat_session = types.SimpleNamespace(command_trace_queue=queue.Queue())
    adapter.chat_session.command_trace_queue.put(
        types.SimpleNamespace(
            direction="A2W",
            command_name="_tool_one",
            parameters={"x": 1},
            response_text="skip",
            success=True,
        )
    )
    adapter.chat_session.command_trace_queue.put(
        types.SimpleNamespace(
            direction="W2A",
            command_name="_tool_one",
            parameters={"x": 2},
            response_text="ok",
            success=True,
        )
    )
    adapter.chat_session.command_trace_queue.put(
        types.SimpleNamespace(
            direction="W2A",
            command_name=None,
            parameters={"x": 3},
            response_text="bad",
            success=True,
        )
    )
    out = adapter._drain_command_trace()
    assert out == [("_tool_one", {"x": 2}, "ok", True)]


def test_drain_command_trace_when_uninitialized(adapter):
    adapter.is_initialized = False
    adapter.chat_session = None
    assert adapter._drain_command_trace() == []


def test_drain_agent_outputs(adapter):
    adapter.is_initialized = True
    q = queue.Queue()
    q.put(
        types.SimpleNamespace(
            command_responses=[
                types.SimpleNamespace(response=" one "),
                types.SimpleNamespace(response=""),
                types.SimpleNamespace(response="two"),
            ]
        )
    )
    q.put("three")
    adapter.chat_session = types.SimpleNamespace(command_output_queue=q)
    assert adapter._drain_agent_outputs() == ["one\ntwo", "three"]


def test_drain_queues_until_idle(monkeypatch, adapter):
    calls = {"trace": 0, "out": 0}

    def fake_trace(max_drain=200):
        calls["trace"] += 1
        if calls["trace"] == 1:
            return [("_tool_one", {"x": 1}, "ok", True)]
        return []

    def fake_out(max_drain=200):
        calls["out"] += 1
        if calls["out"] == 1:
            return ["hello"]
        return []

    monkeypatch.setattr(adapter, "_drain_command_trace", fake_trace)
    monkeypatch.setattr(adapter, "_drain_agent_outputs", fake_out)
    monkeypatch.setattr("tau2.agent.fastworkflow_adapter.time.sleep", lambda _: None)
    commands, texts = adapter._drain_queues_until_idle(idle_limit=2, max_num_steps=10)
    assert commands == [("_tool_one", {"x": 1}, "ok", True)]
    assert texts == ["hello"]


def test_convert_commands_to_tool_calls_filters(adapter):
    calls = adapter._convert_commands_to_tool_calls(
        [
            ("_tool_one", {"x": 1}, "ok", True),
            ("missing_tool", {"x": 2}, "ok", True),
            ("_tool_two", {"y": 3}, "bad", False),
        ]
    )
    assert len(calls) == 1
    assert calls[0].name == "_tool_one"
    assert calls[0].arguments == {"x": 1}
    assert calls[0].id.startswith("call_")
    assert len(calls[0].id) == 17


def test_generate_user_first_turn_tool_calls(monkeypatch, adapter):
    init_seen = {}

    def fake_init(initial_message=None):
        init_seen["initial"] = initial_message
        adapter.is_initialized = True

    monkeypatch.setattr(adapter, "_initialize_fastworkflow", fake_init)
    monkeypatch.setattr(
        adapter,
        "_drain_queues_until_idle",
        lambda **kwargs: ([("_tool_one", {"x": 1}, "ok", True)], ["pending text"]),
    )
    converted = [
        ToolCall(
            id="call_test123456",
            name="_tool_one",
            arguments={"x": 1},
            requestor="assistant",
        )
    ]
    monkeypatch.setattr(
        adapter, "_convert_commands_to_tool_calls", lambda commands: converted
    )
    msg, state = adapter.generate_next_message(
        UserMessage(role="user", content="hi"), None
    )
    assert init_seen["initial"] == "hi"
    assert msg.tool_calls == converted
    assert msg.content is None
    assert state["pending_texts"] == ["pending text"]


def test_generate_user_text_only(monkeypatch, adapter):
    adapter.is_initialized = True
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    monkeypatch.setattr(
        adapter, "_drain_queues_until_idle", lambda **kwargs: ([], ["a", "b"])
    )
    msg, state = adapter.generate_next_message(
        UserMessage(role="user", content="hi"), None
    )
    assert msg.tool_calls is None
    assert msg.content == "a\nb"
    assert state["pending_texts"] == []


def test_generate_user_invalid_commands_with_text_fallback(monkeypatch, adapter):
    adapter.is_initialized = True
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    monkeypatch.setattr(
        adapter,
        "_drain_queues_until_idle",
        lambda **kwargs: ([("missing_tool", {}, "x", True)], ["fallback"]),
    )
    monkeypatch.setattr(adapter, "_convert_commands_to_tool_calls", lambda commands: [])
    msg, _ = adapter.generate_next_message(UserMessage(role="user", content="hi"), None)
    assert msg.content == "fallback"
    assert msg.tool_calls is None


def test_generate_tool_message_uses_pending_texts(monkeypatch, adapter):
    monkeypatch.setattr(adapter, "_drain_queues_until_idle", lambda **kwargs: ([], []))
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": ["done"]}
    msg, state = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="ok"),
        state,
    )
    assert msg.content == "done"
    assert state["pending_texts"] == []


def test_generate_tool_message_with_extra_commands(monkeypatch, adapter):
    monkeypatch.setattr(
        adapter,
        "_drain_queues_until_idle",
        lambda **kwargs: ([("_tool_two", {"y": 2}, "ok", True)], ["fresh"]),
    )
    converted = [
        ToolCall(
            id="call_test654321",
            name="_tool_two",
            arguments={"y": 2},
            requestor="assistant",
        )
    ]
    monkeypatch.setattr(
        adapter, "_convert_commands_to_tool_calls", lambda commands: converted
    )
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": ["old"]}
    msg, state = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="ok"),
        state,
    )
    assert msg.tool_calls == converted
    assert state["pending_texts"] == ["fresh"]


def test_generate_tool_message_retry_summary(monkeypatch, adapter):
    calls = {"n": 0}
    pushed = {}

    def fake_drain(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([], [])
        return ([], ["summary text"])

    monkeypatch.setattr(adapter, "_drain_queues_until_idle", fake_drain)
    monkeypatch.setattr(
        adapter, "_push_user_message", lambda text: pushed.setdefault("text", text)
    )
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": []}
    msg, _ = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="tool output"),
        state,
    )
    assert "Tool results:" in pushed["text"]
    assert msg.content == "summary text"


def test_generate_multi_tool_message(monkeypatch, adapter):
    monkeypatch.setattr(adapter, "_drain_queues_until_idle", lambda **kwargs: ([], []))
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": ["done"]}
    msg, state = adapter.generate_next_message(
        MultiToolMessage(
            role="tool",
            tool_messages=[
                ToolMessage(id="1", role="tool", content="a"),
                ToolMessage(id="2", role="tool", content="b"),
            ],
        ),
        state,
    )
    assert msg.content == "done"
    assert state["pending_texts"] == []


def test_generate_unexpected_message_type(adapter):
    msg, _ = adapter.generate_next_message(
        SystemMessage(role="system", content="bad"),  # type: ignore[arg-type]
        {"message_history": [], "last_tool_calls": [], "pending_texts": []},
    )
    assert "unexpected situation" in msg.content.lower()


def test_generate_exception_path(monkeypatch, adapter):
    def fake_init(initial_message=None):
        adapter.is_initialized = True

    monkeypatch.setattr(adapter, "_initialize_fastworkflow", fake_init)
    monkeypatch.setattr(
        adapter,
        "_drain_queues_until_idle",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    msg, state = adapter.generate_next_message(
        UserMessage(role="user", content="hi"), None
    )
    assert "encountered an error: boom" in msg.content
    assert state is not None


def test_get_init_state_and_stop_and_reset(adapter):
    state = adapter.get_init_state(
        message_history=[UserMessage(role="user", content="hello")]
    )
    assert state["message_history"][0].content == "hello"
    assert state["pending_texts"] == []
    assert state["last_tool_calls"] == []

    cleared = {"ok": False}

    class FakeChatSessionClass:
        @staticmethod
        def clear_workflow_stack():
            cleared["ok"] = True

    adapter.is_initialized = True
    adapter.fastworkflow = types.SimpleNamespace(ChatSession=FakeChatSessionClass)
    adapter.chat_session = object()
    adapter.stop()
    assert cleared["ok"] is True
    assert adapter.is_initialized is False
    assert adapter.chat_session is None


def test_is_stop(adapter):
    assert adapter.is_stop(types.SimpleNamespace(content="done ###STOP### now"))
    assert not adapter.is_stop(types.SimpleNamespace(content="keep going"))


def test_is_stop_none_content(adapter):
    """is_stop should return False when content is None."""
    assert not adapter.is_stop(types.SimpleNamespace(content=None))


def test_generate_user_followup_pushes_message(monkeypatch, adapter):
    """Second UserMessage (already initialized) should call _push_user_message."""
    adapter.is_initialized = True
    pushed = {}
    monkeypatch.setattr(
        adapter, "_push_user_message", lambda text: pushed.setdefault("text", text)
    )
    monkeypatch.setattr(
        adapter, "_drain_queues_until_idle", lambda **kwargs: ([], ["response"])
    )
    msg, state = adapter.generate_next_message(
        UserMessage(role="user", content="follow up"), None
    )
    assert pushed["text"] == "follow up"
    assert msg.content == "response"
    assert msg.tool_calls is None


def test_generate_user_commands_no_valid_calls_no_texts(monkeypatch, adapter):
    """Commands present, but none map to valid tools AND no texts — 'still processing' fallback."""
    adapter.is_initialized = True
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    monkeypatch.setattr(
        adapter,
        "_drain_queues_until_idle",
        lambda **kwargs: ([("internal_cmd", {}, "x", True)], []),
    )
    monkeypatch.setattr(adapter, "_convert_commands_to_tool_calls", lambda commands: [])
    msg, _ = adapter.generate_next_message(UserMessage(role="user", content="hi"), None)
    assert "still processing" in msg.content.lower()
    assert msg.tool_calls is None


def test_generate_user_no_commands_no_texts(monkeypatch, adapter):
    """No commands and no texts from FW — 'still processing' fallback."""
    adapter.is_initialized = True
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    monkeypatch.setattr(adapter, "_drain_queues_until_idle", lambda **kwargs: ([], []))
    msg, _ = adapter.generate_next_message(UserMessage(role="user", content="hi"), None)
    assert "still processing" in msg.content.lower()
    assert msg.tool_calls is None


def test_generate_tool_extra_commands_no_valid_calls_no_texts(monkeypatch, adapter):
    """ToolMessage path: extra commands but none valid, no pending texts — generic fallback."""
    monkeypatch.setattr(
        adapter,
        "_drain_queues_until_idle",
        lambda **kwargs: ([("internal_cmd", {}, "x", True)], []),
    )
    monkeypatch.setattr(adapter, "_convert_commands_to_tool_calls", lambda commands: [])
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": []}
    msg, _ = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="ok"),
        state,
    )
    assert "processed the information" in msg.content.lower()
    assert msg.tool_calls is None


def test_generate_tool_extra_commands_no_valid_calls_with_pending(monkeypatch, adapter):
    """ToolMessage path: extra commands but none valid, pending texts available — use texts."""
    monkeypatch.setattr(
        adapter,
        "_drain_queues_until_idle",
        lambda **kwargs: ([("internal_cmd", {}, "x", True)], ["fresh text"]),
    )
    monkeypatch.setattr(adapter, "_convert_commands_to_tool_calls", lambda commands: [])
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": ["old"]}
    msg, state = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="ok"),
        state,
    )
    assert msg.content == "fresh text"
    assert msg.tool_calls is None
    assert state["pending_texts"] == []


def test_generate_tool_retry_produces_tool_calls(monkeypatch, adapter):
    """ToolMessage retry path: no pending texts, retry drain yields new valid tool calls."""
    calls = {"n": 0}

    def fake_drain(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([], [])
        return ([("_tool_one", {"x": 5}, "ok", True)], ["stashed"])

    converted = [
        ToolCall(
            id="call_retry123456",
            name="_tool_one",
            arguments={"x": 5},
            requestor="assistant",
        )
    ]
    monkeypatch.setattr(adapter, "_drain_queues_until_idle", fake_drain)
    monkeypatch.setattr(adapter, "_push_user_message", lambda text: None)
    monkeypatch.setattr(
        adapter, "_convert_commands_to_tool_calls", lambda commands: converted
    )
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": []}
    msg, state = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="tool output"),
        state,
    )
    assert msg.tool_calls == converted
    assert msg.content is None
    assert state["pending_texts"] == ["stashed"]


def test_generate_tool_retry_commands_no_valid_calls_with_texts(monkeypatch, adapter):
    """ToolMessage retry path: retry drain yields commands but none valid, texts available."""
    calls = {"n": 0}

    def fake_drain(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([], [])
        return ([("internal_cmd", {}, "x", True)], ["retry text"])

    monkeypatch.setattr(adapter, "_drain_queues_until_idle", fake_drain)
    monkeypatch.setattr(adapter, "_push_user_message", lambda text: None)
    monkeypatch.setattr(adapter, "_convert_commands_to_tool_calls", lambda commands: [])
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": []}
    msg, _ = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="tool output"),
        state,
    )
    assert msg.content == "retry text"
    assert msg.tool_calls is None


def test_generate_tool_retry_commands_no_valid_calls_no_texts(monkeypatch, adapter):
    """ToolMessage retry path: retry drain yields commands but none valid, no texts — generic fallback."""
    calls = {"n": 0}

    def fake_drain(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([], [])
        return ([("internal_cmd", {}, "x", True)], [])

    monkeypatch.setattr(adapter, "_drain_queues_until_idle", fake_drain)
    monkeypatch.setattr(adapter, "_push_user_message", lambda text: None)
    monkeypatch.setattr(adapter, "_convert_commands_to_tool_calls", lambda commands: [])
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": []}
    msg, _ = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="tool output"),
        state,
    )
    assert "processed the information" in msg.content.lower()
    assert msg.tool_calls is None


def test_generate_tool_retry_no_commands_no_texts(monkeypatch, adapter):
    """ToolMessage retry path: retry drain yields nothing — generic fallback."""
    calls = {"n": 0}

    def fake_drain(**kwargs):
        calls["n"] += 1
        return ([], [])

    monkeypatch.setattr(adapter, "_drain_queues_until_idle", fake_drain)
    monkeypatch.setattr(adapter, "_push_user_message", lambda text: None)
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": []}
    msg, _ = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="tool output"),
        state,
    )
    assert "processed the information" in msg.content.lower()
    assert msg.tool_calls is None


def test_generate_tool_retry_texts_only(monkeypatch, adapter):
    """ToolMessage retry path: retry drain yields only texts, no commands."""
    calls = {"n": 0}

    def fake_drain(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([], [])
        return ([], ["retry only text"])

    monkeypatch.setattr(adapter, "_drain_queues_until_idle", fake_drain)
    monkeypatch.setattr(adapter, "_push_user_message", lambda text: None)
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": []}
    msg, _ = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="tool output"),
        state,
    )
    assert msg.content == "retry only text"
    assert msg.tool_calls is None


def test_generate_multi_tool_retry_summary(monkeypatch, adapter):
    """MultiToolMessage with no pending texts triggers retry with combined tool summaries."""
    calls = {"n": 0}
    pushed = {}

    def fake_drain(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return ([], [])
        return ([], ["multi summary"])

    monkeypatch.setattr(adapter, "_drain_queues_until_idle", fake_drain)
    monkeypatch.setattr(
        adapter, "_push_user_message", lambda text: pushed.setdefault("text", text)
    )
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": []}
    msg, _ = adapter.generate_next_message(
        MultiToolMessage(
            role="tool",
            tool_messages=[
                ToolMessage(id="1", role="tool", content="result_a"),
                ToolMessage(id="2", role="tool", content="result_b"),
            ],
        ),
        state,
    )
    assert "result_a" in pushed["text"]
    assert "result_b" in pushed["text"]
    assert msg.content == "multi summary"


def test_drain_agent_outputs_when_uninitialized(adapter):
    """_drain_agent_outputs should return empty list when not initialized."""
    adapter.is_initialized = False
    adapter.chat_session = None
    assert adapter._drain_agent_outputs() == []


def test_reset_when_not_initialized(adapter):
    """reset when already not initialized should be a no-op."""
    adapter.is_initialized = False
    adapter.chat_session = None
    adapter.fastworkflow = None
    adapter.reset()  # should not raise
    assert adapter.is_initialized is False
    assert adapter.chat_session is None


def test_initialize_fastworkflow_failure(monkeypatch, adapter):
    """_initialize_fastworkflow should re-raise on import/init failure."""
    monkeypatch.setattr(
        "tau2.agent.fastworkflow_adapter.dotenv_values",
        lambda path: {},
    )

    def fake_import_fail(*args, **kwargs):
        raise ImportError("fastworkflow not found")

    import builtins

    original_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "fastworkflow":
            raise ImportError("fastworkflow not found")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)
    with pytest.raises(ImportError, match="fastworkflow not found"):
        adapter._initialize_fastworkflow(initial_message="hi")
    assert adapter.is_initialized is False


def test_convert_commands_empty_list(adapter):
    """_convert_commands_to_tool_calls with empty input returns empty list."""
    assert adapter._convert_commands_to_tool_calls([]) == []


def test_convert_commands_all_valid(adapter):
    """All commands map to valid tools and succeed."""
    calls = adapter._convert_commands_to_tool_calls(
        [
            ("_tool_one", {"x": 1}, "ok", True),
            ("_tool_two", {"y": 2}, "ok", True),
        ]
    )
    assert len(calls) == 2
    assert calls[0].name == "_tool_one"
    assert calls[1].name == "_tool_two"


def test_convert_commands_all_filtered(adapter):
    """All commands are filtered out (invalid names or failed)."""
    calls = adapter._convert_commands_to_tool_calls(
        [
            ("missing_tool", {"x": 1}, "ok", True),
            ("_tool_one", {"x": 2}, "bad", False),
        ]
    )
    assert calls == []


def test_get_init_state_empty(adapter):
    """get_init_state with no message_history returns empty lists."""
    state = adapter.get_init_state()
    assert state["message_history"] == []
    assert state["pending_texts"] == []
    assert state["last_tool_calls"] == []


def test_state_backward_compat_missing_pending_texts(monkeypatch, adapter):
    """generate_next_message adds pending_texts to old state dicts missing it."""
    adapter.is_initialized = True
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    monkeypatch.setattr(
        adapter, "_drain_queues_until_idle", lambda **kwargs: ([], ["text"])
    )
    old_state = {"message_history": [], "last_tool_calls": []}
    msg, state = adapter.generate_next_message(
        UserMessage(role="user", content="hi"), old_state
    )
    assert "pending_texts" in state
    assert msg.content == "text"


def test_generate_appends_to_message_history(monkeypatch, adapter):
    """Messages (user + assistant) are appended to state's message_history."""
    adapter.is_initialized = True
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    monkeypatch.setattr(
        adapter, "_drain_queues_until_idle", lambda **kwargs: ([], ["reply"])
    )
    state = {"message_history": [], "last_tool_calls": [], "pending_texts": []}
    msg, state = adapter.generate_next_message(
        UserMessage(role="user", content="hi"), state
    )
    assert len(state["message_history"]) == 2
    assert state["message_history"][0].role == "user"
    assert state["message_history"][1].role == "assistant"


def test_drain_command_trace_max_drain_limit(adapter):
    """_drain_command_trace respects max_drain limit."""

    class Dir:
        AGENT_TO_WORKFLOW = "A2W"

    adapter.fastworkflow = types.SimpleNamespace(CommandTraceEventDirection=Dir)
    adapter.is_initialized = True
    adapter.chat_session = types.SimpleNamespace(command_trace_queue=queue.Queue())
    for i in range(10):
        adapter.chat_session.command_trace_queue.put(
            types.SimpleNamespace(
                direction="W2A",
                command_name="_tool_one",
                parameters={"x": i},
                response_text="ok",
                success=True,
            )
        )
    out = adapter._drain_command_trace(max_drain=3)
    assert len(out) == 3
    # 7 items should remain in queue
    assert adapter.chat_session.command_trace_queue.qsize() == 7


def test_drain_agent_outputs_max_drain_limit(adapter):
    """_drain_agent_outputs respects max_drain limit."""
    adapter.is_initialized = True
    q = queue.Queue()
    for i in range(10):
        q.put(f"text_{i}")
    adapter.chat_session = types.SimpleNamespace(command_output_queue=q)
    out = adapter._drain_agent_outputs(max_drain=4)
    assert len(out) == 4
    assert q.qsize() == 6


def test_set_seed_is_noop(adapter):
    """set_seed should not raise."""
    adapter.set_seed(42)  # should be a no-op


# ---------------------------------------------------------------------------
# Message format validation (orchestrator compatibility)
# ---------------------------------------------------------------------------


def test_text_response_passes_message_validation(monkeypatch, adapter):
    """Text-only responses must pass orchestrator's validate_message_format."""
    adapter.is_initialized = True
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    monkeypatch.setattr(
        adapter, "_drain_queues_until_idle", lambda **kwargs: ([], ["hello"])
    )
    msg, _ = adapter.generate_next_message(UserMessage(role="user", content="hi"), None)
    valid, err = validate_message_format(msg)
    assert valid, f"Text response failed validation: {err}"


def test_tool_call_response_passes_message_validation(monkeypatch, adapter):
    """Tool-call-only responses must pass orchestrator's validate_message_format."""
    adapter.is_initialized = True
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    monkeypatch.setattr(
        adapter,
        "_drain_queues_until_idle",
        lambda **kwargs: ([("_tool_one", {"x": 1}, "ok", True)], ["pending"]),
    )
    converted = [
        ToolCall(
            id="call_abc123456789",
            name="_tool_one",
            arguments={"x": 1},
            requestor="assistant",
        )
    ]
    monkeypatch.setattr(
        adapter, "_convert_commands_to_tool_calls", lambda cmds: converted
    )
    msg, _ = adapter.generate_next_message(UserMessage(role="user", content="hi"), None)
    valid, err = validate_message_format(msg)
    assert valid, f"Tool call response failed validation: {err}"


def test_fallback_responses_pass_message_validation(monkeypatch, adapter):
    """All fallback/error responses must also pass validation."""
    adapter.is_initialized = True
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)
    # No commands, no texts → "still processing" fallback
    monkeypatch.setattr(adapter, "_drain_queues_until_idle", lambda **kwargs: ([], []))
    msg, _ = adapter.generate_next_message(UserMessage(role="user", content="hi"), None)
    valid, err = validate_message_format(msg)
    assert valid, f"Fallback response failed validation: {err}"


def test_error_response_passes_message_validation(monkeypatch, adapter):
    """Exception-path responses must pass validation."""

    def fake_init(initial_message=None):
        adapter.is_initialized = True

    monkeypatch.setattr(adapter, "_initialize_fastworkflow", fake_init)
    monkeypatch.setattr(
        adapter,
        "_drain_queues_until_idle",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    msg, _ = adapter.generate_next_message(UserMessage(role="user", content="hi"), None)
    valid, err = validate_message_format(msg)
    assert valid, f"Error response failed validation: {err}"


# ---------------------------------------------------------------------------
# Resume from message history (orchestrator restart compatibility)
# ---------------------------------------------------------------------------


def test_get_init_state_with_history_preserves_messages(adapter):
    """get_init_state with message_history should preserve all messages for restart."""
    history = [
        AssistantMessage(role="assistant", content="Hi!", cost=0.0),
        UserMessage(role="user", content="Help me"),
        AssistantMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_123",
                    name="_tool_one",
                    arguments={"x": 1},
                    requestor="assistant",
                )
            ],
            cost=0.0,
        ),
    ]
    state = adapter.get_init_state(message_history=history)
    assert len(state["message_history"]) == 3
    assert state["message_history"][0].role == "assistant"
    assert state["message_history"][1].role == "user"
    assert state["message_history"][2].tool_calls is not None
    assert state["pending_texts"] == []
    assert state["last_tool_calls"] == []


def test_stop_then_reuse(monkeypatch, adapter):
    """After stop(), adapter can be re-initialized for a new task."""
    # Simulate initialized state
    cleared = {"ok": False}

    class FakeChatSessionClass:
        @staticmethod
        def clear_workflow_stack():
            cleared["ok"] = True

    adapter.is_initialized = True
    adapter.fastworkflow = types.SimpleNamespace(ChatSession=FakeChatSessionClass)
    adapter.chat_session = object()

    adapter.stop()
    assert adapter.is_initialized is False

    # Now simulate re-initialization for a new task
    init_called = {}

    def fake_init(initial_message=None):
        init_called["msg"] = initial_message
        adapter.is_initialized = True

    monkeypatch.setattr(adapter, "_initialize_fastworkflow", fake_init)
    monkeypatch.setattr(
        adapter, "_drain_queues_until_idle", lambda **kwargs: ([], ["new task reply"])
    )
    msg, state = adapter.generate_next_message(
        UserMessage(role="user", content="new task"), None
    )
    assert init_called["msg"] == "new task"
    assert msg.content == "new task reply"


# ---------------------------------------------------------------------------
# Multi-turn conversation flow
# ---------------------------------------------------------------------------


def test_multi_turn_user_tool_user_flow(monkeypatch, adapter):
    """Simulate a realistic multi-turn flow: user → tool_calls → tool_result → user → text."""
    init_called = {}

    def fake_init(initial_message=None):
        init_called["msg"] = initial_message
        adapter.is_initialized = True

    drain_calls = {"n": 0}

    def fake_drain(**kwargs):
        drain_calls["n"] += 1
        if drain_calls["n"] == 1:
            # First call: user message → tool calls
            return ([("_tool_one", {"x": 1}, "ok", True)], ["pending"])
        elif drain_calls["n"] == 2:
            # Second call: after tool result, quick drain → nothing new
            return ([], [])
        elif drain_calls["n"] == 3:
            # Third call: second user message → text only
            return ([], ["final answer"])
        return ([], [])

    converted = [
        ToolCall(
            id="call_flow12345678",
            name="_tool_one",
            arguments={"x": 1},
            requestor="assistant",
        )
    ]
    monkeypatch.setattr(adapter, "_initialize_fastworkflow", fake_init)
    monkeypatch.setattr(adapter, "_drain_queues_until_idle", fake_drain)
    monkeypatch.setattr(
        adapter,
        "_convert_commands_to_tool_calls",
        lambda cmds: converted if cmds else [],
    )
    monkeypatch.setattr(adapter, "_push_user_message", lambda _: None)

    # Turn 1: User message → tool calls
    msg1, state = adapter.generate_next_message(
        UserMessage(role="user", content="do something"), None
    )
    assert msg1.tool_calls == converted
    assert msg1.content is None
    assert state["pending_texts"] == ["pending"]

    # Turn 2: Tool result → pending text
    msg2, state = adapter.generate_next_message(
        ToolMessage(id="1", role="tool", content="tool output"), state
    )
    assert msg2.content == "pending"
    assert msg2.tool_calls is None
    assert state["pending_texts"] == []

    # Turn 3: Follow-up user message → text
    msg3, state = adapter.generate_next_message(
        UserMessage(role="user", content="thanks"), state
    )
    assert msg3.content == "final answer"
    assert msg3.tool_calls is None

    # Verify all messages accumulated in history
    assert (
        len(state["message_history"]) == 6
    )  # user, assistant(tool), tool, assistant(text), user, assistant(text)


# ---------------------------------------------------------------------------
# _initialize_fastworkflow success paths
# ---------------------------------------------------------------------------


def test_initialize_fastworkflow_with_initial_message(monkeypatch, adapter):
    """Successful init with initial_message calls init, ChatSession, domain_policy, load_domain_db, start_workflow."""
    call_log = []

    class FakeChatSession:
        def __init__(self, run_as_agent=False):
            call_log.append(("ChatSession", run_as_agent))
            self.domain_policy = None

        def start_workflow(self, path, **kwargs):
            call_log.append(("start_workflow", path, kwargs))

        @staticmethod
        def clear_workflow_stack():
            call_log.append(("clear_workflow_stack",))

    fake_fw = types.SimpleNamespace(
        init=lambda env_vars: call_log.append(("fw_init", set(env_vars.keys()))),
        ChatSession=FakeChatSession,
    )

    monkeypatch.setattr(
        "tau2.agent.fastworkflow_adapter.dotenv_values",
        lambda path: {"KEY_FROM_" + path.split("/")[-1]: "val"},
    )

    import builtins

    original_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "fastworkflow":
            return fake_fw
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)
    monkeypatch.setattr(adapter, "_load_domain_db", lambda: {"mock": "db"})

    adapter._initialize_fastworkflow(initial_message="hello")

    assert adapter.is_initialized is True
    assert adapter.fastworkflow is fake_fw

    # Verify fw.init was called with merged env vars
    assert call_log[0][0] == "fw_init"
    env_keys = call_log[0][1]
    assert "KEY_FROM_fastworkflow.env" in env_keys
    assert "KEY_FROM_fastworkflow.passwords.env" in env_keys

    # Verify ChatSession created with run_as_agent=True
    cs_call = [c for c in call_log if c[0] == "ChatSession"]
    assert len(cs_call) == 1
    assert cs_call[0][1] is True

    # Verify start_workflow called with correct args including domain_policy in workflow_context
    sw_call = [c for c in call_log if c[0] == "start_workflow"]
    assert len(sw_call) == 1
    assert sw_call[0][1] == adapter.workflow_path
    assert sw_call[0][2]["workflow_context"] == {
        "db": {"mock": "db"},
        "domain_policy": "policy",
    }
    assert sw_call[0][2]["startup_command"] == "hello"
    assert sw_call[0][2]["keep_alive"] is True


def test_initialize_fastworkflow_without_initial_message(monkeypatch, adapter):
    """Init without initial_message should NOT call start_workflow."""
    call_log = []

    class FakeChatSession:
        def __init__(self, run_as_agent=False):
            self.domain_policy = None

        def start_workflow(self, path, **kwargs):
            call_log.append("start_workflow")

        @staticmethod
        def clear_workflow_stack():
            pass

    fake_fw = types.SimpleNamespace(
        init=lambda env_vars: None,
        ChatSession=FakeChatSession,
    )

    monkeypatch.setattr(
        "tau2.agent.fastworkflow_adapter.dotenv_values", lambda path: {}
    )

    import builtins

    original_import = builtins.__import__

    def patched_import(name, *args, **kwargs):
        if name == "fastworkflow":
            return fake_fw
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", patched_import)
    monkeypatch.setattr(adapter, "_load_domain_db", lambda: {"mock": "db"})

    adapter._initialize_fastworkflow(initial_message=None)

    assert adapter.is_initialized is True
    assert "start_workflow" not in call_log


def test_initialize_fastworkflow_already_initialized_is_noop(monkeypatch, adapter):
    """Calling _initialize_fastworkflow when already initialized should be a no-op."""
    adapter.is_initialized = True
    adapter.chat_session = object()  # sentinel

    # If init runs, this would blow up — proving it's a no-op
    monkeypatch.setattr(
        "tau2.agent.fastworkflow_adapter.dotenv_values",
        lambda path: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    adapter._initialize_fastworkflow(initial_message="hello")
    # Still the same sentinel
    assert adapter.chat_session is not None


# ---------------------------------------------------------------------------
# create_fastworkflow_agent factory
# ---------------------------------------------------------------------------


def test_create_fastworkflow_agent_passes_env_db(monkeypatch, tools):
    """Factory function should forward env_db kwarg to the adapter."""
    monkeypatch.setattr(
        FastWorkflowAgentAdapter,
        "_find_workflow_path",
        lambda self, wt: f"/tmp/{wt}_workflow",
    )
    fake_db = _FakeDB(flag=True)
    agent = create_fastworkflow_agent(
        tools=tools,
        domain_policy="policy",
        domain="telecom",
        env_db=fake_db,
    )
    assert isinstance(agent, FastWorkflowAgentAdapter)
    assert agent._env_db is fake_db
    assert agent.workflow_type == "telecom"


def test_create_fastworkflow_agent_without_env_db(monkeypatch, tools):
    """Factory function without env_db should set _env_db to None."""
    monkeypatch.setattr(
        FastWorkflowAgentAdapter,
        "_find_workflow_path",
        lambda self, wt: f"/tmp/{wt}_workflow",
    )
    agent = create_fastworkflow_agent(
        tools=tools,
        domain_policy="policy",
        domain="retail",
    )
    assert agent._env_db is None
