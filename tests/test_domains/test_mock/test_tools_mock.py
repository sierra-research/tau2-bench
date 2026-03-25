import pytest

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall
from tau2.domains.mock.data_model import MockDB, Task, User
from tau2.domains.mock.environment import get_environment
from tau2.environment.environment import Environment


@pytest.fixture
def mock_db() -> MockDB:
    return MockDB(
        tasks={
            "task_1": Task(
                task_id="task_1",
                title="Test task",
                description="A test task",
                status="pending",
            )
        },
        users={"user_1": User(user_id="user_1", name="Test User", tasks=["task_1"])},
    )


@pytest.fixture
def environment(mock_db: MockDB) -> Environment:
    return get_environment(mock_db)


@pytest.fixture
def create_task_call() -> ToolCall:
    return ToolCall(
        id="1",
        name="create_task",
        arguments={
            "user_id": "user_1",
            "title": "New task",
            "description": "A new test task",
        },
    )


@pytest.fixture
def update_task_status_call() -> ToolCall:
    return ToolCall(
        id="2",
        name="update_task_status",
        arguments={"task_id": "task_1", "status": "completed"},
    )


def test_create_task(environment: Environment, create_task_call: ToolCall):
    response = environment.get_response(create_task_call)
    assert not response.error
    task = environment.tools.db.tasks["task_2"]
    assert task.title == "New task"
    assert task.description == "A new test task"
    assert task.status == "pending"

    # Test error case
    create_task_call.arguments["user_id"] = "nonexistent"
    response = environment.get_response(create_task_call)
    assert response.error


def test_update_task_status(
    environment: Environment, update_task_status_call: ToolCall
):
    response = environment.get_response(update_task_status_call)
    assert not response.error
    task = environment.tools.db.tasks["task_1"]
    assert task.status == "completed"

    # Test error case
    update_task_status_call.arguments["task_id"] = "nonexistent"
    response = environment.get_response(update_task_status_call)
    assert response.error


def test_user_tools(environment: Environment):
    """Test user tools: check_notifications, dismiss_notification, and helpers."""
    # Add a notification via initialization action
    environment.run_env_function_call(
        EnvFunctionCall(
            env_type="user",
            func_name="add_notification",
            arguments={
                "notification_id": "notif_1",
                "message": "Task assigned to you.",
                "task_id": "task_1",
            },
        )
    )

    # User calls check_notifications
    response = environment.get_response(
        ToolCall(
            id="1",
            name="check_notifications",
            arguments={},
            requestor="user",
        )
    )
    assert not response.error
    assert "notif_1" in response.content
    assert "Task assigned to you." in response.content

    # User dismisses the notification
    response = environment.get_response(
        ToolCall(
            id="2",
            name="dismiss_notification",
            arguments={"notification_id": "notif_1"},
            requestor="user",
        )
    )
    assert not response.error
    assert "dismissed" in response.content

    # Verify via env assertion
    result = environment.run_env_assertion(
        EnvAssertion(
            env_type="user",
            func_name="assert_notification_status",
            arguments={
                "notification_id": "notif_1",
                "expected_status": "read",
            },
        )
    )
    assert result is True

    # Dismiss nonexistent notification
    response = environment.get_response(
        ToolCall(
            id="3",
            name="dismiss_notification",
            arguments={"notification_id": "nonexistent"},
            requestor="user",
        )
    )
    assert response.error
