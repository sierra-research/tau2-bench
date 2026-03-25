from tau2.domains.mock.user_data_model import MockUserDB, Notification
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class MockUserTools(ToolKitBase):
    """User-side tools for the mock domain.

    Simulates a notification inbox where the user can check and manage
    notifications about their tasks.
    """

    db: MockUserDB

    def __init__(self, db: MockUserDB) -> None:
        super().__init__(db)

    @is_tool(ToolType.READ)
    def check_notifications(self) -> list[Notification]:
        """Check all notifications in the user's inbox.

        Returns:
            A list of all notifications.
        """
        return list(self.db.notifications.values())

    @is_tool(ToolType.WRITE)
    def dismiss_notification(self, notification_id: str) -> str:
        """Dismiss (mark as read) a notification.

        Args:
            notification_id: The ID of the notification to dismiss.

        Returns:
            A confirmation message.

        Raises:
            ValueError: If the notification is not found.
        """
        if notification_id not in self.db.notifications:
            raise ValueError(f"Notification {notification_id} not found")
        self.db.notifications[notification_id].status = "read"
        return f"Notification {notification_id} dismissed"

    # --- Non-tool helpers (for initialization_actions / env_assertions) ---

    def add_notification(
        self,
        notification_id: str,
        message: str,
        task_id: str | None = None,
    ) -> Notification:
        """Add a notification to the user's inbox (used by initialization_actions)."""
        notification = Notification(
            notification_id=notification_id,
            message=message,
            task_id=task_id,
        )
        self.db.notifications[notification_id] = notification
        return notification

    def assert_notification_status(
        self, notification_id: str, expected_status: str
    ) -> bool:
        """Check if a notification has the expected status (used by env_assertions)."""
        if notification_id not in self.db.notifications:
            raise ValueError(f"Notification {notification_id} not found")
        return self.db.notifications[notification_id].status == expected_status
