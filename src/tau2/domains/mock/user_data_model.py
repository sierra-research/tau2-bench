from typing import Dict, Literal, Optional

from pydantic import Field

from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra

NotificationStatus = Literal["unread", "read"]


class Notification(BaseModelNoExtra):
    """A notification in the user's inbox."""

    notification_id: str = Field(description="Unique identifier for the notification")
    message: str = Field(description="The notification message")
    status: NotificationStatus = Field(
        default="unread", description="Whether the notification has been read"
    )
    task_id: Optional[str] = Field(
        None, description="Associated task ID, if applicable"
    )


class MockUserDB(DB):
    """Simple user database with a notification inbox."""

    notifications: Dict[str, Notification] = Field(
        default_factory=dict,
        description="Dictionary of notifications indexed by notification ID",
    )
