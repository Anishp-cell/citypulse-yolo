"""Status-change notifications. Routes call `notify_event` after committing a transition; delivery happens
in a background task so requests never wait on push services."""

from fastapi import BackgroundTasks


def deliver_event(app, event_id: int, exclude_user_id: int | None = None) -> None:
    """Send push notifications for one StatusEvent (implemented with web push in services/push.py)."""
    push = getattr(app.state, "push", None)
    if push is not None:
        push.send_for_event(event_id, exclude_user_id=exclude_user_id)


def notify_event(background: BackgroundTasks, app, event_id: int, exclude_user_id: int | None = None) -> None:
    background.add_task(deliver_event, app, event_id, exclude_user_id)
