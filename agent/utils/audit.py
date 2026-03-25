"""Structured audit logging for Webex-triggered agent runs.

Emits JSON log records to a dedicated "openswe.audit" logger so they can be
routed to a separate file, log aggregator, or SIEM without mixing into the
normal application logs.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

audit_logger = logging.getLogger("openswe.audit")

if not audit_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(_handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False


def log_audit_event(
    event: str,
    *,
    user_email: str = "",
    room_id: str = "",
    repo: str = "",
    thread_id: str = "",
    run_id: str = "",
    outcome: str = "",
    detail: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    """Emit a single structured audit log entry as JSON."""
    record: dict[str, Any] = {
        "ts": time.time(),
        "event": event,
        "user_email": user_email,
        "room_id": room_id,
        "repo": repo,
        "thread_id": thread_id,
        "run_id": run_id,
        "outcome": outcome,
    }
    if detail:
        record["detail"] = detail
    if extra:
        record.update(extra)
    audit_logger.info(json.dumps(record, default=str))
