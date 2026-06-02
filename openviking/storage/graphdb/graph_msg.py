# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph extraction queue message model."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class GraphMsg:
    """Message for graph extraction queue.

    Attributes:
        id: Unique identifier (UUID).
        messages: Archived message list for conversation assembly.
        written_entries: entity/event written fields data.
        account_id: Tenant identifier.
        user_id: User identifier.
        source: Source identifier (session_id).
        timestamp: Creation timestamp (seconds since epoch).
    """

    messages: List[Any]
    written_entries: List[Dict]
    account_id: str
    user_id: str
    source: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphMsg":
        raw_messages = data.get("messages", [])
        messages: List[Any] = []
        if raw_messages and isinstance(raw_messages[0], dict):
            from openviking.message import Message

            messages = [Message.from_dict(m) for m in raw_messages]
        else:
            messages = raw_messages

        return cls(
            id=data.get("id", str(uuid4())),
            messages=messages,
            written_entries=data.get("written_entries", []),
            account_id=data.get("account_id", ""),
            user_id=data.get("user_id", ""),
            source=data.get("source", ""),
            timestamp=data.get("timestamp", int(datetime.now().timestamp())),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "GraphMsg":
        return cls.from_dict(json.loads(json_str))
