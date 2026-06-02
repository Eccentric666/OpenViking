# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph extraction queue."""

from typing import Any, Dict, Optional

from openviking_cli.utils.logger import get_logger

from ..queuefs.named_queue import NamedQueue
from .graph_msg import GraphMsg

logger = get_logger(__name__)


class GraphQueue(NamedQueue):
    """Graph extraction queue supporting GraphMsg enqueue/dequeue."""

    async def enqueue(self, msg: Optional[GraphMsg]) -> str:
        """Serialize GraphMsg and enqueue."""
        if msg is None:
            logger.warning("Graph extraction message is None, skipping enqueue")
            return ""
        logger.debug(f"Enqueued graph extraction message: {msg.id}")
        return await super().enqueue(msg.to_dict())

    async def dequeue(self) -> Optional[Dict[str, Any]]:
        """Dequeue and return raw dict (deserialization handled by GraphHandler)."""
        return await super().dequeue()

    async def peek(self) -> Optional[Dict[str, Any]]:
        """Peek at head message, return raw dict."""
        return await super().peek()
