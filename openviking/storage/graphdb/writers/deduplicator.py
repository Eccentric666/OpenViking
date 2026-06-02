# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Node deduplicator based on source + name composite key."""

from typing import Any, Dict, Optional

from openviking_cli.utils.logger import get_logger

from ..neo4j_backend import Neo4jBackend

logger = get_logger(__name__)


class NodeDeduplicator:
    """Match existing nodes by source + name composite key."""

    def __init__(self, backend: Neo4jBackend):
        self._backend = backend

    async def find_existing_node(
        self,
        name: str,
        source: str,
        account_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Query existing node by composite key.

        Returns:
            Node dict if found, None otherwise.
        """
        cypher = """
            MATCH (n:Node)
            WHERE n.account_id = $aid AND n.user_id = $uid
              AND toLower(n.source) = toLower($source)
              AND toLower(n.name) = toLower($name)
            RETURN n.name AS name, n.source AS source
            LIMIT 1
        """
        result = await self._backend.execute(cypher, {
            "aid": account_id,
            "uid": user_id,
            "source": source,
            "name": name,
        })
        return result[0] if result else None
