# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph schema DDL for Neo4j.

Creates indexes on first initialization; silently skips if they already exist.
"""

from typing import Any

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class GraphSchema:
    """Schema initializer for Neo4j graph database."""

    INDEXES = [
        "CREATE INDEX node_name_idx IF NOT EXISTS FOR (n:Node) ON (n.name)",
        "CREATE INDEX node_account_user_idx IF NOT EXISTS FOR (n:Node) ON (n.account_id, n.user_id)",
        "CREATE INDEX node_source_name_idx IF NOT EXISTS FOR (n:Node) ON (n.source, n.name)",
    ]

    @classmethod
    async def initialize(cls, backend: Any) -> None:
        """Execute DDL statements to set up indexes."""
        for ddl in cls.INDEXES:
            try:
                await backend.execute(ddl)
                logger.debug(f"[GraphSchema] Created index: {ddl[:60]}")
            except Exception as e:
                msg = str(e).lower()
                if "already exists" in msg or "equivalent" in msg:
                    logger.debug(f"[GraphSchema] Skipped (already exists): {ddl[:60]}")
                else:
                    logger.warning(f"[GraphSchema] DDL failed: {ddl[:60]} - {e}")
