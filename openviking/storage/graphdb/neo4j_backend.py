# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Neo4j graph database connection layer.

Provides unified async Cypher execution interface.
"""

import asyncio
from typing import Any, Dict, List, Optional

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


class Neo4jBackend:
    """Neo4j async backend. Manages driver lifecycle and Cypher execution."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ):
        self._uri = uri
        self._username = username
        self._password = password
        self._database = database
        self._driver: Optional[Any] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_driver(self) -> Any:
        """Lazy-load neo4j package and create async driver bound to current loop."""
        current_loop = asyncio.get_running_loop()
        if self._driver is not None and self._loop is current_loop:
            return self._driver

        try:
            from neo4j import AsyncGraphDatabase
        except ImportError as e:
            raise RuntimeError(
                "neo4j package is not installed. "
                "Install it with: pip install neo4j"
            ) from e

        # Close old driver if loop changed
        if self._driver is not None and self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._driver.close(), self._loop
                ).result(timeout=5)
            except Exception:
                pass

        self._driver = AsyncGraphDatabase.driver(
            self._uri,
            auth=(self._username, self._password),
        )
        self._loop = current_loop
        logger.info(
            f"[Neo4jBackend] Driver created for {self._uri}, "
            f"database={self._database}"
        )
        return self._driver

    async def close(self) -> None:
        """Close driver and release connections."""
        if self._driver:
            try:
                await self._driver.close()
            except Exception:
                pass
            self._driver = None
            self._loop = None
            logger.info("[Neo4jBackend] Driver closed")

    async def execute(
        self,
        cypher: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute Cypher query and return list of dict records.

        Args:
            cypher: Cypher query string.
            params: Named parameters for the query.

        Returns:
            List of dicts, one per record. Empty list for no results.
        """
        params = params or {}
        records: List[Dict[str, Any]] = []

        driver = self._ensure_driver()
        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(cypher, **params)
                async for record in result:
                    records.append(dict(record))
        except Exception as e:
            logger.error(
                f"[Neo4jBackend] Cypher error: {e} | "
                f"query: {cypher[:200]}"
            )
            raise

        return records
