"""Streamlined Memory recall logic.

Queries the SQLite-backed observation store using FTS5 full-text search
and returns a state_block assembled from matching observations.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """Result of a recall operation."""

    recalled: bool
    state_block: str
    local_timeline: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]


class StreamlinedMemoryRecall:
    """Recall engine for Streamlined Memory observations."""

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        if not self._db_path.exists():
            raise FileNotFoundError(f"Streamlined Memory database not found: {db_path}")
        logger.info("StreamlinedMemoryRecall initialized (db=%s)", db_path)

    def recall(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10,
        scope: str = "auto",
    ) -> RecallResult:
        """Recall observations matching the query.

        Args:
            query: User query text.
            session_id: Optional session ID for session-scoped recall.
            limit: Maximum number of observations to return.
            scope: Recall scope (session | thread | global | auto).

        Returns:
            RecallResult with state_block assembled from matching observations.
        """
        diagnostics: Dict[str, Any] = {"query": query, "scope": scope}

        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Build FTS5 query from user query
            # Escape special FTS5 characters and join terms with AND
            fts_query = self._build_fts_query(query)
            diagnostics["fts_query"] = fts_query

            # Query observations via FTS5
            observations = self._search_observations(cursor, fts_query, session_id, limit)
            diagnostics["match_count"] = len(observations)

            if not observations:
                diagnostics["empty_reason"] = "no_matching_observations"
                return RecallResult(
                    recalled=False,
                    state_block="",
                    local_timeline=[],
                    diagnostics=diagnostics,
                )

            # Assemble state_block from observations
            timeline = self._build_timeline(observations)
            state_block = self._assemble_state_block(observations)

            diagnostics["state_block_chars"] = len(state_block)
            diagnostics["timeline_entries"] = len(timeline)

            return RecallResult(
                recalled=True,
                state_block=state_block,
                local_timeline=timeline,
                diagnostics=diagnostics,
            )

        except Exception as exc:
            logger.exception("Streamlined Memory recall failed")
            diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            return RecallResult(
                recalled=False,
                state_block="",
                local_timeline=[],
                diagnostics=diagnostics,
            )
        finally:
            if "conn" in locals():
                conn.close()

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """Build an FTS5 query from natural language query.

        Escapes FTS5 special characters and joins tokens with AND.
        """
        # Remove common question words and punctuation
        stopwords = {"when", "did", "what", "where", "who", "how", "the", "a", "an",
                     "is", "was", "were", "are", "to", "of", "in", "on", "at", "for",
                     "with", "about", "from", "by", "and", "or", "?", ".", ",", "!"}
        tokens = []
        for token in query.lower().split():
            token = token.strip(".,?!;:\"'")
            if token and token not in stopwords and len(token) > 2:
                # Escape FTS5 special characters
                token = token.replace('"', '""')
                tokens.append(f'"{token}"')
        if not tokens:
            # Fallback: use all non-stopword tokens
            for token in query.lower().split():
                token = token.strip(".,?!;:\"'")
                if token and len(token) > 2:
                    token = token.replace('"', '""')
                    tokens.append(f'"{token}"')
        if not tokens:
            return query.replace('"', '""')
        return " AND ".join(tokens[:8])  # Limit to top 8 tokens

    def _search_observations(
        self,
        cursor: sqlite3.Cursor,
        fts_query: str,
        session_id: Optional[str],
        limit: int,
    ) -> List[sqlite3.Row]:
        """Search observations using FTS5.

        If FTS5 returns no results, falls back to a fast prefix LIKE on title only.
        """
        # FTS5 search
        try:
            if session_id:
                cursor.execute(
                    """
                    SELECT o.* FROM observations o
                    JOIN observations_fts fts ON o.rowid = fts.rowid
                    WHERE observations_fts MATCH ? AND o.session_id = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, session_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT o.* FROM observations o
                    JOIN observations_fts fts ON o.rowid = fts.rowid
                    WHERE observations_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
            rows = cursor.fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError as exc:
            logger.warning("FTS5 search failed (%s), falling back to title LIKE", exc)

        # Fast fallback: search only title column (indexed, much faster)
        # Extract the first meaningful token for LIKE
        tokens = [t.strip('"') for t in fts_query.split(" AND ") if len(t.strip('"')) > 2]
        if not tokens:
            return []

        conditions = []
        params = []
        for token in tokens[:3]:  # Use top 3 tokens
            conditions.append("title LIKE ?")
            params.append(f"%{token}%")

        where_clause = " OR ".join(conditions)
        if session_id:
            where_clause = f"session_id = ? AND ({where_clause})"
            params = [session_id] + params

        params.append(limit)
        cursor.execute(
            f"SELECT * FROM observations WHERE {where_clause} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return cursor.fetchall()

    @staticmethod
    def _build_timeline(observations: List[sqlite3.Row]) -> List[Dict[str, Any]]:
        """Build a timeline from observations."""
        timeline = []
        for obs in observations:
            timeline.append({
                "obs_id": obs["obs_id"],
                "created_at": obs["created_at"],
                "type": obs["type"],
                "title": obs["title"],
                "summary": obs["summary"],
                "content": obs["content"][:200] if obs["content"] else "",
                "thread_id": obs["thread_id"],
            })
        return timeline

    @staticmethod
    def _assemble_state_block(observations: List[sqlite3.Row]) -> str:
        """Assemble a state_block string from observations.

        The state_block is a natural-language summary of the recalled
        observations, suitable for injection into an LLM prompt.
        """
        lines: List[str] = []
        lines.append("## Relevant Context from Past Conversations\n")

        for i, obs in enumerate(observations, 1):
            title = obs["title"] or "Untitled"
            summary = obs["summary"] or ""
            content = obs["content"] or ""
            created_at = obs["created_at"] or ""
            obs_type = obs["type"] or "observation"

            lines.append(f"### [{i}] {title}")
            if created_at:
                lines.append(f"*Time: {created_at}*")
            if obs_type:
                lines.append(f"*Type: {obs_type}*")
            if summary:
                lines.append(summary)
            elif content:
                # Truncate long content
                content_preview = content[:500] + "..." if len(content) > 500 else content
                lines.append(content_preview)
            lines.append("")

        return "\n".join(lines)
