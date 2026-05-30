"""Streamlined Memory FastAPI sidecar.

Lightweight SQLite-backed observation store providing:
- POST /recall_state  — recall observations matching a query
- GET  /health         — health check
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add OpenViking root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Fallback: define minimal classes for type checking
    class BaseModel:  # type: ignore
        pass
    class Field:  # type: ignore
        @staticmethod
        def default_factory(*args, **kwargs):
            return None
    class FastAPI:  # type: ignore
        pass
    class HTTPException(Exception):  # type: ignore
        pass

from openviking.streamlined_memory.recall import StreamlinedMemoryRecall

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RecallStateRequest(BaseModel):
    """Request body for /recall_state."""

    query: str
    session_id: Optional[str] = None
    scope: str = "auto"
    view: str = "compact"
    limit: int = 10
    recall_intent: str = "resume_task"


class RecallStateResponse(BaseModel):
    """Response from /recall_state."""

    recalled: bool
    ok: bool
    enabled: bool = True
    state_block: str = ""
    local_timeline: list = Field(default_factory=list)
    resolved_thread: dict = Field(default_factory=dict)
    diagnostics: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(db_path: str) -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not HAS_FASTAPI:
        raise RuntimeError(
            "FastAPI is required for the Streamlined Memory sidecar. "
            "Install it with: pip install fastapi uvicorn pydantic"
        )

    app = FastAPI(
        title="OpenViking Streamlined Memory Sidecar",
        version="0.1.0",
        docs_url="/docs" if os.environ.get("DEBUG") else None,
    )

    recall_engine: Optional[StreamlinedMemoryRecall] = None

    @app.on_event("startup")
    async def startup() -> None:
        nonlocal recall_engine
        try:
            recall_engine = StreamlinedMemoryRecall(db_path)
            logger.info("Streamlined Memory sidecar ready (db=%s)", db_path)
        except Exception as exc:
            logger.error("Failed to initialize recall engine: %s", exc)
            raise

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Health check endpoint."""
        healthy = recall_engine is not None
        return {
            "status": "ok" if healthy else "error",
            "service": "streamlined_memory",
            "db_path": db_path,
            "db_exists": Path(db_path).exists(),
        }

    @app.post("/recall_state", response_model=RecallStateResponse)
    async def recall_state(request: RecallStateRequest) -> RecallStateResponse:
        """Recall observations matching the query."""
        if recall_engine is None:
            raise HTTPException(status_code=503, detail="Recall engine not initialized")

        result = recall_engine.recall(
            query=request.query,
            session_id=request.session_id,
            limit=request.limit,
            scope=request.scope,
        )

        return RecallStateResponse(
            recalled=result.recalled,
            ok=result.recalled,
            enabled=True,
            state_block=result.state_block,
            local_timeline=result.local_timeline,
            resolved_thread={},
            diagnostics=result.diagnostics,
        )

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Streamlined Memory Sidecar")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=1944, help="Bind port")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not HAS_FASTAPI:
        print("ERROR: FastAPI is required. Install: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)

    app = create_app(args.db)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())


if __name__ == "__main__":
    main()
