# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tool output compression router —— detects tool type and dispatches to handler."""

from typing import Dict, List, Optional, Tuple

from openviking_cli.utils.logger import get_logger

from .handlers.base import CompressHandler
from .handlers.build import CMakeBuildHandler
from .handlers.cargo import CargoBuildHandler
from .handlers.docker import DockerBuildHandler
from .handlers.git import GitLogHandler, GitStatusHandler
from .handlers.js_test import JestHandler, PytestHandler, TscHandler, VitestHandler
from .handlers.npm import NpmInstallHandler

logger = get_logger(__name__)

# Tool signatures: keywords/patterns for quick identification
# Order matters: more specific signatures should come first
_TOOL_SIGNATURES: List[Tuple[str, str]] = [
    ("git status", "git_status"),
    ("git log", "git_log"),
    ("npm install", "npm_install"),
    ("npm ci", "npm_install"),
    ("cargo build", "cargo_build"),
    ("cargo test", "cargo_build"),
    ("docker build", "docker_build"),
    ("pytest", "pytest"),
    ("vitest", "vitest"),
    ("tsc ", "tsc"),
    ("jest", "jest"),
    ("cmake", "cmake_build"),
    ("make ", "cmake_build"),
]

# Handler name → class mapping
_HANDLER_CLASSES: Dict[str, type] = {
    "git_status": GitStatusHandler,
    "git_log": GitLogHandler,
    "npm_install": NpmInstallHandler,
    "cargo_build": CargoBuildHandler,
    "docker_build": DockerBuildHandler,
    "pytest": PytestHandler,
    "vitest": VitestHandler,
    "tsc": TscHandler,
    "jest": JestHandler,
    "cmake_build": CMakeBuildHandler,
}


class ToolCompressorRouter:
    """Tool output compression router —— detects tool type and dispatches to handler."""

    def __init__(self, enabled_handlers: Optional[List[str]] = None):
        """
        Args:
            enabled_handlers: List of enabled handler names, None means all enabled.
        """
        self._handlers: Dict[str, CompressHandler] = {}
        self._enabled = set(enabled_handlers) if enabled_handlers else None
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default handlers."""
        for name, cls in _HANDLER_CLASSES.items():
            if self._enabled is None or name in self._enabled:
                try:
                    self._handlers[name] = cls()
                except Exception as e:
                    logger.warning("[ToolCompressorRouter] Failed to register handler '%s': %s", name, e)

        logger.info(
            "[ToolCompressorRouter] Registered handlers: %s",
            list(self._handlers.keys()),
        )

    def detect(self, content: str) -> Optional[str]:
        """Detect the most likely tool type for the content.

        Returns:
            handler_name or None (not recognized)
        """
        content_len = len(content)
        first_lines = "\n".join(content.strip().split("\n")[:20]).lower()

        # Heuristic 1: check if content starts with known command signatures
        for signature, handler_name in _TOOL_SIGNATURES:
            if signature in first_lines:
                handler = self._handlers.get(handler_name)
                if handler and handler.can_handle(content):
                    logger.info(
                        "[ToolCompressorRouter] Detected tool='%s' via signature='%s', "
                        "content_len=%d",
                        handler_name,
                        signature,
                        content_len,
                    )
                    return handler_name

        # Heuristic 2: check can_handle for all registered handlers
        for name, handler in self._handlers.items():
            if handler.can_handle(content):
                logger.info(
                    "[ToolCompressorRouter] Detected tool='%s' via can_handle(), "
                    "content_len=%d",
                    name,
                    content_len,
                )
                return name

        logger.debug(
            "[ToolCompressorRouter] No tool detected, content_len=%d",
            content_len,
        )
        return None

    def compress(self, content: str, handler_name: Optional[str] = None) -> Optional[str]:
        """Compress content.

        Args:
            content: Raw content
            handler_name: Specific handler name, None for auto-detect

        Returns:
            Compressed content, or None (not recognized / no compression needed)
        """
        if handler_name is None:
            handler_name = self.detect(content)

        if not handler_name:
            logger.debug("[ToolCompressorRouter] compress skipped: no handler detected")
            return None

        handler = self._handlers.get(handler_name)
        if not handler:
            logger.warning(
                "[ToolCompressorRouter] Handler '%s' not registered",
                handler_name,
            )
            return None

        logger.info(
            "[ToolCompressorRouter] Compressing with handler='%s', content_len=%d",
            handler_name,
            len(content),
        )
        try:
            result = handler.compress(content)
            if result is None:
                logger.info(
                    "[ToolCompressorRouter] Handler '%s' returned None "
                    "(no compression applied)",
                    handler_name,
                )
            else:
                logger.info(
                    "[ToolCompressorRouter] Handler '%s' compressed: %d -> %d chars",
                    handler_name,
                    len(content),
                    len(result),
                )
            return result
        except Exception as e:
            logger.warning(
                "[ToolCompressorRouter] Handler '%s' compression failed: %s",
                handler_name,
                e,
                exc_info=True,
            )
            return None
