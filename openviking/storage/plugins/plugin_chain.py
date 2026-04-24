# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""PreWritePlugin chain manager."""

from typing import List

from openviking.server.identity import RequestContext
from openviking_cli.utils.config import get_openviking_config
from openviking_cli.utils.logger import get_logger

from .pre_write_plugin import PreWritePlugin

logger = get_logger(__name__)


class PreWritePluginChain:
    """管理并执行所有 PreWritePlugin。"""

    def __init__(self):
        self._plugins: List[PreWritePlugin] = []
        self._load_plugins()

    def _load_plugins(self) -> None:
        """根据配置加载启用的插件。"""
        config = get_openviking_config()
        compressor_cfg = getattr(config, "tool_compressor", None)

        # 如果配置存在且显式禁用，则不加载
        if compressor_cfg and not getattr(compressor_cfg, "enabled", True):
            logger.info("[PreWritePluginChain] Tool compressor disabled by config")
            return

        # 默认启用工具压缩插件
        from .tool_compressor_plugin import ToolOutputCompressorPlugin

        self._plugins.append(ToolOutputCompressorPlugin())
        self._plugins.sort(key=lambda p: p.priority)

        logger.info(
            "[PreWritePluginChain] Loaded plugins: %s",
            [p.name for p in self._plugins],
        )

    async def process(self, uri: str, content: str, ctx: RequestContext) -> str:
        """依次执行所有插件。"""
        logger.info(
            "[PreWritePluginChain] Start processing uri=%s, "
            "plugins=%s, content_len=%d",
            uri,
            [p.name for p in self._plugins],
            len(content),
        )
        for plugin in self._plugins:
            try:
                processed = await plugin.process(uri, content, ctx)
                if processed != content:
                    logger.info(
                        "[PreWritePluginChain] %s transformed: %d -> %d chars",
                        plugin.name,
                        len(content),
                        len(processed),
                    )
                else:
                    logger.debug(
                        "[PreWritePluginChain] %s skipped (no change)",
                        plugin.name,
                    )
                content = processed
            except Exception as e:
                logger.warning(
                    "[PreWritePluginChain] %s failed: %s. Using original content.",
                    plugin.name,
                    e,
                    exc_info=True,
                )
        logger.info(
            "[PreWritePluginChain] Finish processing uri=%s, final_len=%d",
            uri,
            len(content),
        )
        return content
