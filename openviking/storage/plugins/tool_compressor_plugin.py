# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tool output compressor plugin —— pre-write plugin for compressing tool outputs."""

from openviking.server.identity import RequestContext
from openviking_cli.utils.config import get_openviking_config
from openviking_cli.utils.logger import get_logger

from .compressors.router import ToolCompressorRouter
from .compressors.savings_checker import SavingsChecker, SavingsThreshold
from .pre_write_plugin import PreWritePlugin

logger = get_logger(__name__)


class ToolOutputCompressorPlugin(PreWritePlugin):
    """工具输出压缩插件 —— 移植 model-gate 的智能工具压缩能力。"""

    name = "tool_output_compressor"
    priority = 10  # 较早执行

    def __init__(self):
        config = get_openviking_config()
        # 读取压缩配置
        compressor_cfg = getattr(config, "tool_compressor", None) or {}

        enabled_handlers = compressor_cfg.get("handlers")
        self._router = ToolCompressorRouter(enabled_handlers=enabled_handlers)

        threshold_cfg = compressor_cfg.get("threshold", {})
        self._checker = SavingsChecker(
            SavingsThreshold(
                min_bytes=threshold_cfg.get("min_bytes", 40),
                min_lines=threshold_cfg.get("min_lines", 10),
                min_ratio=threshold_cfg.get("min_ratio", 0.1),
                max_lines=threshold_cfg.get("max_lines", 500),
            )
        )

        self._min_content_length = compressor_cfg.get("min_content_length", 500)

    async def process(self, uri: str, content: str, ctx: RequestContext) -> str:
        content_len = len(content)

        # 快速过滤：太短的内容不需要压缩
        if content_len < self._min_content_length:
            logger.debug(
                "[ToolOutputCompressor] Skipped: content too short "
                "(%d < %d), uri=%s",
                content_len,
                self._min_content_length,
                uri,
            )
            return content

        logger.info(
            "[ToolOutputCompressor] Processing uri=%s, "
            "content_len=%d, min_length=%d",
            uri,
            content_len,
            self._min_content_length,
        )

        # 尝试压缩
        compressed = self._router.compress(content)
        if compressed is None:
            logger.info(
                "[ToolOutputCompressor] No compression applied for uri=%s "
                "(no matching handler or handler returned None)",
                uri,
            )
            return content

        # 验证节省门槛
        if self._checker.check(content, compressed):
            stats = self._checker.get_stats(content, compressed)
            logger.info(
                "[ToolOutputCompressor] COMPRESSED uri=%s: "
                "%dB/%dL -> %dB/%dL "
                "(save %dB/%dL, ratio %.1f%%)",
                uri,
                stats["orig_bytes"],
                stats["orig_lines"],
                stats["comp_bytes"],
                stats["comp_lines"],
                stats["byte_savings"],
                stats["line_savings"],
                stats["ratio"] * 100,
            )
            return compressed

        logger.info(
            "[ToolOutputCompressor] Rejected compression for uri=%s: "
            "savings below threshold, keeping original (%d chars)",
            uri,
            content_len,
        )
        return content
