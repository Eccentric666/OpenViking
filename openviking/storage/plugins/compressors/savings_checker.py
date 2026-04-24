# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Savings checker for tool output compression."""

from dataclasses import dataclass
from typing import Optional

from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SavingsThreshold:
    """压缩节省门槛配置。"""

    min_bytes: int = 40  # 最少节省 40 字节才接受压缩
    min_lines: int = 10  # 最少节省 10 行才接受压缩
    min_ratio: float = 0.1  # 最少压缩比例 10%
    max_lines: int = 500  # 压缩结果最多保留 500 行（防反向膨胀）


class SavingsChecker:
    """验证压缩是否真正节省了空间。"""

    def __init__(self, threshold: Optional[SavingsThreshold] = None):
        self.threshold = threshold or SavingsThreshold()

    def check(self, original: str, compressed: str) -> bool:
        """检查压缩结果是否满足节省门槛。

        Returns:
            True: 接受压缩结果
            False: 放弃压缩，使用原文
        """
        if not compressed:
            logger.debug("[SavingsChecker] Compression rejected: empty result")
            return False

        orig_bytes = len(original.encode("utf-8"))
        comp_bytes = len(compressed.encode("utf-8"))
        orig_lines = original.count("\n") + 1
        comp_lines = compressed.count("\n") + 1

        # 防反向膨胀
        if comp_bytes >= orig_bytes:
            logger.debug("[SavingsChecker] Compression rejected: no byte savings (%dB -> %dB)", orig_bytes, comp_bytes)
            return False

        if comp_lines > self.threshold.max_lines:
            logger.debug(
                "[SavingsChecker] Compression rejected: result too long (%d lines > %d)",
                comp_lines,
                self.threshold.max_lines,
            )
            return False

        byte_savings = orig_bytes - comp_bytes
        line_savings = orig_lines - comp_lines
        ratio = byte_savings / orig_bytes if orig_bytes > 0 else 0

        if byte_savings < self.threshold.min_bytes:
            logger.debug(
                "[SavingsChecker] Compression rejected: byte savings %d < %d",
                byte_savings,
                self.threshold.min_bytes,
            )
            return False

        if line_savings < self.threshold.min_lines:
            logger.debug(
                "[SavingsChecker] Compression rejected: line savings %d < %d",
                line_savings,
                self.threshold.min_lines,
            )
            return False

        if ratio < self.threshold.min_ratio:
            logger.debug(
                "[SavingsChecker] Compression rejected: ratio %.2f%% < %.2f%%",
                ratio * 100,
                self.threshold.min_ratio * 100,
            )
            return False

        logger.info(
            "[SavingsChecker] ACCEPTED: %dB/%dL -> %dB/%dL "
            "(save %dB/%dL, ratio %.1f%%)",
            orig_bytes,
            orig_lines,
            comp_bytes,
            comp_lines,
            byte_savings,
            line_savings,
            ratio * 100,
        )
        return True

    def get_stats(self, original: str, compressed: str) -> dict:
        """获取压缩统计信息（用于日志和监控）。"""
        orig_bytes = len(original.encode("utf-8"))
        comp_bytes = len(compressed.encode("utf-8"))
        orig_lines = original.count("\n") + 1
        comp_lines = compressed.count("\n") + 1
        return {
            "orig_bytes": orig_bytes,
            "comp_bytes": comp_bytes,
            "orig_lines": orig_lines,
            "comp_lines": comp_lines,
            "byte_savings": orig_bytes - comp_bytes,
            "line_savings": orig_lines - comp_lines,
            "ratio": (orig_bytes - comp_bytes) / orig_bytes if orig_bytes > 0 else 0,
        }
