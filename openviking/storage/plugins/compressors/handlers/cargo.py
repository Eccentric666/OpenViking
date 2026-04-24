# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Cargo command output compress handler."""

import re
from typing import Optional

from openviking_cli.utils.logger import get_logger

from .base import CompressHandler

logger = get_logger(__name__)


class CargoBuildHandler(CompressHandler):
    """压缩 cargo build / cargo test 输出。

    保留：编译错误、测试结果摘要、警告
    移除：编译进度、依赖下载、通过测试的详细输出
    """

    name = "cargo_build"

    def can_handle(self, content: str) -> bool:
        return (
            "cargo build" in content[:500]
            or "cargo test" in content[:500]
            or "cargo check" in content[:500]
            or "Compiling" in content[:500]
            or "Finished" in content[:500]
            or "Running" in content[:500]
            or "test result:" in content
        )

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[CargoBuildHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        keep_patterns = [
            re.compile(r"^error\["),
            re.compile(r"^warning\["),
            re.compile(r"^test result:"),
            re.compile(r"^running \d+ tests?"),
            re.compile(r"^Finished"),
            re.compile(r"^error:"),
            re.compile(r"^Build FAILED"),
            re.compile(r"^FAILED"),
        ]

        skip_patterns = [
            re.compile(r"^\s*Compiling"),
            re.compile(r"^\s*Downloading"),
            re.compile(r"^\s*Downloaded"),
            re.compile(r"^\s*Updating"),
            re.compile(r"^\s*Blocking"),
            re.compile(r"^\s*Fresh"),
            re.compile(r"^\s*Documenting"),
        ]

        kept = []
        skipped = 0
        in_error_block = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                in_error_block = False
                continue

            # 保留模式
            if any(p.match(stripped) for p in keep_patterns):
                kept.append(stripped)
                in_error_block = True
                continue

            # 明确丢弃
            if any(p.match(stripped) for p in skip_patterns):
                skipped += 1
                continue

            # 保留编译错误上下文（error: 行后的几行）
            if in_error_block:
                kept.append(stripped)
                continue

            # 保留失败的测试名
            if "FAILED" in stripped and "test" in stripped.lower():
                kept.append(stripped)
                continue

            # 保留少量其他行
            if len(kept) < 20:
                kept.append(stripped)
            else:
                skipped += 1

        if not kept:
            logger.info("[CargoBuildHandler] Compression yielded empty result")
            return None

        compressed = "\n".join(kept)
        logger.info(
            "[CargoBuildHandler] Compressed: %dL/%dC -> %dL/%dC, kept=%d, skipped=%d",
            len(lines),
            len(content),
            len(kept),
            len(compressed),
            skipped,
        )
        return compressed
