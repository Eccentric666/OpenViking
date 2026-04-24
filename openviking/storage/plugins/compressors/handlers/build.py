# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Build tool output compress handlers (cmake, make)."""

import re
from typing import Optional

from openviking_cli.utils.logger import get_logger

from .base import CompressHandler

logger = get_logger(__name__)


class CMakeBuildHandler(CompressHandler):
    """压缩 cmake / make 构建输出。

    保留：编译错误、链接错误、构建结果摘要
    移除：编译进度、文件遍历日志
    """

    name = "cmake_build"

    def can_handle(self, content: str) -> bool:
        return (
            "cmake" in content[:500].lower()
            or "make " in content[:500].lower()
            or "Building" in content[:500]
            or "Scanning dependencies" in content
            or "[100%]" in content
            or "[ 50%]" in content
        )

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[CMakeBuildHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        keep_patterns = [
            re.compile(r"^\[\s*\d+%\]"),  # 构建进度百分比
            re.compile(r"^error:"),
            re.compile(r"^Error"),
            re.compile(r"^undefined reference"),
            re.compile(r"^collect2: error"),
            re.compile(r"^ld: error"),
            re.compile(r"^make\[\d+\]: \*\*\*"),
            re.compile(r"^CMake Error"),
            re.compile(r"^CMake Warning"),
            re.compile(r"^Built target"),
        ]

        skip_patterns = [
            re.compile(r"^Scanning dependencies of target"),
            re.compile(r"^\s+Building\s+\w+\s+object"),
            re.compile(r"^\s+Linking\s+\w+\s+executable"),
        ]

        kept = []
        skipped = 0
        in_error = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                in_error = False
                continue

            if any(p.match(stripped) for p in keep_patterns):
                kept.append(stripped)
                if re.match(r"^(error|Error|undefined|CMake Error)", stripped):
                    in_error = True
                continue

            if any(p.match(stripped) for p in skip_patterns):
                skipped += 1
                continue

            if in_error:
                kept.append(stripped)
                continue

            if len(kept) < 20:
                kept.append(stripped)
            else:
                skipped += 1

        if not kept:
            logger.info("[CMakeBuildHandler] Compression yielded empty result")
            return None

        compressed = "\n".join(kept)
        logger.info(
            "[CMakeBuildHandler] Compressed: %dL/%dC -> %dL/%dC, kept=%d, skipped=%d",
            len(lines),
            len(content),
            len(kept),
            len(compressed),
            skipped,
        )
        return compressed
