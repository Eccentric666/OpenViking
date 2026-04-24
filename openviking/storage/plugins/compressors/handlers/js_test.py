# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""JavaScript/TypeScript test and build output compress handlers."""

import re
from typing import Optional

from openviking_cli.utils.logger import get_logger

from .base import CompressHandler

logger = get_logger(__name__)


class PytestHandler(CompressHandler):
    """压缩 pytest 输出。

    保留：测试结果摘要、失败用例、错误堆栈
    移除：通过的测试详细输出、进度条
    """

    name = "pytest"

    def can_handle(self, content: str) -> bool:
        return (
            "pytest" in content[:500]
            or "test session starts" in content
            or "collected" in content and "items" in content
            or "PASSED" in content
            or "FAILED" in content
        )

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[PytestHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        keep_patterns = [
            re.compile(r"^=+\s+.*\s+=+"),  # 结果分隔线
            re.compile(r"^collected\s+\d+\s+items?"),
            re.compile(r"^\d+\s+passed"),
            re.compile(r"^\d+\s+failed"),
            re.compile(r"^\d+\s+error"),
            re.compile(r"^\d+\s+skipped"),
            re.compile(r"^FAILED\s+"),
            re.compile(r"^ERROR\s+"),
            re.compile(r"^\s+assert"),
            re.compile(r"^\s+def\s+test_"),
        ]

        skip_patterns = [
            re.compile(r"^\s*\d+%\|"),  # 进度条
            re.compile(r"^\[\s*\d+%\]"),
        ]

        kept = []
        skipped = 0
        in_failure = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                in_failure = False
                continue

            if any(p.match(stripped) for p in keep_patterns):
                kept.append(stripped)
                if "FAILED" in stripped or "ERROR" in stripped or "assert" in stripped:
                    in_failure = True
                continue

            if any(p.match(stripped) for p in skip_patterns):
                skipped += 1
                continue

            # 保留失败上下文
            if in_failure:
                kept.append(stripped)
                continue

            # 保留错误行
            if stripped.startswith("E ") or stripped.startswith(">"):
                kept.append(stripped)
                continue

            if len(kept) < 20:
                kept.append(stripped)
            else:
                skipped += 1

        if not kept:
            logger.info("[PytestHandler] Compression yielded empty result")
            return None

        compressed = "\n".join(kept)
        logger.info(
            "[PytestHandler] Compressed: %dL/%dC -> %dL/%dC, kept=%d, skipped=%d",
            len(lines),
            len(content),
            len(kept),
            len(compressed),
            skipped,
        )
        return compressed


class VitestHandler(CompressHandler):
    """压缩 vitest 输出。"""

    name = "vitest"

    def can_handle(self, content: str) -> bool:
        return (
            "vitest" in content[:500]
            or "Test Files" in content
            or "Tests" in content and "Duration" in content
            or "FAIL" in content and "PASS" in content
        )

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[VitestHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        keep_patterns = [
            re.compile(r"^\s*(PASS|FAIL)\s+"),
            re.compile(r"^\s*Test Files\s+"),
            re.compile(r"^\s*Tests\s+"),
            re.compile(r"^\s*Duration\s+"),
            re.compile(r"^\s*Expected:"),
            re.compile(r"^\s*Received:"),
            re.compile(r"^AssertionError"),
        ]

        skip_patterns = [
            re.compile(r"^\s*[⣿⣷⣦⣄\|/\\-]"),  # 进度动画字符
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
                if "FAIL" in stripped or "AssertionError" in stripped or "Expected:" in stripped:
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
            logger.info("[VitestHandler] Compression yielded empty result")
            return None

        compressed = "\n".join(kept)
        logger.info(
            "[VitestHandler] Compressed: %dL/%dC -> %dL/%dC, kept=%d, skipped=%d",
            len(lines),
            len(content),
            len(kept),
            len(compressed),
            skipped,
        )
        return compressed


class TscHandler(CompressHandler):
    """压缩 TypeScript compiler (tsc) 输出。"""

    name = "tsc"

    def can_handle(self, content: str) -> bool:
        return (
            "tsc " in content[:500]
            or "TS" in content and ".ts(" in content
            or "error TS" in content
            or content.strip().startswith("error TS")
        )

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[TscHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        kept = []
        skipped = 0
        in_error = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                in_error = False
                continue

            # 保留错误和警告
            if re.match(r"^.*\.ts\(\d+,\d+\):\s+(error|warning)\s+TS\d+", stripped):
                kept.append(stripped)
                in_error = True
                continue

            if re.match(r"^error TS\d+:", stripped):
                kept.append(stripped)
                in_error = True
                continue

            if stripped.startswith("Found") and ("error" in stripped or "warning" in stripped):
                kept.append(stripped)
                continue

            if in_error:
                kept.append(stripped)
                continue

            if len(kept) < 20:
                kept.append(stripped)
            else:
                skipped += 1

        if not kept:
            logger.info("[TscHandler] Compression yielded empty result")
            return None

        compressed = "\n".join(kept)
        logger.info(
            "[TscHandler] Compressed: %dL/%dC -> %dL/%dC, kept=%d, skipped=%d",
            len(lines),
            len(content),
            len(kept),
            len(compressed),
            skipped,
        )
        return compressed


class JestHandler(CompressHandler):
    """压缩 jest 输出。"""

    name = "jest"

    def can_handle(self, content: str) -> bool:
        return (
            "jest" in content[:500]
            or "PASS" in content and "FAIL" in content
            or "Test Suites:" in content
            or "Tests:" in content
            or "Snapshot" in content and "test" in content.lower()
        )

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[JestHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        keep_patterns = [
            re.compile(r"^\s*(PASS|FAIL)\s+"),
            re.compile(r"^\s*Test Suites:"),
            re.compile(r"^\s*Tests:"),
            re.compile(r"^\s*Snapshots:"),
            re.compile(r"^\s*Time:"),
            re.compile(r"^\s*Expected:"),
            re.compile(r"^\s*Received:"),
            re.compile(r"^\s*●\s+"),
        ]

        skip_patterns = [
            re.compile(r"^\s*[⣿⣷⣦⣄\|/\\-]"),
        ]

        kept = []
        skipped = 0
        in_failure = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                in_failure = False
                continue

            if any(p.match(stripped) for p in keep_patterns):
                kept.append(stripped)
                if "FAIL" in stripped or "●" in stripped:
                    in_failure = True
                continue

            if any(p.match(stripped) for p in skip_patterns):
                skipped += 1
                continue

            if in_failure:
                kept.append(stripped)
                continue

            if len(kept) < 20:
                kept.append(stripped)
            else:
                skipped += 1

        if not kept:
            logger.info("[JestHandler] Compression yielded empty result")
            return None

        compressed = "\n".join(kept)
        logger.info(
            "[JestHandler] Compressed: %dL/%dC -> %dL/%dC, kept=%d, skipped=%d",
            len(lines),
            len(content),
            len(kept),
            len(compressed),
            skipped,
        )
        return compressed
