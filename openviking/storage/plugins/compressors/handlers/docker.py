# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Docker command output compress handler."""

import re
from typing import Optional

from openviking_cli.utils.logger import get_logger

from .base import CompressHandler

logger = get_logger(__name__)


class DockerBuildHandler(CompressHandler):
    """压缩 docker build 输出。

    保留：每层构建结果、最终镜像 ID、错误信息
    移除：逐层下载进度、apt 安装日志、构建上下文发送进度
    """

    name = "docker_build"

    def can_handle(self, content: str) -> bool:
        return (
            "docker build" in content[:500]
            or "docker compose" in content[:500]
            or content.strip().startswith("Sending build context")
            or content.strip().startswith("Step ")
            or "dockerfile" in content[:200].lower()
        )

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[DockerBuildHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        keep_patterns = [
            re.compile(r"^Step \d+/\d+"),
            re.compile(r"^Successfully built"),
            re.compile(r"^Successfully tagged"),
            re.compile(r"^error"),
            re.compile(r"^Error"),
            re.compile(r"^--->"),
            re.compile(r"^Removing intermediate container"),
            re.compile(r"^The command .* returned a non-zero code"),
        ]

        skip_patterns = [
            re.compile(r"^\s*Get:\d+"),  # apt 安装
            re.compile(r"^\s*Unpacking"),
            re.compile(r"^\s*Setting up"),
            re.compile(r"^\s*Processing triggers"),
            re.compile(r"^\s*\d+MB/\d+MB"),  # 下载进度
            re.compile(r"^\s*Downloading"),
            re.compile(r"^\s*Extracting"),
            re.compile(r"^Sending build context"),
        ]

        kept = []
        skipped = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if any(p.match(stripped) for p in keep_patterns):
                kept.append(stripped)
                continue

            if any(p.match(stripped) for p in skip_patterns):
                skipped += 1
                continue

            # 保留错误行
            if "error" in stripped.lower() or "failed" in stripped.lower():
                kept.append(stripped)
                continue

            # 保留少量其他行
            if len(kept) < 30:
                kept.append(stripped)
            else:
                skipped += 1

        if not kept:
            logger.info("[DockerBuildHandler] Compression yielded empty result")
            return None

        compressed = "\n".join(kept)
        logger.info(
            "[DockerBuildHandler] Compressed: %dL/%dC -> %dL/%dC, kept=%d, skipped=%d",
            len(lines),
            len(content),
            len(kept),
            len(compressed),
            skipped,
        )
        return compressed
