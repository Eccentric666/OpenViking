# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""NPM command output compress handler."""

import re
from typing import Optional

from openviking_cli.utils.logger import get_logger

from .base import CompressHandler

logger = get_logger(__name__)


class NpmInstallHandler(CompressHandler):
    """压缩 npm install / npm ci 输出。

    保留：安装的包数量、最终的 audit/security 信息、错误信息
    移除：逐行下载进度、解压进度、gyp 编译日志
    """

    name = "npm_install"

    def can_handle(self, content: str) -> bool:
        return (
            "npm install" in content[:500]
            or "npm ci" in content[:500]
            or ("added " in content and "packages" in content)
            or ("removed " in content and "packages" in content)
        )

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[NpmInstallHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        keep_patterns = [
            re.compile(r"^added\s+\d+\s+packages?"),
            re.compile(r"^removed\s+\d+\s+packages?"),
            re.compile(r"^updated\s+\d+\s+packages?"),
            re.compile(r"^audited\s+\d+\s+packages?"),
            re.compile(r"^found\s+\d+\s+vulnerabilities?"),
            re.compile(r"^\d+\s+(high|critical|moderate|low)\s+severity"),
            re.compile(r"npm ERR!"),
            re.compile(r"npm WARN.*deprecated"),
        ]

        skip_patterns = [
            re.compile(r"^\[.*\]\s+\|\s+.*\[.*\]"),  # 进度条
            re.compile(r"^\s*[-|/\\]\s+.*fetch"),  # fetch 进度
            re.compile(r"^gyp\s+"),  # gyp 编译
            re.compile(r"^\s*>\s+.*install"),  # postinstall
        ]

        kept = []
        skipped = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                skipped += 1
                continue

            # 优先保留
            if any(p.match(stripped) for p in keep_patterns):
                kept.append(stripped)
                continue

            # 明确丢弃
            if any(p.match(stripped) for p in skip_patterns):
                skipped += 1
                continue

            # 保留错误行
            if "ERR!" in stripped or "error" in stripped.lower():
                kept.append(stripped)
                continue

            # 保留警告行（去重）
            if "WARN" in stripped or "warning" in stripped.lower():
                if stripped not in kept:
                    kept.append(stripped)
                continue

            # 保留少量其他行（限制数量防止膨胀）
            if len(kept) < 20:
                kept.append(stripped)
            else:
                skipped += 1

        if not kept:
            logger.info("[NpmInstallHandler] Compression yielded empty result")
            return None

        compressed = "\n".join(kept)
        logger.info(
            "[NpmInstallHandler] Compressed: %dL/%dC -> %dL/%dC, kept=%d, skipped=%d",
            len(lines),
            len(content),
            len(kept),
            len(compressed),
            skipped,
        )
        return compressed
