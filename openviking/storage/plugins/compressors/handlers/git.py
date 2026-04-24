# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Git command output compress handlers."""

import re
from typing import List, Optional

from openviking_cli.utils.logger import get_logger

from .base import CompressHandler

logger = get_logger(__name__)


class GitStatusHandler(CompressHandler):
    """压缩 git status 输出。

    保留：分支信息、分类统计、未跟踪文件列表
    移除：大量重复的状态前缀行（可通过统计重建）
    """

    name = "git_status"

    # 识别模式
    _PATTERNS = {
        "branch": re.compile(r"^(On branch|HEAD detached|Your branch is)\s+.+"),
        "modified": re.compile(r"^\s*modified:\s+(.+)"),
        "deleted": re.compile(r"^\s*deleted:\s+(.+)"),
        "added": re.compile(r"^\s*(new file|added):\s+(.+)"),
    }

    def can_handle(self, content: str) -> bool:
        lines = content.split("\n")
        if len(lines) < 3:
            return False
        first = lines[0].strip()
        return (
            first.startswith("On branch")
            or first.startswith("HEAD detached")
            or "Changes to be committed" in content
            or "Changes not staged" in content
        )

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[GitStatusHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        modified = []
        deleted = []
        added = []
        untracked = []
        branch_info = []
        other = []

        in_untracked = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 分支信息
            if self._PATTERNS["branch"].match(stripped):
                branch_info.append(stripped)
                continue

            # Untracked files 区域标记
            if "Untracked files:" in stripped:
                in_untracked = True
                continue
            if in_untracked and stripped.startswith('(use "git add'):
                continue
            if in_untracked and not stripped.startswith("\t") and not stripped.startswith("  "):
                in_untracked = False

            # 分类收集
            m = self._PATTERNS["modified"].match(stripped)
            if m:
                modified.append(m.group(1).strip())
                continue
            m = self._PATTERNS["deleted"].match(stripped)
            if m:
                deleted.append(m.group(1).strip())
                continue
            m = self._PATTERNS["added"].match(stripped)
            if m:
                added.append(m.group(2).strip() if m.group(2) else m.group(1).strip())
                continue

            if in_untracked:
                untracked.append(stripped.strip())
                continue

            other.append(stripped)

        # 生成压缩输出
        parts = []
        if branch_info:
            parts.append("\n".join(branch_info))

        stats = []
        if modified:
            stats.append(f"modified: {len(modified)} files")
        if deleted:
            stats.append(f"deleted: {len(deleted)} files")
        if added:
            stats.append(f"added: {len(added)} files")
        if untracked:
            stats.append(f"untracked: {len(untracked)} files")

        if stats:
            parts.append(" | ".join(stats))
            if modified:
                parts.append(
                    f"Modified: {', '.join(modified[:5])}"
                    + ("..." if len(modified) > 5 else "")
                )
            if untracked:
                parts.append(
                    f"Untracked: {', '.join(untracked[:5])}"
                    + ("..." if len(untracked) > 5 else "")
                )

        if other and len(other) < 5:
            parts.append("\n".join(other))

        compressed = "\n".join(parts)
        if compressed:
            comp_lines = compressed.count("\n") + 1
            logger.info(
                "[GitStatusHandler] Compressed: %dL/%dC -> %dL/%dC, "
                "stats={modified=%d, deleted=%d, added=%d, untracked=%d}",
                len(lines),
                len(content),
                comp_lines,
                len(compressed),
                len(modified),
                len(deleted),
                len(added),
                len(untracked),
            )
        else:
            logger.info("[GitStatusHandler] Compression yielded empty result")
        return compressed if compressed else None


class GitLogHandler(CompressHandler):
    """压缩 git log 输出。

    保留：提交哈希、作者、日期、提交信息的前几行
    移除：完整 diff、过长描述
    """

    name = "git_log"

    _COMMIT_PATTERN = re.compile(r"^commit\s+([a-f0-9]{40})" )

    def can_handle(self, content: str) -> bool:
        lines = content.split("\n")
        if len(lines) < 3:
            return False
        first = lines[0].strip()
        return first.startswith("commit ") and len(first) > 45

    def compress(self, content: str) -> Optional[str]:
        lines = content.split("\n")
        logger.info(
            "[GitLogHandler] Start compressing: %d lines, %d chars",
            len(lines),
            len(content),
        )

        commits = []
        current_commit: List[str] = []
        in_diff = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            if self._COMMIT_PATTERN.match(stripped):
                if current_commit:
                    commits.append(current_commit)
                current_commit = [stripped]
                in_diff = False
                continue

            if stripped.startswith("diff --git"):
                in_diff = True
                continue

            if not in_diff and len(current_commit) < 6:
                current_commit.append(stripped)

        if current_commit:
            commits.append(current_commit)

        if not commits:
            return None

        result_lines = []
        for commit in commits[:20]:  # 最多保留 20 个提交
            result_lines.extend(commit)
            result_lines.append("")

        compressed = "\n".join(result_lines).strip()
        logger.info(
            "[GitLogHandler] Compressed: %dL/%dC -> %dL/%dC, commits=%d",
            len(lines),
            len(content),
            compressed.count("\n") + 1,
            len(compressed),
            len(commits),
        )
        return compressed
