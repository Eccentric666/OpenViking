# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Base class for tool output compress handlers."""

from abc import ABC, abstractmethod
from typing import Optional


class CompressHandler(ABC):
    """工具输出压缩 Handler 基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Handler 唯一标识名称。"""
        ...

    @abstractmethod
    def can_handle(self, content: str) -> bool:
        """判断内容是否为本工具的输出。"""
        ...

    @abstractmethod
    def compress(self, content: str) -> Optional[str]:
        """执行压缩，返回压缩后的内容。"""
        ...
