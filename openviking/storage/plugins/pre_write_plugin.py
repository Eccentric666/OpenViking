# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""PreWritePlugin abstract base class."""

from abc import ABC, abstractmethod

from openviking.server.identity import RequestContext


class PreWritePlugin(ABC):
    """写入 VikingFS 前的内容预处理插件接口。

    插件在 content 序列化之后、VikingFS.write_file 之前执行。
    对于 memory 文件，插件处理的是分离 metadata 后的纯 content。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """插件唯一标识名称。"""
        ...

    @property
    def priority(self) -> int:
        """执行优先级，数字越小越早执行。"""
        return 100

    @abstractmethod
    async def process(self, uri: str, content: str, ctx: RequestContext) -> str:
        """处理内容，返回处理后的字符串。

        Args:
            uri: 目标文件 URI
            content: 纯内容（memory 文件已去除 metadata）
            ctx: 请求上下文

        Returns:
            处理后的 content。如果无需处理，返回原字符串。
        """
        ...
