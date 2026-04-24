# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tool output compressor configuration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config_utils import raise_unknown_config_fields


@dataclass
class CompressorThresholdConfig:
    """压缩节省门槛配置。"""

    min_bytes: int = 40
    min_lines: int = 10
    min_ratio: float = 0.1
    max_lines: int = 500

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressorThresholdConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        raise_unknown_config_fields(
            data=data, valid_fields=valid_fields, context_name=cls.__name__
        )
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class ToolCompressorConfig:
    """工具输出压缩配置。"""

    enabled: bool = True
    handlers: List[str] = field(
        default_factory=lambda: [
            "git_status",
            "git_log",
            "npm_install",
            "cargo_build",
            "docker_build",
            "pytest",
            "vitest",
            "tsc",
            "jest",
            "cmake_build",
        ]
    )
    min_content_length: int = 500
    threshold: CompressorThresholdConfig = field(
        default_factory=CompressorThresholdConfig
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCompressorConfig":
        data = data.copy()
        threshold_data = data.pop("threshold", {})
        threshold = CompressorThresholdConfig.from_dict(threshold_data)

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        raise_unknown_config_fields(
            data=data, valid_fields=valid_fields, context_name=cls.__name__
        )
        return cls(threshold=threshold, **data)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        result = asdict(self)
        return result
