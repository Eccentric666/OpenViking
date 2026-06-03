# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""MemRouter configuration for embedded routing in OpenViking."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemRouterEmbeddingConfig(BaseModel):
    """Embedding provider config for MemRouter."""

    provider: str = Field(default="openai", description="Embedding provider")
    model: str = Field(default="text-embedding-v3", description="Embedding model name")
    api_key: str = Field(default="", description="API key")
    api_base: str = Field(default="", description="API base URL")


class MemRouterLLMConfig(BaseModel):
    """LLM fallback config for MemRouter."""

    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="deepseek-v4-flash", description="LLM model name")
    api_key: str = Field(default="", description="API key")
    base_url: str = Field(default="", description="API base URL")


class MemRouterConfig(BaseModel):
    """MemRouter embedded routing configuration."""

    enabled: bool = Field(default=False, description="Enable MemRouter embedded routing")
    echomem_path: str = Field(
        default="",
        description="Path to EchoMem repository (for importing echomem package)",
    )
    template_dir: str = Field(
        default="",
        description="Directory containing route template YAML files",
    )
    embedding: MemRouterEmbeddingConfig = Field(
        default_factory=MemRouterEmbeddingConfig,
        description="Embedding provider config",
    )
    llm_fallback: MemRouterLLMConfig = Field(
        default_factory=MemRouterLLMConfig,
        description="LLM fallback config",
    )
    enabled_backends: List[str] = Field(
        default_factory=lambda: ["openviking_memory_backend"],
        description="Enabled backend IDs for routing",
    )
    route_events_path: str = Field(
        default="",
        description="Path to route events JSONL file for observability",
    )
