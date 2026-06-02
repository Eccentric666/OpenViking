# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph database configuration."""

from typing import List, Optional

from pydantic import BaseModel, Field


class GraphDatabaseConfig(BaseModel):
    """Configuration for Neo4j graph database backend."""

    enabled: bool = Field(default=False, description="Enable graph database")
    uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt URI")
    username: str = Field(default="neo4j", description="Neo4j username")
    password: str = Field(default="", description="Neo4j password")
    database: str = Field(default="neo4j", description="Neo4j database name")
    confidence_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Relation confidence threshold"
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Vector similarity threshold for retrieval"
    )
