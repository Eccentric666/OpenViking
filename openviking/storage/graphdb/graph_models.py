# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph data models for Neo4j graph storage."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RawEntity:
    """LLM raw extracted entity (before post-processing)."""

    entity: str
    entity_type: str


@dataclass
class RawRelation:
    """LLM raw extracted relation (before post-processing)."""

    from_entity: str
    rel_desc: str
    to_entity: str
    confidence: float = 1.0


@dataclass
class RawRelationTriple:
    """LLM raw extracted triple with full entity info (no external entity list)."""

    from_entity: str
    from_entity_type: str
    to_entity: str
    to_entity_type: str
    rel_desc: str
    confidence: float = 1.0
    rel_from: str = ""                 # source URI of the text segment
    rel_date: str = ""
    rel_content: str = ""


@dataclass
class GraphEntity:
    """Graph entity aligned with Neo4j node properties."""

    name: str
    source: str = ""                   # data source, combined with name as node match key
    tag: str = ""                      # entity type tag, e.g. "person", "organization"
    properties: Optional[Dict[str, Any]] = None   # extended properties
    account_id: str = ""               # tenant isolation
    user_id: str = ""                  # user isolation
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class GraphRelation:
    """Neo4j relationship edge properties."""

    rel_desc: str = ""                 # relationship description
    rel_date: str = ""                 # when the relationship occurred
    rel_content: str = ""              # the specific text snippet that supports this relation
    rel_from: str = ""                 # source URI of the text segment
    valid: bool = True                 # true = active
    invalidated_at: Optional[datetime] = None
    history: Optional[List[Dict[str, Any]]] = None


@dataclass
class RelationTriple:
    """Relation triple (post-processing output)."""

    from_entity: str
    to_entity: str
    relation: GraphRelation


@dataclass
class ExtractionResult:
    """Complete two-stage extraction result."""

    entities: List[GraphEntity] = field(default_factory=list)
    relations: List[RelationTriple] = field(default_factory=list)
