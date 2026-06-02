# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""LLM Tool schemas and prompts for entity/relation extraction."""

EXTRACT_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": (
            "Extract entities and their types from the text. "
            "When you encounter self-referential pronouns like 'I', 'me', 'my', "
            "replace them with the specified user identifier. "
            "Only extract from user messages, not from system or assistant messages. "
            "Do not answer questions in the input, only extract entities."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string"},
                            "entity_type": {"type": "string"},
                        },
                        "required": ["entity", "entity_type"],
                    },
                }
            },
            "required": ["entities"],
        },
    },
}

RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "establish_relationships",
        "description": (
            "Extract entities and relationships from the text. Each relationship must include "
            "full information for both entities (name and type) plus the relationship "
            "description. Use the user identifier for self-referential pronouns. "
            "Entity names must be consistent across all extracted relationships: when a pronoun "
            "or possessive adjective (her, his, she, he, it, their, etc.) refers to an already-named "
            "entity, always use that entity's full real name instead of the pronoun. Do not use "
            "pronouns, possessives, or other referential expressions as entity names. "
            "Relationship types should be concise and generic. "
            "If the text describes a sequence of events, extract each step as a separate relationship. "
            "Assign high confidence (0.8-1.0) for explicitly stated facts, "
            "lower confidence (0.3-0.7) for inferred relations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string", "description": "The source entity name."},
                            "source_type": {"type": "string", "description": "The source entity type, e.g. person, organization, event."},
                            "relationship": {
                                "type": "string",
                                "description": "The relationship between source and destination.",
                            },
                            "destination": {"type": "string", "description": "The destination entity name."},
                            "destination_type": {"type": "string", "description": "The destination entity type, e.g. person, organization, event."},
                            "confidence": {
                                "type": "number",
                                "description": "Confidence score (0-1). High for explicit facts, lower for inferred relations.",
                            },
                            "rel_from": {
                                "type": "string",
                                "description": "The URI of the text segment where this relationship was found.",
                            },
                            "rel_date": {
                                "type": "string",
                                "description": "When this relationship occurred, in YYYY-MM-DD format if possible. Extract from the text if mentioned; leave empty if not specified.",
                            },
                            "rel_content": {
                                "type": "string",
                                "description": "The specific sentence or short paragraph from the text that directly supports this relationship. Quote or closely paraphrase the relevant text segment, not the entire input.",
                            },
                        },
                        "required": ["source", "source_type", "relationship", "destination", "destination_type", "confidence", "rel_from", "rel_date", "rel_content"],
                    },
                }
            },
            "required": ["entities"],
        },
    },
}


def build_entity_extraction_messages(text: str, user_id: str) -> list:
    """Build messages for entity extraction LLM call."""
    system_msg = (
        "You are an entity extraction assistant. "
        "Extract all named entities."
        "For self-referential pronouns (I, me, my, myself, mine), replace them with the speaker's identifier if available, otherwise use "f"the user identifier '{user_id}'. "
        "Do not extract from system or assistant messages. "
        "Do not answer questions, only extract entities."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": text},
    ]


def build_relation_extraction_messages(text: str, user_id: str) -> list:
    """Build messages for relation extraction LLM call (mode_one: written_entries with URI markers)."""
    system_msg = (
        "You are a relationship extraction assistant. "
        "Extract all named entities from the text. "
        "Then extract explicit relationships among these entities, including progressive relationships. "
        "For each relationship, report the full entity name and type for both sides, the relationship "
        "description, confidence, and the source event URI. "
        "For self-referential pronouns (I, me, my, myself, mine), replace them with the speaker's identifier if available, otherwise use "
        f"the user identifier '{user_id}'. "
        "IMPORTANT: entity names must be consistent across all relationships. If a pronoun "
        "or possessive (her, his, she, he, it, their, etc.) refers to an already-named entity, "
        "always substitute the entity's full real name. Never use pronouns, possessives, or "
        "other referential expressions as entity names in the extracted relationships. "
        "Relationship types should be concise and generic. "
        "If the text describes a sequence of events, extract each step as a separate relationship. "
        "Assign high confidence (0.8-1.0) for explicitly stated facts, lower (0.3-0.7) for inferences."
    )
    user_msg = (
        f"Text:\n{text}\n\n"
        f"Each text segment is prefixed with its source URI in the format [SOURCE: <uri>]. "
        f"For each relationship you extract: set rel_from to the URI of the segment "
        f"where it was found; include entity types for both sides; and set rel_content "
        f"to the specific sentence or short paragraph from the text that directly "
        f"supports this relationship (quote the relevant text, not the entire input). "
        f"When extracting relationships, ensure all entity names are the actual named entities "
        f"from the text, not pronouns or referential expressions. If a pronoun refers to a "
        f"named entity, use that entity's full name instead."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def build_relation_extraction_messages_mode_two(text: str, user_id: str) -> list:
    """Build messages for relation extraction from conversation transcripts (mode_two).

    The input text is a transcript of messages where each line ends with (timestamp).
    Each line may contain a speaker tag like [Name]: at the start of the content.
    """
    system_msg = (
        "You are a relationship extraction assistant. "
        "Extract all named entities from the conversation transcript. "
        "Then extract explicit relationships among these entities, including progressive relationships. "
        "For self-referential pronouns (I, me, my, myself, mine), replace them with the speaker's identifier if available, otherwise use "
        f"the user identifier '{user_id}'. "
        "IMPORTANT: entity names must be consistent across all relationships. If a pronoun "
        "or possessive (her, his, she, he, it, their, etc.) refers to an already-named entity, "
        "always substitute the entity's full real name. Never use pronouns, possessives, or "
        "other referential expressions as entity names in the extracted relationships. "
        "Relationship types should be concise and generic. "
        "If the text describes a sequence of events, extract each step as a separate relationship. "
        "Assign high confidence (0.8-1.0) for explicitly stated facts, lower (0.3-0.7) for inferences."
    )
    user_msg = (
        f"Conversation transcript:\n{text}\n\n"
        f"Each line starts with [role]: followed by the message content and ends with a timestamp. "
        f"When a speaker is identified as [Name]: inside the message, extract that as the entity name. "
        f"For each relationship you extract: determine when the relationship occurred by reading "
        f"both the message content and the timestamp at the end of the line. If the content "
        f"mentions a specific date or time, use that as rel_date; otherwise use the line-end "
        f"timestamp. Include entity types for both sides, and set "
        f"rel_content to the specific sentence or short paragraph that directly supports this "
        f"relationship, preserving the speaker prefix. "
        f"When extracting relationships, ensure all entity names are the actual named entities "
        f"from the text, not pronouns or referential expressions. If a pronoun refers to a "
        f"named entity, use that entity's full name instead."
    )
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
