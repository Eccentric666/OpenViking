# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Graph retriever: vector similarity-based subgraph retrieval."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from openviking_cli.utils.logger import get_logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from ..neo4j_backend import Neo4jBackend

logger = get_logger(__name__)


@dataclass
class GraphSearchResult:
    """Single graph search result."""

    source: str
    source_uri: str
    source_tag: str
    rel_desc: str
    target: str
    target_uri: str
    target_tag: str
    source_properties: Optional[Dict[str, Any]] = None
    target_properties: Optional[Dict[str, Any]] = None
    rel_from: str = ""
    rel_date: str = ""
    rel_content: str = ""
    score: float = 0.0
    history: Optional[List[Dict[str, Any]]] = None


class GraphRetriever:
    """Graph retriever with vector similarity node matching."""

    def __init__(
        self,
        backend: Neo4jBackend,
        embedder: Any,
        entity_extractor: Any,
        similarity_threshold: float = 0.7,
        top_k: int = 10,
    ):
        self._backend = backend
        self._embedder = embedder
        self._entity_extractor = entity_extractor
        self._similarity_threshold = similarity_threshold
        self._top_k = top_k

    async def search(
        self,
        query: str,
        account_id: str,
        user_id: str,
        top_k: int = 10,
    ) -> List[GraphSearchResult]:
        """Main retrieval entry.

        Flow:
        1. LLM entity extraction from query.
        2. If >= 2 entities: try multi-hop path search.
        3. Fallback: embedding + vector match + 1-hop relation expansion per entity.
        4. Merge multi-hop and fallback results, deduplicate, keyword rerank, and return top_k.
        """
        raw_entities = await self._entity_extractor.raw_extract(query, user_id)
        entity_names = [r.entity for r in raw_entities]
        logger.info(f"[GraphRetriever] Query='{query}' extracted entities: {entity_names}")
        if not raw_entities:
            return []

        all_results: List[GraphSearchResult] = []

        # Multi-hop: if >= 2 entities, try to find paths between them.
        if len(raw_entities) >= 2:
            multi_hop = await self._multi_hop_search(
                entity_names, account_id, user_id, top_k
            )
            if multi_hop:
                logger.info(
                    f"[GraphRetriever] Multi-hop found {len(multi_hop)} results"
                )
                all_results.extend(multi_hop)

        # Fallback: vector similarity + 1-hop per entity (always run to catch single-hop edges).
        for raw in raw_entities:
            normalized = raw.entity.lower().replace(" ", "_")
            try:
                entity_emb = self._embedder.embed(normalized).dense_vector
            except Exception as e:
                logger.warning(f"[GraphRetriever] Embedding failed for '{normalized}': {e}")
                continue

            cypher = self._build_search_cypher()
            params = {
                "entity_embedding": entity_emb,
                "account_id": account_id,
                "user_id": user_id,
                "similarity_threshold": self._similarity_threshold,
            }
            records = await self._backend.execute(cypher, params)
            for record in records:
                all_results.append(self._record_to_result(record))

        unique_results = self._deduplicate_results(all_results)
        return self._keyword_rerank(query, unique_results, top_k)

    # Common English stop words to filter from the query before BM25 scoring.
    # Removing these prevents high-IDF noise (e.g. "her", "did") from distorting ranks.
    _STOP_WORDS = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "about", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "now", "when", "what", "where", "why", "how", "who", "which",
        "did", "do", "does", "have", "has", "had", "is", "are", "was",
        "were", "be", "been", "being", "will", "would", "could", "should",
        "may", "might", "can", "shall", "her", "his", "he", "she", "him",
        "they", "them", "their", "we", "us", "our", "you", "your", "it",
        "its", "my", "me", "mine", "this", "that", "these", "those",
        "i", "am", "get", "got", "gets",
    })

    def _bm25_rerank(
        self,
        query: str,
        results: List[GraphSearchResult],
        top_k: int,
    ) -> List[GraphSearchResult]:
        """Rerank deduplicated results with BM25 (mem0-style)."""
        if not BM25Okapi or not results:
            return results[:top_k]

        # Build corpus: each result is a single document string.
        # Underscores are replaced with spaces so that compound tokens
        # (e.g. "lost_job") can match against space-separated queries.
        corpus: List[str] = []
        for r in results:
            doc = " ".join(
                filter(
                    None,
                    [r.source, r.rel_desc, r.target],
                )
            ).replace("_", " ")
            corpus.append(doc)

        # Tokenise corpus and query with Porter stemming so that
        # e.g. "start" matches "started" and "go" matches "going".
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            tokenized_corpus = [
                [stemmer.stem(w) for w in doc.lower().split()] for doc in corpus
            ]
            tokenized_query = [
                stemmer.stem(w)
                for w in query.lower().replace("_", " ").replace("?", " ").replace("!", " ").replace(".", " ").replace(",", " ").split()
                if w not in self._STOP_WORDS
            ]
        except ImportError:
            tokenized_corpus = [doc.lower().split() for doc in corpus]
            tokenized_query = [
                w
                for w in query.lower().replace("_", " ").replace("?", " ").replace("!", " ").replace(".", " ").replace(",", " ").split()
                if w not in self._STOP_WORDS
            ]

        try:
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(tokenized_query)
        except Exception as e:
            logger.debug(f"[GraphRetriever] BM25 reranking failed: {e}")
            return results[:top_k]

        # Sort by BM25 score descending and slice.
        scored: List[Tuple[float, GraphSearchResult]] = list(zip(scores, results))
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked: List[GraphSearchResult] = []
        for s, r in scored[:top_k]:
            r.score = round(s, 4)
            reranked.append(r)
        logger.info(
            f"[GraphRetriever] BM25 reranked {len(results)} → {len(reranked)} results"
        )
        return reranked

    def _keyword_rerank(
        self,
        query: str,
        results: List[GraphSearchResult],
        top_k: int,
    ) -> List[GraphSearchResult]:
        """Rerank by counting query keyword occurrences in rel_desc + rel_content."""
        if not results:
            return results[:top_k]

        raw_words = (
            query.lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace("?", " ")
            .replace("!", " ")
            .replace(".", " ")
            .replace(",", " ")
            .split()
        )
        keywords = {w for w in raw_words if w not in self._STOP_WORDS}
        if not keywords:
            return results[:top_k]

        def _count(text: str) -> int:
            if not text:
                return 0
            normalized = (
                text.lower()
                .replace("_", " ")
                .replace("-", " ")
                .replace("?", " ")
                .replace("!", " ")
                .replace(".", " ")
                .replace(",", " ")
            )
            tokens = normalized.split()
            return sum(1 for t in tokens if t in keywords)

        scored: List[Tuple[int, GraphSearchResult]] = []
        for r in results:
            score = (
                _count(r.source)
                + _count(r.rel_desc)
                + _count(r.target)
                + _count(r.rel_content)
            )
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        reranked: List[GraphSearchResult] = []
        for s, r in scored[:top_k]:
            r.score = float(s)
            reranked.append(r)
        logger.info(
            f"[GraphRetriever] Keyword reranked {len(results)} → {len(reranked)} results"
        )
        return reranked

    async def _search_by_entity(
        self,
        entity_name: str,
        account_id: str,
        user_id: str,
        top_k: int = 10,
    ) -> List[GraphSearchResult]:
        """Exact search by entity name (no embedding)."""
        cypher = """
        MATCH (n:Node {account_id: $account_id, user_id: $user_id})
        WHERE n.name = $entity_name
        CALL {
            WITH n
            MATCH (n)-[r:RELATION]-(m:Node)
            WHERE r.valid = true
              AND m.account_id = $account_id
              AND m.user_id = $user_id
            RETURN properties(n) AS source_node,
                   properties(r) AS rel_props,
                   properties(m) AS target_node
            UNION
            WITH n
            MATCH (n)<-[r:RELATION]-(m:Node)
            WHERE r.valid = true
              AND m.account_id = $account_id
              AND m.user_id = $user_id
            RETURN properties(m) AS source_node,
                   properties(r) AS rel_props,
                   properties(n) AS target_node
        }
        RETURN source_node, rel_props, target_node
        LIMIT $limit
        """
        params = {
            "entity_name": entity_name.lower().replace(" ", "_"),
            "account_id": account_id,
            "user_id": user_id,
            "limit": top_k,
        }
        records = await self._backend.execute(cypher, params)
        return [self._record_to_result(r) for r in records]

    async def _multi_hop_search(
        self,
        entity_names: List[str],
        account_id: str,
        user_id: str,
        top_k: int,
    ) -> List[GraphSearchResult]:
        """Find multi-hop paths between entities (up to 4 hops) via vector matching."""
        # 1. Vector match each query entity to DB nodes
        matched_names: set = set()
        for name in entity_names:
            normalized = name.lower().replace(" ", "_")
            try:
                entity_emb = self._embedder.embed(normalized).dense_vector
            except Exception as e:
                logger.warning(
                    f"[GraphRetriever] Embedding failed for '{normalized}': {e}"
                )
                continue
            records = await self._backend.execute(
                """
                MATCH (n:Node)
                WHERE n.embedding IS NOT NULL
                  AND n.account_id = $account_id
                  AND n.user_id = $user_id
                WITH n, vector.similarity.cosine(n.embedding, $embedding) AS similarity
                WHERE similarity >= $threshold
                RETURN n.name AS name
                ORDER BY similarity DESC
                LIMIT $limit
                """,
                {
                    "embedding": entity_emb,
                    "account_id": account_id,
                    "user_id": user_id,
                    "threshold": self._similarity_threshold,
                    "limit": 1,
                },
            )
            for r in records:
                matched_names.add(r["name"])

        if len(matched_names) < 2:
            return []

        # 2. Multi-hop between matched nodes
        cypher = self._build_multi_hop_cypher()
        params = {
            "entity_names": list(matched_names),
            "account_id": account_id,
            "user_id": user_id,
            "limit": top_k,
        }
        try:
            records = await self._backend.execute(cypher, params)
        except Exception as e:
            logger.warning(f"[GraphRetriever] Multi-hop search failed: {e}")
            return []
        return [self._record_to_result(r) for r in records]

    def _build_multi_hop_cypher(self) -> str:
        return """
        MATCH path = (a:Node)-[r:RELATION*1..4]-(b:Node)
        WHERE a.name IN $entity_names
          AND b.name IN $entity_names
          AND a <> b
          AND a.account_id = $account_id
          AND a.user_id = $user_id
          AND b.account_id = $account_id
          AND b.user_id = $user_id
          AND all(rel IN r WHERE rel.valid = true)
        UNWIND relationships(path) AS rel
        WITH DISTINCT rel, startNode(rel) AS src, endNode(rel) AS dst
        WHERE src.account_id = $account_id
          AND src.user_id = $user_id
          AND dst.account_id = $account_id
          AND dst.user_id = $user_id
        RETURN properties(src) AS source_node,
               properties(rel) AS rel_props,
               properties(dst) AS target_node
        LIMIT $limit
        """

    def _build_search_cypher(self) -> str:
        return """
        MATCH (n:Node)
        WHERE n.embedding IS NOT NULL
          AND n.account_id = $account_id
          AND n.user_id = $user_id
        WITH n, vector.similarity.cosine(n.embedding, $entity_embedding) AS similarity
        WHERE similarity >= $similarity_threshold
        WITH n, similarity
        ORDER BY similarity DESC

        CALL {
            WITH n, similarity
            MATCH (n)-[r:RELATION]-(m:Node)
            WHERE r.valid = true
              AND m.account_id = $account_id
              AND m.user_id = $user_id
            RETURN properties(n) AS source_node,
                   properties(r) AS rel_props,
                   properties(m) AS target_node,
                   similarity AS node_similarity
        }
        RETURN source_node, rel_props, target_node, node_similarity
        ORDER BY node_similarity DESC
        """

    def _deduplicate_results(
        self, results: List[GraphSearchResult]
    ) -> List[GraphSearchResult]:
        seen: set = set()
        unique: List[GraphSearchResult] = []
        for r in results:
            key = (r.source_uri, r.target_uri, r.rel_desc)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def _record_to_result(self, record: Dict[str, Any]) -> GraphSearchResult:
        import json

        rel_props = record.get("rel_props", {}) or {}
        source_node = record.get("source_node", {}) or {}
        target_node = record.get("target_node", {}) or {}
        source_ext = source_node.get("properties")
        target_ext = target_node.get("properties")

        def _parse_props(raw):
            if not raw:
                return None
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except Exception:
                    return {"content": raw}
            return raw

        return GraphSearchResult(
            source=source_node.get("name", ""),
            source_uri=source_node.get("source", ""),
            source_tag=source_node.get("tag", ""),
            source_properties=_parse_props(source_ext),
            rel_desc=rel_props.get("rel_desc", ""),
            target=target_node.get("name", ""),
            target_uri=target_node.get("source", ""),
            target_tag=target_node.get("tag", ""),
            target_properties=_parse_props(target_ext),
            rel_from=rel_props.get("rel_from", ""),
            rel_date=rel_props.get("rel_date", ""),
            rel_content=rel_props.get("rel_content", ""),
            history=rel_props.get("history"),
        )
