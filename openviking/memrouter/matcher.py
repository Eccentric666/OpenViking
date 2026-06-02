"""TemplateMatcher — multi-prototype vector matching with hard-negative penalty.

Implements the v1.3/v1.4 scoring algorithm:
    S_pos  = 0.50 * S_max + 0.30 * S_mean@3 + 0.20 * S_centroid
    S_final = S_pos - lambda * max(0, delta_neg - M_neg)

Backend aggregation runs for all enabled templates regardless of backend.
The algorithm is kept backend-agnostic so that multi-backend expansion
requires zero code changes here.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from openviking.memrouter.embeddings.base import EmbeddingProvider
from openviking.memrouter.features import QueryFeatures
from openviking.memrouter.normalizer import QueryNormalizer
from openviking.memrouter.templates import BackendRouteTemplateIndex, MemoryBackendRouteTemplate

logger = logging.getLogger(__name__)

# Score combination weights (from v1.3/v1.4 design doc)
_WEIGHT_MAX = 0.50
_WEIGHT_MEAN3 = 0.30
_WEIGHT_CENTROID = 0.20
_TOP_K_MEAN = 3


@dataclass
class TemplateCandidate:
    """Scoring result for a single template."""

    template_id: str
    primary_backend_id: str
    score: float
    score_components: Dict[str, float]


@dataclass
class BackendCandidate:
    """Aggregated candidate per backend (used in multi-backend scenarios)."""

    backend_id: str
    best_template_id: str
    score: float


class TemplateMatcher:
    """Match user queries against template prototypes using vector similarity."""

    def __init__(self, embedder: EmbeddingProvider, template_index: BackendRouteTemplateIndex) -> None:
        self._embedder = embedder
        self._template_index = template_index
        # Cache for prototype embeddings: {template_id: (prototype_embeddings, centroid)}
        self._proto_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # Cache for hard-negative embeddings: {template_id: negative_embeddings}
        self._neg_cache: Dict[str, np.ndarray] = {}
        self._normalizer = QueryNormalizer()
        logger.info(
            "TemplateMatcher initialized with %d templates",
            len(template_index.enabled_templates()),
        )

    def match(self, features: QueryFeatures) -> Tuple[List[TemplateCandidate], List[BackendCandidate]]:
        """Run full template matching pipeline.

        Args:
            features: Extracted query features including embedding.

        Returns:
            (template_candidates, backend_candidates)
            template_candidates: all enabled templates sorted by final score.
            backend_candidates: per-backend best scores, sorted descending.
        """
        query_vec = features.query_embedding
        normalized_query = features.normalized_query.lower()
        logger.debug("Starting template matching for query: %s", features.normalized_query)

        template_candidates: List[TemplateCandidate] = []
        for template in self._template_index.enabled_templates():
            score, components = self._score_template(template, query_vec)
            # Apply lightweight lexical boost based on query keywords
            boosted_score = self._apply_lexical_boost(normalized_query, template, score)
            if boosted_score != score:
                components["lexical_boost"] = round(boosted_score - score, 4)
                components["s_final"] = round(boosted_score, 4)
            template_candidates.append(
                TemplateCandidate(
                    template_id=template.template_id,
                    primary_backend_id=template.target.primary_backend_id,
                    score=boosted_score,
                    score_components=components,
                )
            )

        # Sort by score descending
        template_candidates.sort(key=lambda c: c.score, reverse=True)
        logger.debug(
            "Template ranking (top3): %s",
            [(c.template_id, round(c.score, 4)) for c in template_candidates[:3]],
        )

        # Aggregate to backend candidates
        backend_candidates = self._aggregate_backends(template_candidates)
        logger.debug(
            "Backend ranking: %s",
            [(c.backend_id, round(c.score, 4)) for c in backend_candidates],
        )

        return template_candidates, backend_candidates

    def _apply_lexical_boost(
        self, query: str, template: MemoryBackendRouteTemplate, score: float
    ) -> float:
        """Apply lightweight lexical boost to template score.

        Rules are conservative (+0.06 max) to avoid overriding embedding signal.
        Only boosts when keyword pattern strongly indicates a specific intent family.
        """
        boost = 0.0
        template_id = template.template_id

        # Streamlined: temporal facts (when, what date, which year)
        if template_id == "streamlined.timeline_fact.v1":
            if any(kw in query for kw in ("when did", "what date", "which year", "what year")):
                boost = 0.06
            elif query.startswith("when ") and "when did" not in query:
                boost = 0.04
            # v1.5: boost queries containing explicit month/year/season anchors
            # even when the answer is a place/state/item (e.g. "What state did X visit in July 2023?")
            elif any(kw in query for kw in (
                "in january", "in february", "in march", "in april", "in may",
                "in june", "in july", "in august", "in september", "in october",
                "in november", "in december", "in summer", "in spring", "in fall",
                "in winter", "in 2021", "in 2022", "in 2023", "in 2024",
                "during january", "during february", "during march", "during april",
                "during may", "during june", "during july", "during august",
                "during september", "during october", "during november", "during december",
                "between august", "between september", "between october", "between november",
            )):
                boost = 0.05

        # Streamlined: duration comparison
        elif template_id == "streamlined.duration_comparison.v1":
            if any(kw in query for kw in ("how long", "how many days", "how many months",
                                            "how many years", "how many weeks", "how much time",
                                            "passed between", "duration", "how many days ago",
                                            "how many months ago", "how many years ago",
                                            "how many weeks ago")):
                boost = 0.06

        # Streamlined: sequence reasoning
        elif template_id == "streamlined.sequence_reasoning.v1":
            if any(kw in query for kw in ("which happened first", "what happened first",
                                            "before or after", "earlier or later")):
                boost = 0.06
            elif any(kw in query for kw in (" before ", " after ", " earlier ", " later ")):
                # Only boost if no temporal anchor — avoid boosting "what did X do before Y"
                if not any(kw in query for kw in ("when did", "what date", "which year")):
                    boost = 0.04

        # OpenViking: subjective reasoning
        elif template_id == "openviking.subjective_reasoning.v1":
            if any(kw in query for kw in ("would ", "why did", "why does", "how does",
                                            "how did", "how do", "how would", "what would",
                                            "feel about", "think about", "why ")):
                boost = 0.06
            elif query.startswith("would "):
                boost = 0.05

        # Graph: entity relation — only if non-temporal
        elif template_id == "graph.entity_relation.v1":
            if any(kw in query for kw in ("relationship", "both ", " share ", " common ",
                                            " together", " collaborated", " connected to",
                                            "relation between")):
                # Do NOT boost if query contains temporal keywords
                if not any(kw in query for kw in ("when", "how long", "before", "after",
                                                    "first", "last time")):
                    boost = 0.06
            elif any(kw in query for kw in ("with ", "and ", "between ")):
                # Weak signal — smaller boost, still exclude temporal
                if not any(kw in query for kw in ("when", "how long", "before", "after")):
                    boost = 0.03
            # v1.5: stronger boost for multi-entity preference/subjective comparison
            # that involves "both X and Y" or "with his family"
            elif any(kw in query for kw in ("both person", "with their family", "with his family",
                                             "with her family", "for both", "between person")):
                boost = 0.05
        # OpenViking: count/list fact
        elif template_id == "openviking.count_list_fact.v1":
            if any(kw in query for kw in ("how many", "what types", "what kinds",
                                            "which books", "which bands", "which items")):
                boost = 0.05

        return score + boost

    def _score_template(
        self,
        template: MemoryBackendRouteTemplate,
        query_vec: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute final score for a single template.

        Returns:
            (final_score, component_dict)
        """
        # --- Positive prototype scoring ---
        proto_embeddings, centroid = self._get_prototype_embeddings(template)
        s_max = float(np.max(proto_embeddings @ query_vec))
        # Top-K mean (guard against templates with fewer than _TOP_K_MEAN prototypes)
        sims = proto_embeddings @ query_vec
        k = min(_TOP_K_MEAN, len(sims))
        if k > 0:
            top_k_indices = np.argpartition(sims, -k)[-k:]
            s_mean_k = float(np.mean(sims[top_k_indices]))
        else:
            s_mean_k = 0.0
        s_centroid = float(centroid @ query_vec)

        s_pos = _WEIGHT_MAX * s_max + _WEIGHT_MEAN3 * s_mean_k + _WEIGHT_CENTROID * s_centroid

        # --- Hard-negative penalty ---
        neg_embeddings = self._get_negative_embeddings(template)
        if len(neg_embeddings) > 0:
            s_neg = float(np.max(neg_embeddings @ query_vec))
            m_neg = s_pos - s_neg
            delta_neg = template.thresholds.hard_negative_margin
            lambda_pen = template.thresholds.hard_negative_penalty
            penalty = lambda_pen * max(0.0, delta_neg - m_neg)
            s_final = s_pos - penalty
        else:
            s_neg = 0.0
            penalty = 0.0
            s_final = s_pos

        components = {
            "s_max": round(s_max, 4),
            "s_mean@3": round(s_mean_k, 4),
            "s_centroid": round(s_centroid, 4),
            "s_pos": round(s_pos, 4),
            "s_neg": round(s_neg, 4),
            "penalty": round(penalty, 4),
            "s_final": round(s_final, 4),
        }
        return s_final, components

    def _get_prototype_embeddings(
        self, template: MemoryBackendRouteTemplate
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return cached prototype embeddings and centroid for a template."""
        if template.template_id in self._proto_cache:
            return self._proto_cache[template.template_id]

        texts = [self._normalizer.normalize(text) for text in template.query_prototypes]
        if not texts:
            # No prototypes: create zero embeddings so the template scores 0
            dim = self._embedder.dimension()
            embeddings = np.zeros((1, dim), dtype=np.float32)
            centroid = np.zeros(dim, dtype=np.float32)
            self._proto_cache[template.template_id] = (embeddings, centroid)
            return embeddings, centroid

        embeddings = self._embedder.embed(texts)
        # Normalize each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        # Centroid = normalized mean of all prototypes
        centroid = np.mean(embeddings, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm
        else:
            centroid = centroid  # zeros

        self._proto_cache[template.template_id] = (embeddings, centroid)
        return embeddings, centroid

    def _get_negative_embeddings(self, template: MemoryBackendRouteTemplate) -> np.ndarray:
        """Return cached hard-negative embeddings for a template."""
        if template.template_id in self._neg_cache:
            return self._neg_cache[template.template_id]

        texts = [self._normalizer.normalize(hn.query) for hn in template.hard_negatives]
        if not texts:
            self._neg_cache[template.template_id] = np.zeros((0, self._embedder.dimension()), dtype=np.float32)
            return self._neg_cache[template.template_id]

        embeddings = self._embedder.embed(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        self._neg_cache[template.template_id] = embeddings
        return embeddings

    @staticmethod
    def _aggregate_backends(
        template_candidates: List[TemplateCandidate],
    ) -> List[BackendCandidate]:
        """Aggregate template candidates into per-backend best scores.

        If multiple templates map to the same backend, only the highest-scoring
        template contributes to that backend's candidate. This ensures that
        top1/top2 in RouteDecision represent distinct backends.
        """
        best_by_backend: Dict[str, TemplateCandidate] = {}
        for cand in template_candidates:
            bid = cand.primary_backend_id
            if bid not in best_by_backend or cand.score > best_by_backend[bid].score:
                best_by_backend[bid] = cand

        backend_cands = [
            BackendCandidate(
                backend_id=cand.primary_backend_id,
                best_template_id=cand.template_id,
                score=cand.score,
            )
            for cand in best_by_backend.values()
        ]
        backend_cands.sort(key=lambda c: c.score, reverse=True)
        return backend_cands
