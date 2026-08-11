"""Deterministic lexical and semantic retrieval fusion over exact evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Protocol

from corpusdock.contracts import EvidenceResult
from corpusdock.retrieval import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    MatchMode,
    RetrievalError,
    SearchBackend,
    SearchResponse,
    VerificationReport,
)


DEFAULT_FUSION_CANDIDATES = 60
DEFAULT_RRF_K = 60
DEFAULT_LEXICAL_WEIGHT = 1.0
DEFAULT_SEMANTIC_WEIGHT = 1.0


class SemanticSearchBackend(SearchBackend, Protocol):
    """Semantic search plus the non-content provenance required by evaluation."""

    def evaluation_metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Validated deterministic reciprocal-rank-fusion settings."""

    candidate_limit: int = DEFAULT_FUSION_CANDIDATES
    rrf_k: int = DEFAULT_RRF_K
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT

    def __post_init__(self) -> None:
        if (
            not isinstance(self.candidate_limit, int)
            or isinstance(self.candidate_limit, bool)
            or not 1 <= self.candidate_limit <= MAX_SEARCH_LIMIT
        ):
            raise RetrievalError(
                "hybrid_candidate_limit_invalid",
                f"Hybrid candidate limit must be between 1 and {MAX_SEARCH_LIMIT}.",
            )
        if (
            not isinstance(self.rrf_k, int)
            or isinstance(self.rrf_k, bool)
            or self.rrf_k < 1
            or self.rrf_k > 1_000_000
        ):
            raise RetrievalError(
                "hybrid_rrf_k_invalid",
                "Hybrid reciprocal-rank constant must be between 1 and 1000000.",
            )
        for label, weight in (
            ("lexical", self.lexical_weight),
            ("semantic", self.semantic_weight),
        ):
            if (
                not isinstance(weight, (int, float))
                or isinstance(weight, bool)
                or not math.isfinite(float(weight))
                or float(weight) <= 0
                or float(weight) > 1_000_000
            ):
                raise RetrievalError(
                    "hybrid_weight_invalid",
                    f"Hybrid {label} weight must be finite and between 0 and 1000000.",
                )

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "algorithm": "reciprocal_rank_fusion",
            "candidate_limit": self.candidate_limit,
            "rrf_k": self.rrf_k,
            "lexical_weight": float(self.lexical_weight),
            "semantic_weight": float(self.semantic_weight),
            "tie_break": "lexical_rank,semantic_rank,evidence_id",
        }


@dataclass(frozen=True, slots=True)
class _Candidate:
    evidence: EvidenceResult
    lexical_rank: int | None = None
    semantic_rank: int | None = None

    @property
    def channels(self) -> int:
        return int(self.lexical_rank is not None) + int(self.semantic_rank is not None)


class HybridSearchBackend:
    """Fuse exact lexical and persistent dense rankings without changing evidence."""

    def __init__(
        self,
        lexical_backend: SearchBackend,
        semantic_backend: SemanticSearchBackend,
        *,
        config: FusionConfig | None = None,
    ) -> None:
        self._lexical_backend = lexical_backend
        self._semantic_backend = semantic_backend
        self.config = config or FusionConfig()

    def evaluation_metadata(self) -> dict[str, Any]:
        metadata = self._semantic_backend.evaluation_metadata()
        return {**metadata, "fusion": self.config.to_dict()}

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        source_id: str | None = None,
        match_mode: MatchMode = "all",
    ) -> SearchResponse:
        _validate_search_limit(limit)
        candidate_limit = max(limit, self.config.candidate_limit)
        lexical = self._lexical_backend.search(
            query,
            limit=candidate_limit,
            source_id=source_id,
            match_mode=match_mode,
        )
        semantic = self._semantic_backend.search(
            query,
            limit=candidate_limit,
            source_id=source_id,
            match_mode=match_mode,
        )
        _validate_response_pair(lexical, semantic)

        candidates: dict[str, _Candidate] = {}
        _add_channel_results(candidates, lexical.results, channel="lexical")
        _add_channel_results(candidates, semantic.results, channel="semantic")
        ranked = sorted(
            candidates.values(),
            key=lambda candidate: _ranking_key(candidate, self.config),
        )[:limit]
        results = tuple(
            replace(
                candidate.evidence,
                score=round(_fusion_score(candidate, self.config), 12),
            )
            for candidate in ranked
        )
        return SearchResponse(
            query=query,
            match_mode=match_mode,
            results=results,
            index_built_at=lexical.index_built_at,
            indexed_sources=lexical.indexed_sources,
            indexed_chunks=lexical.indexed_chunks,
            partial_sources=lexical.partial_sources,
        )

    def verify(self, evidence_id: str) -> VerificationReport:
        return self._lexical_backend.verify(evidence_id)


def _validate_search_limit(limit: int) -> None:
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_SEARCH_LIMIT
    ):
        raise RetrievalError(
            "search_limit_invalid",
            f"Search limit must be between 1 and {MAX_SEARCH_LIMIT}.",
        )


def _validate_response_pair(
    lexical: SearchResponse,
    semantic: SearchResponse,
) -> None:
    lexical_signature = (
        lexical.query,
        lexical.match_mode,
        lexical.index_built_at,
        lexical.indexed_sources,
        lexical.indexed_chunks,
        lexical.partial_sources,
    )
    semantic_signature = (
        semantic.query,
        semantic.match_mode,
        semantic.index_built_at,
        semantic.indexed_sources,
        semantic.indexed_chunks,
        semantic.partial_sources,
    )
    if lexical_signature != semantic_signature:
        raise RetrievalError(
            "hybrid_index_mismatch",
            "Lexical and semantic candidates were not produced from the same exact index.",
        )


def _add_channel_results(
    candidates: dict[str, _Candidate],
    results: tuple[EvidenceResult, ...],
    *,
    channel: str,
) -> None:
    seen: set[str] = set()
    for rank, evidence in enumerate(results, start=1):
        if evidence.evidence_id in seen:
            raise RetrievalError(
                "hybrid_candidate_duplicate",
                f"The {channel} backend returned duplicate evidence.",
            )
        seen.add(evidence.evidence_id)
        existing = candidates.get(evidence.evidence_id)
        if existing is not None:
            _validate_evidence_pair(existing.evidence, evidence)
            base = existing
        else:
            base = _Candidate(evidence=replace(evidence, score=None))
        if channel == "lexical":
            candidates[evidence.evidence_id] = replace(base, lexical_rank=rank)
        else:
            candidates[evidence.evidence_id] = replace(base, semantic_rank=rank)


def _validate_evidence_pair(first: EvidenceResult, second: EvidenceResult) -> None:
    if replace(first, score=None) != replace(second, score=None):
        raise RetrievalError(
            "hybrid_evidence_mismatch",
            "Lexical and semantic backends returned conflicting exact evidence.",
        )


def _fusion_score(candidate: _Candidate, config: FusionConfig) -> float:
    score = 0.0
    if candidate.lexical_rank is not None:
        score += float(config.lexical_weight) / (config.rrf_k + candidate.lexical_rank)
    if candidate.semantic_rank is not None:
        score += float(config.semantic_weight) / (
            config.rrf_k + candidate.semantic_rank
        )
    return score


def _ranking_key(
    candidate: _Candidate,
    config: FusionConfig,
) -> tuple[float, int, int, int, str]:
    missing_rank = MAX_SEARCH_LIMIT + 1
    return (
        -_fusion_score(candidate, config),
        -candidate.channels,
        candidate.lexical_rank or missing_rank,
        candidate.semantic_rank or missing_rank,
        candidate.evidence.evidence_id,
    )
