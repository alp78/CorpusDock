from __future__ import annotations

from dataclasses import replace

import pytest

from corpusdock.contracts import CitationLocator, EvidenceResult
from corpusdock.hybrid import FusionConfig, HybridSearchBackend
from corpusdock.retrieval import RetrievalError, SearchResponse, VerificationReport


TIMESTAMP = "2026-08-11T12:00:00Z"


class _StaticBackend:
    def __init__(self, results: tuple[EvidenceResult, ...]) -> None:
        self.results = results
        self.requested_limits: list[int] = []

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        source_id: str | None = None,
        match_mode: str = "all",
    ) -> SearchResponse:
        self.requested_limits.append(limit)
        results = tuple(
            result
            for result in self.results
            if source_id is None or result.locator.source_id == source_id
        )[:limit]
        return SearchResponse(
            query=query,
            match_mode=match_mode,  # type: ignore[arg-type]
            results=results,
            index_built_at=TIMESTAMP,
            indexed_sources=3,
            indexed_chunks=3,
            partial_sources=0,
        )

    def verify(self, evidence_id: str) -> VerificationReport:
        evidence = next(
            item for item in self.results if item.evidence_id == evidence_id
        )
        return VerificationReport(TIMESTAMP, evidence, ("fixture",))

    def evaluation_metadata(self):  # type: ignore[no-untyped-def]
        return {"embedding": {}, "semantic_index": {}}


def _evidence(token: str, source: str) -> EvidenceResult:
    source_id = f"src-{source * 64}"
    locator = CitationLocator(
        source_id=source_id,
        locator_type="text_line",
        label="line 1",
        line_start=1,
        line_end=1,
        start_offset=0,
        end_offset=10,
    )
    return EvidenceResult(
        evidence_id=f"ev-{token * 64}",
        excerpt=f"Fixture {token}",
        citation=f"Fixture {token}, line 1",
        locator=locator,
        source_path=f"fixture-{token}.txt",
        verification_status="artifact-anchor-confirmed",
        score=0.5,
        chunk_id=f"chk-{token * 64}",
        locators=(locator,),
    )


def test_rrf_promotes_consensus_and_uses_stable_lexical_tie_break() -> None:
    first = _evidence("a", "1")
    consensus = _evidence("b", "2")
    semantic_only = _evidence("c", "3")
    lexical = _StaticBackend((first, consensus))
    semantic = _StaticBackend((semantic_only, consensus))
    backend = HybridSearchBackend(lexical, semantic)

    response = backend.search("fixture query", limit=3)

    assert lexical.requested_limits == [60]
    assert semantic.requested_limits == [60]
    assert [result.evidence_id for result in response.results] == [
        consensus.evidence_id,
        first.evidence_id,
        semantic_only.evidence_id,
    ]
    assert response.results[0].score == round(2 / 62, 12)
    assert response.results[1].score == round(1 / 61, 12)
    assert response.results[2].score == round(1 / 61, 12)
    assert response.results[0].excerpt == consensus.excerpt
    assert backend.verify(first.evidence_id).checks == ("fixture",)


def test_hybrid_preserves_source_filter_and_expands_for_large_output() -> None:
    first = _evidence("a", "1")
    second = _evidence("b", "2")
    lexical = _StaticBackend((first, second))
    semantic = _StaticBackend((second, first))
    backend = HybridSearchBackend(lexical, semantic)

    response = backend.search(
        "fixture query",
        limit=80,
        source_id=second.locator.source_id,
        match_mode="any",
    )

    assert lexical.requested_limits == [80]
    assert semantic.requested_limits == [80]
    assert response.match_mode == "any"
    assert [result.evidence_id for result in response.results] == [second.evidence_id]


def test_hybrid_rejects_index_and_evidence_disagreement() -> None:
    evidence = _evidence("a", "1")

    class _ChangedIndexBackend(_StaticBackend):
        def search(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return replace(super().search(*args, **kwargs), indexed_chunks=4)

    with pytest.raises(RetrievalError, match="same exact index"):
        HybridSearchBackend(
            _StaticBackend((evidence,)),
            _ChangedIndexBackend((evidence,)),
        ).search("fixture query")

    conflicting = replace(evidence, excerpt="Conflicting text")
    with pytest.raises(RetrievalError, match="conflicting exact evidence"):
        HybridSearchBackend(
            _StaticBackend((evidence,)),
            _StaticBackend((conflicting,)),
        ).search("fixture query")


@pytest.mark.parametrize(
    "config",
    (
        {"candidate_limit": 0},
        {"candidate_limit": 101},
        {"rrf_k": 0},
        {"lexical_weight": 0.0},
        {"semantic_weight": float("nan")},
    ),
)
def test_fusion_config_rejects_invalid_values(config: dict[str, object]) -> None:
    with pytest.raises(RetrievalError):
        FusionConfig(**config)  # type: ignore[arg-type]
