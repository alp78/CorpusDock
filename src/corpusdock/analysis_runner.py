"""End-to-end local extraction runner over exact CorpusDock evidence."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from corpusdock.analysis_contracts import (
    AnalysisContractError,
    parse_and_validate_analysis,
    rejection_analysis,
)
from corpusdock.analysis_models import StructuredExtractionProvider
from corpusdock.analysis_store import (
    AnalysisRunDescriptor,
    AnalysisStore,
    AnalysisStoreError,
)


MAX_ANALYSIS_EVIDENCE_LIMIT = 10_000_000

ProgressCallback = Callable[[int, int, AnalysisRunDescriptor], None]


def run_corpus_analysis(
    project_root: str,
    provider: StructuredExtractionProvider,
    *,
    source_ids: Sequence[str] = (),
    limit: int | None = None,
    resume: bool = True,
    progress: ProgressCallback | None = None,
) -> AnalysisRunDescriptor:
    """Extract, ground, and transactionally persist candidates for exact chunks."""

    _validate_limit(limit)
    clean_source_ids = _source_ids(source_ids)
    store = AnalysisStore(project_root, reconcile=True)
    unknown_sources = set(clean_source_ids) - set(store.snapshot.source_ids)
    if unknown_sources:
        unknown = sorted(unknown_sources)[0]
        raise AnalysisStoreError(
            "analysis_source_unknown",
            f"No indexed source has ID '{unknown}'.",
        )
    selected = tuple(
        evidence
        for evidence in store.snapshot.evidence
        if not clean_source_ids or evidence.locator.source_id in clean_source_ids
    )
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise AnalysisStoreError(
            "analysis_scope_empty",
            "The selected exact-index scope contains no evidence chunks.",
        )

    info = provider.info
    extractor = info.to_dict()
    scope: dict[str, Any] = {
        "selection": "exact-index-order-v1",
        "source_ids": list(clean_source_ids),
        "limit": limit,
        "selected_evidence": len(selected),
    }
    descriptor = store.begin_run(
        extractor,
        prompt_version=info.prompt_version,
        prompt_sha256=info.prompt_sha256,
        scope=scope,
        resume=resume,
    )
    store.prepare_evidence_scope(
        descriptor.run_id,
        tuple(evidence.evidence_id for evidence in selected),
    )
    descriptor = store.run_descriptor(descriptor.run_id)
    completed = store.analyzed_evidence_ids(descriptor.run_id)
    pending = tuple(
        evidence for evidence in selected if evidence.evidence_id not in completed
    )
    total = len(selected)
    done = total - len(pending)
    if progress is not None:
        progress(done, total, descriptor)

    batch_size = info.batch_size
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        outputs = provider.extract(tuple(evidence.excerpt for evidence in batch))
        if len(outputs) != len(batch):
            raise AnalysisStoreError(
                "analysis_provider_invalid",
                "The extraction provider returned the wrong number of results.",
            )
        analyses = []
        for evidence, output in zip(batch, outputs, strict=True):
            try:
                analysis = parse_and_validate_analysis(
                    output.raw_output,
                    evidence,
                    run_id=descriptor.run_id,
                    evidence_units=output.evidence_units or None,
                    inference_ms=output.inference_ms,
                    output_tokens=output.output_tokens,
                    output_truncated=output.truncated,
                )
            except AnalysisContractError as error:
                analysis = rejection_analysis(
                    output.raw_output,
                    evidence,
                    run_id=descriptor.run_id,
                    error=error,
                    inference_ms=output.inference_ms,
                    output_tokens=output.output_tokens,
                    output_truncated=output.truncated,
                )
            analyses.append(analysis)
        store.write_batch(descriptor.run_id, analyses)
        done += len(batch)
        descriptor = store.run_descriptor(descriptor.run_id)
        if progress is not None:
            progress(done, total, descriptor)

    return store.finish_run(descriptor.run_id, extractor=provider.info.to_dict())


def _validate_limit(limit: int | None) -> None:
    if limit is None:
        return
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_ANALYSIS_EVIDENCE_LIMIT
    ):
        raise AnalysisStoreError(
            "analysis_limit_invalid",
            f"Analysis limit must be between 1 and {MAX_ANALYSIS_EVIDENCE_LIMIT}.",
        )


def _source_ids(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise AnalysisStoreError(
            "analysis_source_invalid", "Source IDs must be a sequence of strings."
        )
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 100:
            raise AnalysisStoreError(
                "analysis_source_invalid", "Analysis source ID is invalid."
            )
        if value not in result:
            result.append(value)
    return tuple(sorted(result))
