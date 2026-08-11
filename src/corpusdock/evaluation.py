"""Versioned, non-content retrieval evaluation for local CorpusDock indexes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import ceil
from pathlib import Path
import re
from statistics import fmean
import sys
from time import perf_counter
from typing import Any

from corpusdock import __version__
from corpusdock.contracts import CitationLocator, EvidenceResult
from corpusdock.manifest import utc_now
from corpusdock.retrieval import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    MatchMode,
    RetrievalError,
    SearchBackend,
    SearchResponse,
)


EVALUATION_DATASET_SCHEMA_VERSION = 1
EVALUATION_REPORT_SCHEMA_VERSION = 1
MAX_EVALUATION_DATASET_BYTES = 10 * 1024 * 1024
MAX_EVALUATION_CASES = 10_000

_SOURCE_ID_PATTERN = re.compile(r"src-[0-9a-f]{64}")
_CASE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_MATCH_MODES = {"all", "any", "phrase"}
_RETRIEVAL_METADATA_RESERVED = {
    "backend",
    "mode",
    "limit",
    "verification_enabled",
    "latency_scope",
    "index",
    "process_peak_rss_bytes",
}
_LOCATOR_FIELD_TYPES: dict[str, type[str] | type[int]] = {
    "locator_type": str,
    "label": str,
    "page": int,
    "page_label": str,
    "chapter": str,
    "heading": str,
    "spine_item": str,
    "paragraph_id": str,
    "line_start": int,
    "line_end": int,
    "start_offset": int,
    "end_offset": int,
    "extraction_method": str,
}
_ONE_BASED_LOCATOR_FIELDS = {"page", "line_start", "line_end"}
_ZERO_BASED_LOCATOR_FIELDS = {"start_offset", "end_offset"}


class EvaluationError(Exception):
    """A malformed evaluation dataset or invalid evaluation request."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class LocatorExpectation:
    """Fields that must match one stored locator for a relevant result."""

    fields: tuple[tuple[str, str | int], ...]

    def matches(self, locators: Sequence[CitationLocator]) -> bool:
        return any(
            all(getattr(locator, field) == expected for field, expected in self.fields)
            for locator in locators
        )

    def to_dict(self) -> dict[str, str | int]:
        return dict(self.fields)


@dataclass(frozen=True, slots=True)
class RelevanceJudgment:
    """One source-level relevance judgment and optional locator expectation."""

    source_id: str
    locator: LocatorExpectation | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"source_id": self.source_id}
        if self.locator is not None:
            result["locator"] = self.locator.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One query with explicit, human-authored relevance judgments."""

    case_id: str
    category: str
    query: str
    match_mode: MatchMode
    relevance: tuple[RelevanceJudgment, ...]

    @property
    def relevant_source_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.source_id for item in self.relevance))


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    """A versioned collection of retrieval cases."""

    name: str
    description: str
    cases: tuple[EvaluationCase, ...]
    sha256: str
    schema_version: int = EVALUATION_DATASET_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class LatencySummary:
    """Search-only wall-clock latency in milliseconds."""

    mean: float
    p50: float
    p95: float
    maximum: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "max": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Aggregate source retrieval, locator, verification, and latency metrics."""

    cases: int
    cases_with_results: int
    relevant_sources: int
    matched_relevant_sources: int
    recall_at_k: float
    mean_reciprocal_rank_at_k: float
    locator_judgments: int
    matched_locator_judgments: int
    locator_accuracy: float | None
    verification_attempts: int
    verification_successes: int
    verification_rate: float | None
    latency_ms: LatencySummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "cases_with_results": self.cases_with_results,
            "relevant_sources": self.relevant_sources,
            "matched_relevant_sources": self.matched_relevant_sources,
            "recall_at_k": self.recall_at_k,
            "mean_reciprocal_rank_at_k": self.mean_reciprocal_rank_at_k,
            "locator_judgments": self.locator_judgments,
            "matched_locator_judgments": self.matched_locator_judgments,
            "locator_accuracy": self.locator_accuracy,
            "verification_attempts": self.verification_attempts,
            "verification_successes": self.verification_successes,
            "verification_rate": self.verification_rate,
            "latency_ms": self.latency_ms.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """Non-content diagnostics for one evaluated query."""

    case_id: str
    category: str
    query: str
    match_mode: MatchMode
    result_count: int
    relevant_source_ids: tuple[str, ...]
    retrieved_source_ids: tuple[str, ...]
    retrieved_evidence_ids: tuple[str, ...]
    matched_source_ids: tuple[str, ...]
    recall_at_k: float
    reciprocal_rank_at_k: float
    locator_judgments: int
    matched_locator_judgments: int
    locator_accuracy: float | None
    verification_attempts: int
    verification_successes: int
    verification_error_codes: tuple[str, ...]
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "query": self.query,
            "match_mode": self.match_mode,
            "result_count": self.result_count,
            "relevant_source_ids": list(self.relevant_source_ids),
            "retrieved_source_ids": list(self.retrieved_source_ids),
            "retrieved_evidence_ids": list(self.retrieved_evidence_ids),
            "matched_source_ids": list(self.matched_source_ids),
            "metrics": {
                "recall_at_k": self.recall_at_k,
                "reciprocal_rank_at_k": self.reciprocal_rank_at_k,
                "locator_judgments": self.locator_judgments,
                "matched_locator_judgments": self.matched_locator_judgments,
                "locator_accuracy": self.locator_accuracy,
            },
            "verification": {
                "attempts": self.verification_attempts,
                "successes": self.verification_successes,
                "error_codes": list(self.verification_error_codes),
            },
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Versioned evaluation output that deliberately excludes retrieved excerpts."""

    generated_at: str
    dataset: EvaluationDataset
    backend_name: str
    retrieval_mode: str
    limit: int
    verification_enabled: bool
    index_built_at: str
    indexed_sources: int
    indexed_chunks: int
    partial_sources: int
    index_size_bytes: int | None
    process_peak_rss_bytes: int | None
    summary: MetricSummary
    by_category: tuple[tuple[str, MetricSummary], ...]
    cases: tuple[CaseEvaluation, ...]
    retrieval_metadata: Mapping[str, Any] | None = None
    schema_version: int = EVALUATION_REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        retrieval: dict[str, Any] = {
            "backend": self.backend_name,
            "mode": self.retrieval_mode,
            "limit": self.limit,
            "verification_enabled": self.verification_enabled,
            "latency_scope": "search_only",
            "index": {
                "built_at": self.index_built_at,
                "sources": self.indexed_sources,
                "chunks": self.indexed_chunks,
                "partial_sources": self.partial_sources,
                "size_bytes": self.index_size_bytes,
            },
            "process_peak_rss_bytes": self.process_peak_rss_bytes,
        }
        if self.retrieval_metadata is not None:
            retrieval.update(self.retrieval_metadata)
        return {
            "schema_version": self.schema_version,
            "corpusdock_version": __version__,
            "generated_at": self.generated_at,
            "dataset": {
                "schema_version": self.dataset.schema_version,
                "name": self.dataset.name,
                "description": self.dataset.description,
                "sha256": self.dataset.sha256,
                "cases": len(self.dataset.cases),
            },
            "retrieval": retrieval,
            "summary": self.summary.to_dict(),
            "by_category": {
                category: metrics.to_dict() for category, metrics in self.by_category
            },
            "cases": [case.to_dict() for case in self.cases],
        }


def load_evaluation_dataset(path: Path | str) -> EvaluationDataset:
    """Load and strictly validate a versioned JSON relevance dataset."""

    dataset_path = Path(path).expanduser()
    try:
        resolved_path = dataset_path.resolve(strict=True)
        if not resolved_path.is_file():
            raise EvaluationError(
                "evaluation_dataset_not_file",
                f"Evaluation dataset is not a regular file: '{dataset_path}'.",
            )
        if resolved_path.stat().st_size > MAX_EVALUATION_DATASET_BYTES:
            raise EvaluationError(
                "evaluation_dataset_too_large",
                "Evaluation dataset exceeds the 10 MiB safety limit.",
            )
        with resolved_path.open("rb") as dataset_file:
            raw_bytes = dataset_file.read(MAX_EVALUATION_DATASET_BYTES + 1)
        if len(raw_bytes) > MAX_EVALUATION_DATASET_BYTES:
            raise EvaluationError(
                "evaluation_dataset_too_large",
                "Evaluation dataset exceeds the 10 MiB safety limit.",
            )
        payload = json.loads(
            raw_bytes.decode("utf-8"), object_pairs_hook=_unique_json_object
        )
    except EvaluationError:
        raise
    except FileNotFoundError as error:
        raise EvaluationError(
            "evaluation_dataset_missing",
            f"Evaluation dataset does not exist: '{dataset_path}'.",
        ) from error
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError(
            "evaluation_dataset_invalid",
            f"Could not read evaluation dataset: {error}.",
        ) from error

    root = _mapping(payload, "evaluation dataset")
    _reject_unknown_fields(
        root,
        {"schema_version", "name", "description", "cases"},
        "evaluation dataset",
    )
    schema_version = root.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != EVALUATION_DATASET_SCHEMA_VERSION
    ):
        raise EvaluationError(
            "evaluation_schema_unsupported",
            f"Evaluation dataset schema must be {EVALUATION_DATASET_SCHEMA_VERSION}.",
        )
    name = _string(root.get("name"), "dataset name", maximum=200)
    description = _string(
        root.get("description", ""),
        "dataset description",
        maximum=2_000,
        allow_empty=True,
    )
    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationError(
            "evaluation_cases_invalid",
            "Evaluation dataset cases must be a non-empty array.",
        )
    if len(raw_cases) > MAX_EVALUATION_CASES:
        raise EvaluationError(
            "evaluation_cases_invalid",
            f"Evaluation dataset cannot exceed {MAX_EVALUATION_CASES} cases.",
        )
    cases = tuple(_case(raw_case, index) for index, raw_case in enumerate(raw_cases))
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationError(
            "evaluation_case_duplicate", "Evaluation case IDs must be unique."
        )
    return EvaluationDataset(
        name=name,
        description=description,
        cases=cases,
        sha256=sha256(raw_bytes).hexdigest(),
    )


def evaluate_retrieval(
    dataset: EvaluationDataset,
    backend: SearchBackend,
    *,
    limit: int = DEFAULT_SEARCH_LIMIT,
    verify: bool = True,
    backend_name: str = "sqlite_fts5",
    retrieval_mode: str = "lexical",
    index_size_bytes: int | None = None,
    retrieval_metadata: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = perf_counter,
    now: Callable[[], str] = utc_now,
) -> EvaluationReport:
    """Evaluate retrieval without returning excerpts, paths, or citation text."""

    if not dataset.cases:
        raise EvaluationError(
            "evaluation_cases_invalid",
            "Evaluation dataset cases must be non-empty.",
        )

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_SEARCH_LIMIT
    ):
        raise EvaluationError(
            "evaluation_limit_invalid",
            f"Evaluation limit must be between 1 and {MAX_SEARCH_LIMIT}.",
        )

    if index_size_bytes is not None and (
        not isinstance(index_size_bytes, int)
        or isinstance(index_size_bytes, bool)
        or index_size_bytes < 0
    ):
        raise EvaluationError(
            "evaluation_index_size_invalid",
            "Index size must be a non-negative integer when supplied.",
        )

    if retrieval_metadata is not None:
        reserved = sorted(set(retrieval_metadata) & _RETRIEVAL_METADATA_RESERVED)
        if reserved:
            raise EvaluationError(
                "evaluation_retrieval_metadata_invalid",
                "Retrieval metadata cannot replace report field(s): "
                + ", ".join(reserved)
                + ".",
            )

    timed_responses: list[tuple[EvaluationCase, SearchResponse, float]] = []
    index_signature: tuple[str, int, int, int] | None = None
    for case in dataset.cases:
        started = clock()
        response = backend.search(
            case.query,
            limit=limit,
            match_mode=case.match_mode,
        )
        elapsed_ms = _rounded(max(0.0, clock() - started) * 1_000)
        current_signature = _index_signature(response)
        if index_signature is None:
            index_signature = current_signature
        elif current_signature != index_signature:
            raise EvaluationError(
                "evaluation_index_changed",
                "The search index changed while evaluation was running.",
            )
        timed_responses.append((case, response, elapsed_ms))

    assert index_signature is not None
    _assert_judged_sources_indexed(dataset, backend, index_signature)

    verification_cache: dict[str, str | None] = {}
    case_results: list[CaseEvaluation] = []
    for case, response, elapsed_ms in timed_responses:
        results = response.results
        retrieved_source_ids = tuple(result.locator.source_id for result in results)
        relevant_source_ids = case.relevant_source_ids
        relevant_set = set(relevant_source_ids)
        matched_source_ids = tuple(
            source_id
            for source_id in relevant_source_ids
            if source_id in set(retrieved_source_ids)
        )
        reciprocal_rank = next(
            (
                1.0 / rank
                for rank, result in enumerate(results, start=1)
                if result.locator.source_id in relevant_set
            ),
            0.0,
        )
        locator_judgments = tuple(
            judgment for judgment in case.relevance if judgment.locator is not None
        )
        matched_locator_judgments = sum(
            _judgment_matches(judgment, results) for judgment in locator_judgments
        )

        verification_attempts = 0
        verification_successes = 0
        verification_error_codes: list[str] = []
        if verify:
            verification_attempts = len(results)
            for result in results:
                error_code = verification_cache.get(result.evidence_id)
                if result.evidence_id not in verification_cache:
                    try:
                        report = backend.verify(result.evidence_id)
                        error_code = (
                            None
                            if report.evidence.verification_status
                            == "source-anchor-confirmed"
                            else "verification_status_invalid"
                        )
                    except RetrievalError as error:
                        error_code = error.code
                    verification_cache[result.evidence_id] = error_code
                if error_code is None:
                    verification_successes += 1
                else:
                    verification_error_codes.append(error_code)

        locator_count = len(locator_judgments)
        case_results.append(
            CaseEvaluation(
                case_id=case.case_id,
                category=case.category,
                query=case.query,
                match_mode=case.match_mode,
                result_count=len(results),
                relevant_source_ids=relevant_source_ids,
                retrieved_source_ids=retrieved_source_ids,
                retrieved_evidence_ids=tuple(result.evidence_id for result in results),
                matched_source_ids=matched_source_ids,
                recall_at_k=_ratio(len(matched_source_ids), len(relevant_source_ids)),
                reciprocal_rank_at_k=_rounded(reciprocal_rank),
                locator_judgments=locator_count,
                matched_locator_judgments=matched_locator_judgments,
                locator_accuracy=(
                    _ratio(matched_locator_judgments, locator_count)
                    if locator_count
                    else None
                ),
                verification_attempts=verification_attempts,
                verification_successes=verification_successes,
                verification_error_codes=tuple(sorted(verification_error_codes)),
                latency_ms=elapsed_ms,
            )
        )

    completed_cases = tuple(case_results)
    categories = sorted({case.category for case in completed_cases})
    return EvaluationReport(
        generated_at=now(),
        dataset=dataset,
        backend_name=backend_name,
        retrieval_mode=retrieval_mode,
        limit=limit,
        verification_enabled=verify,
        index_built_at=index_signature[0],
        indexed_sources=index_signature[1],
        indexed_chunks=index_signature[2],
        partial_sources=index_signature[3],
        index_size_bytes=index_size_bytes,
        process_peak_rss_bytes=_process_peak_rss_bytes(),
        summary=_aggregate_metrics(completed_cases, verification_enabled=verify),
        by_category=tuple(
            (
                category,
                _aggregate_metrics(
                    tuple(
                        case for case in completed_cases if case.category == category
                    ),
                    verification_enabled=verify,
                ),
            )
            for category in categories
        ),
        cases=completed_cases,
        retrieval_metadata=dict(retrieval_metadata)
        if retrieval_metadata is not None
        else None,
    )


def _assert_judged_sources_indexed(
    dataset: EvaluationDataset,
    backend: SearchBackend,
    expected_signature: tuple[str, int, int, int],
) -> None:
    representatives: dict[str, EvaluationCase] = {}
    for case in dataset.cases:
        for source_id in case.relevant_source_ids:
            representatives.setdefault(source_id, case)

    for source_id, case in representatives.items():
        try:
            response = backend.search(
                case.query,
                limit=1,
                source_id=source_id,
                match_mode=case.match_mode,
            )
        except RetrievalError as error:
            if error.code == "source_not_indexed":
                raise EvaluationError(
                    "evaluation_source_not_indexed",
                    f"Judged source '{source_id}' is not present in the search index.",
                ) from error
            raise
        current_signature = _index_signature(response)
        if current_signature != expected_signature:
            raise EvaluationError(
                "evaluation_index_changed",
                "The search index changed while evaluation was validating sources.",
            )


def _index_signature(response: SearchResponse) -> tuple[str, int, int, int]:
    return (
        response.index_built_at,
        response.indexed_sources,
        response.indexed_chunks,
        response.partial_sources,
    )


def _process_peak_rss_bytes() -> int | None:
    """Return the process high-water RSS where the standard library exposes it."""

    try:
        import resource

        maximum_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError, OSError, ValueError):
        return None
    return maximum_rss if sys.platform == "darwin" else maximum_rss * 1024


def _case(value: object, index: int) -> EvaluationCase:
    item = _mapping(value, f"case {index + 1}")
    _reject_unknown_fields(
        item,
        {"case_id", "category", "query", "match_mode", "relevance"},
        f"case {index + 1}",
    )
    case_id = _string(item.get("case_id"), "case ID", maximum=128)
    if _CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise EvaluationError(
            "evaluation_case_id_invalid",
            f"Evaluation case ID '{case_id}' must use lowercase letters, digits, '.', '_', or '-'.",
        )
    category = _string(item.get("category"), f"category for '{case_id}'", maximum=100)
    query = _string(item.get("query"), f"query for '{case_id}'", maximum=4_096)
    raw_match_mode = item.get("match_mode", "all")
    if not isinstance(raw_match_mode, str) or raw_match_mode not in _MATCH_MODES:
        raise EvaluationError(
            "evaluation_match_mode_invalid",
            f"Match mode for '{case_id}' must be all, any, or phrase.",
        )
    raw_relevance = item.get("relevance")
    if not isinstance(raw_relevance, list) or not raw_relevance:
        raise EvaluationError(
            "evaluation_relevance_invalid",
            f"Relevance judgments for '{case_id}' must be a non-empty array.",
        )
    relevance = tuple(
        _judgment(raw_judgment, case_id, judgment_index)
        for judgment_index, raw_judgment in enumerate(raw_relevance)
    )
    judgment_keys = [
        (
            judgment.source_id,
            judgment.locator.fields if judgment.locator is not None else (),
        )
        for judgment in relevance
    ]
    if len(judgment_keys) != len(set(judgment_keys)):
        raise EvaluationError(
            "evaluation_relevance_duplicate",
            f"Relevance judgments for '{case_id}' contain a duplicate.",
        )
    return EvaluationCase(
        case_id=case_id,
        category=category,
        query=query,
        match_mode=raw_match_mode,
        relevance=relevance,
    )


def _judgment(value: object, case_id: str, index: int) -> RelevanceJudgment:
    item = _mapping(value, f"judgment {index + 1} for '{case_id}'")
    _reject_unknown_fields(
        item,
        {"source_id", "locator"},
        f"judgment {index + 1} for '{case_id}'",
    )
    source_id = _string(item.get("source_id"), "relevant source ID", maximum=68)
    if _SOURCE_ID_PATTERN.fullmatch(source_id) is None:
        raise EvaluationError(
            "evaluation_source_id_invalid",
            f"Relevant source ID '{source_id}' is not a CorpusDock source ID.",
        )
    locator = (
        _locator_expectation(item["locator"], case_id) if "locator" in item else None
    )
    return RelevanceJudgment(source_id=source_id, locator=locator)


def _locator_expectation(value: object, case_id: str) -> LocatorExpectation:
    item = _mapping(value, f"locator expectation for '{case_id}'")
    if not item:
        raise EvaluationError(
            "evaluation_locator_invalid",
            f"Locator expectation for '{case_id}' cannot be empty.",
        )
    _reject_unknown_fields(
        item,
        set(_LOCATOR_FIELD_TYPES),
        f"locator expectation for '{case_id}'",
    )
    fields: list[tuple[str, str | int]] = []
    for field in sorted(item):
        expected_type = _LOCATOR_FIELD_TYPES[field]
        field_value = item[field]
        if expected_type is int:
            if not isinstance(field_value, int) or isinstance(field_value, bool):
                raise EvaluationError(
                    "evaluation_locator_invalid",
                    f"Locator field '{field}' for '{case_id}' must be an integer.",
                )
            if field in _ONE_BASED_LOCATOR_FIELDS and field_value < 1:
                raise EvaluationError(
                    "evaluation_locator_invalid",
                    f"Locator field '{field}' for '{case_id}' must be at least 1.",
                )
            if field in _ZERO_BASED_LOCATOR_FIELDS and field_value < 0:
                raise EvaluationError(
                    "evaluation_locator_invalid",
                    f"Locator field '{field}' for '{case_id}' cannot be negative.",
                )
        elif not isinstance(field_value, str) or not field_value.strip():
            raise EvaluationError(
                "evaluation_locator_invalid",
                f"Locator field '{field}' for '{case_id}' must be a non-empty string.",
            )
        fields.append((field, field_value))
    normalized_fields = dict(fields)
    for start_field, end_field in (
        ("line_start", "line_end"),
        ("start_offset", "end_offset"),
    ):
        if (
            start_field in normalized_fields
            and end_field in normalized_fields
            and normalized_fields[start_field] > normalized_fields[end_field]
        ):
            raise EvaluationError(
                "evaluation_locator_invalid",
                f"Locator field '{start_field}' for '{case_id}' cannot exceed '{end_field}'.",
            )
    return LocatorExpectation(tuple(fields))


def _judgment_matches(
    judgment: RelevanceJudgment, results: Sequence[EvidenceResult]
) -> bool:
    assert judgment.locator is not None
    for result in results:
        if result.locator.source_id != judgment.source_id:
            continue
        locators = tuple(dict.fromkeys((result.locator, *result.locators)))
        if judgment.locator.matches(locators):
            return True
    return False


def _aggregate_metrics(
    cases: Sequence[CaseEvaluation], *, verification_enabled: bool
) -> MetricSummary:
    relevant_sources = sum(len(case.relevant_source_ids) for case in cases)
    matched_sources = sum(len(case.matched_source_ids) for case in cases)
    locator_judgments = sum(case.locator_judgments for case in cases)
    matched_locators = sum(case.matched_locator_judgments for case in cases)
    verification_attempts = sum(case.verification_attempts for case in cases)
    verification_successes = sum(case.verification_successes for case in cases)
    latencies = tuple(case.latency_ms for case in cases)
    return MetricSummary(
        cases=len(cases),
        cases_with_results=sum(case.result_count > 0 for case in cases),
        relevant_sources=relevant_sources,
        matched_relevant_sources=matched_sources,
        recall_at_k=_ratio(matched_sources, relevant_sources),
        mean_reciprocal_rank_at_k=_rounded(
            fmean(case.reciprocal_rank_at_k for case in cases) if cases else 0.0
        ),
        locator_judgments=locator_judgments,
        matched_locator_judgments=matched_locators,
        locator_accuracy=(
            _ratio(matched_locators, locator_judgments) if locator_judgments else None
        ),
        verification_attempts=verification_attempts,
        verification_successes=verification_successes,
        verification_rate=(
            _ratio(verification_successes, verification_attempts)
            if verification_enabled and verification_attempts
            else None
        ),
        latency_ms=LatencySummary(
            mean=_rounded(fmean(latencies) if latencies else 0.0),
            p50=_percentile(latencies, 50),
            p95=_percentile(latencies, 95),
            maximum=_rounded(max(latencies, default=0.0)),
        ),
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise EvaluationError(
            "evaluation_dataset_invalid", f"{label.capitalize()} must be an object."
        )
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError(
                "evaluation_json_key_duplicate",
                f"Evaluation dataset contains duplicate JSON key '{key}'.",
            )
        result[key] = value
    return result


def _reject_unknown_fields(
    value: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise EvaluationError(
            "evaluation_field_unknown",
            f"{label.capitalize()} has unknown field(s): {', '.join(unexpected)}.",
        )


def _string(
    value: object,
    label: str,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise EvaluationError(
            "evaluation_field_invalid", f"{label.capitalize()} must be a string."
        )
    if len(value) > maximum:
        raise EvaluationError(
            "evaluation_field_invalid",
            f"{label.capitalize()} cannot exceed {maximum} characters.",
        )
    return value


def _ratio(numerator: int, denominator: int) -> float:
    return _rounded(numerator / denominator) if denominator else 0.0


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, ceil((percentile / 100) * len(ordered)))
    return _rounded(ordered[rank - 1])


def _rounded(value: float) -> float:
    return round(value, 6)
