"""Project-authored quality evaluation for local structured extraction models."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from corpusdock.analysis_contracts import (
    AnalysisContractError,
    EvidenceAnalysis,
    parse_and_validate_analysis,
    rejection_analysis,
)
from corpusdock.analysis_models import StructuredExtractionProvider
from corpusdock.contracts import CitationLocator, EvidenceResult


ANALYSIS_BENCHMARK_SCHEMA_VERSION = 1
ANALYSIS_EVALUATION_SCHEMA_VERSION = 1
MAX_BENCHMARK_CASES = 1_000
MAX_BENCHMARK_TEXT_CHARACTERS = 100_000

_CASE_FIELDS = {"case_id", "language", "text", "expected"}
_EXPECTED_FIELDS = {"concepts", "claims", "relations"}
_CONCEPT_GOLD_FIELDS = {"gold_id", "support_contains", "label_aliases"}
_CLAIM_GOLD_FIELDS = {
    "gold_id",
    "support_contains",
    "claim_type",
    "polarity",
    "certainty",
    "conditional",
    "attribution",
    "normative_force",
}
_RELATION_GOLD_FIELDS = {
    "gold_id",
    "support_contains",
    "subject_gold_id",
    "relation_type",
    "object_gold_id",
    "polarity",
    "certainty",
    "conditional",
    "attribution",
    "normative_force",
}
_EXPECTED_ENUMS = {
    "claim_type": {
        "observation",
        "definition",
        "causal",
        "recommendation",
        "comparison",
        "prediction",
        "value_judgment",
        "other",
    },
    "relation_type": {
        "is_a",
        "part_of",
        "causes",
        "enables",
        "inhibits",
        "associated_with",
        "contrasts_with",
        "depends_on",
        "precedes",
        "uses",
        "measures",
        "other",
    },
    "polarity": {"affirmed", "negated", "mixed"},
    "certainty": {"asserted", "possible", "probable", "uncertain"},
    "attribution": {"source", "reported", "quoted", "unclear"},
    "normative_force": {
        "none",
        "recommended",
        "required",
        "permitted",
        "prohibited",
    },
}


class AnalysisEvaluationError(Exception):
    """An invalid extraction benchmark or evaluation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AnalysisBenchmarkCase:
    case_id: str
    language: str
    text: str
    concepts: tuple[Mapping[str, Any], ...]
    claims: tuple[Mapping[str, Any], ...]
    relations: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AnalysisBenchmark:
    path: str
    schema_version: int
    benchmark_id: str
    description: str
    sha256: str
    cases: tuple[AnalysisBenchmarkCase, ...]


@dataclass(frozen=True, slots=True)
class CandidateMetrics:
    expected: int
    produced: int
    matched: int

    @property
    def precision(self) -> float:
        return (
            self.matched / self.produced if self.produced else float(self.expected == 0)
        )

    @property
    def recall(self) -> float:
        return (
            self.matched / self.expected if self.expected else float(self.produced == 0)
        )

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        return (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "expected": self.expected,
            "produced": self.produced,
            "matched": self.matched,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
        }


@dataclass(frozen=True, slots=True)
class AnalysisCaseResult:
    case_id: str
    language: str
    response_valid: bool
    validation_status: str
    concepts: CandidateMetrics
    claims: CandidateMetrics
    relations: CandidateMetrics
    rejected_candidates: int
    rejection_codes: tuple[str, ...]
    inference_ms: float
    output_tokens: int | None
    output_truncated: bool

    @property
    def exact(self) -> bool:
        return (
            self.response_valid
            and self.rejected_candidates == 0
            and all(
                metrics.expected == metrics.produced == metrics.matched
                for metrics in (self.concepts, self.claims, self.relations)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "language": self.language,
            "response_valid": self.response_valid,
            "validation_status": self.validation_status,
            "exact": self.exact,
            "concepts": self.concepts.to_dict(),
            "claims": self.claims.to_dict(),
            "relations": self.relations.to_dict(),
            "rejected_candidates": self.rejected_candidates,
            "rejection_codes": list(self.rejection_codes),
            "inference_ms": self.inference_ms,
            "output_tokens": self.output_tokens,
            "output_truncated": self.output_truncated,
        }


@dataclass(frozen=True, slots=True)
class AnalysisEvaluationReport:
    benchmark: AnalysisBenchmark
    extractor: Mapping[str, Any]
    cases: tuple[AnalysisCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        concepts = _sum_metrics(tuple(case.concepts for case in self.cases))
        claims = _sum_metrics(tuple(case.claims for case in self.cases))
        relations = _sum_metrics(tuple(case.relations for case in self.cases))
        latencies = tuple(case.inference_ms for case in self.cases)
        output_tokens = tuple(
            case.output_tokens for case in self.cases if case.output_tokens is not None
        )
        accepted = sum(
            metrics.produced
            for case in self.cases
            for metrics in (case.concepts, case.claims, case.relations)
        )
        rejected = sum(case.rejected_candidates for case in self.cases)
        return {
            "schema_version": ANALYSIS_EVALUATION_SCHEMA_VERSION,
            "benchmark": {
                "schema_version": self.benchmark.schema_version,
                "benchmark_id": self.benchmark.benchmark_id,
                "sha256": self.benchmark.sha256,
                "cases": len(self.cases),
            },
            "extractor": dict(self.extractor),
            "summary": {
                "valid_response_rate": round(
                    sum(case.response_valid for case in self.cases) / len(self.cases), 6
                ),
                "fully_grounded_response_rate": round(
                    sum(case.rejected_candidates == 0 for case in self.cases)
                    / len(self.cases),
                    6,
                ),
                "accepted_candidate_rate": round(
                    accepted / (accepted + rejected) if accepted + rejected else 1.0,
                    6,
                ),
                "exact_case_rate": round(
                    sum(case.exact for case in self.cases) / len(self.cases), 6
                ),
                "concepts": concepts.to_dict(),
                "claims": claims.to_dict(),
                "relations": relations.to_dict(),
                "macro_candidate_f1": round(
                    (concepts.f1 + claims.f1 + relations.f1) / 3, 6
                ),
                "latency_ms": {
                    "total": round(sum(latencies), 6),
                    "p50": _percentile(latencies, 0.50),
                    "p95": _percentile(latencies, 0.95),
                },
                "generation": {
                    "measured_responses": len(output_tokens),
                    "total_output_tokens": sum(output_tokens),
                    "output_tokens_p50": (
                        _percentile(output_tokens, 0.50) if output_tokens else None
                    ),
                    "output_tokens_p95": (
                        _percentile(output_tokens, 0.95) if output_tokens else None
                    ),
                    "truncated_responses": sum(
                        case.output_truncated for case in self.cases
                    ),
                },
            },
            "cases": [case.to_dict() for case in self.cases],
        }


def load_analysis_benchmark(path: Path | str) -> AnalysisBenchmark:
    """Load a strict, versioned, project-authored extraction benchmark."""

    benchmark_path = Path(path).expanduser().resolve()
    try:
        raw = benchmark_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisEvaluationError(
            "analysis_benchmark_read_failed",
            f"Could not read analysis benchmark '{benchmark_path}': {error}.",
        ) from error
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "benchmark_id",
        "description",
        "cases",
    }:
        raise AnalysisEvaluationError(
            "analysis_benchmark_schema_invalid",
            "Analysis benchmark top-level fields are invalid.",
        )
    if payload["schema_version"] != ANALYSIS_BENCHMARK_SCHEMA_VERSION:
        raise AnalysisEvaluationError(
            "analysis_benchmark_schema_invalid",
            "Analysis benchmark schema version is unsupported.",
        )
    benchmark_id = _short_text(payload.get("benchmark_id"), "benchmark_id", 120)
    description = _short_text(payload.get("description"), "description", 1_000)
    raw_cases = payload.get("cases")
    if (
        not isinstance(raw_cases, list)
        or not raw_cases
        or len(raw_cases) > MAX_BENCHMARK_CASES
    ):
        raise AnalysisEvaluationError(
            "analysis_benchmark_schema_invalid",
            "Analysis benchmark cases must be a non-empty bounded array.",
        )
    cases: list[AnalysisBenchmarkCase] = []
    seen_ids: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict) or set(raw_case) != _CASE_FIELDS:
            raise AnalysisEvaluationError(
                "analysis_benchmark_schema_invalid",
                f"Analysis benchmark case {index} has invalid fields.",
            )
        case_id = _short_text(raw_case.get("case_id"), "case_id", 120)
        if case_id in seen_ids:
            raise AnalysisEvaluationError(
                "analysis_benchmark_schema_invalid",
                f"Analysis benchmark repeats case ID '{case_id}'.",
            )
        seen_ids.add(case_id)
        language = _short_text(raw_case.get("language"), "language", 40)
        text = _short_text(raw_case.get("text"), "text", MAX_BENCHMARK_TEXT_CHARACTERS)
        expected = raw_case.get("expected")
        if not isinstance(expected, dict) or set(expected) != _EXPECTED_FIELDS:
            raise AnalysisEvaluationError(
                "analysis_benchmark_schema_invalid",
                f"Expected candidates for case '{case_id}' have invalid fields.",
            )
        concepts = _gold_items(
            expected.get("concepts"), _CONCEPT_GOLD_FIELDS, case_id, "concepts", text
        )
        claims = _gold_items(
            expected.get("claims"), _CLAIM_GOLD_FIELDS, case_id, "claims", text
        )
        relations = _gold_items(
            expected.get("relations"),
            _RELATION_GOLD_FIELDS,
            case_id,
            "relations",
            text,
        )
        concept_gold_ids = {str(item["gold_id"]) for item in concepts}
        if any(
            relation["subject_gold_id"] not in concept_gold_ids
            or relation["object_gold_id"] not in concept_gold_ids
            for relation in relations
        ):
            raise AnalysisEvaluationError(
                "analysis_benchmark_schema_invalid",
                f"A relation in case '{case_id}' refers to an unknown gold concept.",
            )
        cases.append(
            AnalysisBenchmarkCase(
                case_id=case_id,
                language=language,
                text=text,
                concepts=concepts,
                claims=claims,
                relations=relations,
            )
        )
    return AnalysisBenchmark(
        path=str(benchmark_path),
        schema_version=ANALYSIS_BENCHMARK_SCHEMA_VERSION,
        benchmark_id=benchmark_id,
        description=description,
        sha256=sha256(raw).hexdigest(),
        cases=tuple(cases),
    )


def evaluate_analysis_benchmark(
    benchmark: AnalysisBenchmark,
    provider: StructuredExtractionProvider,
) -> AnalysisEvaluationReport:
    """Run a local provider and score strict grounding plus candidate fidelity."""

    outputs = provider.extract(tuple(case.text for case in benchmark.cases))
    if len(outputs) != len(benchmark.cases):
        raise AnalysisEvaluationError(
            "analysis_provider_invalid",
            "Extraction provider returned the wrong number of benchmark results.",
        )
    run_id = (
        "run-"
        + sha256(
            (
                benchmark.sha256
                + provider.info.model_fingerprint
                + provider.info.prompt_sha256
            ).encode("utf-8")
        ).hexdigest()
    )
    results: list[AnalysisCaseResult] = []
    for case, output in zip(benchmark.cases, outputs, strict=True):
        evidence = _benchmark_evidence(case)
        response_valid = True
        try:
            analysis = parse_and_validate_analysis(
                output.raw_output,
                evidence,
                run_id=run_id,
                evidence_units=output.evidence_units or None,
                inference_ms=output.inference_ms,
                output_tokens=output.output_tokens,
                output_truncated=output.truncated,
            )
        except AnalysisContractError as error:
            response_valid = False
            analysis = rejection_analysis(
                output.raw_output,
                evidence,
                run_id=run_id,
                error=error,
                inference_ms=output.inference_ms,
                output_tokens=output.output_tokens,
                output_truncated=output.truncated,
            )
        results.append(_score_case(case, analysis, response_valid=response_valid))
    return AnalysisEvaluationReport(
        benchmark=benchmark,
        extractor=provider.info.to_dict(),
        cases=tuple(results),
    )


def _score_case(
    case: AnalysisBenchmarkCase,
    analysis: EvidenceAnalysis,
    *,
    response_valid: bool,
) -> AnalysisCaseResult:
    concepts = _match_candidates(
        case.concepts,
        analysis.concepts,
        lambda gold, candidate: _support_matches(case.text, gold, candidate.support),
    )
    claims = _match_candidates(
        case.claims,
        analysis.claims,
        lambda gold, candidate: (
            _support_matches(case.text, gold, candidate.support)
            and all(
                getattr(candidate, field) == gold[field]
                for field in (
                    "claim_type",
                    "polarity",
                    "certainty",
                    "conditional",
                    "attribution",
                    "normative_force",
                )
            )
        ),
    )
    concepts_by_id = {
        candidate.concept_id: candidate for candidate in analysis.concepts
    }
    gold_concepts_by_id = {
        str(concept["gold_id"]): concept for concept in case.concepts
    }
    relations = _match_candidates(
        case.relations,
        analysis.relations,
        lambda gold, candidate: (
            _support_matches(case.text, gold, candidate.support)
            and candidate.subject_concept_id in concepts_by_id
            and candidate.object_concept_id in concepts_by_id
            and _relation_concepts_match(
                case.text,
                gold,
                candidate,
                concepts_by_id,
                gold_concepts_by_id,
            )
            and all(
                getattr(candidate, field) == gold[field]
                for field in (
                    "relation_type",
                    "polarity",
                    "certainty",
                    "conditional",
                    "attribution",
                    "normative_force",
                )
            )
        ),
    )
    return AnalysisCaseResult(
        case_id=case.case_id,
        language=case.language,
        response_valid=response_valid,
        validation_status=analysis.status,
        concepts=concepts,
        claims=claims,
        relations=relations,
        rejected_candidates=len(analysis.rejections),
        rejection_codes=tuple(rejection.code for rejection in analysis.rejections),
        inference_ms=analysis.inference_ms or 0.0,
        output_tokens=analysis.output_tokens,
        output_truncated=analysis.output_truncated,
    )


def _relation_concepts_match(
    text: str,
    gold: Mapping[str, Any],
    candidate: Any,
    concepts_by_id: Mapping[str, Any],
    gold_concepts_by_id: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Match endpoints while treating only explicitly symmetric relations as such."""

    candidate_subject = concepts_by_id[candidate.subject_concept_id]
    candidate_object = concepts_by_id[candidate.object_concept_id]
    gold_subject = gold_concepts_by_id[str(gold["subject_gold_id"])]
    gold_object = gold_concepts_by_id[str(gold["object_gold_id"])]
    direct = _support_matches(text, gold_subject, candidate_subject.support) and (
        _support_matches(text, gold_object, candidate_object.support)
    )
    if direct or gold["relation_type"] != "contrasts_with":
        return direct
    return _support_matches(text, gold_object, candidate_subject.support) and (
        _support_matches(text, gold_subject, candidate_object.support)
    )


def _match_candidates(
    gold: Sequence[Mapping[str, Any]],
    candidates: Sequence[Any],
    matches: Any,
) -> CandidateMetrics:
    available = set(range(len(candidates)))
    matched = 0
    for expected in gold:
        match = next(
            (
                index
                for index in sorted(available)
                if matches(expected, candidates[index])
            ),
            None,
        )
        if match is not None:
            available.remove(match)
            matched += 1
    return CandidateMetrics(
        expected=len(gold), produced=len(candidates), matched=matched
    )


def _support_matches(text: str, gold: Mapping[str, Any], span: Any) -> bool:
    support = text[span.start_offset : span.end_offset]
    return str(gold["support_contains"]) in support


def _benchmark_evidence(case: AnalysisBenchmarkCase) -> EvidenceResult:
    case_digest = sha256(case.case_id.encode("utf-8")).hexdigest()
    text_digest = sha256(case.text.encode("utf-8")).hexdigest()
    source_id = f"src-{case_digest}"
    locator = CitationLocator(
        source_id=source_id,
        locator_type="benchmark_case",
        label=case.case_id,
        start_offset=0,
        end_offset=len(case.text),
        extraction_method="project-authored-fixture",
    )
    return EvidenceResult(
        evidence_id=f"ev-{text_digest}",
        excerpt=case.text,
        citation=f"Project-authored analysis fixture {case.case_id}",
        locator=locator,
        source_path="benchmark://project-authored",
        verification_status="fixture-confirmed",
        chunk_id=f"chk-{text_digest}",
        locators=(locator,),
    )


def _gold_items(
    value: object,
    fields: set[str],
    case_id: str,
    category: str,
    text: str,
) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or len(value) > 100:
        raise AnalysisEvaluationError(
            "analysis_benchmark_schema_invalid",
            f"Expected {category} for case '{case_id}' must be a bounded array.",
        )
    result: list[Mapping[str, Any]] = []
    gold_ids: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != fields:
            raise AnalysisEvaluationError(
                "analysis_benchmark_schema_invalid",
                f"Expected {category} item {index} for case '{case_id}' has invalid fields.",
            )
        gold_id = _short_text(item.get("gold_id"), "gold_id", 120)
        support = _short_text(item.get("support_contains"), "support_contains", 4_000)
        if gold_id in gold_ids or support not in text:
            raise AnalysisEvaluationError(
                "analysis_benchmark_schema_invalid",
                f"Expected {category} for case '{case_id}' has duplicate IDs or absent support.",
            )
        gold_ids.add(gold_id)
        for aliases_key in ("label_aliases",):
            if aliases_key in item:
                aliases = item[aliases_key]
                if (
                    not isinstance(aliases, list)
                    or not aliases
                    or len(aliases) > 20
                    or any(
                        not isinstance(alias, str)
                        or not alias.strip()
                        or len(alias) > 240
                        for alias in aliases
                    )
                ):
                    raise AnalysisEvaluationError(
                        "analysis_benchmark_schema_invalid",
                        f"Expected aliases for case '{case_id}' are invalid.",
                    )
        for field, allowed in _EXPECTED_ENUMS.items():
            if field in item and item[field] not in allowed:
                raise AnalysisEvaluationError(
                    "analysis_benchmark_schema_invalid",
                    f"Expected field '{field}' for case '{case_id}' is invalid.",
                )
        if "conditional" in item and not isinstance(item["conditional"], bool):
            raise AnalysisEvaluationError(
                "analysis_benchmark_schema_invalid",
                f"Expected conditionality for case '{case_id}' must be boolean.",
            )
        result.append(dict(item))
    return tuple(result)


def _short_text(value: object, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or "\x00" in value
    ):
        raise AnalysisEvaluationError(
            "analysis_benchmark_schema_invalid",
            f"Analysis benchmark {label} must be a non-empty bounded string.",
        )
    return value


def _sum_metrics(metrics: Sequence[CandidateMetrics]) -> CandidateMetrics:
    return CandidateMetrics(
        expected=sum(metric.expected for metric in metrics),
        produced=sum(metric.produced for metric in metrics),
        matched=sum(metric.matched for metric in metrics),
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)
