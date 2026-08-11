"""Strict, evidence-grounded contracts for derived corpus analysis.

The records in this module are model-produced *candidates*, not accepted facts. A
candidate is valid only when it cites deterministic evidence units that resolve to an
exact span in one CorpusDock evidence excerpt. Persisted records retain that span and
its digest, but do not copy the excerpt into the analysis database.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping, Sequence

from corpusdock.chunking import RuleSentenceProcessor
from corpusdock.contracts import EvidenceResult


ANALYSIS_CONTRACT_SCHEMA_VERSION = 3
ANALYSIS_PROMPT_VERSION = "analysis-extraction-v17"

MAX_MODEL_OUTPUT_CHARACTERS = 200_000
MAX_CANDIDATES_PER_KIND = 64
MAX_PROMPT_CONCEPTS = 8
MAX_PROMPT_CLAIMS = 8
MAX_PROMPT_RELATIONS = 3
MAX_LOCAL_ID_CHARACTERS = 80
MAX_LABEL_CHARACTERS = 240
MAX_DESCRIPTION_CHARACTERS = 1_500
MAX_STATEMENT_CHARACTERS = 2_000
MAX_PREDICATE_CHARACTERS = 160
MAX_CONCEPT_LINKS_PER_CLAIM = 24
MAX_EVIDENCE_UNITS = 128
MAX_SUPPORT_UNITS = 8

ClaimType = Literal[
    "observation",
    "definition",
    "causal",
    "recommendation",
    "comparison",
    "prediction",
    "value_judgment",
    "other",
]
Polarity = Literal["affirmed", "negated", "mixed"]
Certainty = Literal["asserted", "possible", "probable", "uncertain"]
Attribution = Literal["source", "reported", "quoted", "unclear"]
NormativeForce = Literal["none", "recommended", "required", "permitted", "prohibited"]
RelationType = Literal[
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
]
ReviewState = Literal["unreviewed", "accepted", "rejected", "needs_review"]
AnalysisStatus = Literal["accepted", "partial", "empty", "rejected"]

_CLAIM_TYPES = {
    "observation",
    "definition",
    "causal",
    "recommendation",
    "comparison",
    "prediction",
    "value_judgment",
    "other",
}
_POLARITIES = {"affirmed", "negated", "mixed"}
_CERTAINTIES = {"asserted", "possible", "probable", "uncertain"}
_ATTRIBUTIONS = {"source", "reported", "quoted", "unclear"}
_NORMATIVE_FORCES = {
    "none",
    "recommended",
    "required",
    "permitted",
    "prohibited",
}
_RELATION_TYPES = {
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
}
_TOP_LEVEL_FIELDS = {"schema_version", "concepts", "claims", "relations"}
_SUPPORT_FIELDS = {"unit_ids"}
_CONCEPT_FIELDS = {
    "local_id",
    "label",
    "description",
    "concept_type",
    "confidence",
    "support",
}
_OPTIONAL_CONCEPT_FIELDS = {"description"}
_CLAIM_FIELDS = {
    "local_id",
    "statement",
    "claim_type",
    "polarity",
    "certainty",
    "conditional",
    "attribution",
    "normative_force",
    "confidence",
    "support",
    "concept_ids",
}
_OPTIONAL_CLAIM_FIELDS = {"concept_ids"}
_RELATION_FIELDS = {
    "local_id",
    "subject_concept_id",
    "relation_type",
    "predicate",
    "object_concept_id",
    "claim_local_id",
    "confidence",
}


class AnalysisContractError(Exception):
    """A malformed, ungrounded, or internally inconsistent extraction result."""

    def __init__(self, code: str, message: str, *, path: str = "$") -> None:
        super().__init__(message)
        self.code = code
        self.path = path


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    """One exact, model-addressable support unit within an evidence excerpt."""

    unit_id: str
    start_offset: int
    end_offset: int

    def prompt_item(self, text: str) -> dict[str, str]:
        return {
            "unit_id": self.unit_id,
            "text": text[self.start_offset : self.end_offset],
        }


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    """An exact chunk-relative support span without duplicated source content."""

    span_id: str
    evidence_id: str
    start_offset: int
    end_offset: int
    text_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConceptCandidate:
    """One local concept mention awaiting cross-evidence resolution and review."""

    concept_id: str
    local_id: str
    label: str
    description: str
    concept_type: str
    confidence: float
    support: EvidenceSpan
    review_state: ReviewState = "unreviewed"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["support"] = self.support.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class ClaimCandidate:
    """A source-attributed proposition awaiting human or downstream review."""

    claim_id: str
    local_id: str
    statement: str
    claim_type: ClaimType
    polarity: Polarity
    certainty: Certainty
    conditional: bool
    attribution: Attribution
    normative_force: NormativeForce
    confidence: float
    support: EvidenceSpan
    concept_ids: tuple[str, ...]
    review_state: ReviewState = "unreviewed"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["support"] = self.support.to_dict()
        result["concept_ids"] = list(self.concept_ids)
        return result


@dataclass(frozen=True, slots=True)
class RelationCandidate:
    """A directed relation between two local concept candidates."""

    relation_id: str
    local_id: str
    claim_id: str
    subject_concept_id: str
    relation_type: RelationType
    predicate: str
    object_concept_id: str
    polarity: Polarity
    certainty: Certainty
    conditional: bool
    attribution: Attribution
    normative_force: NormativeForce
    confidence: float
    support: EvidenceSpan
    review_state: ReviewState = "unreviewed"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["support"] = self.support.to_dict()
        return result


@dataclass(frozen=True, slots=True)
class CandidateRejection:
    """A content-free reason why one model-produced candidate was discarded."""

    category: Literal["concept", "claim", "relation", "response"]
    code: str
    path: str
    local_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceAnalysis:
    """Validated candidates and validation outcomes for one exact evidence chunk."""

    run_id: str
    evidence_id: str
    chunk_id: str
    source_id: str
    status: AnalysisStatus
    raw_output_sha256: str
    concepts: tuple[ConceptCandidate, ...]
    claims: tuple[ClaimCandidate, ...]
    relations: tuple[RelationCandidate, ...]
    rejections: tuple[CandidateRejection, ...]
    inference_ms: float | None = None
    output_tokens: int | None = None
    output_truncated: bool = False

    @property
    def accepted_count(self) -> int:
        return len(self.concepts) + len(self.claims) + len(self.relations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ANALYSIS_CONTRACT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "evidence_id": self.evidence_id,
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "status": self.status,
            "raw_output_sha256": self.raw_output_sha256,
            "inference_ms": self.inference_ms,
            "output_tokens": self.output_tokens,
            "output_truncated": self.output_truncated,
            "counts": {
                "concepts": len(self.concepts),
                "claims": len(self.claims),
                "relations": len(self.relations),
                "rejected": len(self.rejections),
            },
            "concepts": [candidate.to_dict() for candidate in self.concepts],
            "claims": [candidate.to_dict() for candidate in self.claims],
            "relations": [candidate.to_dict() for candidate in self.relations],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
        }


def evidence_units_for(
    text: str, segments: Sequence[str] | None = None
) -> tuple[EvidenceUnit, ...]:
    """Create stable exact support units from reconstructing sentence segments."""

    if not isinstance(text, str) or not text or "\x00" in text:
        raise AnalysisContractError(
            "evidence_units_invalid",
            "Evidence units require non-empty Unicode source text.",
        )
    if segments is None:
        segments = RuleSentenceProcessor().split_many((text,))[0]
    if isinstance(segments, (str, bytes)):
        raise AnalysisContractError(
            "evidence_units_invalid",
            "Evidence sentence segments must be a sequence of strings.",
        )
    clean_segments = tuple(segments)
    if (
        not clean_segments
        or any(not isinstance(segment, str) for segment in clean_segments)
        or "".join(clean_segments) != text
    ):
        raise AnalysisContractError(
            "evidence_units_invalid",
            "Evidence sentence segments must reconstruct the exact source text.",
        )
    units: list[EvidenceUnit] = []
    cursor = 0
    for segment in clean_segments:
        segment_start = cursor
        cursor += len(segment)
        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment) - len(segment.rstrip())
        start = segment_start + leading
        end = cursor - trailing
        if start < end:
            units.append(EvidenceUnit(f"u{len(units) + 1}", start, end))
    if not units or len(units) > MAX_EVIDENCE_UNITS:
        raise AnalysisContractError(
            "evidence_units_invalid",
            f"Evidence must produce between 1 and {MAX_EVIDENCE_UNITS} support units.",
        )
    return tuple(units)


def parse_and_validate_analysis(
    raw_output: str,
    evidence: EvidenceResult,
    *,
    run_id: str,
    evidence_units: Sequence[EvidenceUnit] | None = None,
    inference_ms: float | None = None,
    output_tokens: int | None = None,
    output_truncated: bool = False,
) -> EvidenceAnalysis:
    """Parse strict JSON and retain only candidates grounded in ``evidence``."""

    _validate_evidence(evidence)
    units = _validated_evidence_units(evidence, evidence_units)
    unit_positions = {unit.unit_id: index for index, unit in enumerate(units)}
    if not isinstance(raw_output, str):
        raise AnalysisContractError(
            "analysis_output_invalid", "Model output must be a JSON string."
        )
    encoded = raw_output.encode("utf-8")
    output_digest = sha256(encoded).hexdigest()
    if len(raw_output) > MAX_MODEL_OUTPUT_CHARACTERS:
        raise AnalysisContractError(
            "analysis_output_too_large",
            f"Model output exceeds {MAX_MODEL_OUTPUT_CHARACTERS} characters.",
        )
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise AnalysisContractError(
            "analysis_json_invalid",
            "Model output is not one complete JSON value.",
            path=f"$ at character {error.pos}",
        ) from error
    if not isinstance(payload, dict):
        raise AnalysisContractError(
            "analysis_schema_invalid", "Analysis output must be a JSON object."
        )
    _require_exact_fields(payload, _TOP_LEVEL_FIELDS, path="$")
    if payload["schema_version"] != ANALYSIS_CONTRACT_SCHEMA_VERSION:
        raise AnalysisContractError(
            "analysis_schema_invalid",
            "Analysis output uses an unsupported schema version.",
            path="$.schema_version",
        )
    raw_concepts = _candidate_array(payload, "concepts")
    raw_claims = _candidate_array(payload, "claims")
    raw_relations = _candidate_array(payload, "relations")

    rejections: list[CandidateRejection] = []
    concepts: list[ConceptCandidate] = []
    concept_ids_by_local: dict[str, str] = {}
    seen_concept_local_ids: set[str] = set()
    for index, raw in enumerate(raw_concepts):
        path = f"$.concepts[{index}]"
        local_id = _candidate_local_id(raw)
        if local_id is not None and local_id in seen_concept_local_ids:
            rejections.append(
                CandidateRejection(
                    "concept", "local_id_duplicate", f"{path}.local_id", local_id
                )
            )
            continue
        if local_id is not None:
            seen_concept_local_ids.add(local_id)
        try:
            concept = _validate_concept(
                raw,
                evidence,
                units=units,
                unit_positions=unit_positions,
                run_id=run_id,
                path=path,
            )
        except AnalysisContractError as error:
            rejections.append(
                CandidateRejection("concept", error.code, error.path, local_id)
            )
            continue
        concepts.append(concept)
        concept_ids_by_local[concept.local_id] = concept.concept_id

    claims: list[ClaimCandidate] = []
    claims_by_local: dict[str, ClaimCandidate] = {}
    seen_claim_local_ids: set[str] = set()
    for index, raw in enumerate(raw_claims):
        path = f"$.claims[{index}]"
        local_id = _candidate_local_id(raw)
        if local_id is not None and local_id in seen_claim_local_ids:
            rejections.append(
                CandidateRejection(
                    "claim", "local_id_duplicate", f"{path}.local_id", local_id
                )
            )
            continue
        if local_id is not None:
            seen_claim_local_ids.add(local_id)
        try:
            claim = _validate_claim(
                raw,
                evidence,
                units=units,
                unit_positions=unit_positions,
                run_id=run_id,
                concept_ids_by_local=concept_ids_by_local,
                path=path,
            )
        except AnalysisContractError as error:
            rejections.append(
                CandidateRejection("claim", error.code, error.path, local_id)
            )
            continue
        claims.append(claim)
        claims_by_local[claim.local_id] = claim

    relations: list[RelationCandidate] = []
    seen_relation_local_ids: set[str] = set()
    for index, raw in enumerate(raw_relations):
        path = f"$.relations[{index}]"
        local_id = _candidate_local_id(raw)
        if local_id is not None and local_id in seen_relation_local_ids:
            rejections.append(
                CandidateRejection(
                    "relation", "local_id_duplicate", f"{path}.local_id", local_id
                )
            )
            continue
        if local_id is not None:
            seen_relation_local_ids.add(local_id)
        try:
            relations.append(
                _validate_relation(
                    raw,
                    evidence,
                    run_id=run_id,
                    concept_ids_by_local=concept_ids_by_local,
                    claims_by_local=claims_by_local,
                    path=path,
                )
            )
        except AnalysisContractError as error:
            rejections.append(
                CandidateRejection("relation", error.code, error.path, local_id)
            )

    accepted_count = len(concepts) + len(claims) + len(relations)
    requested_count = len(raw_concepts) + len(raw_claims) + len(raw_relations)
    status: AnalysisStatus
    if accepted_count and rejections:
        status = "partial"
    elif accepted_count:
        status = "accepted"
    elif requested_count:
        status = "rejected"
    else:
        status = "empty"

    return EvidenceAnalysis(
        run_id=run_id,
        evidence_id=evidence.evidence_id,
        chunk_id=evidence.chunk_id or "",
        source_id=evidence.locator.source_id,
        status=status,
        raw_output_sha256=output_digest,
        concepts=tuple(concepts),
        claims=tuple(claims),
        relations=tuple(relations),
        rejections=tuple(rejections),
        inference_ms=_optional_duration(inference_ms),
        output_tokens=_optional_token_count(output_tokens),
        output_truncated=_boolean_metadata(output_truncated),
    )


def rejection_analysis(
    raw_output: str,
    evidence: EvidenceResult,
    *,
    run_id: str,
    error: AnalysisContractError,
    inference_ms: float | None = None,
    output_tokens: int | None = None,
    output_truncated: bool = False,
) -> EvidenceAnalysis:
    """Create a content-free rejected record for an invalid whole response."""

    _validate_evidence(evidence)
    raw_bytes = raw_output.encode("utf-8") if isinstance(raw_output, str) else b""
    return EvidenceAnalysis(
        run_id=run_id,
        evidence_id=evidence.evidence_id,
        chunk_id=evidence.chunk_id or "",
        source_id=evidence.locator.source_id,
        status="rejected",
        raw_output_sha256=sha256(raw_bytes).hexdigest(),
        concepts=(),
        claims=(),
        relations=(),
        rejections=(CandidateRejection("response", error.code, error.path),),
        inference_ms=_optional_duration(inference_ms),
        output_tokens=_optional_token_count(output_tokens),
        output_truncated=_boolean_metadata(output_truncated),
    )


def _validate_concept(
    raw: object,
    evidence: EvidenceResult,
    *,
    units: Sequence[EvidenceUnit],
    unit_positions: Mapping[str, int],
    run_id: str,
    path: str,
) -> ConceptCandidate:
    item = _candidate_object(
        raw,
        _CONCEPT_FIELDS,
        optional_fields=_OPTIONAL_CONCEPT_FIELDS,
        path=path,
    )
    local_id = _local_id(item.get("local_id"), path=f"{path}.local_id")
    label = _bounded_text(item.get("label"), MAX_LABEL_CHARACTERS, path=f"{path}.label")
    description = _bounded_text(
        item.get("description", ""),
        MAX_DESCRIPTION_CHARACTERS,
        path=f"{path}.description",
        allow_empty=True,
    )
    concept_type = _bounded_text(
        item.get("concept_type"), 100, path=f"{path}.concept_type"
    )
    confidence = _confidence(item.get("confidence"), path=f"{path}.confidence")
    support = _support_span(
        item.get("support"),
        evidence,
        units=units,
        unit_positions=unit_positions,
        path=f"{path}.support",
    )
    concept_id = _derived_id(
        "cm",
        run_id,
        evidence.evidence_id,
        local_id,
        label,
        description,
        concept_type,
        str(support.start_offset),
        str(support.end_offset),
    )
    return ConceptCandidate(
        concept_id=concept_id,
        local_id=local_id,
        label=label,
        description=description,
        concept_type=concept_type,
        confidence=confidence,
        support=support,
    )


def _validate_claim(
    raw: object,
    evidence: EvidenceResult,
    *,
    units: Sequence[EvidenceUnit],
    unit_positions: Mapping[str, int],
    run_id: str,
    concept_ids_by_local: Mapping[str, str],
    path: str,
) -> ClaimCandidate:
    item = _candidate_object(
        raw,
        _CLAIM_FIELDS,
        optional_fields=_OPTIONAL_CLAIM_FIELDS,
        path=path,
    )
    local_id = _local_id(item.get("local_id"), path=f"{path}.local_id")
    statement = _bounded_text(
        item.get("statement"), MAX_STATEMENT_CHARACTERS, path=f"{path}.statement"
    )
    claim_type = _enum_value(
        item.get("claim_type"), _CLAIM_TYPES, path=f"{path}.claim_type"
    )
    polarity = _enum_value(item.get("polarity"), _POLARITIES, path=f"{path}.polarity")
    certainty = _enum_value(
        item.get("certainty"), _CERTAINTIES, path=f"{path}.certainty"
    )
    conditional = _boolean(item.get("conditional"), path=f"{path}.conditional")
    attribution = _enum_value(
        item.get("attribution"), _ATTRIBUTIONS, path=f"{path}.attribution"
    )
    normative_force = _enum_value(
        item.get("normative_force"),
        _NORMATIVE_FORCES,
        path=f"{path}.normative_force",
    )
    confidence = _confidence(item.get("confidence"), path=f"{path}.confidence")
    support = _support_span(
        item.get("support"),
        evidence,
        units=units,
        unit_positions=unit_positions,
        path=f"{path}.support",
    )
    raw_links = item.get("concept_ids", [])
    if not isinstance(raw_links, list) or len(raw_links) > MAX_CONCEPT_LINKS_PER_CLAIM:
        raise AnalysisContractError(
            "concept_links_invalid",
            "Claim concept_ids must be a bounded JSON array.",
            path=f"{path}.concept_ids",
        )
    local_links: list[str] = []
    for link_index, raw_link in enumerate(raw_links):
        link = _local_id(raw_link, path=f"{path}.concept_ids[{link_index}]")
        if link not in concept_ids_by_local:
            raise AnalysisContractError(
                "concept_reference_invalid",
                "Claim refers to a concept that was not accepted in this response.",
                path=f"{path}.concept_ids[{link_index}]",
            )
        if link in local_links:
            raise AnalysisContractError(
                "concept_reference_duplicate",
                "Claim contains a duplicate concept reference.",
                path=f"{path}.concept_ids[{link_index}]",
            )
        local_links.append(link)
    concept_ids = tuple(concept_ids_by_local[link] for link in local_links)
    claim_id = _derived_id(
        "clm",
        run_id,
        evidence.evidence_id,
        local_id,
        statement,
        claim_type,
        polarity,
        certainty,
        str(conditional),
        attribution,
        normative_force,
        str(support.start_offset),
        str(support.end_offset),
        *concept_ids,
    )
    return ClaimCandidate(
        claim_id=claim_id,
        local_id=local_id,
        statement=statement,
        claim_type=claim_type,  # type: ignore[arg-type]
        polarity=polarity,  # type: ignore[arg-type]
        certainty=certainty,  # type: ignore[arg-type]
        conditional=conditional,
        attribution=attribution,  # type: ignore[arg-type]
        normative_force=normative_force,  # type: ignore[arg-type]
        confidence=confidence,
        support=support,
        concept_ids=concept_ids,
    )


def _validate_relation(
    raw: object,
    evidence: EvidenceResult,
    *,
    run_id: str,
    concept_ids_by_local: Mapping[str, str],
    claims_by_local: Mapping[str, ClaimCandidate],
    path: str,
) -> RelationCandidate:
    item = _candidate_object(raw, _RELATION_FIELDS, path=path)
    local_id = _local_id(item.get("local_id"), path=f"{path}.local_id")
    subject_local = _local_id(
        item.get("subject_concept_id"), path=f"{path}.subject_concept_id"
    )
    object_local = _local_id(
        item.get("object_concept_id"), path=f"{path}.object_concept_id"
    )
    claim_local = _local_id(item.get("claim_local_id"), path=f"{path}.claim_local_id")
    if subject_local == object_local:
        raise AnalysisContractError(
            "relation_self_reference",
            "A relation must connect two distinct concept candidates.",
            path=path,
        )
    if (
        subject_local not in concept_ids_by_local
        or object_local not in concept_ids_by_local
    ):
        raise AnalysisContractError(
            "concept_reference_invalid",
            "Relation refers to a concept that was not accepted in this response.",
            path=path,
        )
    if claim_local not in claims_by_local:
        raise AnalysisContractError(
            "claim_reference_invalid",
            "Relation refers to a claim that was not accepted in this response.",
            path=f"{path}.claim_local_id",
        )
    relation_type = _enum_value(
        item.get("relation_type"),
        _RELATION_TYPES,
        path=f"{path}.relation_type",
    )
    predicate = _bounded_text(
        item.get("predicate"),
        MAX_PREDICATE_CHARACTERS,
        path=f"{path}.predicate",
    )
    confidence = _confidence(item.get("confidence"), path=f"{path}.confidence")
    claim = claims_by_local[claim_local]
    support = claim.support
    subject_id = concept_ids_by_local[subject_local]
    object_id = concept_ids_by_local[object_local]
    relation_id = _derived_id(
        "rel",
        run_id,
        evidence.evidence_id,
        local_id,
        subject_id,
        relation_type,
        predicate,
        object_id,
        claim.claim_id,
        str(support.start_offset),
        str(support.end_offset),
    )
    return RelationCandidate(
        relation_id=relation_id,
        local_id=local_id,
        claim_id=claim.claim_id,
        subject_concept_id=subject_id,
        relation_type=relation_type,  # type: ignore[arg-type]
        predicate=predicate,
        object_concept_id=object_id,
        polarity=claim.polarity,
        certainty=claim.certainty,
        conditional=claim.conditional,
        attribution=claim.attribution,
        normative_force=claim.normative_force,
        confidence=confidence,
        support=support,
    )


def _support_span(
    raw: object,
    evidence: EvidenceResult,
    *,
    units: Sequence[EvidenceUnit],
    unit_positions: Mapping[str, int],
    path: str,
) -> EvidenceSpan:
    if not isinstance(raw, dict):
        raise AnalysisContractError(
            "support_invalid", "Candidate support must be a JSON object.", path=path
        )
    _require_exact_fields(raw, _SUPPORT_FIELDS, path=path)
    raw_unit_ids = raw.get("unit_ids")
    if (
        not isinstance(raw_unit_ids, list)
        or not raw_unit_ids
        or len(raw_unit_ids) > MAX_SUPPORT_UNITS
    ):
        raise AnalysisContractError(
            "support_units_invalid",
            f"Support must cite between 1 and {MAX_SUPPORT_UNITS} evidence units.",
            path=f"{path}.unit_ids",
        )
    unit_ids: list[str] = []
    positions: list[int] = []
    for index, raw_unit_id in enumerate(raw_unit_ids):
        unit_id = _local_id(raw_unit_id, path=f"{path}.unit_ids[{index}]")
        if unit_id not in unit_positions:
            raise AnalysisContractError(
                "support_unit_unknown",
                "Candidate support refers to an unknown evidence unit.",
                path=f"{path}.unit_ids[{index}]",
            )
        if unit_id in unit_ids:
            raise AnalysisContractError(
                "support_unit_duplicate",
                "Candidate support repeats an evidence unit.",
                path=f"{path}.unit_ids[{index}]",
            )
        unit_ids.append(unit_id)
        positions.append(unit_positions[unit_id])
    if positions != sorted(positions):
        raise AnalysisContractError(
            "support_units_out_of_order",
            "Support units must be listed in source order.",
            path=f"{path}.unit_ids",
        )
    if positions[-1] - positions[0] + 1 > MAX_SUPPORT_UNITS:
        raise AnalysisContractError(
            "support_range_too_large",
            f"Support cannot span more than {MAX_SUPPORT_UNITS} evidence units.",
            path=path,
        )
    start = units[positions[0]].start_offset
    end = units[positions[-1]].end_offset
    support_text = evidence.excerpt[start:end]
    text_digest = sha256(support_text.encode("utf-8")).hexdigest()
    span_id = _derived_id(
        "esp", evidence.evidence_id, str(start), str(end), text_digest
    )
    return EvidenceSpan(
        span_id=span_id,
        evidence_id=evidence.evidence_id,
        start_offset=start,
        end_offset=end,
        text_sha256=text_digest,
    )


def _validate_evidence(evidence: EvidenceResult) -> None:
    if not isinstance(evidence, EvidenceResult):
        raise AnalysisContractError(
            "evidence_invalid", "Analysis requires a CorpusDock evidence record."
        )
    if not evidence.evidence_id or not evidence.chunk_id or not evidence.excerpt:
        raise AnalysisContractError(
            "evidence_invalid",
            "Evidence must have stable evidence and chunk IDs plus a non-empty excerpt.",
        )
    if not evidence.locator.source_id:
        raise AnalysisContractError(
            "evidence_invalid", "Evidence must retain a stable source ID."
        )


def _validated_evidence_units(
    evidence: EvidenceResult, units: Sequence[EvidenceUnit] | None
) -> tuple[EvidenceUnit, ...]:
    result = evidence_units_for(evidence.excerpt) if units is None else tuple(units)
    if not result or len(result) > MAX_EVIDENCE_UNITS:
        raise AnalysisContractError(
            "evidence_units_invalid",
            f"Evidence must contain between 1 and {MAX_EVIDENCE_UNITS} support units.",
        )
    previous_end = -1
    for index, unit in enumerate(result):
        if (
            not isinstance(unit, EvidenceUnit)
            or unit.unit_id != f"u{index + 1}"
            or unit.start_offset < 0
            or unit.end_offset <= unit.start_offset
            or unit.end_offset > len(evidence.excerpt)
            or unit.start_offset < previous_end
            or not evidence.excerpt[unit.start_offset : unit.end_offset].strip()
        ):
            raise AnalysisContractError(
                "evidence_units_invalid",
                "Evidence units must be ordered, non-overlapping exact excerpt spans.",
            )
        previous_end = unit.end_offset
    return result


def _candidate_array(payload: Mapping[str, object], name: str) -> list[object]:
    value = payload.get(name)
    if not isinstance(value, list):
        raise AnalysisContractError(
            "analysis_schema_invalid",
            f"Analysis field '{name}' must be an array.",
            path=f"$.{name}",
        )
    if len(value) > MAX_CANDIDATES_PER_KIND:
        raise AnalysisContractError(
            "analysis_candidate_limit_exceeded",
            f"Analysis field '{name}' exceeds {MAX_CANDIDATES_PER_KIND} items.",
            path=f"$.{name}",
        )
    return value


def _candidate_object(
    raw: object,
    expected_fields: set[str],
    *,
    optional_fields: set[str] | None = None,
    path: str,
) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise AnalysisContractError(
            "candidate_invalid", "Candidate must be a JSON object.", path=path
        )
    optional = optional_fields or set()
    fields = set(raw)
    missing = sorted((expected_fields - optional) - fields)
    unknown = sorted(fields - expected_fields)
    if missing:
        raise AnalysisContractError(
            "field_missing",
            f"Analysis schema mismatch (missing fields: {', '.join(missing)}).",
            path=path,
        )
    if unknown:
        raise AnalysisContractError(
            "field_unknown",
            f"Analysis schema mismatch (unknown fields: {', '.join(unknown)}).",
            path=path,
        )
    return raw


def _candidate_local_id(raw: object) -> str | None:
    if not isinstance(raw, dict):
        return None
    value = raw.get("local_id")
    return value.strip() if isinstance(value, str) else None


def _require_exact_fields(
    value: Mapping[str, object], expected: set[str], *, path: str
) -> None:
    fields = set(value)
    if fields == expected:
        return
    missing = sorted(expected - fields)
    unknown = sorted(fields - expected)
    if missing:
        code = "field_missing"
        detail = f"missing fields: {', '.join(missing)}"
    else:
        code = "field_unknown"
        detail = f"unknown fields: {', '.join(unknown)}"
    raise AnalysisContractError(
        code, f"Analysis schema mismatch ({detail}).", path=path
    )


def _local_id(value: object, *, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.strip()) > MAX_LOCAL_ID_CHARACTERS
        or "\x00" in value
    ):
        raise AnalysisContractError(
            "local_id_invalid",
            f"Local ID must be a string up to {MAX_LOCAL_ID_CHARACTERS} characters.",
            path=path,
        )
    return value.strip()


def _bounded_text(
    value: object,
    maximum: int,
    *,
    path: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise AnalysisContractError(
            "text_invalid", "Expected a string value.", path=path
        )
    clean = value.strip()
    if (not clean and not allow_empty) or len(clean) > maximum or "\x00" in clean:
        raise AnalysisContractError(
            "text_invalid",
            f"Text must contain at most {maximum} characters.",
            path=path,
        )
    return clean


def _confidence(value: object, *, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise AnalysisContractError(
            "confidence_invalid", "Confidence must be between 0 and 1.", path=path
        )
    return round(float(value), 6)


def _enum_value(value: object, allowed: set[str], *, path: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise AnalysisContractError(
            "enum_invalid",
            f"Value must be one of: {', '.join(sorted(allowed))}.",
            path=path,
        )
    return value


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise AnalysisContractError(
            "boolean_invalid", "Value must be true or false.", path=path
        )
    return value


def _derived_id(prefix: str, *parts: str) -> str:
    digest = sha256()
    digest.update(f"corpusdock-{prefix}-v1".encode("utf-8"))
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"{prefix}-{digest.hexdigest()}"


def _optional_duration(value: float | None) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise AnalysisContractError(
            "inference_duration_invalid",
            "Inference duration must be a non-negative number.",
        )
    return round(float(value), 6)


def _optional_token_count(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AnalysisContractError(
            "output_tokens_invalid",
            "Generated output token count must be a non-negative integer.",
        )
    return value


def _boolean_metadata(value: bool) -> bool:
    if not isinstance(value, bool):
        raise AnalysisContractError(
            "output_truncation_invalid", "Output truncation metadata must be boolean."
        )
    return value


def analysis_output_template() -> dict[str, object]:
    """Return the exact JSON shape supplied to local extraction models."""

    return {
        "schema_version": ANALYSIS_CONTRACT_SCHEMA_VERSION,
        "concepts": [
            {
                "local_id": "c1",
                "label": "short normalized label",
                "concept_type": "open vocabulary type",
                "confidence": 0.0,
                "support": {"unit_ids": ["u1"]},
            },
            {
                "local_id": "c2",
                "label": "another normalized label",
                "concept_type": "open vocabulary type",
                "confidence": 0.0,
                "support": {"unit_ids": ["u1"]},
            },
        ],
        "claims": [
            {
                "local_id": "q1",
                "statement": "faithful standalone paraphrase",
                "claim_type": "observation",
                "polarity": "affirmed",
                "certainty": "asserted",
                "conditional": False,
                "attribution": "source",
                "normative_force": "none",
                "confidence": 0.0,
                "support": {"unit_ids": ["u1"]},
                "concept_ids": ["c1"],
            }
        ],
        "relations": [
            {
                "local_id": "r1",
                "subject_concept_id": "c1",
                "relation_type": "associated_with",
                "predicate": "short directed relation",
                "object_concept_id": "c2",
                "claim_local_id": "q1",
                "confidence": 0.0,
            }
        ],
    }


def analysis_json_schema() -> dict[str, object]:
    """Return the strict model-output JSON Schema with semantic field guidance."""

    support = {
        "type": "object",
        "additionalProperties": False,
        "required": ["unit_ids"],
        "properties": {
            "unit_ids": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_SUPPORT_UNITS,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 2, "maxLength": 80},
                "description": "Shortest contiguous source-unit range that supports the candidate.",
            }
        },
    }
    stance = {
        "polarity": {
            "enum": ["affirmed", "negated", "mixed"],
            "description": "Whether the proposition is affirmed or negated.",
        },
        "certainty": {
            "enum": ["asserted", "possible", "probable", "uncertain"],
            "description": "Epistemic certainty expressed by the passage.",
        },
        "conditional": {
            "type": "boolean",
            "description": "True when the proposition depends on an explicit condition.",
        },
        "attribution": {
            "enum": ["source", "reported", "quoted", "unclear"],
            "description": "source for the passage voice; reported/quoted for another voice.",
        },
        "normative_force": {
            "enum": ["none", "recommended", "required", "permitted", "prohibited"],
            "description": "Normative force; must/shall is required, should is recommended.",
        },
    }
    concept = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_CONCEPT_FIELDS - _OPTIONAL_CONCEPT_FIELDS),
        "properties": {
            "local_id": {"type": "string", "minLength": 1, "maxLength": 80},
            "label": {"type": "string", "minLength": 1, "maxLength": 240},
            "concept_type": {"type": "string", "minLength": 1, "maxLength": 100},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "support": support,
        },
    }
    claim_properties: dict[str, object] = {
        "local_id": {"type": "string", "minLength": 1, "maxLength": 80},
        "statement": {
            "type": "string",
            "minLength": 1,
            "maxLength": 2000,
            "description": "Standalone affirmative-base proposition; polarity carries negation.",
        },
        "claim_type": {
            "enum": sorted(_CLAIM_TYPES),
            "description": "causal for stated influence; recommendation for guidance.",
        },
        **stance,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "support": support,
        "concept_ids": {
            "type": "array",
            "maxItems": MAX_CONCEPT_LINKS_PER_CLAIM,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 80},
        },
    }
    relation_properties: dict[str, object] = {
        "local_id": {"type": "string", "minLength": 1, "maxLength": 80},
        "subject_concept_id": {"type": "string", "minLength": 1, "maxLength": 80},
        "relation_type": {"enum": sorted(_RELATION_TYPES)},
        "predicate": {"type": "string", "minLength": 1, "maxLength": 160},
        "object_concept_id": {"type": "string", "minLength": 1, "maxLength": 80},
        "claim_local_id": {"type": "string", "minLength": 1, "maxLength": 80},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "concepts", "claims", "relations"],
        "properties": {
            "schema_version": {
                "type": "integer",
                "minimum": ANALYSIS_CONTRACT_SCHEMA_VERSION,
                "maximum": ANALYSIS_CONTRACT_SCHEMA_VERSION,
            },
            "concepts": {
                "type": "array",
                "maxItems": MAX_PROMPT_CONCEPTS,
                "items": concept,
            },
            "claims": {
                "type": "array",
                "maxItems": MAX_PROMPT_CLAIMS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_CLAIM_FIELDS - _OPTIONAL_CLAIM_FIELDS),
                    "properties": claim_properties,
                },
            },
            "relations": {
                "type": "array",
                "maxItems": MAX_PROMPT_RELATIONS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_RELATION_FIELDS),
                    "properties": relation_properties,
                },
            },
        },
    }
