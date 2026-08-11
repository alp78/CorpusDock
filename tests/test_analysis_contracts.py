from __future__ import annotations

from hashlib import sha256
import json

import pytest

from corpusdock.analysis_contracts import (
    AnalysisContractError,
    analysis_output_template,
    parse_and_validate_analysis,
    rejection_analysis,
)
from corpusdock.contracts import CitationLocator, EvidenceResult


RUN_ID = "run-" + "a" * 64
EVIDENCE_ID = "ev-" + "b" * 64
CHUNK_ID = "chk-" + "c" * 64
SOURCE_ID = "src-" + "d" * 64
EXCERPT = (
    "A sealed ledger reduces moisture damage during storage. "
    "The guide does not claim that labels prevent every error. "
    "If humidity rises, operators should inspect the cabinet twice daily. "
    "A marker repeats. A marker repeats."
)


def _evidence() -> EvidenceResult:
    locator = CitationLocator(
        source_id=SOURCE_ID,
        locator_type="text_lines",
        label="lines 1-4",
        line_start=1,
        line_end=4,
        start_offset=0,
        end_offset=len(EXCERPT),
    )
    return EvidenceResult(
        evidence_id=EVIDENCE_ID,
        excerpt=EXCERPT,
        citation="Project-authored fixture, lines 1-4",
        locator=locator,
        source_path="/private/fixture.txt",
        verification_status="artifact-anchor-confirmed",
        chunk_id=CHUNK_ID,
        locators=(locator,),
    )


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": 3,
        "concepts": [
            {
                "local_id": "c1",
                "label": "sealed ledger",
                "description": "A storage record kept closed.",
                "concept_type": "artifact",
                "confidence": 0.96,
                "support": {"unit_ids": ["u1"]},
            },
            {
                "local_id": "c2",
                "label": "moisture damage",
                "description": "Damage associated with storage moisture.",
                "concept_type": "risk",
                "confidence": 0.94,
                "support": {"unit_ids": ["u1"]},
            },
        ],
        "claims": [
            {
                "local_id": "q1",
                "statement": "A sealed ledger reduces storage moisture damage.",
                "claim_type": "causal",
                "polarity": "affirmed",
                "certainty": "asserted",
                "conditional": False,
                "attribution": "source",
                "normative_force": "none",
                "confidence": 0.95,
                "support": {"unit_ids": ["u1"]},
                "concept_ids": ["c1", "c2"],
            },
            {
                "local_id": "q2",
                "statement": "The guide rejects a universal error-prevention claim for labels.",
                "claim_type": "observation",
                "polarity": "negated",
                "certainty": "asserted",
                "conditional": False,
                "attribution": "source",
                "normative_force": "none",
                "confidence": 0.91,
                "support": {"unit_ids": ["u2"]},
                "concept_ids": [],
            },
        ],
        "relations": [
            {
                "local_id": "r1",
                "subject_concept_id": "c1",
                "relation_type": "inhibits",
                "predicate": "reduces",
                "object_concept_id": "c2",
                "claim_local_id": "q1",
                "confidence": 0.93,
            }
        ],
    }


def test_analysis_contract_derives_exact_spans_without_persisting_unit_ids() -> None:
    raw = json.dumps(_valid_payload())

    result = parse_and_validate_analysis(
        raw, _evidence(), run_id=RUN_ID, inference_ms=7.5
    )

    assert result.status == "accepted"
    assert len(result.concepts) == 2
    assert len(result.claims) == 2
    assert len(result.relations) == 1
    assert result.raw_output_sha256 == sha256(raw.encode()).hexdigest()
    assert result.inference_ms == 7.5
    first = result.claims[0]
    quote = "A sealed ledger reduces moisture damage during storage."
    assert EXCERPT[first.support.start_offset : first.support.end_offset] == quote
    assert first.support.text_sha256 == sha256(quote.encode()).hexdigest()
    assert first.concept_ids == tuple(concept.concept_id for concept in result.concepts)
    assert result.relations[0].subject_concept_id == result.concepts[0].concept_id
    assert result.relations[0].claim_id == result.claims[0].claim_id
    assert result.relations[0].support == result.claims[0].support
    assert "unit_ids" not in result.to_dict()["claims"][0]["support"]
    assert result.claims[1].polarity == "negated"


def test_claim_concept_links_are_optional_without_inventing_links() -> None:
    payload = _valid_payload()
    claims = payload["claims"]
    assert isinstance(claims, list)
    claims[0].pop("concept_ids")

    result = parse_and_validate_analysis(
        json.dumps(payload), _evidence(), run_id=RUN_ID
    )

    assert result.status == "accepted"
    assert result.claims[0].concept_ids == ()


def test_unit_ranges_disambiguate_repeated_source_text() -> None:
    payload = _valid_payload()
    payload["concepts"] = [
        {
            "local_id": "c1",
            "label": "marker",
            "description": "A repeated marker.",
            "concept_type": "artifact",
            "confidence": 0.8,
            "support": {"unit_ids": ["u4"]},
        }
    ]
    payload["claims"] = []
    payload["relations"] = []

    result = parse_and_validate_analysis(
        json.dumps(payload), _evidence(), run_id=RUN_ID
    )

    assert result.status == "accepted"
    span = result.concepts[0].support
    assert EXCERPT[span.start_offset : span.end_offset] == "A marker repeats."


def test_support_units_must_follow_source_order() -> None:
    payload = _valid_payload()
    payload["concepts"] = [
        {
            "local_id": "c1",
            "label": "reversed support",
            "concept_type": "invalid",
            "confidence": 0.8,
            "support": {"unit_ids": ["u2", "u1"]},
        }
    ]
    payload["claims"] = []
    payload["relations"] = []

    result = parse_and_validate_analysis(
        json.dumps(payload), _evidence(), run_id=RUN_ID
    )

    assert result.status == "rejected"
    assert result.rejections[0].code == "support_units_out_of_order"


def test_duplicate_local_ids_are_compared_after_whitespace_normalization() -> None:
    payload = _valid_payload()
    duplicate = dict(payload["concepts"][0])  # type: ignore[index]
    duplicate["local_id"] = " c1 "
    payload["concepts"].append(duplicate)  # type: ignore[union-attr]

    result = parse_and_validate_analysis(
        json.dumps(payload), _evidence(), run_id=RUN_ID
    )

    assert result.status == "partial"
    assert len(result.concepts) == 2
    assert result.rejections[-1].category == "concept"
    assert result.rejections[-1].code == "local_id_duplicate"


def test_support_ids_resolve_to_one_continuous_exact_range() -> None:
    payload = _valid_payload()
    payload["concepts"] = [
        {
            "local_id": "c1",
            "label": "transition",
            "description": "Text spanning a sentence boundary.",
            "concept_type": "textual",
            "confidence": 0.8,
            "support": {"unit_ids": ["u1", "u3"]},
        }
    ]
    payload["claims"] = []
    payload["relations"] = []

    result = parse_and_validate_analysis(
        json.dumps(payload), _evidence(), run_id=RUN_ID
    )

    span = result.concepts[0].support
    assert EXCERPT[span.start_offset : span.end_offset] == (
        "A sealed ledger reduces moisture damage during storage. "
        "The guide does not claim that labels prevent every error. "
        "If humidity rises, operators should inspect the cabinet twice daily."
    )


def test_ungrounded_candidates_and_dangling_links_are_rejected_individually() -> None:
    payload = _valid_payload()
    concepts = payload["concepts"]
    assert isinstance(concepts, list)
    concepts[0]["support"] = {"unit_ids": ["u999"]}  # type: ignore[index]

    result = parse_and_validate_analysis(
        json.dumps(payload), _evidence(), run_id=RUN_ID
    )

    assert result.status == "partial"
    assert [concept.local_id for concept in result.concepts] == ["c2"]
    assert [claim.local_id for claim in result.claims] == ["q2"]
    assert result.relations == ()
    assert [rejection.code for rejection in result.rejections] == [
        "support_unit_unknown",
        "concept_reference_invalid",
        "concept_reference_invalid",
    ]


def test_invalid_top_level_json_is_a_whole_response_error() -> None:
    with pytest.raises(AnalysisContractError) as malformed:
        parse_and_validate_analysis("```json\n{}\n```", _evidence(), run_id=RUN_ID)
    assert malformed.value.code == "analysis_json_invalid"

    payload = _valid_payload()
    payload["summary"] = "not part of the contract"
    with pytest.raises(AnalysisContractError) as unknown:
        parse_and_validate_analysis(json.dumps(payload), _evidence(), run_id=RUN_ID)
    assert unknown.value.code == "field_unknown"


def test_whole_response_failure_can_be_recorded_without_raw_content() -> None:
    error = AnalysisContractError(
        "analysis_json_invalid", "Invalid output.", path="$ at character 2"
    )

    result = rejection_analysis(
        "not json", _evidence(), run_id=RUN_ID, error=error, inference_ms=1.25
    )

    assert result.status == "rejected"
    assert result.accepted_count == 0
    assert result.rejections[0].category == "response"
    assert result.rejections[0].code == "analysis_json_invalid"
    assert "not json" not in json.dumps(result.to_dict())


def test_output_template_matches_the_strict_top_level_contract() -> None:
    assert set(analysis_output_template()) == {
        "schema_version",
        "concepts",
        "claims",
        "relations",
    }
