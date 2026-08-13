from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import pytest

from corpusdock.analysis_contracts import EvidenceAnalysis, parse_and_validate_analysis
from corpusdock.analysis_store import AnalysisStore
from corpusdock.chunking import (
    RuleSentenceProcessor,
    chunk_extraction_artifact,
    write_chunk_artifact,
)
from corpusdock.cli import build_parser, main
from corpusdock.concept_graph import (
    AUTOMATIC_RUN_SELECTION,
    CONCEPT_RESOLUTION_POLICY,
    ConceptGraphError,
    build_concept_graph,
    concept_graph_path_for,
    concept_graph_status_report,
    query_concept_graph,
    read_current_concept_graph_descriptor,
)
from corpusdock.extraction import extract_source, write_extraction_artifact
from corpusdock.manifest import ManifestStore
from corpusdock.retrieval import build_search_index


TIMESTAMP = "2026-08-13T12:00:00Z"
PROMPT_SHA = "f" * 64
MODEL_INFO = {
    "provider": "fixture_local",
    "runtime": "fixture-runtime",
    "runtime_version": "1",
    "model_id": "project-authored-fixture",
    "model_revision": "fixture-v1",
    "model_fingerprint": "sha256:" + "e" * 64,
    "device": "cpu",
    "remote_code_trusted": False,
}


def _add_source(project_root: Path, source_path: Path, text: str) -> None:
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(text, encoding="utf-8")
    registration = ManifestStore(project_root, now=lambda: TIMESTAMP).register(
        [source_path]
    )[0]
    extraction = extract_source(
        registration.source,
        registration.source_path,
        now=lambda: TIMESTAMP,
    )
    write_extraction_artifact(project_root, extraction)
    chunks = chunk_extraction_artifact(
        extraction.to_dict(), RuleSentenceProcessor(), now=lambda: TIMESTAMP
    )
    write_chunk_artifact(project_root, chunks)


def _payload(*, second: bool) -> str:
    latch_label = "brass latch" if second else "Brass Latch."
    effect_label = "shelf vibration" if second else "cabinet vibration"
    concepts: list[dict[str, object]] = [
        {
            "local_id": "c1",
            "label": latch_label,
            "description": "A closure component used during transport.",
            "concept_type": "component" if second else "Component",
            "confidence": 0.96,
            "support": {"unit_ids": ["u1"]},
        },
        {
            "local_id": "c2",
            "label": effect_label,
            "description": "Movement affecting stored equipment.",
            "concept_type": "effect",
            "confidence": 0.93,
            "support": {"unit_ids": ["u1"]},
        },
    ]
    if second:
        concepts.append(
            {
                "local_id": "c3",
                "label": "brass latch",
                "description": "A procedural checkpoint with the same surface label.",
                "concept_type": "procedure",
                "confidence": 0.71,
                "support": {"unit_ids": ["u1"]},
            }
        )
    return json.dumps(
        {
            "schema_version": 3,
            "concepts": concepts,
            "claims": [
                {
                    "local_id": "q1",
                    "statement": "The latch reduces vibration during transport.",
                    "claim_type": "causal",
                    "polarity": "affirmed",
                    "certainty": "asserted",
                    "conditional": False,
                    "attribution": "source",
                    "normative_force": "none",
                    "confidence": 0.94,
                    "support": {"unit_ids": ["u1"]},
                    "concept_ids": ["c1", "c2"],
                }
            ],
            "relations": [
                {
                    "local_id": "r1",
                    "subject_concept_id": "c1",
                    "relation_type": "inhibits",
                    "predicate": "reduces",
                    "object_concept_id": "c2",
                    "claim_local_id": "q1",
                    "confidence": 0.92,
                }
            ],
        }
    )


def _build_project(
    tmp_path: Path,
) -> tuple[Path, AnalysisStore, str, tuple[EvidenceAnalysis, ...]]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _add_source(
        project_root,
        tmp_path / "documents" / "cabinet.txt",
        "A Brass Latch reduces cabinet vibration during transport.\n",
    )
    _add_source(
        project_root,
        tmp_path / "documents" / "shelf.txt",
        "The brass latch limits shelf vibration in transit.\n",
    )
    build_search_index(project_root, now=lambda: TIMESTAMP)
    store = AnalysisStore(project_root)
    run = store.begin_run(
        MODEL_INFO,
        prompt_version="analysis-extraction-v1",
        prompt_sha256=PROMPT_SHA,
        scope={"source_ids": [], "limit": None, "selected_evidence": 2},
        now=lambda: TIMESTAMP,
    )
    analyses = tuple(
        parse_and_validate_analysis(
            _payload(second="shelf" in evidence.excerpt.casefold()),
            evidence,
            run_id=run.run_id,
        )
        for evidence in store.snapshot.evidence
    )
    store.write_batch(run.run_id, analyses)
    store.finish_run(run.run_id, now=lambda: "2026-08-13T12:01:00Z")
    return project_root, store, run.run_id, analyses


def test_concept_graph_resolves_safe_equivalents_and_preserves_exact_support(
    tmp_path: Path,
) -> None:
    project_root, _, run_id, _ = _build_project(tmp_path)

    descriptor = build_concept_graph(project_root, now=lambda: "2026-08-13T12:02:00Z")

    assert descriptor.analysis_run_id == run_id
    assert descriptor.analysis_run_selection == AUTOMATIC_RUN_SELECTION
    assert descriptor.resolution_policy == CONCEPT_RESOLUTION_POLICY
    assert descriptor.indexed_sources == 2
    assert descriptor.represented_sources == 2
    assert descriptor.analyzed_evidence == 2
    assert descriptor.concept_mentions == 5
    assert descriptor.concepts == 4
    assert descriptor.claims == 2
    assert descriptor.claim_concept_links == 4
    assert descriptor.relations == 2

    response = query_concept_graph(
        project_root, "brass latch", limit=5, evidence_limit=3
    )
    assert len(response.results) == 2  # Same label, incompatible types stay distinct.
    resolved = response.results[0]
    assert resolved.canonical_label == "brass latch"
    assert resolved.canonical_type in {"component", "Component"}
    assert resolved.mentions == 2
    assert resolved.sources == 2
    assert resolved.claims == 2
    assert resolved.relations == 2
    assert len(resolved.support) == 2
    assert len({item.evidence.locator.source_id for item in resolved.support}) == 2
    for support in resolved.support:
        assert support.text
        assert sha256(support.text.encode()).hexdigest() == support.text_sha256
        assert support.evidence.citation
        assert support.evidence.verification_status == "artifact-anchor-confirmed"
    assert resolved.neighbors
    assert resolved.neighbors[0].direction == "outgoing"
    assert resolved.neighbors[0].relation_type == "inhibits"

    graph_bytes = concept_graph_path_for(project_root).read_bytes()
    assert str(tmp_path / "documents" / "cabinet.txt").encode() not in graph_bytes
    assert str(tmp_path / "documents" / "shelf.txt").encode() not in graph_bytes
    assert (
        b"A Brass Latch reduces cabinet vibration during transport.\n"
        not in graph_bytes
    )
    assert b"The brass latch limits shelf vibration in transit.\n" not in graph_bytes


def test_graph_excludes_rejected_candidates_and_stales_after_review(
    tmp_path: Path,
) -> None:
    project_root, store, _, analyses = _build_project(tmp_path)
    procedure = next(
        concept
        for analysis in analyses
        for concept in analysis.concepts
        if concept.concept_type == "procedure"
    )
    store.record_review(
        "concept",
        procedure.concept_id,
        "rejected",
        now=lambda: "2026-08-13T12:01:30Z",
    )
    descriptor = build_concept_graph(project_root)
    assert descriptor.concept_mentions == 4
    assert descriptor.concepts == 3
    assert descriptor.excluded_concepts == 1

    accepted = next(
        concept
        for analysis in analyses
        for concept in analysis.concepts
        if concept.concept_type.casefold() == "component"
    )
    store.record_review(
        "concept",
        accepted.concept_id,
        "accepted",
        now=lambda: "2026-08-13T12:03:00Z",
    )

    assert concept_graph_status_report(project_root)["status"] == "stale"
    with pytest.raises(ConceptGraphError) as stale:
        read_current_concept_graph_descriptor(project_root)
    assert stale.value.code == "concept_graph_stale"


def test_automatic_graph_run_selection_prefers_coverage_over_recency(
    tmp_path: Path,
) -> None:
    project_root, store, full_run_id, _ = _build_project(tmp_path)
    pilot_model = {**MODEL_INFO, "model_fingerprint": "sha256:" + "d" * 64}
    pilot = store.begin_run(
        pilot_model,
        prompt_version="analysis-extraction-v1",
        prompt_sha256=PROMPT_SHA,
        scope={"source_ids": [], "limit": 1, "selected_evidence": 1},
        now=lambda: "2026-08-13T12:04:00Z",
    )
    evidence = store.snapshot.evidence[0]
    store.write_batch(
        pilot.run_id,
        [
            parse_and_validate_analysis(
                _payload(second="shelf" in evidence.excerpt.casefold()),
                evidence,
                run_id=pilot.run_id,
            )
        ],
    )
    store.finish_run(pilot.run_id, now=lambda: "2026-08-13T12:05:00Z")

    automatic = build_concept_graph(project_root)
    explicit = build_concept_graph(project_root, run_id=pilot.run_id)

    assert automatic.analysis_run_id == full_run_id
    assert automatic.analyzed_evidence == 2
    assert explicit.analysis_run_id == pilot.run_id
    assert explicit.analyzed_evidence == 1


def test_graph_checksum_detects_tampering_and_missing_state_is_optional(
    tmp_path: Path,
) -> None:
    project_root, _, _, _ = _build_project(tmp_path)
    assert concept_graph_status_report(project_root)["status"] == "missing"
    build_concept_graph(project_root)

    with sqlite3.connect(concept_graph_path_for(project_root)) as connection:
        connection.execute(
            "UPDATE concepts SET canonical_label = 'tampered' WHERE rowid = 1"
        )

    assert concept_graph_status_report(project_root)["status"] == "invalid"


def test_graph_cli_build_query_status_and_parser_contract(
    tmp_path: Path, capsys
) -> None:  # type: ignore[no-untyped-def]
    project_root, _, run_id, _ = _build_project(tmp_path)
    parsed = build_parser().parse_args(
        [
            "graph",
            "query",
            "brass latch",
            "--limit",
            "4",
            "--evidence-limit",
            "2",
            "--json",
        ]
    )
    assert parsed.command == "graph"
    assert parsed.graph_command == "query"
    assert parsed.limit == 4
    assert parsed.evidence_limit == 2

    assert (
        main(
            [
                "graph",
                "build",
                "--project",
                str(project_root),
                "--json",
            ]
        )
        == 0
    )
    built = json.loads(capsys.readouterr().out)
    assert built["analysis"]["run_id"] == run_id
    assert built["counts"]["concepts"] == 4

    assert (
        main(
            [
                "graph",
                "query",
                "brass latch",
                "--project",
                str(project_root),
                "--json",
            ]
        )
        == 0
    )
    queried = json.loads(capsys.readouterr().out)
    assert queried["result_count"] == 2
    assert queried["results"][0]["support"][0]["evidence"]["evidence_id"]

    assert main(["graph", "status", "--project", str(project_root), "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "ready"

    assert main(["doctor", "--project", str(project_root), "--json"]) == 0
    health = json.loads(capsys.readouterr().out)
    assert health["concept_graph"]["status"] == "ready"
