from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import pytest

from corpusdock.analysis_contracts import parse_and_validate_analysis
from corpusdock.analysis_store import (
    AnalysisStore,
    AnalysisStoreError,
    analysis_database_path_for,
    analysis_status_report,
    reconcile_analysis_database,
)
from corpusdock.chunking import (
    RuleSentenceProcessor,
    chunk_extraction_artifact,
    write_chunk_artifact,
)
from corpusdock.extraction import extract_source, write_extraction_artifact
from corpusdock.manifest import ManifestStore
from corpusdock.retrieval import build_search_index


TIMESTAMP = "2026-08-11T12:00:00Z"
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


def _add_source(project_root: Path, source_path: Path, content: str) -> None:
    source_path.write_text(content, encoding="utf-8")
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


def _build_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    _add_source(
        project_root,
        tmp_path / "fixture.txt",
        "The brass latch reduces cabinet vibration during transport.\n",
    )
    build_search_index(project_root, now=lambda: TIMESTAMP)
    return project_root


def _payload() -> str:
    return json.dumps(
        {
            "schema_version": 3,
            "concepts": [
                {
                    "local_id": "c1",
                    "label": "brass latch",
                    "description": "A cabinet closure component.",
                    "concept_type": "component",
                    "confidence": 0.95,
                    "support": {"unit_ids": ["u1"]},
                },
                {
                    "local_id": "c2",
                    "label": "cabinet vibration",
                    "description": "Movement affecting a cabinet.",
                    "concept_type": "effect",
                    "confidence": 0.93,
                    "support": {"unit_ids": ["u1"]},
                },
            ],
            "claims": [
                {
                    "local_id": "q1",
                    "statement": "The latch reduces vibration in transport.",
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


def test_analysis_store_persists_lineage_offsets_runs_and_reviews(
    tmp_path: Path,
) -> None:
    project_root = _build_project(tmp_path)
    store = AnalysisStore(project_root)
    first = store.begin_run(
        MODEL_INFO,
        prompt_version="analysis-extraction-v1",
        prompt_sha256=PROMPT_SHA,
        scope={"source_ids": [], "limit": None},
        now=lambda: TIMESTAMP,
    )
    resumed = store.begin_run(
        MODEL_INFO,
        prompt_version="analysis-extraction-v1",
        prompt_sha256=PROMPT_SHA,
        scope={"source_ids": [], "limit": None},
        now=lambda: "2026-08-11T12:01:00Z",
    )
    assert resumed.run_id == first.run_id

    evidence = store.snapshot.evidence[0]
    analysis = parse_and_validate_analysis(
        _payload(), evidence, run_id=first.run_id, inference_ms=12.5
    )
    store.write_batch(first.run_id, [analysis])

    progress = store.run_descriptor(first.run_id)
    assert progress.status == "running"
    assert progress.analyzed_evidence == 1
    assert progress.concepts == 2
    assert progress.claims == 1
    assert progress.relations == 1
    assert progress.inference_ms == 12.5
    assert progress.output_tokens == 0
    assert progress.truncated_evidence == 0
    assert store.analyzed_evidence_ids(first.run_id) == {evidence.evidence_id}

    review_id = store.record_review(
        "claim",
        analysis.claims[0].claim_id,
        "needs_review",
        reviewer="fixture-reviewer",
        note="Check scope before accepting.",
        now=lambda: "2026-08-11T12:02:00Z",
    )
    assert review_id.startswith("rev-")

    completed = store.finish_run(first.run_id, now=lambda: "2026-08-11T12:03:00Z")
    assert completed.status == "completed"
    assert completed.completed_at == "2026-08-11T12:03:00Z"

    status = analysis_status_report(project_root)
    assert status["status"] == "ready"
    assert status["runs"] == {"running": 0, "completed": 1, "failed": 0}
    assert status["concepts"] == 2
    assert status["claims"] == 1
    assert status["relations"] == 1
    assert status["reviews"] == 1
    assert status["latest_run"]["run_id"] == first.run_id
    assert status["latest_run"]["status"] == "completed"
    assert status["latest_run"]["counts"]["analyzed_evidence"] == 1

    with sqlite3.connect(analysis_database_path_for(project_root)) as connection:
        state = connection.execute(
            "SELECT review_state FROM claim_candidates"
        ).fetchone()
        assert state == ("needs_review",)


def test_analysis_resume_fingerprint_includes_runtime_engine_settings(
    tmp_path: Path,
) -> None:
    store = AnalysisStore(_build_project(tmp_path))
    scope = {"source_ids": [], "limit": None}
    native = {
        **MODEL_INFO,
        "engine_performance_mode": "throughput",
        "prefix_caching_enabled": True,
        "gpu_memory_utilization": 0.9,
        "structured_output_backend": "xgrammar",
        "sampling_backend": "vllm-native",
    }
    flashinfer = {**native, "sampling_backend": "vllm-flashinfer"}

    first = store.begin_run(
        native,
        prompt_version="analysis-extraction-v1",
        prompt_sha256=PROMPT_SHA,
        scope=scope,
        now=lambda: TIMESTAMP,
    )
    changed = store.begin_run(
        flashinfer,
        prompt_version="analysis-extraction-v1",
        prompt_sha256=PROMPT_SHA,
        scope=scope,
        now=lambda: "2026-08-11T12:01:00Z",
    )

    assert changed.run_id != first.run_id
    assert changed.extractor_fingerprint != first.extractor_fingerprint


def test_analysis_database_does_not_copy_excerpt_path_prompt_or_raw_output(
    tmp_path: Path,
) -> None:
    project_root = _build_project(tmp_path)
    store = AnalysisStore(project_root)
    run = store.begin_run(
        MODEL_INFO,
        prompt_version="analysis-extraction-v1",
        prompt_sha256=PROMPT_SHA,
        scope={"source_ids": [], "limit": 1},
        now=lambda: TIMESTAMP,
    )
    evidence = store.snapshot.evidence[0]
    raw = _payload()
    store.write_batch(
        run.run_id, [parse_and_validate_analysis(raw, evidence, run_id=run.run_id)]
    )
    database = analysis_database_path_for(project_root).read_bytes()

    assert evidence.excerpt.encode() not in database
    assert evidence.source_path.encode() not in database
    assert raw.encode() not in database
    assert (
        b"brass latch" in database
    )  # derived labels/statements are intentionally local
    assert sha256(raw.encode()).hexdigest().encode() in database


def test_store_rejects_duplicate_evidence_and_content_bearing_provenance(
    tmp_path: Path,
) -> None:
    project_root = _build_project(tmp_path)
    store = AnalysisStore(project_root)
    with pytest.raises(AnalysisStoreError) as forbidden:
        store.begin_run(
            {**MODEL_INFO, "prompt": "source content would go here"},
            prompt_version="analysis-extraction-v1",
            prompt_sha256=PROMPT_SHA,
            scope={},
        )
    assert forbidden.value.code == "analysis_provenance_content_forbidden"

    run = store.begin_run(
        MODEL_INFO,
        prompt_version="analysis-extraction-v1",
        prompt_sha256=PROMPT_SHA,
        scope={},
        now=lambda: TIMESTAMP,
    )
    evidence = store.snapshot.evidence[0]
    analysis = parse_and_validate_analysis(_payload(), evidence, run_id=run.run_id)
    store.write_batch(run.run_id, [analysis])
    with pytest.raises(AnalysisStoreError) as duplicate:
        store.write_batch(run.run_id, [analysis])
    assert duplicate.value.code == "analysis_database_conflict"


def test_analysis_database_becomes_stale_when_exact_corpus_changes(
    tmp_path: Path,
) -> None:
    project_root = _build_project(tmp_path)
    AnalysisStore(project_root)

    _add_source(
        project_root,
        tmp_path / "second.txt",
        "A felt washer separates the frame from the shelf.\n",
    )
    build_search_index(project_root, now=lambda: "2026-08-11T12:05:00Z")

    status = analysis_status_report(project_root)
    assert status["status"] == "stale"
    with pytest.raises(AnalysisStoreError) as stale:
        AnalysisStore(project_root)
    assert stale.value.code == "analysis_database_stale"


def test_missing_analysis_database_is_optional_health_state(tmp_path: Path) -> None:
    assert analysis_status_report(tmp_path)["status"] == "missing"


def test_analysis_reconciliation_prunes_absent_evidence_and_derived_candidates(
    tmp_path: Path,
) -> None:
    project_root = _build_project(tmp_path)
    store = AnalysisStore(project_root)
    run = store.begin_run(
        MODEL_INFO,
        prompt_version="analysis-extraction-v1",
        prompt_sha256=PROMPT_SHA,
        scope={"source_ids": [], "limit": None},
        now=lambda: TIMESTAMP,
    )
    evidence = store.snapshot.evidence[0]
    store.write_batch(
        run.run_id,
        [parse_and_validate_analysis(_payload(), evidence, run_id=run.run_id)],
    )

    ManifestStore(project_root).reconcile_mirror(())
    build_search_index(project_root)
    assert reconcile_analysis_database(project_root) == 1

    status = analysis_status_report(project_root)
    assert status["status"] == "ready"
    assert status["analyzed_evidence_records"] == 0
    assert status["concepts"] == 0
    assert status["claims"] == 0
    assert status["relations"] == 0
