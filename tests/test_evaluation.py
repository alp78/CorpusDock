from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from corpusdock.chunking import (
    RuleSentenceProcessor,
    chunk_extraction_artifact,
    write_chunk_artifact,
)
from corpusdock.cli import main
from corpusdock.evaluation import (
    EVALUATION_REPORT_SCHEMA_VERSION,
    EvaluationError,
    evaluate_retrieval,
    load_evaluation_dataset,
)
from corpusdock.extraction import extract_source, write_extraction_artifact
from corpusdock.manifest import ManifestStore
from corpusdock.retrieval import SQLiteSearchBackend, build_search_index


TIMESTAMP = "2026-08-11T12:00:00Z"
BENCHMARK_ROOT = Path(__file__).parents[1] / "benchmarks" / "retrieval-v1"


def _build_benchmark(project_root: Path) -> set[str]:
    store = ManifestStore(project_root, now=lambda: TIMESTAMP)
    registrations = store.register(sorted((BENCHMARK_ROOT / "corpus").glob("*.txt")))
    source_ids: set[str] = set()
    for registration in registrations:
        source = registration.source
        source_ids.add(source.source_id)
        artifact = extract_source(
            source,
            registration.source_path,
            now=lambda: TIMESTAMP,
        )
        assert artifact.status == "complete"
        write_extraction_artifact(project_root, artifact)
        chunks = chunk_extraction_artifact(
            artifact.to_dict(),
            RuleSentenceProcessor(),
            now=lambda: TIMESTAMP,
        )
        assert chunks.status == "complete"
        write_chunk_artifact(project_root, chunks)
    build_search_index(project_root, now=lambda: TIMESTAMP)
    return source_ids


def test_generic_benchmark_measures_the_lexical_baseline_without_content(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    registered_source_ids = _build_benchmark(project_root)
    dataset = load_evaluation_dataset(BENCHMARK_ROOT / "judgments.json")
    judged_source_ids = {
        judgment.source_id for case in dataset.cases for judgment in case.relevance
    }
    clock_values = iter(
        [0.000, 0.001, 1.000, 1.002, 2.000, 2.003, 3.000, 3.004, 4.000, 4.005]
    )

    report = evaluate_retrieval(
        dataset,
        SQLiteSearchBackend(project_root),
        clock=lambda: next(clock_values),
        now=lambda: TIMESTAMP,
    )
    payload = report.to_dict()

    assert judged_source_ids == registered_source_ids
    assert payload["schema_version"] == EVALUATION_REPORT_SCHEMA_VERSION
    assert (
        dataset.sha256
        == sha256((BENCHMARK_ROOT / "judgments.json").read_bytes()).hexdigest()
    )
    assert payload["dataset"]["sha256"] == dataset.sha256
    assert report.summary.cases == 5
    assert report.summary.cases_with_results == 4
    assert report.summary.relevant_sources == 6
    assert report.summary.matched_relevant_sources == 5
    assert report.summary.recall_at_k == pytest.approx(5 / 6, abs=1e-6)
    assert report.summary.mean_reciprocal_rank_at_k == 0.8
    assert report.summary.locator_judgments == 6
    assert report.summary.matched_locator_judgments == 5
    assert report.summary.locator_accuracy == pytest.approx(5 / 6, abs=1e-6)
    assert report.summary.verification_attempts == 5
    assert report.summary.verification_successes == 5
    assert report.summary.verification_rate == 1.0
    assert report.summary.latency_ms.to_dict() == {
        "mean": 3.0,
        "p50": 3.0,
        "p95": 5.0,
        "max": 5.0,
    }
    assert dict(report.by_category)["paraphrase"].recall_at_k == 0.0
    assert all(
        evidence_id.startswith("ev-")
        for case in payload["cases"]
        for evidence_id in case["retrieved_evidence_ids"]
    )
    serialized = json.dumps(payload)
    assert '"excerpt"' not in serialized
    assert '"source_path"' not in serialized
    assert all("citation" not in case for case in payload["cases"])


def test_eval_cli_emits_the_versioned_non_content_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)

    exit_code = main(
        [
            "eval",
            str(BENCHMARK_ROOT / "judgments.json"),
            "--project",
            str(project_root),
            "--no-verify",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["retrieval"]["backend"] == "sqlite_fts5"
    assert payload["retrieval"]["latency_scope"] == "search_only"
    assert payload["retrieval"]["limit"] == 10
    assert payload["retrieval"]["mode"] == "lexical"
    assert payload["retrieval"]["verification_enabled"] is False
    assert payload["retrieval"]["index"]["sources"] == 3
    assert payload["retrieval"]["index"]["chunks"] == 3
    assert payload["retrieval"]["index"]["size_bytes"] > 0
    assert payload["summary"]["verification_attempts"] == 0
    assert payload["summary"]["verification_rate"] is None


def test_evaluation_loader_rejects_unknown_fields_and_duplicate_cases(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "invalid.json"
    source_id = "src-" + "a" * 64
    case = {
        "case_id": "duplicate",
        "category": "lexical",
        "query": "generic query",
        "relevance": [{"source_id": source_id}],
    }
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "invalid",
                "cases": [case, case],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="case IDs must be unique"):
        load_evaluation_dataset(dataset_path)

    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "invalid",
                "unexpected": True,
                "cases": [case],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="unknown field"):
        load_evaluation_dataset(dataset_path)


def test_evaluation_loader_rejects_malformed_modes_ranges_and_duplicate_keys(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "invalid.json"
    source_id = "src-" + "a" * 64
    payload = {
        "schema_version": 1,
        "name": "invalid",
        "cases": [
            {
                "case_id": "invalid-mode",
                "category": "lexical",
                "query": "generic query",
                "match_mode": [],
                "relevance": [{"source_id": source_id}],
            }
        ],
    }
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationError, match="Match mode"):
        load_evaluation_dataset(dataset_path)

    payload["cases"][0]["match_mode"] = "all"
    payload["cases"][0]["relevance"][0]["locator"] = {
        "line_start": 3,
        "line_end": 2,
    }
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvaluationError, match="cannot exceed"):
        load_evaluation_dataset(dataset_path)

    dataset_path.write_text(
        '{"schema_version":1,"name":"one","name":"two","cases":[]}',
        encoding="utf-8",
    )
    with pytest.raises(EvaluationError, match="duplicate JSON key"):
        load_evaluation_dataset(dataset_path)


def test_evaluation_rejects_a_judged_source_missing_from_the_index(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)
    payload = json.loads(
        (BENCHMARK_ROOT / "judgments.json").read_text(encoding="utf-8")
    )
    payload["cases"][0]["relevance"][0]["source_id"] = "src-" + "f" * 64
    dataset_path = tmp_path / "missing-source.json"
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError, match="is not present in the search index"):
        evaluate_retrieval(
            load_evaluation_dataset(dataset_path),
            SQLiteSearchBackend(project_root),
        )
