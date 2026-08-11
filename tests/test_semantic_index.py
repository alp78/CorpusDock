from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import pytest

from corpusdock.chunking import (
    RuleSentenceProcessor,
    chunk_extraction_artifact,
    write_chunk_artifact,
)
from corpusdock.cli import main
from corpusdock.embeddings import EmbeddingError, EmbeddingModelInfo
from corpusdock.evaluation import evaluate_retrieval, load_evaluation_dataset
from corpusdock.extraction import extract_source, write_extraction_artifact
from corpusdock.hybrid import HybridSearchBackend
from corpusdock.manifest import ManifestStore
from corpusdock.retrieval import SQLiteSearchBackend, build_search_index
from corpusdock.semantic_index import (
    PersistentSemanticSearchBackend,
    SemanticIndexError,
    build_semantic_index,
    prune_semantic_index_cache,
    read_semantic_index_descriptor,
    semantic_index_path_for,
    semantic_index_status_report,
)


TIMESTAMP = "2026-08-11T12:00:00Z"
BENCHMARK_ROOT = Path(__file__).parents[1] / "benchmarks" / "retrieval-v1"


class _FixtureEmbeddingProvider:
    def __init__(self) -> None:
        self.document_batches: list[tuple[str, ...]] = []
        self.info = EmbeddingModelInfo(
            provider="fixture",
            runtime="fixture-runtime",
            runtime_version="1",
            model_id="project-authored-fixture",
            model_revision="fixture-v1",
            model_fingerprint="sha256:" + "a" * 64,
            model_size_bytes=1_024,
            dimension=3,
            max_sequence_tokens=512,
            device="cpu",
            dtype="float32",
            normalized=True,
            remote_code_trusted=False,
            download_allowed=False,
            query_prompt_name="query",
            document_prompt_name="passage",
            batch_size=4,
            load_ms=1.0,
        )

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        self.document_batches.append(tuple(texts))
        return [self._document_vector(text) for text in texts]

    def embed_queries(self, texts):  # type: ignore[no-untyped-def]
        return [self._query_vector(text) for text in texts]

    @staticmethod
    def _document_vector(text: str) -> list[float]:
        lowered = text.casefold()
        if "harbor operations" in lowered:
            return [1.0, 0.0, 0.0]
        if "observatory operations" in lowered:
            return [0.0, 1.0, 0.0]
        if "urban garden" in lowered:
            return [0.0, 0.0, 1.0]
        if "additional workshop" in lowered:
            return [1.0, 1.0, 1.0]
        raise AssertionError(f"Unmapped fixture document: {text}")

    @staticmethod
    def _query_vector(text: str) -> list[float]:
        lowered = text.casefold()
        if "operators" in lowered:
            return [1.0, 1.0, 0.0]
        if any(term in lowered for term in ("harbor", "cargo", "shipping")):
            return [1.0, 0.0, 0.0]
        if any(term in lowered for term in ("observatory", "winds", "telescope")):
            return [0.0, 1.0, 0.0]
        if any(term in lowered for term in ("garden", "mulch", "soil")):
            return [0.0, 0.0, 1.0]
        raise AssertionError(f"Unmapped fixture query: {text}")


def _build_benchmark(project_root: Path) -> None:
    store = ManifestStore(project_root, now=lambda: TIMESTAMP)
    registrations = store.register(sorted((BENCHMARK_ROOT / "corpus").glob("*.txt")))
    for registration in registrations:
        _process_registration(project_root, registration)
    build_search_index(project_root, now=lambda: TIMESTAMP)


def _process_registration(project_root: Path, registration) -> None:  # type: ignore[no-untyped-def]
    artifact = extract_source(
        registration.source,
        registration.source_path,
        now=lambda: TIMESTAMP,
    )
    write_extraction_artifact(project_root, artifact)
    chunks = chunk_extraction_artifact(
        artifact.to_dict(),
        RuleSentenceProcessor(),
        now=lambda: TIMESTAMP,
    )
    write_chunk_artifact(project_root, chunks)


def test_persistent_semantic_index_preserves_exact_evidence_without_copying_text(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)
    provider = _FixtureEmbeddingProvider()

    descriptor = build_semantic_index(
        project_root,
        provider,
        now=lambda: TIMESTAMP,
    )

    assert descriptor.path == str(semantic_index_path_for(project_root))
    assert descriptor.indexed_sources == 3
    assert descriptor.indexed_chunks == 3
    assert descriptor.dimension == 3
    assert descriptor.vector_size_bytes == 36
    assert descriptor.model_fingerprint == "sha256:" + "a" * 64
    assert descriptor.vectors_sha256.startswith("sha256:")
    database_bytes = semantic_index_path_for(project_root).read_bytes()
    assert b"Harbor operations" not in database_bytes
    assert str(BENCHMARK_ROOT).encode() not in database_bytes

    exact_backend = SQLiteSearchBackend(project_root)
    backend = PersistentSemanticSearchBackend(exact_backend, provider)
    response = backend.search("keep shipping paperwork", limit=1)
    expected = exact_backend.search("cargo inspection records", limit=1).results[0]

    assert len(response.results) == 1
    result = response.results[0]
    assert result.evidence_id == expected.evidence_id
    assert result.chunk_id == expected.chunk_id
    assert result.excerpt == expected.excerpt
    assert result.locators == expected.locators
    assert result.citation == expected.citation
    assert result.source_path == expected.source_path
    assert result.score == 1.0
    assert backend.verify(result.evidence_id).evidence.verification_status == (
        "source-anchor-confirmed"
    )

    report = evaluate_retrieval(
        load_evaluation_dataset(BENCHMARK_ROOT / "judgments.json"),
        backend,
        limit=2,
        backend_name="persistent_dense",
        retrieval_mode="semantic",
    )
    assert report.summary.recall_at_k == 1.0
    assert report.summary.locator_accuracy == 1.0
    assert report.summary.verification_rate == 1.0

    hybrid = HybridSearchBackend(exact_backend, backend)
    hybrid_report = evaluate_retrieval(
        load_evaluation_dataset(BENCHMARK_ROOT / "judgments.json"),
        hybrid,
        limit=2,
        backend_name="sqlite_fts5+persistent_dense_rrf",
        retrieval_mode="hybrid",
        retrieval_metadata=hybrid.evaluation_metadata(),
    )
    assert hybrid_report.summary.recall_at_k == 1.0
    assert hybrid_report.summary.mean_reciprocal_rank_at_k == 1.0
    assert hybrid_report.summary.locator_accuracy == 1.0
    assert hybrid_report.summary.verification_rate == 1.0
    hybrid_payload = hybrid_report.to_dict()
    assert hybrid_payload["retrieval"]["fusion"]["rrf_k"] == 60
    assert "excerpt" not in json.dumps(hybrid_payload)


def test_semantic_index_build_is_atomic_when_new_vectors_are_invalid(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)
    path = semantic_index_path_for(project_root)
    build_semantic_index(project_root, _FixtureEmbeddingProvider())
    original_sha256 = sha256(path.read_bytes()).hexdigest()

    class _ZeroProvider(_FixtureEmbeddingProvider):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            return [[0.0, 0.0, 0.0] for _ in texts]

    additional = tmp_path / "additional.txt"
    additional.write_text(
        "Additional workshop\nThe additional workshop calibrates its tools locally.\n",
        encoding="utf-8",
    )
    registration = ManifestStore(project_root).register((additional,))[0]
    _process_registration(project_root, registration)
    build_search_index(project_root)

    with pytest.raises(EmbeddingError, match="zero-length"):
        build_semantic_index(project_root, _ZeroProvider())

    assert sha256(path.read_bytes()).hexdigest() == original_sha256
    assert not tuple(path.parent.glob(f".{path.name}.*.tmp"))


def test_semantic_index_validates_candidate_before_atomic_replacement(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)
    path = semantic_index_path_for(project_root)
    build_semantic_index(project_root, _FixtureEmbeddingProvider())
    original_sha256 = sha256(path.read_bytes()).hexdigest()

    with pytest.raises(SemanticIndexError, match="built_at"):
        build_semantic_index(
            project_root,
            _FixtureEmbeddingProvider(),
            now=lambda: "",
        )

    assert sha256(path.read_bytes()).hexdigest() == original_sha256
    assert read_semantic_index_descriptor(project_root).indexed_chunks == 3
    assert not tuple(path.parent.glob(f".{path.name}.*.tmp"))


def test_semantic_index_rejects_model_mismatch_staleness_and_corruption(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)
    provider = _FixtureEmbeddingProvider()
    build_semantic_index(project_root, provider)

    mismatched = _FixtureEmbeddingProvider()
    mismatched.info = replace(
        mismatched.info,
        model_fingerprint="sha256:" + "b" * 64,
    )
    with pytest.raises(SemanticIndexError, match="model fingerprint"):
        PersistentSemanticSearchBackend(SQLiteSearchBackend(project_root), mismatched)

    additional = tmp_path / "additional.txt"
    additional.write_text(
        "Additional workshop\nThe additional workshop calibrates its tools locally.\n",
        encoding="utf-8",
    )
    registration = ManifestStore(project_root).register((additional,))[0]
    _process_registration(project_root, registration)
    build_search_index(project_root, now=lambda: "2026-08-11T12:01:00Z")

    status = semantic_index_status_report(project_root)
    assert status["status"] == "stale"
    with pytest.raises(SemanticIndexError, match="changed after semantic indexing"):
        PersistentSemanticSearchBackend(
            SQLiteSearchBackend(project_root), _FixtureEmbeddingProvider()
        )

    incremental = _FixtureEmbeddingProvider()
    rebuilt = build_semantic_index(project_root, incremental)
    assert [len(batch) for batch in incremental.document_batches] == [1]
    assert rebuilt.build["embedded_documents"] == 1
    assert rebuilt.build["reused_documents"] == 3
    path = semantic_index_path_for(project_root)
    with sqlite3.connect(path) as connection:
        vector = connection.execute(
            "SELECT vector FROM vectors WHERE ordinal = 0"
        ).fetchone()[0]
        tampered = bytes((vector[0] ^ 1,)) + vector[1:]
        connection.execute(
            "UPDATE vectors SET vector = ? WHERE ordinal = 0", (tampered,)
        )

    with pytest.raises(SemanticIndexError, match="checksum"):
        read_semantic_index_descriptor(project_root)


def test_semantic_cli_builds_searches_and_reports_index_health(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)

    def fixture_provider(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FixtureEmbeddingProvider()

    monkeypatch.setattr(
        "corpusdock.cli.SentenceTransformersEmbeddingProvider", fixture_provider
    )

    assert (
        main(
            [
                "embed",
                "--project",
                str(project_root),
                "--embedding-model",
                "fixture/model",
                "--device",
                "cpu",
                "--json",
            ]
        )
        == 0
    )
    build_payload = json.loads(capsys.readouterr().out)
    assert build_payload["source_index"]["chunks"] == 3
    assert build_payload["vectors"]["size_bytes"] == 36
    assert "excerpt" not in json.dumps(build_payload)

    assert (
        main(
            [
                "search",
                "keep shipping paperwork",
                "--project",
                str(project_root),
                "--retrieval",
                "semantic",
                "--embedding-model",
                "fixture/model",
                "--limit",
                "1",
                "--json",
            ]
        )
        == 0
    )
    response = json.loads(capsys.readouterr().out)
    assert response["result_count"] == 1
    assert "cargo inspection records" in response["results"][0]["excerpt"].casefold()
    assert response["results"][0]["locator"]["line_start"] == 1

    assert (
        main(
            [
                "search",
                "cargo inspection records",
                "--project",
                str(project_root),
                "--retrieval",
                "hybrid",
                "--embedding-model",
                "fixture/model",
                "--limit",
                "1",
                "--json",
            ]
        )
        == 0
    )
    hybrid_response = json.loads(capsys.readouterr().out)
    assert hybrid_response["result_count"] == 1
    assert hybrid_response["results"][0]["score"] == round(2 / 61, 12)

    assert (
        main(
            [
                "eval",
                str(BENCHMARK_ROOT / "judgments.json"),
                "--project",
                str(project_root),
                "--retrieval",
                "hybrid",
                "--embedding-model",
                "fixture/model",
                "--limit",
                "2",
                "--no-verify",
                "--json",
            ]
        )
        == 0
    )
    hybrid_evaluation = json.loads(capsys.readouterr().out)
    assert hybrid_evaluation["retrieval"]["mode"] == "hybrid"
    assert hybrid_evaluation["retrieval"]["fusion"]["candidate_limit"] == 60
    assert hybrid_evaluation["summary"]["recall_at_k"] == 1.0

    assert main(["doctor", "--project", str(project_root), "--json"]) == 0
    health = json.loads(capsys.readouterr().out)
    assert health["semantic_index"]["status"] == "ready"
    assert health["semantic_index"]["chunks"] == 3


def test_semantic_index_status_is_optional_when_missing(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)

    assert semantic_index_status_report(project_root)["status"] == "missing"


def test_semantic_cache_prunes_removed_evidence_and_reuses_every_retained_vector(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)
    build_semantic_index(project_root, _FixtureEmbeddingProvider())
    retained_paths = tuple(sorted((BENCHMARK_ROOT / "corpus").glob("*.txt")))[1:]
    ManifestStore(project_root).reconcile_mirror(retained_paths)
    build_search_index(project_root)

    assert prune_semantic_index_cache(project_root) == 1
    assert read_semantic_index_descriptor(project_root).indexed_chunks == 2
    provider = _FixtureEmbeddingProvider()
    descriptor = build_semantic_index(project_root, provider)

    assert provider.document_batches == []
    assert descriptor.build["embedded_documents"] == 0
    assert descriptor.build["reused_documents"] == 2
