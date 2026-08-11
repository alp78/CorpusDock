from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpusdock.chunking import (
    RuleSentenceProcessor,
    chunk_extraction_artifact,
    write_chunk_artifact,
)
from corpusdock.embeddings import (
    EmbeddingError,
    EmbeddingModelInfo,
    InMemorySemanticSearchBackend,
    _cached_model_snapshot,
    _download_file_allowlist,
)
from corpusdock.evaluation import evaluate_retrieval, load_evaluation_dataset
from corpusdock.extraction import extract_source, write_extraction_artifact
from corpusdock.manifest import ManifestStore
from corpusdock.retrieval import RetrievalError, SQLiteSearchBackend, build_search_index


TIMESTAMP = "2026-08-11T12:00:00Z"
BENCHMARK_ROOT = Path(__file__).parents[1] / "benchmarks" / "retrieval-v1"


class _FixtureEmbeddingProvider:
    def __init__(self) -> None:
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
    build_search_index(project_root, now=lambda: TIMESTAMP)


def test_semantic_backend_preserves_exact_evidence_and_reports_resources(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)
    exact_backend = SQLiteSearchBackend(project_root)
    backend = InMemorySemanticSearchBackend(
        exact_backend,
        _FixtureEmbeddingProvider(),
    )

    response = backend.search("keep shipping paperwork", limit=1)
    expected = exact_backend.search("cargo inspection records", limit=1).results[0]

    assert len(response.results) == 1
    result = response.results[0]
    assert result.evidence_id == expected.evidence_id
    assert result.excerpt == expected.excerpt
    assert result.locator == expected.locator
    assert result.citation == expected.citation
    assert result.verification_status == "artifact-anchor-confirmed"
    assert result.score == 1.0
    assert backend.build_stats.documents == 3
    assert backend.build_stats.dimension == 3
    assert backend.build_stats.vector_size_bytes == 36
    assert backend.build_stats.source_index_fingerprint.startswith("sha256:")

    verified = backend.verify(result.evidence_id)
    assert verified.evidence.verification_status == "source-anchor-confirmed"


def test_semantic_backend_runs_the_same_non_content_evaluator(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)
    backend = InMemorySemanticSearchBackend(
        SQLiteSearchBackend(project_root),
        _FixtureEmbeddingProvider(),
    )
    report = evaluate_retrieval(
        load_evaluation_dataset(BENCHMARK_ROOT / "judgments.json"),
        backend,
        limit=2,
        backend_name="in_memory_dense",
        retrieval_mode="semantic",
        retrieval_metadata=backend.evaluation_metadata(),
    )
    payload = report.to_dict()

    assert report.summary.recall_at_k == 1.0
    assert report.summary.mean_reciprocal_rank_at_k == 1.0
    assert report.summary.locator_accuracy == 1.0
    assert payload["retrieval"]["mode"] == "semantic"
    assert payload["retrieval"]["embedding"]["dimension"] == 3
    assert (
        payload["retrieval"]["embedding"]["accelerator_peak_memory_allocated_bytes"]
        is None
    )
    assert payload["retrieval"]["semantic_index"]["vector_size_bytes"] == 36
    serialized = json.dumps(payload)
    assert '"excerpt"' not in serialized
    assert '"source_path"' not in serialized


def test_semantic_backend_rejects_invalid_vectors_and_stale_snapshots(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    _build_benchmark(project_root)

    class ZeroProvider(_FixtureEmbeddingProvider):
        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            return [[0.0, 0.0, 0.0] for _ in texts]

    with pytest.raises(EmbeddingError, match="zero-length"):
        InMemorySemanticSearchBackend(SQLiteSearchBackend(project_root), ZeroProvider())

    backend = InMemorySemanticSearchBackend(
        SQLiteSearchBackend(project_root), _FixtureEmbeddingProvider()
    )
    build_search_index(project_root, now=lambda: "2026-08-11T12:01:00Z")
    with pytest.raises(RetrievalError, match="snapshot"):
        backend.search("leaf mulch")


def test_model_download_allowlist_prefers_safe_weights_and_excludes_code() -> None:
    selected = _download_file_allowlist(
        (
            "config.json",
            "chat_template.jinja",
            "tokenizer.json",
            "modules.json",
            "model.safetensors",
            "pytorch_model.bin",
            "onnx/model.onnx",
            "openvino/openvino_model.bin",
            ".eval_results/score.json",
            "custom_model.py",
        )
    )

    assert selected == (
        "config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "modules.json",
        "model.safetensors",
    )
    with pytest.raises(ValueError, match="no supported"):
        _download_file_allowlist(("config.json", "custom_model.py"))


def test_cached_model_snapshot_resolves_one_revision_without_a_network_ref(
    tmp_path: Path,
) -> None:
    snapshots = tmp_path / "models--example--embedding-model" / "snapshots"
    revision = "a" * 40
    snapshot = snapshots / revision
    snapshot.mkdir(parents=True)

    assert (
        _cached_model_snapshot(
            tmp_path,
            model="example/embedding-model",
            revision=None,
        )
        == snapshot
    )
    assert (
        _cached_model_snapshot(
            tmp_path,
            model="example/embedding-model",
            revision=revision,
        )
        == snapshot
    )

    (snapshots / ("b" * 40)).mkdir()
    with pytest.raises(ValueError, match="multiple revisions"):
        _cached_model_snapshot(
            tmp_path,
            model="example/embedding-model",
            revision=None,
        )
