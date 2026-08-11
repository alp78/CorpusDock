"""Atomic local vector persistence and citation-preserving semantic search."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from time import perf_counter
from typing import Any
from uuid import uuid4

from corpusdock import __version__
from corpusdock.embeddings import (
    MAX_EMBEDDING_BATCH_SIZE,
    EmbeddingModelInfo,
    EmbeddingProvider,
    _normalized_matrix,
    _numpy,
)
from corpusdock.manifest import utc_now
from corpusdock.retrieval import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    MatchMode,
    RetrievalError,
    SQLiteSearchBackend,
    SearchCorpusSnapshot,
    SearchResponse,
    VerificationReport,
)


SEMANTIC_INDEX_SCHEMA_VERSION = 1
SEMANTIC_INDEX_FILE_NAME = "semantic.sqlite3"
VECTOR_DTYPE = "float32-le"
MAX_EMBEDDING_DIMENSION = 65_536

_MATCH_MODES = {"all", "any", "phrase"}
_SOURCE_ID_PATTERN = re.compile(r"src-[0-9a-f]{64}")
_CHUNK_ID_PATTERN = re.compile(r"chk-[0-9a-f]{64}")
_EVIDENCE_ID_PATTERN = re.compile(r"ev-[0-9a-f]{64}")
_FINGERPRINT_PATTERN = re.compile(r"(?:sha256|hf-revision):[0-9a-f]{40,64}")
_EMBEDDING_FIELDS = {
    "provider",
    "runtime",
    "runtime_version",
    "model_id",
    "model_revision",
    "model_fingerprint",
    "model_size_bytes",
    "dimension",
    "max_sequence_tokens",
    "device",
    "dtype",
    "normalized",
    "remote_code_trusted",
    "download_allowed",
    "query_prompt_name",
    "document_prompt_name",
    "batch_size",
    "load_ms",
    "framework_version",
    "accelerator_runtime_version",
    "accelerator_name",
    "accelerator_peak_memory_allocated_bytes",
    "accelerator_peak_memory_reserved_bytes",
}


class SemanticIndexError(Exception):
    """A missing, stale, corrupt, or incompatible local semantic index."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SemanticIndexDescriptor:
    """Non-content provenance and resource metadata for a persistent vector index."""

    path: str
    schema_version: int
    built_at: str
    source_index_built_at: str
    source_index_fingerprint: str
    indexed_sources: int
    indexed_chunks: int
    partial_sources: int
    vector_dtype: str
    vector_size_bytes: int
    vectors_sha256: str
    index_size_bytes: int
    embedding: Mapping[str, Any]
    build: Mapping[str, int | float]

    @property
    def model_id(self) -> str:
        return str(self.embedding["model_id"])

    @property
    def model_revision(self) -> str:
        return str(self.embedding["model_revision"])

    @property
    def model_fingerprint(self) -> str:
        return str(self.embedding["model_fingerprint"])

    @property
    def dimension(self) -> int:
        return int(self.embedding["dimension"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "schema_version": self.schema_version,
            "built_at": self.built_at,
            "source_index": {
                "built_at": self.source_index_built_at,
                "fingerprint": self.source_index_fingerprint,
                "sources": self.indexed_sources,
                "chunks": self.indexed_chunks,
                "partial_sources": self.partial_sources,
            },
            "embedding": dict(self.embedding),
            "vectors": {
                "dtype": self.vector_dtype,
                "count": self.indexed_chunks,
                "size_bytes": self.vector_size_bytes,
                "sha256": self.vectors_sha256,
            },
            "build": dict(self.build),
            "index_size_bytes": self.index_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class _VectorRecord:
    ordinal: int
    chunk_id: str
    evidence_id: str
    source_id: str
    vector: bytes


def semantic_index_path_for(project_root: Path | str) -> Path:
    """Return the ignored project-local persistent vector-index path."""

    return (
        Path(project_root).expanduser().resolve()
        / ".corpusdock"
        / SEMANTIC_INDEX_FILE_NAME
    )


def build_semantic_index(
    project_root: Path | str,
    provider: EmbeddingProvider,
    *,
    now: Callable[[], str] = utc_now,
    clock: Callable[[], float] = perf_counter,
) -> SemanticIndexDescriptor:
    """Embed an exact-index snapshot and atomically replace its derived vector index."""

    root = Path(project_root).expanduser().resolve()
    exact_backend = SQLiteSearchBackend(root)
    snapshot = exact_backend.corpus_snapshot()
    if not snapshot.evidence:
        raise SemanticIndexError(
            "semantic_corpus_empty",
            "The exact index has no chunks to embed for semantic retrieval.",
        )
    _validate_provider_policy(provider.info)

    np = _numpy()
    started = clock()
    raw_vectors = provider.embed_documents(
        tuple(evidence.excerpt for evidence in snapshot.evidence)
    )
    matrix = _normalized_matrix(
        raw_vectors,
        rows=len(snapshot.evidence),
        dimension=provider.info.dimension,
        label="document",
        np=np,
    )
    storage_matrix = np.ascontiguousarray(matrix, dtype=np.dtype("<f4"))
    elapsed = max(0.0, clock() - started)
    exact_backend.assert_snapshot_current(snapshot)

    embedding = provider.info.to_dict()
    dimension = int(storage_matrix.shape[1])
    _validate_embedding_metadata(embedding, expected_dimension=dimension)
    build = {
        "document_embedding_ms": _rounded_ms(elapsed),
        "documents_per_second": round(
            len(snapshot.evidence) / elapsed if elapsed else 0.0, 6
        ),
    }
    records = _records_from_matrix(snapshot, storage_matrix)
    vectors_sha256 = _vectors_digest(
        snapshot.index_fingerprint,
        str(embedding["model_fingerprint"]),
        dimension,
        records,
    )
    metadata = {
        "schema_version": str(SEMANTIC_INDEX_SCHEMA_VERSION),
        "tool_version": __version__,
        "built_at": now(),
        "source_index_built_at": snapshot.index_built_at,
        "source_index_fingerprint": snapshot.index_fingerprint,
        "sources": str(snapshot.indexed_sources),
        "chunks": str(snapshot.indexed_chunks),
        "partial_sources": str(snapshot.partial_sources),
        "vector_dtype": VECTOR_DTYPE,
        "vector_size_bytes": str(int(storage_matrix.nbytes)),
        "vectors_sha256": vectors_sha256,
        "embedding_json": _canonical_json(embedding),
        "build_json": _canonical_json(build),
    }
    path = semantic_index_path_for(root)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    connection: sqlite3.Connection | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(temporary_path)
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        _create_schema(connection)
        with connection:
            connection.executemany(
                "INSERT INTO semantic_metadata(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )
            connection.executemany(
                """
                INSERT INTO vectors(
                    ordinal, chunk_id, evidence_id, source_id, vector
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        record.ordinal,
                        record.chunk_id,
                        record.evidence_id,
                        record.source_id,
                        sqlite3.Binary(record.vector),
                    )
                    for record in records
                ),
            )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise SemanticIndexError(
                "semantic_index_integrity_failed",
                "SQLite did not confirm the integrity of the new semantic index.",
            )
        connection.close()
        connection = None
        with temporary_path.open("r+b") as index_file:
            os.fsync(index_file.fileno())
        _read_semantic_index_path(temporary_path)
        exact_backend.assert_snapshot_current(snapshot)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except SemanticIndexError:
        raise
    except (OSError, sqlite3.DatabaseError) as error:
        raise SemanticIndexError(
            "semantic_index_build_failed",
            f"Could not build the local semantic index: {error}.",
        ) from error
    finally:
        if connection is not None:
            connection.close()
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

    return read_semantic_index_descriptor(root)


def read_semantic_index_descriptor(
    project_root: Path | str,
) -> SemanticIndexDescriptor:
    """Read and checksum a persistent semantic index without loading a model."""

    descriptor, _ = _read_semantic_index(project_root)
    return descriptor


def read_current_semantic_index_descriptor(
    project_root: Path | str,
) -> SemanticIndexDescriptor:
    """Read a semantic descriptor and validate all exact-index lineage."""

    descriptor, _, _ = _current_semantic_index(SQLiteSearchBackend(project_root))
    return descriptor


class PersistentSemanticSearchBackend:
    """Cosine retrieval over persisted vectors with exact evidence resolution."""

    def __init__(
        self,
        exact_backend: SQLiteSearchBackend,
        provider: EmbeddingProvider,
    ) -> None:
        descriptor, records, snapshot = _current_semantic_index(exact_backend)
        _validate_provider_compatibility(descriptor, provider.info)

        np = _numpy()
        raw = b"".join(record.vector for record in records)
        matrix = np.frombuffer(raw, dtype=np.dtype("<f4")).reshape(
            descriptor.indexed_chunks, descriptor.dimension
        )
        if not bool(np.isfinite(matrix).all()):
            raise SemanticIndexError(
                "semantic_index_vector_invalid",
                "The semantic index contains a non-finite vector value.",
            )
        norms = np.linalg.norm(matrix, axis=1)
        if not bool(np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5)):
            raise SemanticIndexError(
                "semantic_index_vector_invalid",
                "The semantic index contains a vector that is not normalized.",
            )

        self._np = np
        self._exact_backend = exact_backend
        self._provider = provider
        self._snapshot = snapshot
        self._matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self._source_ids = frozenset(snapshot.source_ids)
        self.descriptor = descriptor

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._provider.info

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        source_id: str | None = None,
        match_mode: MatchMode = "all",
    ) -> SearchResponse:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_SEARCH_LIMIT
        ):
            raise RetrievalError(
                "search_limit_invalid",
                f"Search limit must be between 1 and {MAX_SEARCH_LIMIT}.",
            )
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query_empty", "Search query cannot be empty.")
        if not isinstance(match_mode, str) or match_mode not in _MATCH_MODES:
            raise RetrievalError(
                "match_mode_invalid", f"Unknown search match mode '{match_mode}'."
            )
        if source_id is not None and source_id not in self._source_ids:
            raise RetrievalError(
                "source_not_indexed", f"No indexed source has ID '{source_id}'."
            )

        self._exact_backend.assert_snapshot_current(self._snapshot)
        raw_query = self._provider.embed_queries((query,))
        query_matrix = _normalized_matrix(
            raw_query,
            rows=1,
            dimension=self.descriptor.dimension,
            label="query",
            np=self._np,
        )
        scores = self._matrix @ query_matrix[0]
        candidates = (
            index
            for index, evidence in enumerate(self._snapshot.evidence)
            if source_id is None or evidence.locator.source_id == source_id
        )
        ranked = sorted(
            candidates,
            key=lambda index: (
                -float(scores[index]),
                self._snapshot.evidence[index].chunk_id or "",
            ),
        )[:limit]
        results = tuple(
            replace(
                self._snapshot.evidence[index],
                score=round(float(scores[index]), 12),
            )
            for index in ranked
        )
        self._exact_backend.assert_snapshot_current(self._snapshot)
        return SearchResponse(
            query=query,
            match_mode=match_mode,
            results=results,
            index_built_at=self._snapshot.index_built_at,
            indexed_sources=self._snapshot.indexed_sources,
            indexed_chunks=self._snapshot.indexed_chunks,
            partial_sources=self._snapshot.partial_sources,
        )

    def verify(self, evidence_id: str) -> VerificationReport:
        return self._exact_backend.verify(evidence_id)


def semantic_index_status_report(project_root: Path | str) -> dict[str, Any]:
    """Return non-content persistent semantic-index health metadata."""

    path = semantic_index_path_for(project_root)
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    try:
        descriptor = read_current_semantic_index_descriptor(project_root)
        return {
            "status": "ready",
            "path": descriptor.path,
            "schema_version": descriptor.schema_version,
            "built_at": descriptor.built_at,
            "sources": descriptor.indexed_sources,
            "chunks": descriptor.indexed_chunks,
            "partial_sources": descriptor.partial_sources,
            "model_id": descriptor.model_id,
            "model_revision": descriptor.model_revision,
            "dimension": descriptor.dimension,
            "vector_size_bytes": descriptor.vector_size_bytes,
            "index_size_bytes": descriptor.index_size_bytes,
        }
    except SemanticIndexError as error:
        status = "stale" if error.code == "semantic_index_stale" else "invalid"
        return {
            "status": status,
            "path": str(path),
            "error": str(error),
        }
    except RetrievalError as error:
        status = (
            "stale" if error.code in {"index_missing", "index_stale"} else "invalid"
        )
        return {"status": status, "path": str(path), "error": str(error)}


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        f"""
        PRAGMA user_version = {SEMANTIC_INDEX_SCHEMA_VERSION};

        CREATE TABLE semantic_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE vectors (
            ordinal INTEGER PRIMARY KEY CHECK (ordinal >= 0),
            chunk_id TEXT NOT NULL UNIQUE,
            evidence_id TEXT NOT NULL UNIQUE,
            source_id TEXT NOT NULL,
            vector BLOB NOT NULL CHECK (length(vector) > 0)
        );

        CREATE INDEX vectors_source_id_idx ON vectors(source_id);
        """
    )


def _read_semantic_index(
    project_root: Path | str,
) -> tuple[SemanticIndexDescriptor, tuple[_VectorRecord, ...]]:
    path = semantic_index_path_for(project_root)
    if not path.is_file():
        raise SemanticIndexError(
            "semantic_index_missing",
            f"No semantic index found at '{path}'. Run 'corpusdock embed' first.",
        )
    return _read_semantic_index_path(path)


def _read_semantic_index_path(
    path: Path,
) -> tuple[SemanticIndexDescriptor, tuple[_VectorRecord, ...]]:
    try:
        with closing(_connect_read_only(path)) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise SemanticIndexError(
                    "semantic_index_integrity_failed",
                    "SQLite did not confirm the integrity of the semantic index.",
                )
            user_version = connection.execute("PRAGMA user_version").fetchone()
            if user_version is None or user_version[0] != SEMANTIC_INDEX_SCHEMA_VERSION:
                raise SemanticIndexError(
                    "semantic_index_schema_invalid",
                    "The semantic index schema is unsupported; run 'corpusdock embed'.",
                )
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key, value FROM semantic_metadata"
                ).fetchall()
            }
            records = tuple(
                _record_from_row(row)
                for row in connection.execute(
                    """
                    SELECT ordinal, chunk_id, evidence_id, source_id, vector
                    FROM vectors
                    ORDER BY ordinal
                    """
                ).fetchall()
            )
    except SemanticIndexError:
        raise
    except sqlite3.DatabaseError as error:
        raise SemanticIndexError(
            "semantic_index_read_failed",
            f"Could not read the local semantic index: {error}. Run 'corpusdock embed'.",
        ) from error

    descriptor = _descriptor_from_metadata(path, metadata, records)
    return descriptor, records


def _descriptor_from_metadata(
    path: Path,
    metadata: Mapping[str, str],
    records: Sequence[_VectorRecord],
) -> SemanticIndexDescriptor:
    required = {
        "schema_version",
        "tool_version",
        "built_at",
        "source_index_built_at",
        "source_index_fingerprint",
        "sources",
        "chunks",
        "partial_sources",
        "vector_dtype",
        "vector_size_bytes",
        "vectors_sha256",
        "embedding_json",
        "build_json",
    }
    if set(metadata) != required:
        raise SemanticIndexError(
            "semantic_index_schema_invalid",
            "Semantic index metadata is incomplete; run 'corpusdock embed'.",
        )
    schema_version = _metadata_integer(metadata, "schema_version")
    if schema_version != SEMANTIC_INDEX_SCHEMA_VERSION:
        raise SemanticIndexError(
            "semantic_index_schema_invalid",
            "The semantic index schema is unsupported; run 'corpusdock embed'.",
        )
    sources = _metadata_integer(metadata, "sources")
    chunks = _metadata_integer(metadata, "chunks")
    partial_sources = _metadata_integer(metadata, "partial_sources")
    vector_size_bytes = _metadata_integer(metadata, "vector_size_bytes")
    if (
        sources < 1
        or chunks < 1
        or not 0 <= partial_sources <= sources
        or vector_size_bytes < 1
    ):
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "Semantic index counts and sizes must be positive.",
        )
    _metadata_string(metadata, "tool_version")
    _metadata_string(metadata, "built_at")
    _metadata_string(metadata, "source_index_built_at")
    source_fingerprint = metadata["source_index_fingerprint"]
    if re.fullmatch(r"sha256:[0-9a-f]{64}", source_fingerprint) is None:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "The semantic index source fingerprint is invalid.",
        )
    if metadata["vector_dtype"] != VECTOR_DTYPE:
        raise SemanticIndexError(
            "semantic_index_dtype_invalid",
            f"Semantic vectors must use {VECTOR_DTYPE} storage.",
        )
    vectors_sha256 = metadata["vectors_sha256"]
    if re.fullmatch(r"sha256:[0-9a-f]{64}", vectors_sha256) is None:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "The semantic vector checksum is invalid.",
        )
    embedding = _json_object(metadata["embedding_json"], "embedding metadata")
    dimension = _metadata_integer(embedding, "dimension")
    _validate_embedding_metadata(embedding, expected_dimension=dimension)
    if not 1 <= dimension <= MAX_EMBEDDING_DIMENSION:
        raise SemanticIndexError(
            "semantic_index_dimension_invalid",
            "The semantic index dimension is outside the supported range.",
        )
    build = _json_object(metadata["build_json"], "semantic build metadata")
    _validate_build_metadata(build)
    if len(records) != chunks:
        raise SemanticIndexError(
            "semantic_index_incomplete",
            "The semantic index does not contain every declared vector.",
        )
    expected_vector_bytes = dimension * 4
    for ordinal, record in enumerate(records):
        if record.ordinal != ordinal:
            raise SemanticIndexError(
                "semantic_index_order_invalid",
                "Semantic vector ordinals must be contiguous and zero-based.",
            )
        if len(record.vector) != expected_vector_bytes:
            raise SemanticIndexError(
                "semantic_index_vector_invalid",
                f"Semantic vector {ordinal} has an invalid byte length.",
            )
    measured_vector_bytes = sum(len(record.vector) for record in records)
    if measured_vector_bytes != vector_size_bytes:
        raise SemanticIndexError(
            "semantic_index_vector_invalid",
            "Semantic vector bytes do not match the stored size metadata.",
        )
    measured_digest = _vectors_digest(
        source_fingerprint,
        str(embedding["model_fingerprint"]),
        dimension,
        records,
    )
    if measured_digest != vectors_sha256:
        raise SemanticIndexError(
            "semantic_index_checksum_mismatch",
            "The semantic vector checksum does not match the stored vectors.",
        )
    try:
        index_size_bytes = path.stat().st_size
    except OSError as error:
        raise SemanticIndexError(
            "semantic_index_read_failed",
            f"Could not measure the local semantic index: {error}.",
        ) from error
    return SemanticIndexDescriptor(
        path=str(path),
        schema_version=schema_version,
        built_at=_metadata_string(metadata, "built_at"),
        source_index_built_at=_metadata_string(metadata, "source_index_built_at"),
        source_index_fingerprint=source_fingerprint,
        indexed_sources=sources,
        indexed_chunks=chunks,
        partial_sources=partial_sources,
        vector_dtype=metadata["vector_dtype"],
        vector_size_bytes=vector_size_bytes,
        vectors_sha256=vectors_sha256,
        index_size_bytes=index_size_bytes,
        embedding=dict(embedding),
        build={
            "document_embedding_ms": float(build["document_embedding_ms"]),
            "documents_per_second": float(build["documents_per_second"]),
        },
    )


def _validate_snapshot_link(
    descriptor: SemanticIndexDescriptor,
    records: Sequence[_VectorRecord],
    snapshot: SearchCorpusSnapshot,
) -> None:
    if (
        descriptor.source_index_fingerprint != snapshot.index_fingerprint
        or descriptor.indexed_sources != snapshot.indexed_sources
        or descriptor.indexed_chunks != snapshot.indexed_chunks
        or descriptor.partial_sources != snapshot.partial_sources
    ):
        raise SemanticIndexError(
            "semantic_index_stale",
            "The exact index changed after semantic indexing; run 'corpusdock embed'.",
        )
    for record, evidence in zip(records, snapshot.evidence, strict=True):
        if (
            evidence.chunk_id != record.chunk_id
            or evidence.evidence_id != record.evidence_id
            or evidence.locator.source_id != record.source_id
        ):
            raise SemanticIndexError(
                "semantic_index_lineage_invalid",
                "Semantic vector lineage does not match the exact evidence index.",
            )


def _current_semantic_index(
    exact_backend: SQLiteSearchBackend,
) -> tuple[
    SemanticIndexDescriptor,
    tuple[_VectorRecord, ...],
    SearchCorpusSnapshot,
]:
    descriptor, records = _read_semantic_index(exact_backend.project_root)
    snapshot = exact_backend.corpus_snapshot()
    _validate_snapshot_link(descriptor, records, snapshot)
    return descriptor, records, snapshot


def _validate_provider_policy(info: EmbeddingModelInfo) -> None:
    if not info.normalized:
        raise SemanticIndexError(
            "semantic_model_policy_invalid",
            "Persistent semantic vectors require normalized embeddings.",
        )
    if info.remote_code_trusted:
        raise SemanticIndexError(
            "semantic_model_policy_invalid",
            "Persistent semantic indexing refuses models that trust repository code.",
        )
    if not 1 <= info.dimension <= MAX_EMBEDDING_DIMENSION:
        raise SemanticIndexError(
            "semantic_index_dimension_invalid",
            "The embedding dimension is outside the supported range.",
        )


def _validate_provider_compatibility(
    descriptor: SemanticIndexDescriptor, info: EmbeddingModelInfo
) -> None:
    _validate_provider_policy(info)
    expected = descriptor.embedding
    comparisons = {
        "provider": (expected["provider"], info.provider),
        "model ID": (expected["model_id"], info.model_id),
        "model revision": (expected["model_revision"], info.model_revision),
        "model fingerprint": (expected["model_fingerprint"], info.model_fingerprint),
        "dimension": (expected["dimension"], info.dimension),
        "query prompt": (expected["query_prompt_name"], info.query_prompt_name),
        "document prompt": (
            expected["document_prompt_name"],
            info.document_prompt_name,
        ),
    }
    mismatched = [
        label for label, (stored, current) in comparisons.items() if stored != current
    ]
    if mismatched:
        raise SemanticIndexError(
            "semantic_model_mismatch",
            "The query embedding model does not match the semantic index: "
            + ", ".join(mismatched)
            + ".",
        )


def _validate_embedding_metadata(
    embedding: Mapping[str, Any], *, expected_dimension: int
) -> None:
    if set(embedding) != _EMBEDDING_FIELDS:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "Semantic embedding metadata has unsupported fields.",
        )
    for field in (
        "provider",
        "runtime",
        "runtime_version",
        "model_id",
        "model_revision",
        "model_fingerprint",
        "device",
        "dtype",
    ):
        _metadata_string(embedding, field)
    if _metadata_integer(embedding, "dimension") != expected_dimension:
        raise SemanticIndexError(
            "semantic_index_dimension_invalid",
            "Semantic embedding dimension metadata is inconsistent.",
        )
    if _metadata_integer(embedding, "model_size_bytes") < 0:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "Semantic embedding field 'model_size_bytes' cannot be negative.",
        )
    batch_size = _metadata_integer(embedding, "batch_size")
    if not 1 <= batch_size <= MAX_EMBEDDING_BATCH_SIZE:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "Semantic embedding batch size is outside the supported range.",
        )
    load_ms = embedding["load_ms"]
    if (
        not isinstance(load_ms, (int, float))
        or isinstance(load_ms, bool)
        or not math.isfinite(float(load_ms))
        or float(load_ms) < 0
    ):
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "Semantic embedding load time must be finite and non-negative.",
        )
    max_sequence_tokens = embedding["max_sequence_tokens"]
    if max_sequence_tokens is not None and (
        not isinstance(max_sequence_tokens, int)
        or isinstance(max_sequence_tokens, bool)
        or max_sequence_tokens < 1
    ):
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "Semantic maximum sequence length must be a positive integer or null.",
        )
    for field in (
        "query_prompt_name",
        "document_prompt_name",
        "framework_version",
        "accelerator_runtime_version",
        "accelerator_name",
    ):
        value = embedding[field]
        if value is not None and (
            not isinstance(value, str) or not value or len(value) > 1_000
        ):
            raise SemanticIndexError(
                "semantic_index_metadata_invalid",
                f"Semantic embedding field '{field}' must be a short string or null.",
            )
    for field in (
        "accelerator_peak_memory_allocated_bytes",
        "accelerator_peak_memory_reserved_bytes",
    ):
        value = embedding[field]
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise SemanticIndexError(
                "semantic_index_metadata_invalid",
                f"Semantic embedding field '{field}' must be a non-negative integer or null.",
            )
    if embedding["normalized"] is not True:
        raise SemanticIndexError(
            "semantic_model_policy_invalid",
            "Semantic embedding metadata must declare normalized vectors.",
        )
    if embedding["remote_code_trusted"] is not False:
        raise SemanticIndexError(
            "semantic_model_policy_invalid",
            "Semantic embedding metadata cannot trust repository code.",
        )
    if not isinstance(embedding["download_allowed"], bool):
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "Semantic download policy metadata must be Boolean.",
        )
    if _FINGERPRINT_PATTERN.fullmatch(str(embedding["model_fingerprint"])) is None:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "Semantic model fingerprint metadata is invalid.",
        )


def _validate_build_metadata(build: Mapping[str, Any]) -> None:
    if set(build) != {"document_embedding_ms", "documents_per_second"}:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            "Semantic build metadata has unsupported fields.",
        )
    for field in ("document_embedding_ms", "documents_per_second"):
        value = build[field]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise SemanticIndexError(
                "semantic_index_metadata_invalid",
                f"Semantic build field '{field}' must be non-negative.",
            )


def _records_from_matrix(
    snapshot: SearchCorpusSnapshot, matrix: Any
) -> tuple[_VectorRecord, ...]:
    return tuple(
        _VectorRecord(
            ordinal=ordinal,
            chunk_id=_required_chunk_id(evidence.chunk_id),
            evidence_id=_required_identifier(
                evidence.evidence_id,
                _EVIDENCE_ID_PATTERN,
                "evidence",
            ),
            source_id=_required_identifier(
                evidence.locator.source_id,
                _SOURCE_ID_PATTERN,
                "source",
            ),
            vector=matrix[ordinal].tobytes(order="C"),
        )
        for ordinal, evidence in enumerate(snapshot.evidence)
    )


def _record_from_row(row: sqlite3.Row) -> _VectorRecord:
    ordinal = row["ordinal"]
    vector = row["vector"]
    if not isinstance(ordinal, int) or ordinal < 0 or not isinstance(vector, bytes):
        raise SemanticIndexError(
            "semantic_index_vector_invalid",
            "The semantic index contains an invalid vector record.",
        )
    chunk_id = str(row["chunk_id"])
    evidence_id = str(row["evidence_id"])
    source_id = str(row["source_id"])
    if (
        _CHUNK_ID_PATTERN.fullmatch(chunk_id) is None
        or _EVIDENCE_ID_PATTERN.fullmatch(evidence_id) is None
        or _SOURCE_ID_PATTERN.fullmatch(source_id) is None
    ):
        raise SemanticIndexError(
            "semantic_index_lineage_invalid",
            "The semantic index contains an invalid evidence identifier.",
        )
    return _VectorRecord(ordinal, chunk_id, evidence_id, source_id, vector)


def _vectors_digest(
    source_fingerprint: str,
    model_fingerprint: str,
    dimension: int,
    records: Iterable[_VectorRecord],
) -> str:
    digest = sha256()
    for value in (
        "corpusdock-semantic-v1",
        source_fingerprint,
        model_fingerprint,
        str(dimension),
        VECTOR_DTYPE,
    ):
        _update_digest_text(digest, value)
    for record in records:
        digest.update(record.ordinal.to_bytes(8, "big"))
        _update_digest_text(digest, record.chunk_id)
        _update_digest_text(digest, record.evidence_id)
        _update_digest_text(digest, record.source_id)
        digest.update(len(record.vector).to_bytes(8, "big"))
        digest.update(record.vector)
    return f"sha256:{digest.hexdigest()}"


def _update_digest_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.DatabaseError as error:
        raise SemanticIndexError(
            "semantic_index_open_failed",
            f"Could not open the local semantic index: {error}.",
        ) from error


def _metadata_integer(values: Mapping[str, Any], field: str) -> int:
    value = values.get(field)
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            f"Semantic index field '{field}' must be an integer.",
        )
    try:
        return int(value)
    except ValueError as error:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            f"Semantic index field '{field}' must be an integer.",
        ) from error


def _metadata_string(values: Mapping[str, Any], field: str) -> str:
    value = values.get(field)
    if not isinstance(value, str) or not value or len(value) > 1_000:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            f"Semantic index field '{field}' must be a short non-empty string.",
        )
    return value


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            f"Could not read {label}: {error}.",
        ) from error
    if not isinstance(result, dict):
        raise SemanticIndexError(
            "semantic_index_metadata_invalid", f"{label.title()} must be an object."
        )
    return result


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise SemanticIndexError(
            "semantic_index_metadata_invalid",
            f"Could not serialize semantic index metadata: {error}.",
        ) from error


def _required_chunk_id(value: str | None) -> str:
    if value is None or _CHUNK_ID_PATTERN.fullmatch(value) is None:
        raise SemanticIndexError(
            "semantic_index_lineage_invalid",
            "Exact evidence is missing a valid chunk identifier.",
        )
    return value


def _required_identifier(value: str, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise SemanticIndexError(
            "semantic_index_lineage_invalid",
            f"Exact evidence is missing a valid {label} identifier.",
        )
    return value


def _rounded_ms(seconds: float) -> float:
    return round(max(0.0, seconds) * 1_000, 6)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
