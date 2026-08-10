"""Local citation-first full-text indexing, search, and evidence verification.

The SQLite database in this module is a disposable derived index. Immutable source
files and the versioned extraction/chunk artifacts remain authoritative.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Literal, Protocol
from uuid import uuid4

from corpusdock import __version__
from corpusdock.chunking import (
    CHUNK_SCHEMA_VERSION,
    chunk_artifact_path_for,
    chunk_id_for,
)
from corpusdock.contracts import CitationLocator, EvidenceResult, SourceCoverage
from corpusdock.extraction import EXTRACTION_SCHEMA_VERSION, artifact_path_for
from corpusdock.manifest import (
    CorpusManifest,
    SourceRecord,
    SourceRegistrationError,
    hash_source_file,
    manifest_path_for,
    utc_now,
)


INDEX_SCHEMA_VERSION = 1
SEARCH_RESPONSE_SCHEMA_VERSION = 1
VERIFICATION_SCHEMA_VERSION = 1
INDEX_FILE_NAME = "index.sqlite3"
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 100

MatchMode = Literal["all", "any", "phrase"]


class RetrievalError(Exception):
    """A local indexing, retrieval, or evidence-verification failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class IndexSummary:
    """Non-content metadata describing one complete index build."""

    path: str
    schema_version: int
    built_at: str
    sources: int
    anchors: int
    chunks: int
    complete_sources: int
    partial_sources: int
    failed_sources: int
    unresolved_pdf_pages: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "schema_version": self.schema_version,
            "built_at": self.built_at,
            "sources": self.sources,
            "anchors": self.anchors,
            "chunks": self.chunks,
            "coverage": {
                "complete_sources": self.complete_sources,
                "partial_sources": self.partial_sources,
                "failed_sources": self.failed_sources,
                "unresolved_pdf_pages": self.unresolved_pdf_pages,
            },
        }


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Versioned response envelope containing exact evidence records."""

    query: str
    match_mode: MatchMode
    results: tuple[EvidenceResult, ...]
    index_built_at: str
    indexed_sources: int
    indexed_chunks: int
    partial_sources: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SEARCH_RESPONSE_SCHEMA_VERSION,
            "query": self.query,
            "match_mode": self.match_mode,
            "result_count": len(self.results),
            "index": {
                "built_at": self.index_built_at,
                "sources": self.indexed_sources,
                "chunks": self.indexed_chunks,
                "partial_sources": self.partial_sources,
            },
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """Live verification of indexed evidence against an immutable original."""

    verified_at: str
    evidence: EvidenceResult
    checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VERIFICATION_SCHEMA_VERSION,
            "verified_at": self.verified_at,
            "verification_status": self.evidence.verification_status,
            "checks": list(self.checks),
            "evidence": self.evidence.to_dict(),
        }


class SearchBackend(Protocol):
    """Backend-neutral operations consumed by the CLI and future MCP adapter."""

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        source_id: str | None = None,
        match_mode: MatchMode = "all",
    ) -> SearchResponse: ...

    def verify(self, evidence_id: str) -> VerificationReport: ...


@dataclass(frozen=True, slots=True)
class _ArtifactState:
    size_bytes: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class _PreparedAnchor:
    anchor_id: str
    locator: CitationLocator
    text: str


@dataclass(frozen=True, slots=True)
class _PreparedChunk:
    chunk_id: str
    evidence_id: str
    text: str
    start_offset: int
    end_offset: int
    anchor_ids: tuple[str, ...]
    locators: tuple[CitationLocator, ...]
    sentence_count: int
    lexical_token_count: int


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    source: SourceRecord
    source_path: str
    title: str | None
    authors: tuple[str, ...]
    extraction_status: str
    extraction_warnings: tuple[str, ...]
    extraction_metadata: dict[str, Any]
    unresolved_pdf_pages: tuple[int, ...]
    anchors: tuple[_PreparedAnchor, ...]
    chunks: tuple[_PreparedChunk, ...]
    extraction_state: _ArtifactState
    chunk_state: _ArtifactState


def index_path_for(project_root: Path | str) -> Path:
    """Return the conventional embedded-index path for a project."""

    return Path(project_root).expanduser().resolve() / ".corpusdock" / INDEX_FILE_NAME


def evidence_id_for(chunk_id: str) -> str:
    """Derive a stable evidence ID from a stable exact chunk ID."""

    fingerprint = f"evidence-v{SEARCH_RESPONSE_SCHEMA_VERSION}\0{chunk_id}"
    return f"ev-{sha256(fingerprint.encode('utf-8')).hexdigest()}"


def build_search_index(
    project_root: Path | str,
    *,
    now: Callable[[], str] = utc_now,
) -> IndexSummary:
    """Validate all persisted artifacts and atomically rebuild the local FTS index."""

    root = Path(project_root).expanduser().resolve()
    manifest_payload, manifest_state = _read_json_stably(manifest_path_for(root))
    manifest = CorpusManifest.from_dict(manifest_payload)
    prepared = tuple(
        _prepare_source(root, source)
        for source in sorted(manifest.sources.values(), key=lambda item: item.source_id)
    )
    built_at = now()
    path = index_path_for(root)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")

    connection: sqlite3.Connection | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(temporary_path)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        _create_schema(connection)
        _write_index(
            connection,
            prepared,
            built_at=built_at,
            manifest_digest=manifest_state.sha256,
        )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise RetrievalError(
                "index_integrity_failed",
                "SQLite did not confirm the integrity of the newly built index.",
            )
        connection.close()
        connection = None
        with temporary_path.open("r+b") as index_file:
            os.fsync(index_file.fileno())
        os.replace(temporary_path, path)
    except RetrievalError:
        raise
    except sqlite3.OperationalError as error:
        if "fts5" in str(error).casefold():
            raise RetrievalError(
                "fts5_unavailable",
                "This Python SQLite build does not provide FTS5 full-text search.",
            ) from error
        raise RetrievalError(
            "index_build_failed", f"Could not build the local search index: {error}."
        ) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise RetrievalError(
            "index_build_failed", f"Could not build the local search index: {error}."
        ) from error
    finally:
        if connection is not None:
            connection.close()
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

    return _index_summary(path, built_at, prepared)


class SQLiteSearchBackend:
    """Read-only citation retrieval over the disposable embedded SQLite index."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.path = index_path_for(self.project_root)

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
        fts_query = _compile_fts_query(query, match_mode)
        try:
            with closing(self._connect()) as connection:
                metadata = _read_index_metadata(connection)
                _assert_index_current(connection, self.project_root, metadata)
                parameters: list[object] = [fts_query]
                source_clause = ""
                if source_id is not None:
                    exists = connection.execute(
                        "SELECT 1 FROM sources WHERE source_id = ?", (source_id,)
                    ).fetchone()
                    if exists is None:
                        raise RetrievalError(
                            "source_not_indexed",
                            f"No indexed source has ID '{source_id}'.",
                        )
                    source_clause = " AND c.source_id = ?"
                    parameters.append(source_id)
                parameters.append(limit)
                rows = connection.execute(
                    _SEARCH_SQL.format(source_clause=source_clause), parameters
                ).fetchall()
                results = tuple(_evidence_from_row(row) for row in rows)
                return SearchResponse(
                    query=query,
                    match_mode=match_mode,
                    results=results,
                    index_built_at=metadata["built_at"],
                    indexed_sources=int(metadata["sources"]),
                    indexed_chunks=int(metadata["chunks"]),
                    partial_sources=int(metadata["partial_sources"]),
                )
        except RetrievalError:
            raise
        except sqlite3.DatabaseError as error:
            raise RetrievalError(
                "index_read_failed",
                f"Could not read the local search index: {error}. Rebuild it with 'corpusdock index'.",
            ) from error

    def verify(self, evidence_id: str) -> VerificationReport:
        if re.fullmatch(r"ev-[0-9a-f]{64}", evidence_id) is None:
            raise RetrievalError(
                "evidence_id_invalid",
                "Evidence ID must be an 'ev-' SHA-256 identifier.",
            )
        try:
            with closing(self._connect()) as connection:
                metadata = _read_index_metadata(connection)
                _assert_index_current(connection, self.project_root, metadata)
                row = connection.execute(_VERIFY_SQL, (evidence_id,)).fetchone()
                if row is None:
                    raise RetrievalError(
                        "evidence_not_found",
                        f"No indexed evidence has ID '{evidence_id}'.",
                    )
                indexed_evidence = _evidence_from_row(row)
        except RetrievalError:
            raise
        except sqlite3.DatabaseError as error:
            raise RetrievalError(
                "index_read_failed",
                f"Could not read the local search index: {error}. Rebuild it with 'corpusdock index'.",
            ) from error

        manifest_payload, _ = _read_json_stably(manifest_path_for(self.project_root))
        manifest = CorpusManifest.from_dict(manifest_payload)
        source_id = indexed_evidence.locator.source_id
        source = manifest.sources.get(source_id)
        if source is None:
            raise RetrievalError(
                "evidence_source_missing",
                f"Evidence source '{source_id}' is no longer registered.",
            )
        prepared = _prepare_source(self.project_root, source)
        chunk = next(
            (
                item
                for item in prepared.chunks
                if item.chunk_id == indexed_evidence.chunk_id
            ),
            None,
        )
        if chunk is None or chunk.evidence_id != evidence_id:
            raise RetrievalError(
                "evidence_lineage_invalid",
                "The evidence ID no longer resolves to the persisted chunk artifact.",
            )
        if (
            chunk.text != indexed_evidence.excerpt
            or chunk.anchor_ids != indexed_evidence.anchor_ids
            or chunk.locators != indexed_evidence.locators
        ):
            raise RetrievalError(
                "evidence_index_mismatch",
                "Indexed evidence does not exactly match the persisted chunk artifact.",
            )

        verified_path = _verify_original(source, prepared.source_path)
        verified_evidence = replace(
            indexed_evidence,
            source_path=verified_path,
            verification_status="source-anchor-confirmed",
            score=None,
        )
        return VerificationReport(
            verified_at=utc_now(),
            evidence=verified_evidence,
            checks=(
                "index-current",
                "source-sha256-confirmed",
                "extraction-lineage-confirmed",
                "chunk-exact-slice-confirmed",
                "anchor-locators-confirmed",
            ),
        )

    def _connect(self) -> sqlite3.Connection:
        if not self.path.is_file():
            raise RetrievalError(
                "index_missing",
                f"No search index found at '{self.path}'. Run 'corpusdock index' first.",
            )
        try:
            connection = sqlite3.connect(
                f"{self.path.resolve().as_uri()}?mode=ro", uri=True
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.DatabaseError as error:
            raise RetrievalError(
                "index_open_failed",
                f"Could not open the local search index: {error}.",
            ) from error


def index_status_report(project_root: Path | str) -> dict[str, Any]:
    """Return non-content index health metadata for ``corpusdock doctor``."""

    backend = SQLiteSearchBackend(project_root)
    if not backend.path.is_file():
        return {"status": "missing", "path": str(backend.path)}
    try:
        with closing(backend._connect()) as connection:
            metadata = _read_index_metadata(connection)
            _assert_index_current(connection, backend.project_root, metadata)
            return {
                "status": "ready",
                "path": str(backend.path),
                "schema_version": int(metadata["schema_version"]),
                "built_at": metadata["built_at"],
                "sources": int(metadata["sources"]),
                "anchors": int(metadata["anchors"]),
                "chunks": int(metadata["chunks"]),
                "partial_sources": int(metadata["partial_sources"]),
                "unresolved_pdf_pages": int(metadata["unresolved_pdf_pages"]),
            }
    except RetrievalError as error:
        status = "stale" if error.code == "index_stale" else "invalid"
        return {
            "status": status,
            "path": str(backend.path),
            "error": str(error),
        }
    except sqlite3.DatabaseError as error:
        return {
            "status": "invalid",
            "path": str(backend.path),
            "error": str(error),
        }


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA user_version = 1;

        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE sources (
            source_id TEXT PRIMARY KEY,
            source_sha256 TEXT NOT NULL,
            source_format TEXT NOT NULL,
            source_path TEXT NOT NULL,
            original_paths_json TEXT NOT NULL,
            title TEXT,
            authors_json TEXT NOT NULL,
            extraction_status TEXT NOT NULL,
            extraction_warnings_json TEXT NOT NULL,
            extraction_metadata_json TEXT NOT NULL,
            unresolved_pdf_pages_json TEXT NOT NULL
        );

        CREATE TABLE anchors (
            anchor_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES sources(source_id),
            locator_json TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            CHECK (start_offset >= 0),
            CHECK (end_offset >= start_offset)
        );

        CREATE TABLE chunks (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            evidence_id TEXT NOT NULL UNIQUE,
            source_id TEXT NOT NULL REFERENCES sources(source_id),
            text TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            anchor_ids_json TEXT NOT NULL,
            locators_json TEXT NOT NULL,
            sentence_count INTEGER NOT NULL,
            lexical_token_count INTEGER NOT NULL,
            CHECK (start_offset >= 0),
            CHECK (end_offset > start_offset)
        );

        CREATE INDEX chunks_source_id_idx ON chunks(source_id);
        CREATE INDEX anchors_source_id_idx ON anchors(source_id);

        CREATE VIRTUAL TABLE chunk_fts USING fts5(
            text,
            content='chunks',
            content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        );

        CREATE TABLE artifact_state (
            source_id TEXT PRIMARY KEY REFERENCES sources(source_id),
            extraction_size INTEGER NOT NULL,
            extraction_mtime_ns INTEGER NOT NULL,
            extraction_sha256 TEXT NOT NULL,
            chunk_size INTEGER NOT NULL,
            chunk_mtime_ns INTEGER NOT NULL,
            chunk_sha256 TEXT NOT NULL
        );
        """
    )


def _write_index(
    connection: sqlite3.Connection,
    prepared_sources: Sequence[_PreparedSource],
    *,
    built_at: str,
    manifest_digest: str,
) -> None:
    summary = _index_summary(Path(""), built_at, prepared_sources)
    metadata = {
        "schema_version": str(INDEX_SCHEMA_VERSION),
        "built_at": built_at,
        "tool_version": __version__,
        "extraction_schema_version": str(EXTRACTION_SCHEMA_VERSION),
        "chunk_schema_version": str(CHUNK_SCHEMA_VERSION),
        "manifest_sha256": manifest_digest,
        "sources": str(summary.sources),
        "anchors": str(summary.anchors),
        "chunks": str(summary.chunks),
        "complete_sources": str(summary.complete_sources),
        "partial_sources": str(summary.partial_sources),
        "failed_sources": str(summary.failed_sources),
        "unresolved_pdf_pages": str(summary.unresolved_pdf_pages),
    }
    with connection:
        connection.executemany(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        for prepared in prepared_sources:
            source = prepared.source
            connection.execute(
                """
                INSERT INTO sources(
                    source_id, source_sha256, source_format, source_path,
                    original_paths_json, title, authors_json, extraction_status,
                    extraction_warnings_json, extraction_metadata_json,
                    unresolved_pdf_pages_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.source_id,
                    source.sha256,
                    source.source_format,
                    prepared.source_path,
                    _json(list(source.original_paths)),
                    prepared.title,
                    _json(list(prepared.authors)),
                    prepared.extraction_status,
                    _json(list(prepared.extraction_warnings)),
                    _json(prepared.extraction_metadata),
                    _json(list(prepared.unresolved_pdf_pages)),
                ),
            )
            connection.executemany(
                """
                INSERT INTO anchors(
                    anchor_id, source_id, locator_json, start_offset, end_offset
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        anchor.anchor_id,
                        source.source_id,
                        _json(anchor.locator.to_dict()),
                        anchor.locator.start_offset,
                        anchor.locator.end_offset,
                    )
                    for anchor in prepared.anchors
                ),
            )
            connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, evidence_id, source_id, text, start_offset,
                    end_offset, anchor_ids_json, locators_json, sentence_count,
                    lexical_token_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        chunk.chunk_id,
                        chunk.evidence_id,
                        source.source_id,
                        chunk.text,
                        chunk.start_offset,
                        chunk.end_offset,
                        _json(list(chunk.anchor_ids)),
                        _json([locator.to_dict() for locator in chunk.locators]),
                        chunk.sentence_count,
                        chunk.lexical_token_count,
                    )
                    for chunk in prepared.chunks
                ),
            )
            connection.execute(
                """
                INSERT INTO artifact_state(
                    source_id, extraction_size, extraction_mtime_ns,
                    extraction_sha256, chunk_size, chunk_mtime_ns, chunk_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source.source_id,
                    prepared.extraction_state.size_bytes,
                    prepared.extraction_state.mtime_ns,
                    prepared.extraction_state.sha256,
                    prepared.chunk_state.size_bytes,
                    prepared.chunk_state.mtime_ns,
                    prepared.chunk_state.sha256,
                ),
            )
        connection.execute("INSERT INTO chunk_fts(chunk_fts) VALUES ('rebuild')")


def _prepare_source(project_root: Path, source: SourceRecord) -> _PreparedSource:
    extraction, extraction_state = _read_json_stably(
        artifact_path_for(project_root, source.source_id)
    )
    chunks, chunk_state = _read_json_stably(
        chunk_artifact_path_for(project_root, source.source_id)
    )
    if extraction.get("schema_version") != EXTRACTION_SCHEMA_VERSION:
        raise RetrievalError(
            "extraction_schema_invalid",
            f"Source '{source.source_id}' has an unsupported extraction artifact schema.",
        )
    if chunks.get("schema_version") != CHUNK_SCHEMA_VERSION:
        raise RetrievalError(
            "chunk_schema_invalid",
            f"Source '{source.source_id}' has an unsupported chunk artifact schema.",
        )
    return _validate_artifacts(
        source, extraction, chunks, extraction_state, chunk_state
    )


def _validate_artifacts(
    source: SourceRecord,
    extraction: dict[str, Any],
    chunk_artifact: dict[str, Any],
    extraction_state: _ArtifactState,
    chunk_state: _ArtifactState,
) -> _PreparedSource:
    _require_equal(extraction, "source_id", source.source_id, "extraction")
    _require_equal(extraction, "source_sha256", source.sha256, "extraction")
    _require_equal(extraction, "source_format", source.source_format, "extraction")
    _require_equal(chunk_artifact, "source_id", source.source_id, "chunk")
    _require_equal(chunk_artifact, "source_sha256", source.sha256, "chunk")

    extraction_status = _status(
        extraction.get("status"), "extraction", source.source_id
    )
    chunk_status = _status(chunk_artifact.get("status"), "chunk", source.source_id)
    if extraction_status == "partial" and chunk_status == "complete":
        raise RetrievalError(
            "chunk_coverage_invalid",
            f"Partial source '{source.source_id}' cannot have complete chunk coverage.",
        )
    if extraction.get("text_offset_unit") != "unicode_codepoint":
        raise RetrievalError(
            "extraction_offset_unit_invalid",
            f"Source '{source.source_id}' does not use Unicode code-point offsets.",
        )
    text = extraction.get("text")
    if not isinstance(text, str):
        raise RetrievalError(
            "extraction_text_invalid",
            f"Source '{source.source_id}' extraction text must be a string.",
        )
    source_path = _required_string(extraction, "source_path", "extraction")
    if source_path not in source.original_paths:
        raise RetrievalError(
            "extraction_source_path_invalid",
            f"Source '{source.source_id}' extraction path is not registered in the manifest.",
        )
    metadata = extraction.get("metadata")
    if not isinstance(metadata, dict):
        raise RetrievalError(
            "extraction_metadata_invalid",
            f"Source '{source.source_id}' extraction metadata must be an object.",
        )
    warnings = _string_tuple(
        extraction.get("warnings"), "extraction warnings", source.source_id
    )

    raw_anchors = extraction.get("anchors")
    if not isinstance(raw_anchors, list):
        raise RetrievalError(
            "extraction_anchors_invalid",
            f"Source '{source.source_id}' extraction anchors must be an array.",
        )
    anchors: list[_PreparedAnchor] = []
    anchors_by_id: dict[str, _PreparedAnchor] = {}
    for raw_anchor in raw_anchors:
        if not isinstance(raw_anchor, dict):
            raise RetrievalError(
                "extraction_anchor_invalid",
                f"Source '{source.source_id}' contains a non-object anchor.",
            )
        anchor_id = _required_string(raw_anchor, "anchor_id", "anchor")
        if anchor_id in anchors_by_id:
            raise RetrievalError(
                "anchor_id_duplicate",
                f"Source '{source.source_id}' contains duplicate anchor '{anchor_id}'.",
            )
        anchor_text = raw_anchor.get("text")
        if not isinstance(anchor_text, str):
            raise RetrievalError(
                "anchor_text_invalid", f"Anchor '{anchor_id}' text must be a string."
            )
        locator = _locator_from_object(raw_anchor.get("locator"), anchor_id)
        if locator.source_id != source.source_id:
            raise RetrievalError(
                "anchor_source_invalid",
                f"Anchor '{anchor_id}' points to a different source ID.",
            )
        start, end = _validated_offsets(
            locator.start_offset,
            locator.end_offset,
            text_length=len(text),
            label=f"Anchor '{anchor_id}'",
        )
        if anchor_text != text[start:end]:
            raise RetrievalError(
                "anchor_slice_invalid",
                f"Anchor '{anchor_id}' text does not match its extraction offsets.",
            )
        prepared_anchor = _PreparedAnchor(anchor_id, locator, anchor_text)
        anchors.append(prepared_anchor)
        anchors_by_id[anchor_id] = prepared_anchor

    chunker = chunk_artifact.get("chunker")
    if not isinstance(chunker, dict):
        raise RetrievalError(
            "chunker_metadata_invalid",
            f"Source '{source.source_id}' chunker metadata must be an object.",
        )
    processor_name = _required_string(chunker, "sentence_processor", "chunker")
    sentence_model = _required_string(chunker, "sentence_model", "chunker")
    target_characters = _required_integer(chunker, "target_characters", minimum=1)
    max_characters = _required_integer(
        chunker, "max_characters", minimum=target_characters
    )
    overlap_sentences = _required_integer(chunker, "overlap_sentences", minimum=0)
    raw_chunks = chunk_artifact.get("chunks")
    if not isinstance(raw_chunks, list):
        raise RetrievalError(
            "chunks_invalid", f"Source '{source.source_id}' chunks must be an array."
        )
    if chunk_status == "failed" and raw_chunks:
        raise RetrievalError(
            "failed_chunks_invalid",
            f"Failed chunk artifact '{source.source_id}' cannot contain chunks.",
        )
    _string_tuple(chunk_artifact.get("warnings"), "chunk warnings", source.source_id)
    if extraction_status == "failed" and (
        chunk_status != "failed" or text or anchors or raw_chunks
    ):
        raise RetrievalError(
            "failed_extraction_lineage_invalid",
            f"Failed extraction '{source.source_id}' cannot contain searchable content.",
        )

    prepared_chunks: list[_PreparedChunk] = []
    seen_chunks: set[str] = set()
    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            raise RetrievalError(
                "chunk_invalid",
                f"Source '{source.source_id}' contains a non-object chunk.",
            )
        chunk_id = _required_string(raw_chunk, "chunk_id", "chunk")
        if chunk_id in seen_chunks:
            raise RetrievalError(
                "chunk_id_duplicate",
                f"Source '{source.source_id}' contains duplicate chunk '{chunk_id}'.",
            )
        _require_equal(raw_chunk, "source_id", source.source_id, "chunk")
        chunk_text = raw_chunk.get("text")
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise RetrievalError(
                "chunk_text_invalid", f"Chunk '{chunk_id}' must contain non-empty text."
            )
        start = raw_chunk.get("start_offset")
        end = raw_chunk.get("end_offset")
        start, end = _validated_offsets(
            start,
            end,
            text_length=len(text),
            label=f"Chunk '{chunk_id}'",
            require_nonempty=True,
        )
        if chunk_text != text[start:end]:
            raise RetrievalError(
                "chunk_slice_invalid",
                f"Chunk '{chunk_id}' text does not match its extraction offsets.",
            )
        expected_id = chunk_id_for(
            source_id=source.source_id,
            start_offset=start,
            end_offset=end,
            sentence_processor_name=processor_name,
            sentence_model=sentence_model,
            target_characters=target_characters,
            max_characters=max_characters,
            overlap_sentences=overlap_sentences,
            text=chunk_text,
        )
        if chunk_id != expected_id:
            raise RetrievalError(
                "chunk_id_invalid",
                f"Chunk '{chunk_id}' does not match its exact content fingerprint.",
            )
        raw_anchor_ids = raw_chunk.get("anchor_ids")
        if (
            not isinstance(raw_anchor_ids, list)
            or not raw_anchor_ids
            or any(not isinstance(item, str) or not item for item in raw_anchor_ids)
        ):
            raise RetrievalError(
                "chunk_anchor_ids_invalid",
                f"Chunk '{chunk_id}' must contain one or more anchor IDs.",
            )
        anchor_ids = tuple(raw_anchor_ids)
        expected_anchors = tuple(
            anchor
            for anchor in anchors
            if anchor.locator.start_offset is not None
            and anchor.locator.end_offset is not None
            and anchor.locator.start_offset < end
            and anchor.locator.end_offset > start
        )
        if anchor_ids != tuple(anchor.anchor_id for anchor in expected_anchors):
            raise RetrievalError(
                "chunk_anchor_range_invalid",
                f"Chunk '{chunk_id}' anchor IDs do not match its extraction range.",
            )
        raw_locators = raw_chunk.get("locators")
        if not isinstance(raw_locators, list) or len(raw_locators) != len(anchor_ids):
            raise RetrievalError(
                "chunk_locators_invalid",
                f"Chunk '{chunk_id}' locators do not match its anchor IDs.",
            )
        locators = tuple(
            _locator_from_object(value, f"chunk '{chunk_id}'") for value in raw_locators
        )
        expected_locators = tuple(
            replace(
                anchor.locator,
                start_offset=max(start, anchor.locator.start_offset or 0),
                end_offset=min(end, anchor.locator.end_offset or 0),
            )
            for anchor in expected_anchors
        )
        if locators != expected_locators:
            raise RetrievalError(
                "chunk_locator_range_invalid",
                f"Chunk '{chunk_id}' locators do not match the clamped anchor ranges.",
            )
        sentence_count = _required_integer(raw_chunk, "sentence_count", minimum=1)
        lexical_token_count = _required_integer(
            raw_chunk, "lexical_token_count", minimum=0
        )
        prepared_chunks.append(
            _PreparedChunk(
                chunk_id=chunk_id,
                evidence_id=evidence_id_for(chunk_id),
                text=chunk_text,
                start_offset=start,
                end_offset=end,
                anchor_ids=anchor_ids,
                locators=locators,
                sentence_count=sentence_count,
                lexical_token_count=lexical_token_count,
            )
        )
        seen_chunks.add(chunk_id)

    empty_pages = metadata.get("empty_pages", [])
    unresolved_pdf_pages = _integer_tuple(
        empty_pages, "metadata.empty_pages", source.source_id
    )
    title, authors = _bibliography_from(metadata)
    return _PreparedSource(
        source=source,
        source_path=source_path,
        title=title,
        authors=authors,
        extraction_status=extraction_status,
        extraction_warnings=warnings,
        extraction_metadata=dict(metadata),
        unresolved_pdf_pages=unresolved_pdf_pages,
        anchors=tuple(anchors),
        chunks=tuple(prepared_chunks),
        extraction_state=extraction_state,
        chunk_state=chunk_state,
    )


def _read_json_stably(path: Path) -> tuple[dict[str, Any], _ArtifactState]:
    try:
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise RetrievalError(
            "artifact_read_failed",
            f"Could not read required artifact '{path}': {error}.",
        ) from error
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(raw) != after.st_size
    ):
        raise RetrievalError(
            "artifact_changed",
            f"Artifact '{path}' changed while the index was reading it.",
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetrievalError(
            "artifact_json_invalid", f"Artifact '{path}' is not valid UTF-8 JSON."
        ) from error
    if not isinstance(payload, dict):
        raise RetrievalError(
            "artifact_shape_invalid", f"Artifact '{path}' root must be an object."
        )
    return payload, _ArtifactState(
        size_bytes=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=sha256(raw).hexdigest(),
    )


def _read_index_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        metadata = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT key, value FROM index_metadata")
        }
    except sqlite3.DatabaseError as error:
        raise RetrievalError(
            "index_schema_invalid",
            "Search index schema is missing or damaged; run 'corpusdock index'.",
        ) from error
    required = {
        "schema_version",
        "extraction_schema_version",
        "chunk_schema_version",
        "built_at",
        "manifest_sha256",
        "sources",
        "anchors",
        "chunks",
        "partial_sources",
        "unresolved_pdf_pages",
    }
    if (
        not required.issubset(metadata)
        or metadata.get("schema_version") != str(INDEX_SCHEMA_VERSION)
        or metadata.get("extraction_schema_version") != str(EXTRACTION_SCHEMA_VERSION)
        or metadata.get("chunk_schema_version") != str(CHUNK_SCHEMA_VERSION)
    ):
        raise RetrievalError(
            "index_schema_invalid",
            "Search index schema is unsupported; run 'corpusdock index'.",
        )
    numeric_keys = {
        "schema_version",
        "extraction_schema_version",
        "chunk_schema_version",
        "sources",
        "anchors",
        "chunks",
        "partial_sources",
        "unresolved_pdf_pages",
    }
    try:
        if any(int(metadata[key]) < 0 for key in numeric_keys):
            raise ValueError
    except ValueError as error:
        raise RetrievalError(
            "index_schema_invalid",
            "Search index metadata is damaged; run 'corpusdock index'.",
        ) from error
    user_version = connection.execute("PRAGMA user_version").fetchone()
    if user_version is None or user_version[0] != INDEX_SCHEMA_VERSION:
        raise RetrievalError(
            "index_schema_invalid",
            "Search index user version is unsupported; run 'corpusdock index'.",
        )
    return metadata


def _assert_index_current(
    connection: sqlite3.Connection,
    project_root: Path,
    metadata: dict[str, str],
) -> None:
    _, manifest_state = _read_json_stably(manifest_path_for(project_root))
    if manifest_state.sha256 != metadata["manifest_sha256"]:
        raise RetrievalError(
            "index_stale",
            "The source manifest changed after indexing; run 'corpusdock index'.",
        )
    rows = connection.execute(
        """
        SELECT source_id, extraction_size, extraction_mtime_ns,
               chunk_size, chunk_mtime_ns
        FROM artifact_state
        """
    ).fetchall()
    if len(rows) != int(metadata["sources"]):
        raise RetrievalError(
            "index_schema_invalid",
            "Search index artifact state is incomplete; run 'corpusdock index'.",
        )
    for row in rows:
        source_id = str(row["source_id"])
        paths_and_state = (
            (
                artifact_path_for(project_root, source_id),
                int(row["extraction_size"]),
                int(row["extraction_mtime_ns"]),
            ),
            (
                chunk_artifact_path_for(project_root, source_id),
                int(row["chunk_size"]),
                int(row["chunk_mtime_ns"]),
            ),
        )
        for path, expected_size, expected_mtime in paths_and_state:
            try:
                current = path.stat()
            except OSError as error:
                raise RetrievalError(
                    "index_stale",
                    f"Indexed artifact '{path}' is unavailable; run 'corpusdock index'.",
                ) from error
            if (
                current.st_size != expected_size
                or current.st_mtime_ns != expected_mtime
            ):
                raise RetrievalError(
                    "index_stale",
                    "A derived artifact changed after indexing; run 'corpusdock index'.",
                )


def _compile_fts_query(query: str, match_mode: MatchMode) -> str:
    if match_mode not in {"all", "any", "phrase"}:
        raise RetrievalError(
            "match_mode_invalid", f"Unknown full-text match mode '{match_mode}'."
        )
    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("query_empty", "Search query cannot be empty.")
    cleaned = query.strip()
    if match_mode == "phrase":
        if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) >= 2:
            cleaned = cleaned[1:-1]
        terms = (cleaned,)
    else:
        terms = tuple(
            quoted if quoted else token
            for quoted, token in re.findall(r'"([^"]+)"|([^\s"]+)', cleaned)
        )
    terms = tuple(term.strip() for term in terms if term and term.strip())
    if not terms or not any(re.search(r"\w", term, flags=re.UNICODE) for term in terms):
        raise RetrievalError(
            "query_terms_missing",
            "Search query must contain at least one word or number.",
        )
    operator = " OR " if match_mode == "any" else " AND "
    return operator.join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)


def _evidence_from_row(row: sqlite3.Row) -> EvidenceResult:
    raw_locators = _json_list(row["locators_json"], "indexed locators")
    locators = tuple(
        _locator_from_object(locator, f"indexed chunk '{row['chunk_id']}'")
        for locator in raw_locators
    )
    if not locators:
        raise RetrievalError(
            "index_evidence_invalid",
            f"Indexed chunk '{row['chunk_id']}' has no citation locator.",
        )
    anchor_ids_raw = _json_list(row["anchor_ids_json"], "indexed anchor IDs")
    if any(not isinstance(value, str) or not value for value in anchor_ids_raw):
        raise RetrievalError(
            "index_evidence_invalid",
            f"Indexed chunk '{row['chunk_id']}' has invalid anchor IDs.",
        )
    primary = _range_locator(
        locators,
        start_offset=int(row["start_offset"]),
        end_offset=int(row["end_offset"]),
    )
    warnings_raw = _json_list(row["extraction_warnings_json"], "coverage warnings")
    unresolved_raw = _json_list(
        row["unresolved_pdf_pages_json"], "unresolved PDF pages"
    )
    coverage = SourceCoverage(
        extraction_status=str(row["extraction_status"]),
        unresolved_pdf_pages=tuple(int(value) for value in unresolved_raw),
        warnings=tuple(str(value) for value in warnings_raw),
    )
    raw_rank = row["rank"]
    score = None if raw_rank is None else round(max(0.0, -float(raw_rank)), 12)
    citation = _format_citation(
        source_id=str(row["source_id"]),
        source_format=str(row["source_format"]),
        source_path=str(row["source_path"]),
        title=str(row["title"]) if row["title"] is not None else None,
        authors=tuple(
            str(value) for value in _json_list(row["authors_json"], "indexed authors")
        ),
        locator=primary,
    )
    return EvidenceResult(
        evidence_id=str(row["evidence_id"]),
        excerpt=str(row["text"]),
        citation=citation,
        locator=primary,
        source_path=str(row["source_path"]),
        verification_status="artifact-anchor-confirmed",
        score=score,
        chunk_id=str(row["chunk_id"]),
        anchor_ids=tuple(anchor_ids_raw),
        locators=locators,
        source_coverage=coverage,
    )


def _range_locator(
    locators: Sequence[CitationLocator], *, start_offset: int, end_offset: int
) -> CitationLocator:
    first = locators[0]
    return CitationLocator(
        source_id=first.source_id,
        locator_type=first.locator_type,
        label=_locator_range_label(locators),
        page=_same_value(locators, "page"),
        page_label=_same_value(locators, "page_label"),
        chapter=_same_value(locators, "chapter"),
        heading=_same_value(locators, "heading"),
        spine_item=_same_value(locators, "spine_item"),
        paragraph_id=_same_value(locators, "paragraph_id"),
        line_start=min(
            (
                locator.line_start
                for locator in locators
                if locator.line_start is not None
            ),
            default=None,
        ),
        line_end=max(
            (locator.line_end for locator in locators if locator.line_end is not None),
            default=None,
        ),
        start_offset=start_offset,
        end_offset=end_offset,
        extraction_method=_same_value(locators, "extraction_method"),
    )


def _same_value(locators: Sequence[CitationLocator], field: str) -> str | int | None:
    values = {getattr(locator, field) for locator in locators}
    return values.pop() if len(values) == 1 else None


def _locator_range_label(locators: Sequence[CitationLocator]) -> str:
    first = locators[0]
    last = locators[-1]
    if first.locator_type == "text_line":
        line_start = min(
            (
                locator.line_start
                for locator in locators
                if locator.line_start is not None
            ),
            default=None,
        )
        line_end = max(
            (locator.line_end for locator in locators if locator.line_end is not None),
            default=None,
        )
        if line_start is not None and line_end is not None:
            return (
                f"line {line_start}"
                if line_start == line_end
                else f"lines {line_start}–{line_end}"
            )
    if first.label == last.label:
        return first.label
    return f"{first.label}–{last.label}"


def _format_citation(
    *,
    source_id: str,
    source_format: str,
    source_path: str,
    title: str | None,
    authors: Sequence[str],
    locator: CitationLocator,
) -> str:
    if title:
        identity = f"{'; '.join(authors)}. {title}" if authors else title
    else:
        identity = f"{source_format.upper()} source {Path(source_path).name}"
    return f"{identity} [{source_id}], {locator.label}"


def _verify_original(source: SourceRecord, extracted_path: str) -> str:
    candidates = tuple(dict.fromkeys((extracted_path, *source.original_paths)))
    readable_paths: list[str] = []
    failures: list[str] = []
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_file():
            continue
        readable_paths.append(str(path))
        try:
            digest, size_bytes = hash_source_file(path)
        except SourceRegistrationError as error:
            failures.append(f"{path}: {error}")
            continue
        if digest == source.sha256 and size_bytes == source.size_bytes:
            return str(path.resolve())
    if not readable_paths:
        raise RetrievalError(
            "source_original_unavailable",
            f"No registered original path for source '{source.source_id}' is currently available.",
        )
    detail = f" ({failures[0]})" if failures else ""
    raise RetrievalError(
        "source_checksum_mismatch",
        f"Available original files do not match source '{source.source_id}'{detail}.",
    )


def _index_summary(
    path: Path, built_at: str, prepared_sources: Sequence[_PreparedSource]
) -> IndexSummary:
    status_counts = {"complete": 0, "partial": 0, "failed": 0}
    for source in prepared_sources:
        status_counts[source.extraction_status] += 1
    return IndexSummary(
        path=str(path),
        schema_version=INDEX_SCHEMA_VERSION,
        built_at=built_at,
        sources=len(prepared_sources),
        anchors=sum(len(source.anchors) for source in prepared_sources),
        chunks=sum(len(source.chunks) for source in prepared_sources),
        complete_sources=status_counts["complete"],
        partial_sources=status_counts["partial"],
        failed_sources=status_counts["failed"],
        unresolved_pdf_pages=sum(
            len(source.unresolved_pdf_pages) for source in prepared_sources
        ),
    )


def _bibliography_from(metadata: dict[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    title_value = metadata.get("title")
    title = (
        title_value.strip()
        if isinstance(title_value, str) and title_value.strip()
        else None
    )
    author_value = metadata.get(
        "authors", metadata.get("author", metadata.get("creator"))
    )
    if isinstance(author_value, str):
        authors = (author_value.strip(),) if author_value.strip() else ()
    elif isinstance(author_value, list):
        authors = tuple(
            value.strip()
            for value in author_value
            if isinstance(value, str) and value.strip()
        )
    else:
        authors = ()
    return title, authors


def _locator_from_object(value: object, label: str) -> CitationLocator:
    if not isinstance(value, dict):
        raise RetrievalError(
            "locator_invalid", f"Locator for {label} must be an object."
        )
    source_id = _required_string(value, "source_id", "locator")
    locator_type = _required_string(value, "locator_type", "locator")
    locator_label = _required_string(value, "label", "locator")
    return CitationLocator(
        source_id=source_id,
        locator_type=locator_type,
        label=locator_label,
        page=_optional_integer(value.get("page"), "page"),
        page_label=_optional_string(value.get("page_label"), "page_label"),
        chapter=_optional_string(value.get("chapter"), "chapter"),
        heading=_optional_string(value.get("heading"), "heading"),
        spine_item=_optional_string(value.get("spine_item"), "spine_item"),
        paragraph_id=_optional_string(value.get("paragraph_id"), "paragraph_id"),
        line_start=_optional_integer(value.get("line_start"), "line_start"),
        line_end=_optional_integer(value.get("line_end"), "line_end"),
        start_offset=_optional_integer(value.get("start_offset"), "start_offset"),
        end_offset=_optional_integer(value.get("end_offset"), "end_offset"),
        extraction_method=_optional_string(
            value.get("extraction_method"), "extraction_method"
        ),
    )


def _validated_offsets(
    start: object,
    end: object,
    *,
    text_length: int,
    label: str,
    require_nonempty: bool = False,
) -> tuple[int, int]:
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end < start
        or end > text_length
        or (require_nonempty and end == start)
    ):
        raise RetrievalError(
            "artifact_offsets_invalid", f"{label} has invalid Unicode offsets."
        )
    return start, end


def _required_string(value: dict[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise RetrievalError(
            "artifact_field_invalid",
            f"{label} field '{key}' must be a non-empty string.",
        )
    return result


def _required_integer(value: dict[str, Any], key: str, *, minimum: int) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result < minimum:
        raise RetrievalError(
            "artifact_field_invalid",
            f"Field '{key}' must be an integer no smaller than {minimum}.",
        )
    return result


def _optional_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RetrievalError(
            "locator_field_invalid",
            f"Locator field '{label}' must be a string or null.",
        )
    return value


def _optional_integer(value: object, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise RetrievalError(
            "locator_field_invalid",
            f"Locator field '{label}' must be an integer or null.",
        )
    return value


def _require_equal(value: dict[str, Any], key: str, expected: str, label: str) -> None:
    if value.get(key) != expected:
        raise RetrievalError(
            "artifact_lineage_invalid",
            f"{label.title()} field '{key}' does not match registered source '{expected}'.",
        )


def _status(value: object, label: str, source_id: str) -> str:
    if value not in {"complete", "partial", "failed"}:
        raise RetrievalError(
            "artifact_status_invalid",
            f"Source '{source_id}' has invalid {label} status '{value}'.",
        )
    return str(value)


def _string_tuple(value: object, label: str, source_id: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RetrievalError(
            "artifact_field_invalid",
            f"Source '{source_id}' {label} must be an array of strings.",
        )
    return tuple(value)


def _integer_tuple(value: object, label: str, source_id: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 1
        for item in value
    ):
        raise RetrievalError(
            "artifact_field_invalid",
            f"Source '{source_id}' {label} must be an array of positive integers.",
        )
    return tuple(value)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_list(value: object, label: str) -> list[object]:
    if not isinstance(value, str):
        raise RetrievalError("index_evidence_invalid", f"{label} must be JSON text.")
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise RetrievalError(
            "index_evidence_invalid", f"{label} contains invalid JSON."
        ) from error
    if not isinstance(result, list):
        raise RetrievalError("index_evidence_invalid", f"{label} must be an array.")
    return result


_SEARCH_COLUMNS = """
    c.chunk_id, c.evidence_id, c.source_id, c.text, c.start_offset,
    c.end_offset, c.anchor_ids_json, c.locators_json,
    s.source_format, s.source_path, s.title, s.authors_json,
    s.extraction_status, s.extraction_warnings_json,
    s.unresolved_pdf_pages_json
"""

_SEARCH_SQL = f"""
    SELECT {_SEARCH_COLUMNS}, bm25(chunk_fts) AS rank
    FROM chunk_fts
    JOIN chunks AS c ON c.rowid = chunk_fts.rowid
    JOIN sources AS s ON s.source_id = c.source_id
    WHERE chunk_fts MATCH ?{{source_clause}}
    ORDER BY rank ASC, c.chunk_id ASC
    LIMIT ?
"""

_VERIFY_SQL = f"""
    SELECT {_SEARCH_COLUMNS}, NULL AS rank
    FROM chunks AS c
    JOIN sources AS s ON s.source_id = c.source_id
    WHERE c.evidence_id = ?
"""
