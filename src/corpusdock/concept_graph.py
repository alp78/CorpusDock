"""Evidence-linked corpus concept resolution and graph querying.

The graph in this module is a disposable relational projection over one completed
analysis run.  Exact evidence and the analysis candidate database remain
authoritative.  Resolution is deliberately conservative: only labels and concept
types that are equivalent after Unicode/case/edge-punctuation normalization share a
node.  Broader semantic aliases require a separate reviewed proposal rather than a
silent merge.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Literal
import unicodedata
from uuid import uuid4

from corpusdock import __version__
from corpusdock.analysis_store import (
    ANALYSIS_DATABASE_SCHEMA_VERSION,
    AnalysisStore,
    AnalysisStoreError,
    analysis_database_path_for,
)
from corpusdock.contracts import EvidenceResult
from corpusdock.manifest import utc_now
from corpusdock.retrieval import SQLiteSearchBackend


CONCEPT_GRAPH_SCHEMA_VERSION = 1
CONCEPT_GRAPH_QUERY_SCHEMA_VERSION = 1
CONCEPT_GRAPH_FILE_NAME = "graph.sqlite3"
CONCEPT_RESOLUTION_POLICY = "normalized-label-and-type-v1"
AUTOMATIC_RUN_SELECTION = "maximum-completed-coverage-v1"
EXPLICIT_RUN_SELECTION = "explicit-run-v1"
DEFAULT_GRAPH_QUERY_LIMIT = 10
MAX_GRAPH_QUERY_LIMIT = 100
DEFAULT_GRAPH_EVIDENCE_LIMIT = 3
MAX_GRAPH_EVIDENCE_LIMIT = 20

_RUN_ID_PATTERN = re.compile(r"run-[0-9a-f]{64}")
_DIGEST_PATTERN = re.compile(r"(?:sha256:)?[0-9a-f]{64}")
_REVIEW_STATES = {"unreviewed", "accepted", "rejected", "needs_review"}
_EDGE_PUNCTUATION = " \t\r\n.,;:!?()[]{}<>\"'`´‘’“”«»…"
_METADATA_FIELDS = {
    "schema_version",
    "tool_version",
    "built_at",
    "source_index_fingerprint",
    "analysis_database_schema_version",
    "analysis_run_id",
    "analysis_run_selection",
    "analysis_extractor_fingerprint",
    "analysis_prompt_version",
    "analysis_prompt_sha256",
    "analysis_projection_fingerprint",
    "resolution_policy",
    "indexed_sources",
    "represented_sources",
    "analyzed_evidence",
    "concept_mentions",
    "concepts",
    "claims",
    "claim_concept_links",
    "relations",
    "excluded_concepts",
    "excluded_claims",
    "excluded_relations",
    "graph_content_fingerprint",
}


class ConceptGraphError(Exception):
    """A missing, stale, corrupt, or invalid derived concept graph."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ConceptGraphDescriptor:
    """Non-content provenance and counts for one graph projection."""

    path: str
    schema_version: int
    built_at: str
    source_index_fingerprint: str
    analysis_run_id: str
    analysis_run_selection: str
    analysis_extractor_fingerprint: str
    analysis_prompt_version: str
    analysis_prompt_sha256: str
    analysis_projection_fingerprint: str
    resolution_policy: str
    indexed_sources: int
    represented_sources: int
    analyzed_evidence: int
    concept_mentions: int
    concepts: int
    claims: int
    claim_concept_links: int
    relations: int
    excluded_concepts: int
    excluded_claims: int
    excluded_relations: int
    database_size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "schema_version": self.schema_version,
            "built_at": self.built_at,
            "source_index_fingerprint": self.source_index_fingerprint,
            "analysis": {
                "run_id": self.analysis_run_id,
                "run_selection": self.analysis_run_selection,
                "extractor_fingerprint": self.analysis_extractor_fingerprint,
                "prompt_version": self.analysis_prompt_version,
                "prompt_sha256": self.analysis_prompt_sha256,
                "projection_fingerprint": self.analysis_projection_fingerprint,
            },
            "resolution_policy": self.resolution_policy,
            "counts": {
                "indexed_sources": self.indexed_sources,
                "represented_sources": self.represented_sources,
                "analyzed_evidence": self.analyzed_evidence,
                "concept_mentions": self.concept_mentions,
                "concepts": self.concepts,
                "claims": self.claims,
                "claim_concept_links": self.claim_concept_links,
                "relations": self.relations,
                "excluded_concepts": self.excluded_concepts,
                "excluded_claims": self.excluded_claims,
                "excluded_relations": self.excluded_relations,
            },
            "database_size_bytes": self.database_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ConceptSurfaceForm:
    """One observed spelling/type pair retained beneath a resolved node."""

    label: str
    concept_type: str
    mentions: int
    sources: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "concept_type": self.concept_type,
            "mentions": self.mentions,
            "sources": self.sources,
        }


@dataclass(frozen=True, slots=True)
class ConceptSupport:
    """One exact support span resolved through the authoritative evidence index."""

    concept_candidate_id: str
    candidate_label: str
    candidate_type: str
    confidence: float
    review_state: str
    start_offset: int
    end_offset: int
    text_sha256: str
    text: str
    evidence: EvidenceResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_candidate_id": self.concept_candidate_id,
            "candidate_label": self.candidate_label,
            "candidate_type": self.candidate_type,
            "confidence": self.confidence,
            "review_state": self.review_state,
            "support": {
                "start_offset": self.start_offset,
                "end_offset": self.end_offset,
                "text_sha256": self.text_sha256,
                "text": self.text,
            },
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ConceptRelationSummary:
    """A stance-preserving aggregate of candidate relations to one neighbor."""

    direction: Literal["outgoing", "incoming"]
    relation_type: str
    predicate: str
    polarity: str
    certainty: str
    conditional: bool
    attribution: str
    normative_force: str
    neighbor_concept_id: str
    neighbor_label: str
    relations: int
    sources: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "relation_type": self.relation_type,
            "predicate": self.predicate,
            "polarity": self.polarity,
            "certainty": self.certainty,
            "conditional": self.conditional,
            "attribution": self.attribution,
            "normative_force": self.normative_force,
            "neighbor": {
                "concept_id": self.neighbor_concept_id,
                "label": self.neighbor_label,
            },
            "relations": self.relations,
            "sources": self.sources,
        }


@dataclass(frozen=True, slots=True)
class ConceptGraphResult:
    """One resolved concept with exact support and stance-preserving neighbors."""

    rank: int
    concept_id: str
    canonical_label: str
    canonical_type: str
    resolution_method: str
    mentions: int
    sources: int
    claims: int
    relations: int
    mean_confidence: float
    max_confidence: float
    surface_forms: tuple[ConceptSurfaceForm, ...]
    support: tuple[ConceptSupport, ...]
    neighbors: tuple[ConceptRelationSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "concept_id": self.concept_id,
            "canonical_label": self.canonical_label,
            "canonical_type": self.canonical_type,
            "resolution_method": self.resolution_method,
            "counts": {
                "mentions": self.mentions,
                "sources": self.sources,
                "claims": self.claims,
                "relations": self.relations,
            },
            "confidence": {
                "mean": self.mean_confidence,
                "maximum": self.max_confidence,
            },
            "surface_forms": [item.to_dict() for item in self.surface_forms],
            "support": [item.to_dict() for item in self.support],
            "neighbors": [item.to_dict() for item in self.neighbors],
        }


@dataclass(frozen=True, slots=True)
class ConceptGraphQueryResponse:
    """Versioned concept-query response over one validated graph snapshot."""

    query: str
    results: tuple[ConceptGraphResult, ...]
    graph: ConceptGraphDescriptor

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONCEPT_GRAPH_QUERY_SCHEMA_VERSION,
            "query": self.query,
            "result_count": len(self.results),
            "graph": {
                "built_at": self.graph.built_at,
                "analysis_run_id": self.graph.analysis_run_id,
                "resolution_policy": self.graph.resolution_policy,
                "concepts": self.graph.concepts,
                "concept_mentions": self.graph.concept_mentions,
            },
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(slots=True)
class _ConceptGroup:
    concept_id: str
    normalized_label: str
    normalized_type: str
    labels: Counter[str]
    types: Counter[str]
    accepted_labels: Counter[str]
    accepted_types: Counter[str]
    sources: set[str]
    confidence_sum: float = 0.0
    max_confidence: float = 0.0
    mentions: int = 0

    def add(
        self,
        *,
        label: str,
        concept_type: str,
        source_id: str,
        confidence: float,
        review_state: str,
    ) -> None:
        self.labels[label] += 1
        self.types[concept_type] += 1
        if review_state == "accepted":
            self.accepted_labels[label] += 1
            self.accepted_types[concept_type] += 1
        self.sources.add(source_id)
        self.confidence_sum += confidence
        self.max_confidence = max(self.max_confidence, confidence)
        self.mentions += 1

    @property
    def canonical_label(self) -> str:
        return _preferred_surface(self.labels, self.accepted_labels)

    @property
    def canonical_type(self) -> str:
        return _preferred_surface(self.types, self.accepted_types)


def concept_graph_path_for(project_root: Path | str) -> Path:
    """Return the ignored project-local concept graph path."""

    return (
        Path(project_root).expanduser().resolve()
        / ".corpusdock"
        / CONCEPT_GRAPH_FILE_NAME
    )


def build_concept_graph(
    project_root: Path | str,
    *,
    run_id: str | None = None,
    now: Callable[[], str] = utc_now,
) -> ConceptGraphDescriptor:
    """Atomically build a conservative graph projection from a completed run."""

    root = Path(project_root).expanduser().resolve()
    analysis_path = analysis_database_path_for(root)
    if not analysis_path.is_file():
        raise ConceptGraphError(
            "concept_graph_analysis_missing",
            "No derived analysis database exists. Run 'corpusdock analyze' first.",
        )
    if run_id is not None and _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ConceptGraphError(
            "concept_graph_run_id_invalid",
            "Analysis run ID must be a 'run-' SHA-256 identifier.",
        )

    try:
        analysis_store = AnalysisStore(root)
    except AnalysisStoreError as error:
        raise ConceptGraphError("concept_graph_analysis_invalid", str(error)) from error
    snapshot = analysis_store.snapshot
    selection = (
        EXPLICIT_RUN_SELECTION if run_id is not None else AUTOMATIC_RUN_SELECTION
    )

    with closing(_connect_analysis_read_only(analysis_path)) as analysis:
        selected_run = _select_completed_run(
            analysis,
            source_index_fingerprint=snapshot.index_fingerprint,
            run_id=run_id,
        )
        selected_id = str(selected_run["run_id"])
        projection_fingerprint = _analysis_projection_fingerprint(analysis, selected_id)

    path = concept_graph_path_for(root)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    connection: sqlite3.Connection | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(temporary_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        _create_schema(connection)

        with closing(_connect_analysis_read_only(analysis_path)) as analysis:
            counts = _write_projection(
                connection,
                analysis,
                run_id=selected_id,
                indexed_sources=snapshot.indexed_sources,
            )

        graph_fingerprint = _graph_content_fingerprint(connection)
        metadata = {
            "schema_version": str(CONCEPT_GRAPH_SCHEMA_VERSION),
            "tool_version": __version__,
            "built_at": now(),
            "source_index_fingerprint": snapshot.index_fingerprint,
            "analysis_database_schema_version": str(ANALYSIS_DATABASE_SCHEMA_VERSION),
            "analysis_run_id": selected_id,
            "analysis_run_selection": selection,
            "analysis_extractor_fingerprint": str(
                selected_run["extractor_fingerprint"]
            ),
            "analysis_prompt_version": str(selected_run["prompt_version"]),
            "analysis_prompt_sha256": str(selected_run["prompt_sha256"]),
            "analysis_projection_fingerprint": projection_fingerprint,
            "resolution_policy": CONCEPT_RESOLUTION_POLICY,
            "graph_content_fingerprint": graph_fingerprint,
            **{key: str(value) for key, value in counts.items()},
        }
        connection.executemany(
            "INSERT INTO graph_metadata(key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.commit()
        _validate_graph_connection(connection)
        connection.close()
        connection = None

        with temporary_path.open("r+b") as graph_file:
            os.fsync(graph_file.fileno())

        exact_backend = SQLiteSearchBackend(root)
        exact_backend.assert_snapshot_current(snapshot)
        with closing(_connect_analysis_read_only(analysis_path)) as analysis:
            current_run = _select_completed_run(
                analysis,
                source_index_fingerprint=snapshot.index_fingerprint,
                run_id=selected_id if selection == EXPLICIT_RUN_SELECTION else None,
            )
            if str(current_run["run_id"]) != selected_id:
                raise ConceptGraphError(
                    "concept_graph_analysis_changed",
                    "The preferred completed analysis run changed during graph build.",
                )
            if (
                _analysis_projection_fingerprint(analysis, selected_id)
                != projection_fingerprint
            ):
                raise ConceptGraphError(
                    "concept_graph_analysis_changed",
                    "Analysis candidates or reviews changed during graph build.",
                )
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except ConceptGraphError:
        raise
    except sqlite3.OperationalError as error:
        if "fts5" in str(error).casefold():
            raise ConceptGraphError(
                "concept_graph_fts5_unavailable",
                "This Python SQLite build does not provide FTS5 full-text search.",
            ) from error
        raise ConceptGraphError(
            "concept_graph_build_failed",
            f"Could not build the local concept graph: {error}.",
        ) from error
    except (OSError, sqlite3.DatabaseError) as error:
        raise ConceptGraphError(
            "concept_graph_build_failed",
            f"Could not build the local concept graph: {error}.",
        ) from error
    finally:
        if connection is not None:
            connection.close()
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

    return read_current_concept_graph_descriptor(root)


def read_concept_graph_descriptor(
    project_root: Path | str,
) -> ConceptGraphDescriptor:
    """Read and fully checksum a graph without validating upstream lineage."""

    path = concept_graph_path_for(project_root)
    if not path.is_file():
        raise ConceptGraphError(
            "concept_graph_missing",
            f"No concept graph found at '{path}'. Run 'corpusdock graph build' first.",
        )
    try:
        with closing(_connect_graph_read_only(path)) as connection:
            _validate_graph_connection(connection)
            metadata = _read_metadata(connection)
            _validate_metadata(metadata)
            observed_fingerprint = _graph_content_fingerprint(connection)
            if observed_fingerprint != metadata["graph_content_fingerprint"]:
                raise ConceptGraphError(
                    "concept_graph_content_invalid",
                    "Concept graph content does not match its recorded checksum.",
                )
            counts = _validated_counts(connection, metadata)
    except ConceptGraphError:
        raise
    except (OSError, sqlite3.DatabaseError) as error:
        raise ConceptGraphError(
            "concept_graph_read_failed",
            f"Could not read the local concept graph: {error}.",
        ) from error

    return ConceptGraphDescriptor(
        path=str(path),
        schema_version=CONCEPT_GRAPH_SCHEMA_VERSION,
        built_at=metadata["built_at"],
        source_index_fingerprint=metadata["source_index_fingerprint"],
        analysis_run_id=metadata["analysis_run_id"],
        analysis_run_selection=metadata["analysis_run_selection"],
        analysis_extractor_fingerprint=metadata["analysis_extractor_fingerprint"],
        analysis_prompt_version=metadata["analysis_prompt_version"],
        analysis_prompt_sha256=metadata["analysis_prompt_sha256"],
        analysis_projection_fingerprint=metadata["analysis_projection_fingerprint"],
        resolution_policy=metadata["resolution_policy"],
        indexed_sources=counts["indexed_sources"],
        represented_sources=counts["represented_sources"],
        analyzed_evidence=counts["analyzed_evidence"],
        concept_mentions=counts["concept_mentions"],
        concepts=counts["concepts"],
        claims=counts["claims"],
        claim_concept_links=counts["claim_concept_links"],
        relations=counts["relations"],
        excluded_concepts=counts["excluded_concepts"],
        excluded_claims=counts["excluded_claims"],
        excluded_relations=counts["excluded_relations"],
        database_size_bytes=path.stat().st_size,
    )


def read_current_concept_graph_descriptor(
    project_root: Path | str,
) -> ConceptGraphDescriptor:
    """Read a graph and reject stale exact-index or analysis lineage."""

    root = Path(project_root).expanduser().resolve()
    descriptor = read_concept_graph_descriptor(root)
    analysis_path = analysis_database_path_for(root)
    if not analysis_path.is_file():
        raise ConceptGraphError(
            "concept_graph_stale",
            "The analysis database used by the concept graph is missing.",
        )
    try:
        store = AnalysisStore(root)
    except AnalysisStoreError as error:
        raise ConceptGraphError("concept_graph_stale", str(error)) from error
    if descriptor.source_index_fingerprint != store.snapshot.index_fingerprint:
        raise ConceptGraphError(
            "concept_graph_stale",
            "The exact corpus changed after the concept graph was built.",
        )

    with closing(_connect_analysis_read_only(analysis_path)) as analysis:
        preferred = _select_completed_run(
            analysis,
            source_index_fingerprint=store.snapshot.index_fingerprint,
            run_id=(
                descriptor.analysis_run_id
                if descriptor.analysis_run_selection == EXPLICIT_RUN_SELECTION
                else None
            ),
        )
        if str(preferred["run_id"]) != descriptor.analysis_run_id:
            raise ConceptGraphError(
                "concept_graph_stale",
                "A more complete or newer preferred analysis run is now available.",
            )
        current_fingerprint = _analysis_projection_fingerprint(
            analysis, descriptor.analysis_run_id
        )
    if current_fingerprint != descriptor.analysis_projection_fingerprint:
        raise ConceptGraphError(
            "concept_graph_stale",
            "Analysis candidates or review states changed after graph construction.",
        )
    return descriptor


def concept_graph_status_report(project_root: Path | str) -> dict[str, Any]:
    """Return non-content graph health metadata for ``corpusdock doctor``."""

    path = concept_graph_path_for(project_root)
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    try:
        descriptor = read_current_concept_graph_descriptor(project_root)
        return {"status": "ready", **descriptor.to_dict()}
    except ConceptGraphError as error:
        stale_codes = {
            "concept_graph_stale",
            "concept_graph_analysis_missing",
            "concept_graph_analysis_changed",
            "concept_graph_run_missing",
        }
        return {
            "status": "stale" if error.code in stale_codes else "invalid",
            "path": str(path),
            "error": str(error),
        }


def query_concept_graph(
    project_root: Path | str,
    query: str,
    *,
    limit: int = DEFAULT_GRAPH_QUERY_LIMIT,
    evidence_limit: int = DEFAULT_GRAPH_EVIDENCE_LIMIT,
) -> ConceptGraphQueryResponse:
    """Search resolved concepts and return exact citation-ready support."""

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_GRAPH_QUERY_LIMIT
    ):
        raise ConceptGraphError(
            "concept_graph_query_limit_invalid",
            f"Graph query limit must be between 1 and {MAX_GRAPH_QUERY_LIMIT}.",
        )
    if (
        not isinstance(evidence_limit, int)
        or isinstance(evidence_limit, bool)
        or not 1 <= evidence_limit <= MAX_GRAPH_EVIDENCE_LIMIT
    ):
        raise ConceptGraphError(
            "concept_graph_evidence_limit_invalid",
            f"Graph evidence limit must be between 1 and {MAX_GRAPH_EVIDENCE_LIMIT}.",
        )
    fts_query = _compile_graph_query(query)
    root = Path(project_root).expanduser().resolve()
    descriptor = read_current_concept_graph_descriptor(root)
    snapshot = SQLiteSearchBackend(root).corpus_snapshot()
    evidence_by_id = {item.evidence_id: item for item in snapshot.evidence}

    try:
        with closing(_connect_graph_read_only(Path(descriptor.path))) as connection:
            rows = connection.execute(
                """
                SELECT c.*, bm25(concept_search) AS search_rank
                FROM concept_search
                JOIN concepts c ON c.concept_id = concept_search.concept_id
                WHERE concept_search MATCH ?
                ORDER BY search_rank, c.source_count DESC,
                         c.mention_count DESC, c.concept_id
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
            results = tuple(
                _concept_result(
                    connection,
                    row,
                    rank=ordinal,
                    evidence_limit=evidence_limit,
                    evidence_by_id=evidence_by_id,
                )
                for ordinal, row in enumerate(rows, start=1)
            )
    except ConceptGraphError:
        raise
    except sqlite3.DatabaseError as error:
        raise ConceptGraphError(
            "concept_graph_query_failed",
            f"Could not query the local concept graph: {error}.",
        ) from error

    SQLiteSearchBackend(root).assert_snapshot_current(snapshot)
    return ConceptGraphQueryResponse(query=query, results=results, graph=descriptor)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        f"""
        PRAGMA user_version = {CONCEPT_GRAPH_SCHEMA_VERSION};
        PRAGMA foreign_keys = ON;

        CREATE TABLE graph_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE concepts (
            concept_id TEXT PRIMARY KEY,
            canonical_label TEXT NOT NULL,
            normalized_label TEXT NOT NULL,
            canonical_type TEXT NOT NULL,
            normalized_type TEXT NOT NULL,
            resolution_method TEXT NOT NULL,
            mention_count INTEGER NOT NULL CHECK (mention_count > 0),
            source_count INTEGER NOT NULL CHECK (source_count > 0),
            claim_count INTEGER NOT NULL CHECK (claim_count >= 0),
            relation_count INTEGER NOT NULL CHECK (relation_count >= 0),
            mean_confidence REAL NOT NULL CHECK (
                mean_confidence >= 0 AND mean_confidence <= 1
            ),
            max_confidence REAL NOT NULL CHECK (
                max_confidence >= 0 AND max_confidence <= 1
            ),
            UNIQUE (normalized_label, normalized_type)
        );

        CREATE TABLE concept_mentions (
            concept_candidate_id TEXT PRIMARY KEY,
            concept_id TEXT NOT NULL REFERENCES concepts(concept_id),
            evidence_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            label TEXT NOT NULL,
            description TEXT NOT NULL,
            concept_type TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            review_state TEXT NOT NULL CHECK (
                review_state IN ('unreviewed', 'accepted', 'needs_review')
            ),
            support_span_id TEXT NOT NULL,
            support_start_offset INTEGER NOT NULL CHECK (support_start_offset >= 0),
            support_end_offset INTEGER NOT NULL CHECK (
                support_end_offset > support_start_offset
            ),
            support_text_sha256 TEXT NOT NULL
        );

        CREATE INDEX concept_mentions_concept_idx
            ON concept_mentions(concept_id, source_id, confidence DESC);
        CREATE INDEX concept_mentions_evidence_idx
            ON concept_mentions(evidence_id);

        CREATE TABLE claims (
            claim_id TEXT PRIMARY KEY,
            evidence_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            statement TEXT NOT NULL,
            claim_type TEXT NOT NULL,
            polarity TEXT NOT NULL,
            certainty TEXT NOT NULL,
            conditional INTEGER NOT NULL CHECK (conditional IN (0, 1)),
            attribution TEXT NOT NULL,
            normative_force TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            review_state TEXT NOT NULL CHECK (
                review_state IN ('unreviewed', 'accepted', 'needs_review')
            ),
            support_span_id TEXT NOT NULL,
            support_start_offset INTEGER NOT NULL CHECK (support_start_offset >= 0),
            support_end_offset INTEGER NOT NULL CHECK (
                support_end_offset > support_start_offset
            ),
            support_text_sha256 TEXT NOT NULL
        );

        CREATE INDEX claims_source_idx ON claims(source_id);
        CREATE INDEX claims_stance_idx ON claims(
            claim_type, polarity, certainty, conditional,
            attribution, normative_force
        );

        CREATE TABLE claim_concepts (
            claim_id TEXT NOT NULL REFERENCES claims(claim_id),
            concept_id TEXT NOT NULL REFERENCES concepts(concept_id),
            concept_candidate_id TEXT NOT NULL
                REFERENCES concept_mentions(concept_candidate_id),
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (claim_id, concept_candidate_id),
            UNIQUE (claim_id, ordinal)
        );

        CREATE INDEX claim_concepts_concept_idx
            ON claim_concepts(concept_id, claim_id);

        CREATE TABLE relations (
            relation_id TEXT PRIMARY KEY,
            evidence_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            claim_id TEXT NOT NULL REFERENCES claims(claim_id),
            subject_concept_id TEXT NOT NULL REFERENCES concepts(concept_id),
            subject_candidate_id TEXT NOT NULL
                REFERENCES concept_mentions(concept_candidate_id),
            relation_type TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_concept_id TEXT NOT NULL REFERENCES concepts(concept_id),
            object_candidate_id TEXT NOT NULL
                REFERENCES concept_mentions(concept_candidate_id),
            polarity TEXT NOT NULL,
            certainty TEXT NOT NULL,
            conditional INTEGER NOT NULL CHECK (conditional IN (0, 1)),
            attribution TEXT NOT NULL,
            normative_force TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            review_state TEXT NOT NULL CHECK (
                review_state IN ('unreviewed', 'accepted', 'needs_review')
            ),
            support_span_id TEXT NOT NULL,
            support_start_offset INTEGER NOT NULL CHECK (support_start_offset >= 0),
            support_end_offset INTEGER NOT NULL CHECK (
                support_end_offset > support_start_offset
            ),
            support_text_sha256 TEXT NOT NULL
        );

        CREATE INDEX relations_subject_idx
            ON relations(subject_concept_id, relation_type, object_concept_id);
        CREATE INDEX relations_object_idx
            ON relations(object_concept_id, relation_type, subject_concept_id);
        CREATE INDEX relations_claim_idx ON relations(claim_id);

        CREATE VIRTUAL TABLE concept_search USING fts5(
            concept_id UNINDEXED,
            search_text,
            tokenize = 'unicode61 remove_diacritics 2'
        );
        """
    )


def _write_projection(
    graph: sqlite3.Connection,
    analysis: sqlite3.Connection,
    *,
    run_id: str,
    indexed_sources: int,
) -> dict[str, int]:
    groups: dict[tuple[str, str], _ConceptGroup] = {}
    concept_mapping: dict[str, str] = {}
    total_concepts = 0
    included_concepts = 0

    for row in analysis.execute(_CONCEPT_ROWS_SQL, (run_id,)):
        total_concepts += 1
        review_state = _review_state(row["review_state"])
        if review_state == "rejected":
            continue
        label = str(row["label"])
        concept_type = str(row["concept_type"])
        normalized_label = _normalize_identity_part(label, "concept label")
        normalized_type = _normalize_identity_part(concept_type, "concept type")
        key = (normalized_label, normalized_type)
        group = groups.get(key)
        if group is None:
            group = _ConceptGroup(
                concept_id=_concept_id_for(*key),
                normalized_label=normalized_label,
                normalized_type=normalized_type,
                labels=Counter(),
                types=Counter(),
                accepted_labels=Counter(),
                accepted_types=Counter(),
                sources=set(),
            )
            groups[key] = group
        confidence = float(row["confidence"])
        group.add(
            label=label,
            concept_type=concept_type,
            source_id=str(row["source_id"]),
            confidence=confidence,
            review_state=review_state,
        )
        concept_mapping[str(row["concept_id"])] = group.concept_id
        included_concepts += 1

    graph.executemany(
        """
        INSERT INTO concepts(
            concept_id, canonical_label, normalized_label,
            canonical_type, normalized_type, resolution_method,
            mention_count, source_count, claim_count, relation_count,
            mean_confidence, max_confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
        """,
        (
            (
                group.concept_id,
                group.canonical_label,
                group.normalized_label,
                group.canonical_type,
                group.normalized_type,
                CONCEPT_RESOLUTION_POLICY,
                group.mentions,
                len(group.sources),
                round(group.confidence_sum / group.mentions, 12),
                round(group.max_confidence, 12),
            )
            for _, group in sorted(groups.items())
        ),
    )

    mention_rows: list[tuple[object, ...]] = []
    for row in analysis.execute(_CONCEPT_ROWS_SQL, (run_id,)):
        candidate_id = str(row["concept_id"])
        resolved_id = concept_mapping.get(candidate_id)
        if resolved_id is None:
            continue
        mention_rows.append(
            (
                candidate_id,
                resolved_id,
                str(row["evidence_id"]),
                str(row["source_id"]),
                str(row["label"]),
                str(row["description"]),
                str(row["concept_type"]),
                float(row["confidence"]),
                _review_state(row["review_state"]),
                str(row["support_span_id"]),
                int(row["start_offset"]),
                int(row["end_offset"]),
                str(row["text_sha256"]),
            )
        )
        if len(mention_rows) >= 2_000:
            graph.executemany(_INSERT_MENTION_SQL, mention_rows)
            mention_rows.clear()
    if mention_rows:
        graph.executemany(_INSERT_MENTION_SQL, mention_rows)

    total_claims = 0
    included_claim_ids: set[str] = set()
    claim_rows: list[tuple[object, ...]] = []
    for row in analysis.execute(_CLAIM_ROWS_SQL, (run_id,)):
        total_claims += 1
        if _review_state(row["review_state"]) == "rejected":
            continue
        claim_id = str(row["claim_id"])
        included_claim_ids.add(claim_id)
        claim_rows.append(
            (
                claim_id,
                str(row["evidence_id"]),
                str(row["source_id"]),
                str(row["statement"]),
                str(row["claim_type"]),
                str(row["polarity"]),
                str(row["certainty"]),
                int(row["conditional"]),
                str(row["attribution"]),
                str(row["normative_force"]),
                float(row["confidence"]),
                _review_state(row["review_state"]),
                str(row["support_span_id"]),
                int(row["start_offset"]),
                int(row["end_offset"]),
                str(row["text_sha256"]),
            )
        )
        if len(claim_rows) >= 2_000:
            graph.executemany(_INSERT_CLAIM_SQL, claim_rows)
            claim_rows.clear()
    if claim_rows:
        graph.executemany(_INSERT_CLAIM_SQL, claim_rows)

    claim_ids_by_concept: dict[str, set[str]] = {}
    claim_link_rows: list[tuple[object, ...]] = []
    for row in analysis.execute(_CLAIM_CONCEPT_ROWS_SQL, (run_id,)):
        claim_id = str(row["claim_id"])
        candidate_id = str(row["concept_id"])
        resolved_id = concept_mapping.get(candidate_id)
        if claim_id not in included_claim_ids or resolved_id is None:
            continue
        claim_link_rows.append(
            (claim_id, resolved_id, candidate_id, int(row["ordinal"]))
        )
        claim_ids_by_concept.setdefault(resolved_id, set()).add(claim_id)
        if len(claim_link_rows) >= 2_000:
            graph.executemany(_INSERT_CLAIM_CONCEPT_SQL, claim_link_rows)
            claim_link_rows.clear()
    if claim_link_rows:
        graph.executemany(_INSERT_CLAIM_CONCEPT_SQL, claim_link_rows)

    total_relations = 0
    included_relations = 0
    relation_counts: Counter[str] = Counter()
    relation_rows: list[tuple[object, ...]] = []
    for row in analysis.execute(_RELATION_ROWS_SQL, (run_id,)):
        total_relations += 1
        if _review_state(row["review_state"]) == "rejected":
            continue
        claim_id = str(row["claim_id"])
        subject_candidate = str(row["subject_concept_id"])
        object_candidate = str(row["object_concept_id"])
        subject = concept_mapping.get(subject_candidate)
        object_ = concept_mapping.get(object_candidate)
        if claim_id not in included_claim_ids or subject is None or object_ is None:
            continue
        relation_rows.append(
            (
                str(row["relation_id"]),
                str(row["evidence_id"]),
                str(row["source_id"]),
                claim_id,
                subject,
                subject_candidate,
                str(row["relation_type"]),
                str(row["predicate"]),
                object_,
                object_candidate,
                str(row["polarity"]),
                str(row["certainty"]),
                int(row["conditional"]),
                str(row["attribution"]),
                str(row["normative_force"]),
                float(row["confidence"]),
                _review_state(row["review_state"]),
                str(row["support_span_id"]),
                int(row["start_offset"]),
                int(row["end_offset"]),
                str(row["text_sha256"]),
            )
        )
        claim_ids_by_concept.setdefault(subject, set()).add(claim_id)
        claim_ids_by_concept.setdefault(object_, set()).add(claim_id)
        relation_counts[subject] += 1
        if object_ != subject:
            relation_counts[object_] += 1
        included_relations += 1
        if len(relation_rows) >= 2_000:
            graph.executemany(_INSERT_RELATION_SQL, relation_rows)
            relation_rows.clear()
    if relation_rows:
        graph.executemany(_INSERT_RELATION_SQL, relation_rows)

    graph.executemany(
        "UPDATE concepts SET claim_count = ? WHERE concept_id = ?",
        (
            (len(claim_ids), concept_id)
            for concept_id, claim_ids in claim_ids_by_concept.items()
        ),
    )
    graph.executemany(
        "UPDATE concepts SET relation_count = ? WHERE concept_id = ?",
        ((count, concept_id) for concept_id, count in relation_counts.items()),
    )

    for _, group in sorted(groups.items()):
        search_surfaces = sorted(set(group.labels) | set(group.types))
        graph.execute(
            "INSERT INTO concept_search(concept_id, search_text) VALUES (?, ?)",
            (group.concept_id, " \n".join(search_surfaces)),
        )

    analyzed_evidence = int(
        analysis.execute(
            "SELECT COUNT(*) FROM evidence_analyses WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    )
    represented_sources = len(
        {source for group in groups.values() for source in group.sources}
    )
    return {
        "indexed_sources": indexed_sources,
        "represented_sources": represented_sources,
        "analyzed_evidence": analyzed_evidence,
        "concept_mentions": included_concepts,
        "concepts": len(groups),
        "claims": len(included_claim_ids),
        "claim_concept_links": _table_count(graph, "claim_concepts"),
        "relations": included_relations,
        "excluded_concepts": total_concepts - included_concepts,
        "excluded_claims": total_claims - len(included_claim_ids),
        "excluded_relations": total_relations - included_relations,
    }


def _concept_result(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    rank: int,
    evidence_limit: int,
    evidence_by_id: Mapping[str, EvidenceResult],
) -> ConceptGraphResult:
    concept_id = str(row["concept_id"])
    surfaces = tuple(
        ConceptSurfaceForm(
            label=str(surface["label"]),
            concept_type=str(surface["concept_type"]),
            mentions=int(surface["mentions"]),
            sources=int(surface["sources"]),
        )
        for surface in connection.execute(
            """
            SELECT label, concept_type, COUNT(*) AS mentions,
                   COUNT(DISTINCT source_id) AS sources
            FROM concept_mentions
            WHERE concept_id = ?
            GROUP BY label, concept_type
            ORDER BY sources DESC, mentions DESC,
                     label COLLATE NOCASE, concept_type COLLATE NOCASE
            LIMIT 20
            """,
            (concept_id,),
        ).fetchall()
    )
    supports = tuple(
        _support_from_row(item, evidence_by_id)
        for item in connection.execute(
            """
            WITH ranked AS (
                SELECT m.*,
                       ROW_NUMBER() OVER (
                           PARTITION BY source_id
                           ORDER BY CASE review_state
                               WHEN 'accepted' THEN 0
                               WHEN 'needs_review' THEN 1
                               ELSE 2 END,
                               confidence DESC, evidence_id,
                               concept_candidate_id
                       ) AS source_rank
                FROM concept_mentions m
                WHERE concept_id = ?
            )
            SELECT * FROM ranked
            WHERE source_rank = 1
            ORDER BY CASE review_state
                WHEN 'accepted' THEN 0
                WHEN 'needs_review' THEN 1
                ELSE 2 END,
                confidence DESC, source_id, evidence_id
            LIMIT ?
            """,
            (concept_id, evidence_limit),
        ).fetchall()
    )
    neighbors = tuple(
        ConceptRelationSummary(
            direction=str(item["direction"]),  # type: ignore[arg-type]
            relation_type=str(item["relation_type"]),
            predicate=str(item["predicate"]),
            polarity=str(item["polarity"]),
            certainty=str(item["certainty"]),
            conditional=bool(item["conditional"]),
            attribution=str(item["attribution"]),
            normative_force=str(item["normative_force"]),
            neighbor_concept_id=str(item["neighbor_concept_id"]),
            neighbor_label=str(item["neighbor_label"]),
            relations=int(item["relation_count"]),
            sources=int(item["source_count"]),
        )
        for item in connection.execute(
            _NEIGHBOR_SQL,
            (concept_id, concept_id, concept_id, concept_id),
        ).fetchall()
    )
    return ConceptGraphResult(
        rank=rank,
        concept_id=concept_id,
        canonical_label=str(row["canonical_label"]),
        canonical_type=str(row["canonical_type"]),
        resolution_method=str(row["resolution_method"]),
        mentions=int(row["mention_count"]),
        sources=int(row["source_count"]),
        claims=int(row["claim_count"]),
        relations=int(row["relation_count"]),
        mean_confidence=round(float(row["mean_confidence"]), 12),
        max_confidence=round(float(row["max_confidence"]), 12),
        surface_forms=surfaces,
        support=supports,
        neighbors=neighbors,
    )


def _support_from_row(
    row: sqlite3.Row, evidence_by_id: Mapping[str, EvidenceResult]
) -> ConceptSupport:
    evidence_id = str(row["evidence_id"])
    evidence = evidence_by_id.get(evidence_id)
    if evidence is None or evidence.locator.source_id != str(row["source_id"]):
        raise ConceptGraphError(
            "concept_graph_evidence_invalid",
            "A graph support record no longer resolves to exact indexed evidence.",
        )
    start = int(row["support_start_offset"])
    end = int(row["support_end_offset"])
    if start < 0 or end <= start or end > len(evidence.excerpt):
        raise ConceptGraphError(
            "concept_graph_evidence_invalid",
            "A graph support span falls outside its exact evidence excerpt.",
        )
    support_text = evidence.excerpt[start:end]
    digest = sha256(support_text.encode("utf-8")).hexdigest()
    if digest != str(row["support_text_sha256"]):
        raise ConceptGraphError(
            "concept_graph_evidence_invalid",
            "A graph support span does not match its exact evidence digest.",
        )
    return ConceptSupport(
        concept_candidate_id=str(row["concept_candidate_id"]),
        candidate_label=str(row["label"]),
        candidate_type=str(row["concept_type"]),
        confidence=round(float(row["confidence"]), 12),
        review_state=str(row["review_state"]),
        start_offset=start,
        end_offset=end,
        text_sha256=digest,
        text=support_text,
        evidence=evidence,
    )


def _select_completed_run(
    connection: sqlite3.Connection,
    *,
    source_index_fingerprint: str,
    run_id: str | None,
) -> sqlite3.Row:
    if run_id is not None:
        row = connection.execute(
            """
            SELECT run_id, status, extractor_fingerprint,
                   prompt_version, prompt_sha256, source_index_fingerprint
            FROM extraction_runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise ConceptGraphError(
                "concept_graph_run_missing",
                f"No analysis run has ID '{run_id}'.",
            )
        if str(row["status"]) != "completed":
            raise ConceptGraphError(
                "concept_graph_run_incomplete",
                f"Analysis run '{run_id}' is not completed.",
            )
        if str(row["source_index_fingerprint"]) != source_index_fingerprint:
            raise ConceptGraphError(
                "concept_graph_run_stale",
                f"Analysis run '{run_id}' belongs to a different exact corpus.",
            )
        return row

    row = connection.execute(
        """
        SELECT r.run_id, r.status, r.extractor_fingerprint,
               r.prompt_version, r.prompt_sha256, r.source_index_fingerprint
        FROM extraction_runs r
        WHERE r.status = 'completed' AND r.source_index_fingerprint = ?
        ORDER BY (
            SELECT COUNT(*) FROM evidence_analyses ea WHERE ea.run_id = r.run_id
        ) DESC,
        COALESCE(r.completed_at, '') DESC,
        r.run_id DESC
        LIMIT 1
        """,
        (source_index_fingerprint,),
    ).fetchone()
    if row is None:
        raise ConceptGraphError(
            "concept_graph_run_missing",
            "No completed analysis run matches the current exact corpus.",
        )
    return row


def _analysis_projection_fingerprint(
    connection: sqlite3.Connection, run_id: str
) -> str:
    digest = sha256()
    digest.update(b"corpusdock-analysis-projection-v1\0")
    queries = (
        ("run", _FINGERPRINT_RUN_SQL),
        ("evidence", _FINGERPRINT_EVIDENCE_SQL),
        ("concepts", _FINGERPRINT_CONCEPT_SQL),
        ("claims", _FINGERPRINT_CLAIM_SQL),
        ("claim_concepts", _FINGERPRINT_CLAIM_CONCEPT_SQL),
        ("relations", _FINGERPRINT_RELATION_SQL),
    )
    for label, query in queries:
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        for row in connection.execute(query, (run_id,)):
            digest.update(_canonical_json(tuple(row)).encode("utf-8"))
            digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _graph_content_fingerprint(connection: sqlite3.Connection) -> str:
    digest = sha256()
    digest.update(b"corpusdock-concept-graph-v1\0")
    queries = (
        ("concepts", "SELECT * FROM concepts ORDER BY concept_id"),
        (
            "mentions",
            "SELECT * FROM concept_mentions ORDER BY concept_candidate_id",
        ),
        ("claims", "SELECT * FROM claims ORDER BY claim_id"),
        (
            "claim_concepts",
            "SELECT * FROM claim_concepts ORDER BY claim_id, ordinal",
        ),
        ("relations", "SELECT * FROM relations ORDER BY relation_id"),
        (
            "search",
            "SELECT concept_id, search_text FROM concept_search ORDER BY concept_id",
        ),
    )
    for label, query in queries:
        digest.update(label.encode("ascii"))
        digest.update(b"\0")
        for row in connection.execute(query):
            digest.update(_canonical_json(tuple(row)).encode("utf-8"))
            digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def _validate_graph_connection(connection: sqlite3.Connection) -> None:
    integrity = connection.execute("PRAGMA integrity_check").fetchone()
    if integrity is None or integrity[0] != "ok":
        raise ConceptGraphError(
            "concept_graph_integrity_failed",
            "SQLite did not confirm concept graph integrity.",
        )
    foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_keys:
        raise ConceptGraphError(
            "concept_graph_integrity_failed",
            "Concept graph contains broken relational links.",
        )
    version = connection.execute("PRAGMA user_version").fetchone()
    if version is None or version[0] != CONCEPT_GRAPH_SCHEMA_VERSION:
        raise ConceptGraphError(
            "concept_graph_schema_invalid",
            "Concept graph schema version is unsupported.",
        )


def _read_metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key, value FROM graph_metadata")
    }


def _validate_metadata(metadata: Mapping[str, str]) -> None:
    if set(metadata) != _METADATA_FIELDS:
        raise ConceptGraphError(
            "concept_graph_schema_invalid",
            "Concept graph metadata is incomplete or contains unknown fields.",
        )
    if _metadata_integer(metadata, "schema_version") != CONCEPT_GRAPH_SCHEMA_VERSION:
        raise ConceptGraphError(
            "concept_graph_schema_invalid",
            "Concept graph metadata has an unsupported schema version.",
        )
    if (
        _metadata_integer(metadata, "analysis_database_schema_version")
        != ANALYSIS_DATABASE_SCHEMA_VERSION
    ):
        raise ConceptGraphError(
            "concept_graph_schema_invalid",
            "Concept graph references an unsupported analysis database schema.",
        )
    if _RUN_ID_PATTERN.fullmatch(metadata["analysis_run_id"]) is None:
        raise ConceptGraphError(
            "concept_graph_schema_invalid", "Concept graph analysis run ID is invalid."
        )
    if metadata["analysis_run_selection"] not in {
        AUTOMATIC_RUN_SELECTION,
        EXPLICIT_RUN_SELECTION,
    }:
        raise ConceptGraphError(
            "concept_graph_schema_invalid", "Concept graph run selection is invalid."
        )
    if metadata["resolution_policy"] != CONCEPT_RESOLUTION_POLICY:
        raise ConceptGraphError(
            "concept_graph_schema_invalid", "Concept resolution policy is unsupported."
        )
    for field in (
        "source_index_fingerprint",
        "analysis_extractor_fingerprint",
        "analysis_prompt_sha256",
        "analysis_projection_fingerprint",
        "graph_content_fingerprint",
    ):
        if _DIGEST_PATTERN.fullmatch(metadata[field]) is None:
            raise ConceptGraphError(
                "concept_graph_schema_invalid",
                f"Concept graph metadata field '{field}' is not a digest.",
            )


def _validated_counts(
    connection: sqlite3.Connection, metadata: Mapping[str, str]
) -> dict[str, int]:
    names = (
        "indexed_sources",
        "represented_sources",
        "analyzed_evidence",
        "concept_mentions",
        "concepts",
        "claims",
        "claim_concept_links",
        "relations",
        "excluded_concepts",
        "excluded_claims",
        "excluded_relations",
    )
    counts = {name: _metadata_integer(metadata, name) for name in names}
    observed = {
        "concept_mentions": _table_count(connection, "concept_mentions"),
        "concepts": _table_count(connection, "concepts"),
        "claims": _table_count(connection, "claims"),
        "claim_concept_links": _table_count(connection, "claim_concepts"),
        "relations": _table_count(connection, "relations"),
        "represented_sources": int(
            connection.execute(
                "SELECT COUNT(DISTINCT source_id) FROM concept_mentions"
            ).fetchone()[0]
        ),
    }
    if any(counts[name] != value for name, value in observed.items()):
        raise ConceptGraphError(
            "concept_graph_content_invalid",
            "Concept graph row counts do not match recorded metadata.",
        )
    if counts["represented_sources"] > counts["indexed_sources"]:
        raise ConceptGraphError(
            "concept_graph_content_invalid",
            "Concept graph represents more sources than the exact corpus.",
        )
    return counts


def _connect_analysis_read_only(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
    except sqlite3.DatabaseError as error:
        raise ConceptGraphError(
            "concept_graph_analysis_invalid",
            f"Could not open the local analysis database: {error}.",
        ) from error


def _connect_graph_read_only(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
    except sqlite3.DatabaseError as error:
        raise ConceptGraphError(
            "concept_graph_open_failed",
            f"Could not open the local concept graph: {error}.",
        ) from error


def _normalize_identity_part(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConceptGraphError(
            "concept_graph_candidate_invalid", f"A {label} is invalid."
        )
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u00a0": " ",
                "\u2007": " ",
                "\u202f": " ",
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u2212": "-",
                "\u2018": "'",
                "\u2019": "'",
            }
        )
    )
    normalized = " ".join(normalized.casefold().split()).strip(_EDGE_PUNCTUATION)
    if not normalized:
        raise ConceptGraphError(
            "concept_graph_candidate_invalid",
            f"A {label} becomes empty after safe normalization.",
        )
    return normalized


def _concept_id_for(normalized_label: str, normalized_type: str) -> str:
    digest = sha256()
    for part in (
        CONCEPT_RESOLUTION_POLICY,
        normalized_label,
        normalized_type,
    ):
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return f"gcon-{digest.hexdigest()}"


def _preferred_surface(counts: Counter[str], accepted_counts: Counter[str]) -> str:
    if not counts:
        raise ConceptGraphError(
            "concept_graph_candidate_invalid", "A concept group has no surface form."
        )
    return min(
        counts,
        key=lambda value: (
            -accepted_counts[value],
            -counts[value],
            len(value),
            value.casefold(),
            value,
        ),
    )


def _review_state(value: object) -> str:
    state = str(value)
    if state not in _REVIEW_STATES:
        raise ConceptGraphError(
            "concept_graph_candidate_invalid",
            "An analysis candidate has an invalid review state.",
        )
    return state


def _compile_graph_query(query: str) -> str:
    if not isinstance(query, str) or not query.strip() or "\x00" in query:
        raise ConceptGraphError(
            "concept_graph_query_empty", "Graph query cannot be empty."
        )
    tokens = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", query).casefold())
    if not tokens:
        raise ConceptGraphError(
            "concept_graph_query_empty",
            "Graph query must contain at least one letter or number.",
        )
    return " AND ".join(f'"{token}"' for token in tokens)


def _metadata_integer(metadata: Mapping[str, str], name: str) -> int:
    try:
        value = int(metadata[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ConceptGraphError(
            "concept_graph_schema_invalid",
            f"Concept graph metadata field '{name}' is not an integer.",
        ) from error
    if value < 0:
        raise ConceptGraphError(
            "concept_graph_schema_invalid",
            f"Concept graph metadata field '{name}' cannot be negative.",
        )
    return value


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


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


_CONCEPT_ROWS_SQL = """
SELECT c.concept_id, c.evidence_id, ea.source_id, c.label, c.description,
       c.concept_type, c.confidence, c.review_state, c.support_span_id,
       s.start_offset, s.end_offset, s.text_sha256
FROM concept_candidates c
JOIN evidence_analyses ea
  ON ea.run_id = c.run_id AND ea.evidence_id = c.evidence_id
JOIN evidence_spans s ON s.span_id = c.support_span_id
WHERE c.run_id = ?
ORDER BY c.concept_id
"""

_CLAIM_ROWS_SQL = """
SELECT c.claim_id, c.evidence_id, ea.source_id, c.statement, c.claim_type,
       c.polarity, c.certainty, c.conditional, c.attribution,
       c.normative_force, c.confidence, c.review_state, c.support_span_id,
       s.start_offset, s.end_offset, s.text_sha256
FROM claim_candidates c
JOIN evidence_analyses ea
  ON ea.run_id = c.run_id AND ea.evidence_id = c.evidence_id
JOIN evidence_spans s ON s.span_id = c.support_span_id
WHERE c.run_id = ?
ORDER BY c.claim_id
"""

_CLAIM_CONCEPT_ROWS_SQL = """
SELECT links.claim_id, links.concept_id, links.ordinal
FROM claim_concepts links
JOIN claim_candidates claims ON claims.claim_id = links.claim_id
WHERE claims.run_id = ?
ORDER BY links.claim_id, links.ordinal
"""

_RELATION_ROWS_SQL = """
SELECT r.relation_id, r.evidence_id, ea.source_id, r.claim_id,
       r.subject_concept_id, r.relation_type, r.predicate,
       r.object_concept_id, r.polarity, r.certainty, r.conditional,
       r.attribution, r.normative_force, r.confidence, r.review_state,
       r.support_span_id, s.start_offset, s.end_offset, s.text_sha256
FROM relation_candidates r
JOIN evidence_analyses ea
  ON ea.run_id = r.run_id AND ea.evidence_id = r.evidence_id
JOIN evidence_spans s ON s.span_id = r.support_span_id
WHERE r.run_id = ?
ORDER BY r.relation_id
"""

_INSERT_MENTION_SQL = """
INSERT INTO concept_mentions(
    concept_candidate_id, concept_id, evidence_id, source_id, label,
    description, concept_type, confidence, review_state, support_span_id,
    support_start_offset, support_end_offset, support_text_sha256
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_CLAIM_SQL = """
INSERT INTO claims(
    claim_id, evidence_id, source_id, statement, claim_type, polarity,
    certainty, conditional, attribution, normative_force, confidence,
    review_state, support_span_id, support_start_offset,
    support_end_offset, support_text_sha256
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_CLAIM_CONCEPT_SQL = """
INSERT INTO claim_concepts(
    claim_id, concept_id, concept_candidate_id, ordinal
) VALUES (?, ?, ?, ?)
"""

_INSERT_RELATION_SQL = """
INSERT INTO relations(
    relation_id, evidence_id, source_id, claim_id,
    subject_concept_id, subject_candidate_id, relation_type, predicate,
    object_concept_id, object_candidate_id, polarity, certainty,
    conditional, attribution, normative_force, confidence, review_state,
    support_span_id, support_start_offset, support_end_offset,
    support_text_sha256
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_NEIGHBOR_SQL = """
WITH edges AS (
    SELECT
        CASE WHEN r.subject_concept_id = ? THEN 'outgoing' ELSE 'incoming' END
            AS direction,
        r.relation_type, r.predicate, r.polarity, r.certainty,
        r.conditional, r.attribution, r.normative_force,
        CASE WHEN r.subject_concept_id = ?
             THEN r.object_concept_id ELSE r.subject_concept_id END
            AS neighbor_concept_id,
        r.source_id
    FROM relations r
    WHERE r.subject_concept_id = ? OR r.object_concept_id = ?
), aggregated AS (
    SELECT direction, relation_type, predicate, polarity, certainty,
           conditional, attribution, normative_force, neighbor_concept_id,
           COUNT(*) AS relation_count,
           COUNT(DISTINCT source_id) AS source_count
    FROM edges
    GROUP BY direction, relation_type, predicate, polarity, certainty,
             conditional, attribution, normative_force, neighbor_concept_id
)
SELECT a.*, c.canonical_label AS neighbor_label
FROM aggregated a
JOIN concepts c ON c.concept_id = a.neighbor_concept_id
ORDER BY a.source_count DESC, a.relation_count DESC,
         a.direction, a.relation_type, a.neighbor_concept_id
LIMIT 12
"""

_FINGERPRINT_RUN_SQL = """
SELECT run_id, status, started_at, completed_at, source_index_fingerprint,
       extractor_fingerprint, prompt_version, prompt_sha256, scope_json
FROM extraction_runs WHERE run_id = ?
"""

_FINGERPRINT_EVIDENCE_SQL = """
SELECT evidence_id, chunk_id, source_id, status, raw_output_sha256,
       output_truncated, rejection_count
FROM evidence_analyses WHERE run_id = ? ORDER BY evidence_id
"""

_FINGERPRINT_CONCEPT_SQL = """
SELECT c.concept_id, c.evidence_id, c.local_id, c.label, c.description,
       c.concept_type, c.confidence, c.support_span_id, c.review_state,
       s.start_offset, s.end_offset, s.text_sha256
FROM concept_candidates c
JOIN evidence_spans s ON s.span_id = c.support_span_id
WHERE c.run_id = ? ORDER BY c.concept_id
"""

_FINGERPRINT_CLAIM_SQL = """
SELECT c.claim_id, c.evidence_id, c.local_id, c.statement, c.claim_type,
       c.polarity, c.certainty, c.conditional, c.attribution,
       c.normative_force, c.confidence, c.support_span_id, c.review_state,
       s.start_offset, s.end_offset, s.text_sha256
FROM claim_candidates c
JOIN evidence_spans s ON s.span_id = c.support_span_id
WHERE c.run_id = ? ORDER BY c.claim_id
"""

_FINGERPRINT_CLAIM_CONCEPT_SQL = """
SELECT links.claim_id, links.concept_id, links.ordinal
FROM claim_concepts links
JOIN claim_candidates claims ON claims.claim_id = links.claim_id
WHERE claims.run_id = ? ORDER BY links.claim_id, links.ordinal
"""

_FINGERPRINT_RELATION_SQL = """
SELECT r.relation_id, r.evidence_id, r.local_id, r.claim_id,
       r.subject_concept_id, r.relation_type, r.predicate,
       r.object_concept_id, r.polarity, r.certainty, r.conditional,
       r.attribution, r.normative_force, r.confidence, r.support_span_id,
       r.review_state, s.start_offset, s.end_offset, s.text_sha256
FROM relation_candidates r
JOIN evidence_spans s ON s.span_id = r.support_span_id
WHERE r.run_id = ? ORDER BY r.relation_id
"""
