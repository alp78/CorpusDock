"""Local SQLite persistence for evidence-grounded analysis candidates.

The exact retrieval index remains authoritative.  This database stores derived
labels, propositions, relations, review events, and evidence-relative offsets.  It
intentionally stores neither source excerpts nor source paths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Literal
from uuid import uuid4

from corpusdock import __version__
from corpusdock.analysis_contracts import (
    ANALYSIS_CONTRACT_SCHEMA_VERSION,
    EvidenceAnalysis,
    ReviewState,
)
from corpusdock.manifest import utc_now
from corpusdock.retrieval import (
    RetrievalError,
    SQLiteSearchBackend,
    SearchCorpusSnapshot,
)


ANALYSIS_DATABASE_SCHEMA_VERSION = 2
ANALYSIS_DATABASE_FILE_NAME = "analysis.sqlite3"

RunStatus = Literal["running", "completed", "failed"]
CandidateKind = Literal["concept", "claim", "relation"]

_RUN_ID_PATTERN = re.compile(r"run-[0-9a-f]{64}")
_MODEL_FINGERPRINT_PATTERN = re.compile(r"(?:sha256|hf-revision):[0-9a-f]{40,64}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_REVIEW_STATES = {"unreviewed", "accepted", "rejected", "needs_review"}
_RUN_STATUSES = {"running", "completed", "failed"}
_CANDIDATE_TABLES = {
    "concept": ("concept_candidates", "concept_id"),
    "claim": ("claim_candidates", "claim_id"),
    "relation": ("relation_candidates", "relation_id"),
}
_METADATA_FIELDS = {
    "schema_version",
    "tool_version",
    "created_at",
    "source_index_built_at",
    "source_index_fingerprint",
    "sources",
    "chunks",
    "partial_sources",
    "analysis_contract_schema_version",
}
_FORBIDDEN_PROVENANCE_KEYS = {
    "excerpt",
    "excerpts",
    "text",
    "texts",
    "source_path",
    "source_paths",
    "prompt",
    "raw_output",
}
_EXTRACTOR_RUN_FIELDS = {
    "provider",
    "runtime",
    "runtime_version",
    "model_id",
    "model_revision",
    "model_fingerprint",
    "prompt_style",
    "prompt_version",
    "prompt_sha256",
    "max_input_tokens",
    "max_output_tokens",
    "batch_size",
    "device",
    "dtype",
    "quantization",
    "quantization_runtime",
    "quantization_runtime_version",
    "structured_output",
    "structured_output_runtime_version",
    "support_unit_processor",
    "support_unit_processor_version",
    "support_unit_model",
    "remote_code_trusted",
    "deterministic",
    "thinking_enabled",
    "framework_version",
    "accelerator_runtime_version",
    "accelerator_name",
    "engine_performance_mode",
    "prefix_caching_enabled",
    "gpu_memory_utilization",
    "structured_output_backend",
    "sampling_backend",
}


class AnalysisStoreError(Exception):
    """A missing, stale, corrupt, or inconsistent derived analysis database."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class AnalysisRunDescriptor:
    """Non-content provenance and progress for one extraction run."""

    path: str
    schema_version: int
    run_id: str
    status: RunStatus
    started_at: str
    completed_at: str | None
    source_index_fingerprint: str
    extractor_fingerprint: str
    extractor: Mapping[str, Any]
    prompt_version: str
    prompt_sha256: str
    scope: Mapping[str, Any]
    analyzed_evidence: int
    accepted_evidence: int
    partial_evidence: int
    empty_evidence: int
    rejected_evidence: int
    concepts: int
    claims: int
    relations: int
    rejected_candidates: int
    inference_ms: float
    output_tokens: int
    truncated_evidence: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "source_index_fingerprint": self.source_index_fingerprint,
            "extractor_fingerprint": self.extractor_fingerprint,
            "extractor": dict(self.extractor),
            "prompt": {
                "version": self.prompt_version,
                "sha256": self.prompt_sha256,
            },
            "scope": dict(self.scope),
            "counts": {
                "analyzed_evidence": self.analyzed_evidence,
                "accepted_evidence": self.accepted_evidence,
                "partial_evidence": self.partial_evidence,
                "empty_evidence": self.empty_evidence,
                "rejected_evidence": self.rejected_evidence,
                "concepts": self.concepts,
                "claims": self.claims,
                "relations": self.relations,
                "rejected_candidates": self.rejected_candidates,
                "truncated_evidence": self.truncated_evidence,
            },
            "inference_ms": self.inference_ms,
            "output_tokens": self.output_tokens,
        }


def analysis_database_path_for(project_root: Path | str) -> Path:
    """Return the ignored project-local derived-analysis database path."""

    return (
        Path(project_root).expanduser().resolve()
        / ".corpusdock"
        / ANALYSIS_DATABASE_FILE_NAME
    )


class AnalysisStore:
    """Transactional analysis persistence over stable exact-evidence identities."""

    def __init__(self, project_root: Path | str, *, reconcile: bool = False) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.path = analysis_database_path_for(self.project_root)
        self._exact_backend = SQLiteSearchBackend(self.project_root)
        self.snapshot = self._exact_backend.corpus_snapshot()
        self._evidence_by_id = {
            evidence.evidence_id: evidence for evidence in self.snapshot.evidence
        }
        self.reconciled_records = 0
        if self.path.is_file():
            _read_and_validate_database(
                self.path,
                self.snapshot,
                allow_stale=reconcile,
            )
            if reconcile:
                self.reconciled_records = self._reconcile_current_database()
        else:
            self._create_database()

    def begin_run(
        self,
        extractor: Mapping[str, Any],
        *,
        prompt_version: str,
        prompt_sha256: str,
        scope: Mapping[str, Any],
        resume: bool = True,
        now: Callable[[], str] = utc_now,
    ) -> AnalysisRunDescriptor:
        """Create, or resume, a run with identical model/prompt/scope provenance."""

        clean_extractor = _provenance_object(extractor, "extractor")
        clean_scope = _provenance_object(scope, "scope")
        model_fingerprint = clean_extractor.get("model_fingerprint")
        if (
            not isinstance(model_fingerprint, str)
            or _MODEL_FINGERPRINT_PATTERN.fullmatch(model_fingerprint) is None
        ):
            raise AnalysisStoreError(
                "analysis_extractor_invalid",
                "Extractor provenance must contain a stable model_fingerprint.",
            )
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise AnalysisStoreError(
                "analysis_prompt_invalid", "Prompt version must be a non-empty string."
            )
        if _DIGEST_PATTERN.fullmatch(prompt_sha256) is None:
            raise AnalysisStoreError(
                "analysis_prompt_invalid",
                "Prompt SHA-256 must be 64 lowercase hex characters.",
            )
        extractor_json = _canonical_json(clean_extractor)
        extractor_fingerprint = _extractor_run_fingerprint(clean_extractor)
        scope_json = _canonical_json(clean_scope)

        self._exact_backend.assert_snapshot_current(self.snapshot)
        try:
            with closing(_connect(self.path)) as connection, connection:
                if resume:
                    rows = connection.execute(
                        """
                        SELECT run_id, status, source_index_fingerprint, scope_json
                        FROM extraction_runs
                        WHERE status IN ('running', 'completed')
                          AND extractor_fingerprint = ?
                          AND prompt_version = ?
                          AND prompt_sha256 = ?
                        ORDER BY started_at DESC, run_id DESC
                        """,
                        (
                            extractor_fingerprint,
                            prompt_version,
                            prompt_sha256,
                        ),
                    ).fetchall()
                    scope_key = _scope_resume_key(clean_scope)
                    row = next(
                        (
                            candidate
                            for candidate in rows
                            if _scope_resume_key(
                                _json_object(
                                    str(candidate["scope_json"]), "analysis scope"
                                )
                            )
                            == scope_key
                        ),
                        None,
                    )
                    if row is not None:
                        if (
                            str(row["source_index_fingerprint"])
                            != self.snapshot.index_fingerprint
                            or str(row["scope_json"]) != scope_json
                        ):
                            connection.execute(
                                """
                                UPDATE extraction_runs
                                SET status = 'running', completed_at = NULL,
                                    source_index_fingerprint = ?, scope_json = ?
                                WHERE run_id = ?
                                """,
                                (
                                    self.snapshot.index_fingerprint,
                                    scope_json,
                                    str(row["run_id"]),
                                ),
                            )
                        return self.run_descriptor(
                            str(row["run_id"]), connection=connection
                        )

                started_at = now()
                run_id = _run_id_for(
                    self.snapshot.index_fingerprint,
                    model_fingerprint,
                    extractor_fingerprint,
                    prompt_version,
                    prompt_sha256,
                    scope_json,
                    started_at,
                    uuid4().hex,
                )
                connection.execute(
                    """
                    INSERT INTO extraction_runs(
                        run_id, status, started_at, completed_at,
                        source_index_fingerprint, extractor_json, prompt_version,
                        extractor_fingerprint, prompt_sha256, scope_json
                    ) VALUES (?, 'running', ?, NULL, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        started_at,
                        self.snapshot.index_fingerprint,
                        extractor_json,
                        prompt_version,
                        extractor_fingerprint,
                        prompt_sha256,
                        scope_json,
                    ),
                )
                descriptor = self.run_descriptor(run_id, connection=connection)
        except AnalysisStoreError:
            raise
        except sqlite3.DatabaseError as error:
            raise AnalysisStoreError(
                "analysis_database_write_failed",
                f"Could not begin the local analysis run: {error}.",
            ) from error
        self._exact_backend.assert_snapshot_current(self.snapshot)
        return descriptor

    def analyzed_evidence_ids(self, run_id: str) -> frozenset[str]:
        """Return stable IDs already committed for a resumable run."""

        _validate_run_id(run_id)
        try:
            with closing(_connect_read_only(self.path)) as connection:
                _require_run(connection, run_id)
                return frozenset(
                    str(row["evidence_id"])
                    for row in connection.execute(
                        """
                        SELECT evidence_id
                        FROM evidence_analyses
                        WHERE run_id = ?
                        """,
                        (run_id,),
                    ).fetchall()
                )
        except AnalysisStoreError:
            raise
        except sqlite3.DatabaseError as error:
            raise AnalysisStoreError(
                "analysis_database_read_failed",
                f"Could not read analysis progress: {error}.",
            ) from error

    def prepare_evidence_scope(self, run_id: str, evidence_ids: Sequence[str]) -> int:
        """Keep reusable records in scope and remove obsolete run membership."""

        _validate_run_id(run_id)
        if isinstance(evidence_ids, (str, bytes)):
            raise AnalysisStoreError(
                "analysis_scope_invalid", "Evidence IDs must be a sequence."
            )
        selected = tuple(dict.fromkeys(evidence_ids))
        if any(evidence_id not in self._evidence_by_id for evidence_id in selected):
            raise AnalysisStoreError(
                "analysis_scope_invalid",
                "Analysis scope refers to evidence outside the current exact index.",
            )
        self._exact_backend.assert_snapshot_current(self.snapshot)
        try:
            with closing(_connect(self.path)) as connection, connection:
                run = _require_run(connection, run_id)
                _replace_temporary_evidence_table(connection, selected)
                removed = _delete_obsolete_analyses(connection, run_id=run_id)
                completed = connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM evidence_analyses
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
                if (
                    run["status"] == "completed"
                    and completed is not None
                    and int(completed["count"]) < len(selected)
                ):
                    connection.execute(
                        """
                        UPDATE extraction_runs
                        SET status = 'running', completed_at = NULL
                        WHERE run_id = ?
                        """,
                        (run_id,),
                    )
        except AnalysisStoreError:
            raise
        except sqlite3.DatabaseError as error:
            raise AnalysisStoreError(
                "analysis_database_write_failed",
                f"Could not prepare the resumable analysis scope: {error}.",
            ) from error
        self._exact_backend.assert_snapshot_current(self.snapshot)
        return removed

    def write_batch(self, run_id: str, analyses: Sequence[EvidenceAnalysis]) -> None:
        """Atomically persist one inference batch after validating exact lineage."""

        _validate_run_id(run_id)
        if isinstance(analyses, (str, bytes)) or not analyses:
            raise AnalysisStoreError(
                "analysis_batch_invalid", "Analysis batch must not be empty."
            )
        seen: set[str] = set()
        for analysis in analyses:
            if not isinstance(analysis, EvidenceAnalysis) or analysis.run_id != run_id:
                raise AnalysisStoreError(
                    "analysis_batch_invalid",
                    "Every analysis record must belong to the selected run.",
                )
            evidence = self._evidence_by_id.get(analysis.evidence_id)
            if evidence is None:
                raise AnalysisStoreError(
                    "analysis_evidence_invalid",
                    "Analysis refers to evidence outside the exact-index snapshot.",
                )
            if (
                analysis.evidence_id in seen
                or analysis.chunk_id != evidence.chunk_id
                or analysis.source_id != evidence.locator.source_id
            ):
                raise AnalysisStoreError(
                    "analysis_evidence_invalid",
                    "Analysis evidence lineage is duplicate or inconsistent.",
                )
            seen.add(analysis.evidence_id)
            _validate_analysis_spans(analysis, len(evidence.excerpt))

        self._exact_backend.assert_snapshot_current(self.snapshot)
        try:
            with closing(_connect(self.path)) as connection, connection:
                run = _require_run(connection, run_id)
                if run["status"] != "running":
                    raise AnalysisStoreError(
                        "analysis_run_closed",
                        f"Analysis run '{run_id}' is not open for writes.",
                    )
                for analysis in analyses:
                    self._insert_analysis(connection, analysis)
        except AnalysisStoreError:
            raise
        except sqlite3.IntegrityError as error:
            raise AnalysisStoreError(
                "analysis_database_conflict",
                "Analysis batch conflicts with already persisted records.",
            ) from error
        except sqlite3.DatabaseError as error:
            raise AnalysisStoreError(
                "analysis_database_write_failed",
                f"Could not persist the analysis batch: {error}.",
            ) from error
        self._exact_backend.assert_snapshot_current(self.snapshot)

    def finish_run(
        self,
        run_id: str,
        *,
        failed: bool = False,
        extractor: Mapping[str, Any] | None = None,
        now: Callable[[], str] = utc_now,
    ) -> AnalysisRunDescriptor:
        """Close a run while preserving every committed batch for audit or resume."""

        _validate_run_id(run_id)
        self._exact_backend.assert_snapshot_current(self.snapshot)
        try:
            with closing(_connect(self.path)) as connection, connection:
                run = _require_run(connection, run_id)
                status = str(run["status"])
                if status == "running":
                    extractor_json: str | None = None
                    if extractor is not None:
                        clean_extractor = _provenance_object(extractor, "extractor")
                        stored = connection.execute(
                            """
                            SELECT extractor_fingerprint
                            FROM extraction_runs
                            WHERE run_id = ?
                            """,
                            (run_id,),
                        ).fetchone()
                        if stored is None or _extractor_run_fingerprint(
                            clean_extractor
                        ) != str(stored["extractor_fingerprint"]):
                            raise AnalysisStoreError(
                                "analysis_extractor_mismatch",
                                "Final extractor provenance does not match the analysis run.",
                            )
                        extractor_json = _canonical_json(clean_extractor)
                    connection.execute(
                        """
                        UPDATE extraction_runs
                        SET status = ?, completed_at = ?,
                            extractor_json = COALESCE(?, extractor_json)
                        WHERE run_id = ?
                        """,
                        (
                            "failed" if failed else "completed",
                            now(),
                            extractor_json,
                            run_id,
                        ),
                    )
                elif failed and status != "failed":
                    raise AnalysisStoreError(
                        "analysis_run_closed",
                        f"Completed analysis run '{run_id}' cannot be marked failed.",
                    )
                descriptor = self.run_descriptor(run_id, connection=connection)
        except AnalysisStoreError:
            raise
        except sqlite3.DatabaseError as error:
            raise AnalysisStoreError(
                "analysis_database_write_failed",
                f"Could not finish the analysis run: {error}.",
            ) from error
        return descriptor

    def run_descriptor(
        self,
        run_id: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> AnalysisRunDescriptor:
        """Return non-content run provenance and aggregate counts."""

        _validate_run_id(run_id)
        owns_connection = connection is None
        active = connection or _connect_read_only(self.path)
        try:
            row = active.execute(_RUN_DESCRIPTOR_SQL, (run_id,)).fetchone()
            if row is None:
                raise AnalysisStoreError(
                    "analysis_run_missing", f"No analysis run has ID '{run_id}'."
                )
            return _descriptor_from_row(self.path, row)
        except AnalysisStoreError:
            raise
        except sqlite3.DatabaseError as error:
            raise AnalysisStoreError(
                "analysis_database_read_failed",
                f"Could not read the analysis run: {error}.",
            ) from error
        finally:
            if owns_connection:
                active.close()

    def record_review(
        self,
        candidate_kind: CandidateKind,
        candidate_id: str,
        state: ReviewState,
        *,
        reviewer: str | None = None,
        note: str | None = None,
        now: Callable[[], str] = utc_now,
    ) -> str:
        """Append an auditable human review and update the candidate's current state."""

        if candidate_kind not in _CANDIDATE_TABLES:
            raise AnalysisStoreError(
                "analysis_candidate_kind_invalid", "Unknown analysis candidate kind."
            )
        if state not in _REVIEW_STATES or state == "unreviewed":
            raise AnalysisStoreError(
                "analysis_review_state_invalid",
                "A review must set accepted, rejected, or needs_review.",
            )
        clean_reviewer = _optional_bounded_text(reviewer, 200, "reviewer")
        clean_note = _optional_bounded_text(note, 4_000, "review note")
        reviewed_at = now()
        review_id = _digest_id(
            "rev",
            candidate_kind,
            candidate_id,
            state,
            reviewed_at,
            clean_reviewer or "",
            clean_note or "",
            uuid4().hex,
        )
        table, identifier = _CANDIDATE_TABLES[candidate_kind]
        try:
            with closing(_connect(self.path)) as connection, connection:
                candidate = connection.execute(
                    f"SELECT 1 FROM {table} WHERE {identifier} = ?", (candidate_id,)
                ).fetchone()
                if candidate is None:
                    raise AnalysisStoreError(
                        "analysis_candidate_missing",
                        f"No {candidate_kind} candidate has ID '{candidate_id}'.",
                    )
                connection.execute(
                    """
                    INSERT INTO candidate_reviews(
                        review_id, candidate_kind, candidate_id, state,
                        reviewed_at, reviewer, note
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_id,
                        candidate_kind,
                        candidate_id,
                        state,
                        reviewed_at,
                        clean_reviewer,
                        clean_note,
                    ),
                )
                connection.execute(
                    f"UPDATE {table} SET review_state = ? WHERE {identifier} = ?",
                    (state, candidate_id),
                )
        except AnalysisStoreError:
            raise
        except sqlite3.DatabaseError as error:
            raise AnalysisStoreError(
                "analysis_database_write_failed",
                f"Could not persist the candidate review: {error}.",
            ) from error
        return review_id

    def _insert_analysis(
        self, connection: sqlite3.Connection, analysis: EvidenceAnalysis
    ) -> None:
        connection.execute(
            """
            INSERT INTO evidence_analyses(
                run_id, evidence_id, chunk_id, source_id, status,
                raw_output_sha256, inference_ms, output_tokens,
                output_truncated, rejection_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis.run_id,
                analysis.evidence_id,
                analysis.chunk_id,
                analysis.source_id,
                analysis.status,
                analysis.raw_output_sha256,
                analysis.inference_ms,
                analysis.output_tokens,
                int(analysis.output_truncated),
                len(analysis.rejections),
            ),
        )
        spans = {
            candidate.support.span_id: candidate.support
            for candidates in (analysis.concepts, analysis.claims, analysis.relations)
            for candidate in candidates
        }
        connection.executemany(
            """
            INSERT OR IGNORE INTO evidence_spans(
                span_id, evidence_id, start_offset, end_offset, text_sha256
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    span.span_id,
                    span.evidence_id,
                    span.start_offset,
                    span.end_offset,
                    span.text_sha256,
                )
                for span in spans.values()
            ),
        )
        connection.executemany(
            """
            INSERT INTO concept_candidates(
                concept_id, run_id, evidence_id, local_id, label, description,
                concept_type, confidence, support_span_id, review_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    concept.concept_id,
                    analysis.run_id,
                    analysis.evidence_id,
                    concept.local_id,
                    concept.label,
                    concept.description,
                    concept.concept_type,
                    concept.confidence,
                    concept.support.span_id,
                    concept.review_state,
                )
                for concept in analysis.concepts
            ),
        )
        connection.executemany(
            """
            INSERT INTO claim_candidates(
                claim_id, run_id, evidence_id, local_id, statement, claim_type,
                polarity, certainty, conditional, attribution, normative_force,
                confidence, support_span_id, review_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    claim.claim_id,
                    analysis.run_id,
                    analysis.evidence_id,
                    claim.local_id,
                    claim.statement,
                    claim.claim_type,
                    claim.polarity,
                    claim.certainty,
                    int(claim.conditional),
                    claim.attribution,
                    claim.normative_force,
                    claim.confidence,
                    claim.support.span_id,
                    claim.review_state,
                )
                for claim in analysis.claims
            ),
        )
        connection.executemany(
            """
            INSERT INTO claim_concepts(claim_id, concept_id, ordinal)
            VALUES (?, ?, ?)
            """,
            (
                (claim.claim_id, concept_id, ordinal)
                for claim in analysis.claims
                for ordinal, concept_id in enumerate(claim.concept_ids)
            ),
        )
        connection.executemany(
            """
            INSERT INTO relation_candidates(
                relation_id, run_id, evidence_id, local_id,
                claim_id, subject_concept_id, relation_type, predicate, object_concept_id,
                polarity, certainty, conditional, attribution, normative_force,
                confidence, support_span_id, review_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    relation.relation_id,
                    analysis.run_id,
                    analysis.evidence_id,
                    relation.local_id,
                    relation.claim_id,
                    relation.subject_concept_id,
                    relation.relation_type,
                    relation.predicate,
                    relation.object_concept_id,
                    relation.polarity,
                    relation.certainty,
                    int(relation.conditional),
                    relation.attribution,
                    relation.normative_force,
                    relation.confidence,
                    relation.support.span_id,
                    relation.review_state,
                )
                for relation in analysis.relations
            ),
        )
        connection.executemany(
            """
            INSERT INTO candidate_rejections(
                run_id, evidence_id, ordinal, category, code, json_path, local_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    analysis.run_id,
                    analysis.evidence_id,
                    ordinal,
                    rejection.category,
                    rejection.code,
                    rejection.path,
                    rejection.local_id,
                )
                for ordinal, rejection in enumerate(analysis.rejections)
            ),
        )

    def _reconcile_current_database(self) -> int:
        """Adopt the current corpus revision while retaining stable evidence work."""

        self._exact_backend.assert_snapshot_current(self.snapshot)
        metadata = {
            "tool_version": __version__,
            "source_index_built_at": self.snapshot.index_built_at,
            "source_index_fingerprint": self.snapshot.index_fingerprint,
            "sources": str(self.snapshot.indexed_sources),
            "chunks": str(self.snapshot.indexed_chunks),
            "partial_sources": str(self.snapshot.partial_sources),
        }
        try:
            with closing(_connect(self.path)) as connection, connection:
                _replace_temporary_evidence_table(
                    connection,
                    tuple(self._evidence_by_id),
                )
                removed = _delete_obsolete_analyses(connection)
                connection.executemany(
                    "UPDATE analysis_metadata SET value = ? WHERE key = ?",
                    ((value, key) for key, value in sorted(metadata.items())),
                )
        except sqlite3.DatabaseError as error:
            raise AnalysisStoreError(
                "analysis_database_write_failed",
                f"Could not reconcile derived analysis with the current corpus: {error}.",
            ) from error
        self._exact_backend.assert_snapshot_current(self.snapshot)
        _read_and_validate_database(self.path, self.snapshot)
        return removed

    def _create_database(self) -> None:
        temporary_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        connection: sqlite3.Connection | None = None
        metadata = {
            "schema_version": str(ANALYSIS_DATABASE_SCHEMA_VERSION),
            "tool_version": __version__,
            "created_at": utc_now(),
            "source_index_built_at": self.snapshot.index_built_at,
            "source_index_fingerprint": self.snapshot.index_fingerprint,
            "sources": str(self.snapshot.indexed_sources),
            "chunks": str(self.snapshot.indexed_chunks),
            "partial_sources": str(self.snapshot.partial_sources),
            "analysis_contract_schema_version": str(ANALYSIS_CONTRACT_SCHEMA_VERSION),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(temporary_path)
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            _create_schema(connection)
            with connection:
                connection.executemany(
                    "INSERT INTO analysis_metadata(key, value) VALUES (?, ?)",
                    sorted(metadata.items()),
                )
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise AnalysisStoreError(
                    "analysis_database_integrity_failed",
                    "SQLite did not confirm the integrity of the new analysis database.",
                )
            connection.close()
            connection = None
            with temporary_path.open("r+b") as database_file:
                os.fsync(database_file.fileno())
            _read_and_validate_database(temporary_path, self.snapshot)
            self._exact_backend.assert_snapshot_current(self.snapshot)
            os.replace(temporary_path, self.path)
            _fsync_directory(self.path.parent)
        except AnalysisStoreError:
            raise
        except (OSError, sqlite3.DatabaseError) as error:
            raise AnalysisStoreError(
                "analysis_database_build_failed",
                f"Could not create the local analysis database: {error}.",
            ) from error
        finally:
            if connection is not None:
                connection.close()
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _validate_current_database(self) -> None:
        _read_and_validate_database(self.path, self.snapshot)


def reconcile_analysis_database(project_root: Path | str) -> int:
    """Prune absent evidence and retain compatible analyses after corpus sync."""

    path = analysis_database_path_for(project_root)
    if not path.is_file():
        return 0
    return AnalysisStore(project_root, reconcile=True).reconciled_records


def analysis_status_report(project_root: Path | str) -> dict[str, Any]:
    """Return non-content analysis health and aggregate progress metadata."""

    path = analysis_database_path_for(project_root)
    if not path.is_file():
        return {"status": "missing", "path": str(path)}
    try:
        store = AnalysisStore(project_root)
        with closing(_connect_read_only(store.path)) as connection:
            run_counts = {
                str(row["status"]): int(row["count"])
                for row in connection.execute(
                    "SELECT status, COUNT(*) AS count FROM extraction_runs GROUP BY status"
                ).fetchall()
            }
            candidate_counts = {
                "concepts": _table_count(connection, "concept_candidates"),
                "claims": _table_count(connection, "claim_candidates"),
                "relations": _table_count(connection, "relation_candidates"),
                "reviews": _table_count(connection, "candidate_reviews"),
            }
            analyzed = _table_count(connection, "evidence_analyses")
            latest_row = connection.execute(
                """
                SELECT run_id
                FROM extraction_runs
                ORDER BY started_at DESC, run_id DESC
                LIMIT 1
                """
            ).fetchone()
            latest_run = (
                store.run_descriptor(str(latest_row["run_id"]), connection=connection)
                if latest_row is not None
                else None
            )
        return {
            "status": "ready",
            "path": str(path),
            "schema_version": ANALYSIS_DATABASE_SCHEMA_VERSION,
            "source_index_fingerprint": store.snapshot.index_fingerprint,
            "runs": {
                status: run_counts.get(status, 0)
                for status in ("running", "completed", "failed")
            },
            "analyzed_evidence_records": analyzed,
            "latest_run": latest_run.to_dict() if latest_run is not None else None,
            **candidate_counts,
        }
    except AnalysisStoreError as error:
        status = "stale" if error.code == "analysis_database_stale" else "invalid"
        return {"status": status, "path": str(path), "error": str(error)}
    except RetrievalError as error:
        status = (
            "stale" if error.code in {"index_missing", "index_stale"} else "invalid"
        )
        return {"status": status, "path": str(path), "error": str(error)}


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        f"""
        PRAGMA user_version = {ANALYSIS_DATABASE_SCHEMA_VERSION};
        PRAGMA foreign_keys = ON;

        CREATE TABLE analysis_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE extraction_runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
            started_at TEXT NOT NULL,
            completed_at TEXT,
            source_index_fingerprint TEXT NOT NULL,
            extractor_json TEXT NOT NULL,
            extractor_fingerprint TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            scope_json TEXT NOT NULL
        );

        CREATE TABLE evidence_analyses (
            run_id TEXT NOT NULL REFERENCES extraction_runs(run_id),
            evidence_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (
                status IN ('accepted', 'partial', 'empty', 'rejected')
            ),
            raw_output_sha256 TEXT NOT NULL,
            inference_ms REAL CHECK (inference_ms IS NULL OR inference_ms >= 0),
            output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
            output_truncated INTEGER NOT NULL CHECK (output_truncated IN (0, 1)),
            rejection_count INTEGER NOT NULL CHECK (rejection_count >= 0),
            PRIMARY KEY (run_id, evidence_id)
        );

        CREATE INDEX evidence_analyses_source_idx
            ON evidence_analyses(run_id, source_id);

        CREATE TABLE evidence_spans (
            span_id TEXT PRIMARY KEY,
            evidence_id TEXT NOT NULL,
            start_offset INTEGER NOT NULL CHECK (start_offset >= 0),
            end_offset INTEGER NOT NULL CHECK (end_offset > start_offset),
            text_sha256 TEXT NOT NULL
        );

        CREATE INDEX evidence_spans_evidence_idx ON evidence_spans(evidence_id);

        CREATE TABLE concept_candidates (
            concept_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            local_id TEXT NOT NULL,
            label TEXT NOT NULL,
            description TEXT NOT NULL,
            concept_type TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            support_span_id TEXT NOT NULL REFERENCES evidence_spans(span_id),
            review_state TEXT NOT NULL CHECK (
                review_state IN ('unreviewed', 'accepted', 'rejected', 'needs_review')
            ),
            FOREIGN KEY (run_id, evidence_id)
                REFERENCES evidence_analyses(run_id, evidence_id),
            UNIQUE (run_id, evidence_id, local_id)
        );

        CREATE INDEX concept_candidates_run_idx ON concept_candidates(run_id);
        CREATE INDEX concept_candidates_label_idx ON concept_candidates(label);

        CREATE TABLE claim_candidates (
            claim_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            local_id TEXT NOT NULL,
            statement TEXT NOT NULL,
            claim_type TEXT NOT NULL,
            polarity TEXT NOT NULL,
            certainty TEXT NOT NULL,
            conditional INTEGER NOT NULL CHECK (conditional IN (0, 1)),
            attribution TEXT NOT NULL,
            normative_force TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            support_span_id TEXT NOT NULL REFERENCES evidence_spans(span_id),
            review_state TEXT NOT NULL CHECK (
                review_state IN ('unreviewed', 'accepted', 'rejected', 'needs_review')
            ),
            FOREIGN KEY (run_id, evidence_id)
                REFERENCES evidence_analyses(run_id, evidence_id),
            UNIQUE (run_id, evidence_id, local_id)
        );

        CREATE INDEX claim_candidates_run_idx ON claim_candidates(run_id);
        CREATE INDEX claim_candidates_type_idx
            ON claim_candidates(
                claim_type, polarity, certainty, conditional,
                attribution, normative_force
            );

        CREATE TABLE claim_concepts (
            claim_id TEXT NOT NULL REFERENCES claim_candidates(claim_id),
            concept_id TEXT NOT NULL REFERENCES concept_candidates(concept_id),
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            PRIMARY KEY (claim_id, concept_id),
            UNIQUE (claim_id, ordinal)
        );

        CREATE TABLE relation_candidates (
            relation_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            local_id TEXT NOT NULL,
            claim_id TEXT NOT NULL REFERENCES claim_candidates(claim_id),
            subject_concept_id TEXT NOT NULL REFERENCES concept_candidates(concept_id),
            relation_type TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object_concept_id TEXT NOT NULL REFERENCES concept_candidates(concept_id),
            polarity TEXT NOT NULL,
            certainty TEXT NOT NULL,
            conditional INTEGER NOT NULL CHECK (conditional IN (0, 1)),
            attribution TEXT NOT NULL,
            normative_force TEXT NOT NULL,
            confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
            support_span_id TEXT NOT NULL REFERENCES evidence_spans(span_id),
            review_state TEXT NOT NULL CHECK (
                review_state IN ('unreviewed', 'accepted', 'rejected', 'needs_review')
            ),
            FOREIGN KEY (run_id, evidence_id)
                REFERENCES evidence_analyses(run_id, evidence_id),
            CHECK (subject_concept_id <> object_concept_id),
            UNIQUE (run_id, evidence_id, local_id)
        );

        CREATE INDEX relation_candidates_run_idx ON relation_candidates(run_id);
        CREATE INDEX relation_candidates_type_idx ON relation_candidates(relation_type);

        CREATE TABLE candidate_rejections (
            run_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
            category TEXT NOT NULL CHECK (
                category IN ('concept', 'claim', 'relation', 'response')
            ),
            code TEXT NOT NULL,
            json_path TEXT NOT NULL,
            local_id TEXT,
            PRIMARY KEY (run_id, evidence_id, ordinal),
            FOREIGN KEY (run_id, evidence_id)
                REFERENCES evidence_analyses(run_id, evidence_id)
        );

        CREATE TABLE candidate_reviews (
            review_id TEXT PRIMARY KEY,
            candidate_kind TEXT NOT NULL CHECK (
                candidate_kind IN ('concept', 'claim', 'relation')
            ),
            candidate_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK (
                state IN ('accepted', 'rejected', 'needs_review')
            ),
            reviewed_at TEXT NOT NULL,
            reviewer TEXT,
            note TEXT
        );

        CREATE INDEX candidate_reviews_candidate_idx
            ON candidate_reviews(candidate_kind, candidate_id, reviewed_at);
        """
    )


_RUN_DESCRIPTOR_SQL = """
SELECT
    r.run_id,
    r.status,
    r.started_at,
    r.completed_at,
    r.source_index_fingerprint,
    r.extractor_json,
    r.extractor_fingerprint,
    r.prompt_version,
    r.prompt_sha256,
    r.scope_json,
    COUNT(DISTINCT ea.evidence_id) AS analyzed_evidence,
    COUNT(DISTINCT CASE WHEN ea.status = 'accepted' THEN ea.evidence_id END)
        AS accepted_evidence,
    COUNT(DISTINCT CASE WHEN ea.status = 'partial' THEN ea.evidence_id END)
        AS partial_evidence,
    COUNT(DISTINCT CASE WHEN ea.status = 'empty' THEN ea.evidence_id END)
        AS empty_evidence,
    COUNT(DISTINCT CASE WHEN ea.status = 'rejected' THEN ea.evidence_id END)
        AS rejected_evidence,
    (SELECT COUNT(*) FROM concept_candidates c WHERE c.run_id = r.run_id)
        AS concepts,
    (SELECT COUNT(*) FROM claim_candidates c WHERE c.run_id = r.run_id)
        AS claims,
    (SELECT COUNT(*) FROM relation_candidates x WHERE x.run_id = r.run_id)
        AS relations,
    COALESCE(SUM(ea.rejection_count), 0) AS rejected_candidates,
    COALESCE(SUM(ea.inference_ms), 0.0) AS inference_ms,
    COALESCE(SUM(ea.output_tokens), 0) AS output_tokens,
    COALESCE(SUM(ea.output_truncated), 0) AS truncated_evidence
FROM extraction_runs r
LEFT JOIN evidence_analyses ea ON ea.run_id = r.run_id
WHERE r.run_id = ?
GROUP BY r.run_id
"""


def _descriptor_from_row(path: Path, row: sqlite3.Row) -> AnalysisRunDescriptor:
    extractor = _json_object(str(row["extractor_json"]), "extractor provenance")
    scope = _json_object(str(row["scope_json"]), "analysis scope")
    status = str(row["status"])
    if status not in _RUN_STATUSES:
        raise AnalysisStoreError(
            "analysis_database_schema_invalid", "Analysis run has an invalid status."
        )
    return AnalysisRunDescriptor(
        path=str(path),
        schema_version=ANALYSIS_DATABASE_SCHEMA_VERSION,
        run_id=str(row["run_id"]),
        status=status,  # type: ignore[arg-type]
        started_at=str(row["started_at"]),
        completed_at=(
            str(row["completed_at"]) if row["completed_at"] is not None else None
        ),
        source_index_fingerprint=str(row["source_index_fingerprint"]),
        extractor_fingerprint=str(row["extractor_fingerprint"]),
        extractor=extractor,
        prompt_version=str(row["prompt_version"]),
        prompt_sha256=str(row["prompt_sha256"]),
        scope=scope,
        analyzed_evidence=int(row["analyzed_evidence"]),
        accepted_evidence=int(row["accepted_evidence"]),
        partial_evidence=int(row["partial_evidence"]),
        empty_evidence=int(row["empty_evidence"]),
        rejected_evidence=int(row["rejected_evidence"]),
        concepts=int(row["concepts"]),
        claims=int(row["claims"]),
        relations=int(row["relations"]),
        rejected_candidates=int(row["rejected_candidates"]),
        inference_ms=round(float(row["inference_ms"]), 6),
        output_tokens=int(row["output_tokens"]),
        truncated_evidence=int(row["truncated_evidence"]),
    )


def _read_and_validate_database(
    path: Path,
    snapshot: SearchCorpusSnapshot,
    *,
    allow_stale: bool = False,
) -> None:
    try:
        with closing(_connect_read_only(path)) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise AnalysisStoreError(
                    "analysis_database_integrity_failed",
                    "SQLite did not confirm the integrity of the analysis database.",
                )
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_keys:
                raise AnalysisStoreError(
                    "analysis_database_integrity_failed",
                    "Analysis database contains broken relational links.",
                )
            user_version = connection.execute("PRAGMA user_version").fetchone()
            if (
                user_version is None
                or user_version[0] != ANALYSIS_DATABASE_SCHEMA_VERSION
            ):
                raise AnalysisStoreError(
                    "analysis_database_schema_invalid",
                    "Analysis database schema is unsupported.",
                )
            metadata = {
                str(row["key"]): str(row["value"])
                for row in connection.execute(
                    "SELECT key, value FROM analysis_metadata"
                ).fetchall()
            }
    except AnalysisStoreError:
        raise
    except sqlite3.DatabaseError as error:
        raise AnalysisStoreError(
            "analysis_database_read_failed",
            f"Could not read the local analysis database: {error}.",
        ) from error

    if set(metadata) != _METADATA_FIELDS:
        raise AnalysisStoreError(
            "analysis_database_schema_invalid",
            "Analysis database metadata is incomplete or contains unknown fields.",
        )
    integers = {
        name: _metadata_integer(metadata, name)
        for name in (
            "schema_version",
            "sources",
            "chunks",
            "partial_sources",
            "analysis_contract_schema_version",
        )
    }
    if (
        integers["schema_version"] != ANALYSIS_DATABASE_SCHEMA_VERSION
        or integers["analysis_contract_schema_version"]
        != ANALYSIS_CONTRACT_SCHEMA_VERSION
    ):
        raise AnalysisStoreError(
            "analysis_database_schema_invalid",
            "Analysis database or candidate contract schema is unsupported.",
        )
    expected = (
        snapshot.index_fingerprint,
        snapshot.indexed_sources,
        snapshot.indexed_chunks,
        snapshot.partial_sources,
    )
    observed = (
        metadata["source_index_fingerprint"],
        integers["sources"],
        integers["chunks"],
        integers["partial_sources"],
    )
    if observed != expected and not allow_stale:
        raise AnalysisStoreError(
            "analysis_database_stale",
            "The exact corpus changed after derived analysis began; synchronize or resume analysis to reconcile it.",
        )


def _connect(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection
    except sqlite3.DatabaseError as error:
        raise AnalysisStoreError(
            "analysis_database_open_failed",
            f"Could not open the local analysis database: {error}.",
        ) from error


def _connect_read_only(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
    except sqlite3.DatabaseError as error:
        raise AnalysisStoreError(
            "analysis_database_open_failed",
            f"Could not open the local analysis database: {error}.",
        ) from error


def _require_run(connection: sqlite3.Connection, run_id: str) -> sqlite3.Row:
    row = connection.execute(
        "SELECT run_id, status FROM extraction_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        raise AnalysisStoreError(
            "analysis_run_missing", f"No analysis run has ID '{run_id}'."
        )
    return row


def _scope_resume_key(scope: Mapping[str, Any]) -> str:
    """Return a scope identity independent of revision-specific progress counts."""

    return _canonical_json(
        {key: value for key, value in scope.items() if key != "selected_evidence"}
    )


def _replace_temporary_evidence_table(
    connection: sqlite3.Connection, evidence_ids: Sequence[str]
) -> None:
    connection.execute("DROP TABLE IF EXISTS temp.current_analysis_evidence")
    connection.execute(
        """
        CREATE TEMP TABLE current_analysis_evidence (
            evidence_id TEXT PRIMARY KEY
        ) WITHOUT ROWID
        """
    )
    connection.executemany(
        "INSERT INTO current_analysis_evidence(evidence_id) VALUES (?)",
        ((evidence_id,) for evidence_id in evidence_ids),
    )


def _delete_obsolete_analyses(
    connection: sqlite3.Connection, *, run_id: str | None = None
) -> int:
    """Delete analysis rows outside a prepared evidence set in FK-safe order."""

    connection.execute("DROP TABLE IF EXISTS temp.obsolete_analysis_keys")
    connection.execute(
        """
        CREATE TEMP TABLE obsolete_analysis_keys (
            run_id TEXT NOT NULL,
            evidence_id TEXT NOT NULL,
            PRIMARY KEY (run_id, evidence_id)
        ) WITHOUT ROWID
        """
    )
    run_clause = "AND ea.run_id = ?" if run_id is not None else ""
    parameters: tuple[str, ...] = (run_id,) if run_id is not None else ()
    connection.execute(
        f"""
        INSERT INTO obsolete_analysis_keys(run_id, evidence_id)
        SELECT ea.run_id, ea.evidence_id
        FROM evidence_analyses ea
        WHERE NOT EXISTS (
            SELECT 1
            FROM current_analysis_evidence current
            WHERE current.evidence_id = ea.evidence_id
        )
        {run_clause}
        """,
        parameters,
    )
    count_row = connection.execute(
        "SELECT COUNT(*) AS count FROM obsolete_analysis_keys"
    ).fetchone()
    removed = int(count_row["count"]) if count_row is not None else 0
    if not removed:
        return 0

    for candidate_kind, table, identifier in (
        ("relation", "relation_candidates", "relation_id"),
        ("claim", "claim_candidates", "claim_id"),
        ("concept", "concept_candidates", "concept_id"),
    ):
        connection.execute(
            f"""
            DELETE FROM candidate_reviews
            WHERE candidate_kind = ?
              AND candidate_id IN (
                  SELECT candidate.{identifier}
                  FROM {table} candidate
                  JOIN obsolete_analysis_keys obsolete
                    ON obsolete.run_id = candidate.run_id
                   AND obsolete.evidence_id = candidate.evidence_id
              )
            """,
            (candidate_kind,),
        )
    connection.execute(
        """
        DELETE FROM claim_concepts
        WHERE claim_id IN (
            SELECT claim.claim_id
            FROM claim_candidates claim
            JOIN obsolete_analysis_keys obsolete
              ON obsolete.run_id = claim.run_id
             AND obsolete.evidence_id = claim.evidence_id
        )
        OR concept_id IN (
            SELECT concept.concept_id
            FROM concept_candidates concept
            JOIN obsolete_analysis_keys obsolete
              ON obsolete.run_id = concept.run_id
             AND obsolete.evidence_id = concept.evidence_id
        )
        """
    )
    for table in (
        "relation_candidates",
        "claim_candidates",
        "concept_candidates",
        "candidate_rejections",
    ):
        connection.execute(
            f"""
            DELETE FROM {table}
            WHERE EXISTS (
                SELECT 1
                FROM obsolete_analysis_keys obsolete
                WHERE obsolete.run_id = {table}.run_id
                  AND obsolete.evidence_id = {table}.evidence_id
            )
            """
        )
    connection.execute(
        """
        DELETE FROM evidence_analyses
        WHERE EXISTS (
            SELECT 1
            FROM obsolete_analysis_keys obsolete
            WHERE obsolete.run_id = evidence_analyses.run_id
              AND obsolete.evidence_id = evidence_analyses.evidence_id
        )
        """
    )
    connection.execute(
        """
        DELETE FROM evidence_spans
        WHERE NOT EXISTS (
            SELECT 1
            FROM evidence_analyses analysis
            WHERE analysis.evidence_id = evidence_spans.evidence_id
        )
        """
    )
    return removed


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise AnalysisStoreError(
            "analysis_run_id_invalid", "Analysis run ID is invalid."
        )


def _validate_analysis_spans(analysis: EvidenceAnalysis, excerpt_length: int) -> None:
    for candidates in (analysis.concepts, analysis.claims, analysis.relations):
        for candidate in candidates:
            span = candidate.support
            if (
                span.evidence_id != analysis.evidence_id
                or span.start_offset < 0
                or span.end_offset <= span.start_offset
                or span.end_offset > excerpt_length
                or _DIGEST_PATTERN.fullmatch(span.text_sha256) is None
            ):
                raise AnalysisStoreError(
                    "analysis_support_invalid",
                    "Candidate support lies outside its exact evidence excerpt.",
                )


def _provenance_object(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AnalysisStoreError(
            "analysis_provenance_invalid", f"Analysis {label} must be an object."
        )
    try:
        serialized = _canonical_json(dict(value))
        clean = json.loads(serialized)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise AnalysisStoreError(
            "analysis_provenance_invalid",
            f"Analysis {label} must contain finite JSON values.",
        ) from error
    if not isinstance(clean, dict) or len(serialized) > 40_000:
        raise AnalysisStoreError(
            "analysis_provenance_invalid", f"Analysis {label} is not a bounded object."
        )
    _reject_content_keys(clean)
    return clean


def _extractor_run_fingerprint(extractor: Mapping[str, Any]) -> str:
    """Fingerprint settings affecting output while excluding per-load measurements."""

    stable = {
        key: extractor[key] for key in sorted(_EXTRACTOR_RUN_FIELDS) if key in extractor
    }
    digest = sha256(_canonical_json(stable).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _reject_content_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key).casefold() in _FORBIDDEN_PROVENANCE_KEYS:
                raise AnalysisStoreError(
                    "analysis_provenance_content_forbidden",
                    "Analysis provenance must not contain excerpts, paths, prompts, or raw model output.",
                )
            _reject_content_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_content_keys(nested)


def _json_object(value: str, label: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise AnalysisStoreError(
            "analysis_database_schema_invalid", f"Stored {label} is invalid JSON."
        ) from error
    if not isinstance(result, dict):
        raise AnalysisStoreError(
            "analysis_database_schema_invalid", f"Stored {label} must be an object."
        )
    _reject_content_keys(result)
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _metadata_integer(metadata: Mapping[str, str], name: str) -> int:
    try:
        value = int(metadata[name])
    except (KeyError, TypeError, ValueError) as error:
        raise AnalysisStoreError(
            "analysis_database_schema_invalid",
            f"Analysis metadata '{name}' must be an integer.",
        ) from error
    if value < 0:
        raise AnalysisStoreError(
            "analysis_database_schema_invalid",
            f"Analysis metadata '{name}' cannot be negative.",
        )
    return value


def _run_id_for(*parts: str) -> str:
    return _digest_id("run", *parts)


def _digest_id(prefix: str, *parts: str) -> str:
    digest = sha256()
    digest.update(f"corpusdock-{prefix}-v1".encode("utf-8"))
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"{prefix}-{digest.hexdigest()}"


def _optional_bounded_text(value: str | None, maximum: int, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise AnalysisStoreError(
            "analysis_review_invalid",
            f"Analysis {label} must be a non-empty string up to {maximum} characters.",
        )
    return value.strip()


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
    return int(row["count"]) if row is not None else 0


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        return
