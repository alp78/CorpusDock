"""Idempotent reconciliation of an authoritative local input directory.

The manifest, extraction artifacts, chunk artifacts, and exact index are each durable
checkpoints. A hard interruption can replay the one uncommitted unit, but committed
sources and evidence are never duplicated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from corpusdock.analysis_store import reconcile_analysis_database
from corpusdock.chunking import (
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_OVERLAP_SENTENCES,
    DEFAULT_SENTENCE_MODEL,
    DEFAULT_TARGET_CHARACTERS,
    ChunkingError,
    chunk_artifact_is_current,
    chunk_extraction_artifact,
    load_chunk_artifact,
    sentence_processor_from,
    sentence_processor_identity,
    write_chunk_artifact,
)
from corpusdock.extraction import (
    ExtractionError,
    extract_source,
    extraction_artifact_is_current,
    load_extraction_artifact,
    repoint_extraction_artifact,
    write_extraction_artifact,
)
from corpusdock.manifest import (
    STATE_DIRECTORY_NAME,
    ManifestError,
    ManifestStore,
    SourceRecord,
    discover_mirror_files,
)
from corpusdock.retrieval import build_search_index, index_status_report
from corpusdock.semantic_index import prune_semantic_index_cache


PIPELINE_CONFIG_SCHEMA_VERSION = 1
PIPELINE_CONFIG_FILE_NAME = "pipeline.json"
_SOURCE_ID_PATTERN = re.compile(r"src-[0-9a-f]{64}")

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Ignored, project-local settings for automatic resume-time scanning."""

    input_root: str
    sentence_processor: str = "sat"
    sentence_model: str = DEFAULT_SENTENCE_MODEL
    target_characters: int = DEFAULT_TARGET_CHARACTERS
    max_characters: int = DEFAULT_MAX_CHARACTERS
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES
    schema_version: int = PIPELINE_CONFIG_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "input_root": self.input_root,
            "chunking": {
                "sentence_processor": self.sentence_processor,
                "sentence_model": self.sentence_model,
                "target_characters": self.target_characters,
                "max_characters": self.max_characters,
                "overlap_sentences": self.overlap_sentences,
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> PipelineConfig:
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise ManifestError("Pipeline configuration schema is unsupported.")
        input_root = value.get("input_root")
        chunking = value.get("chunking")
        if (
            not isinstance(input_root, str)
            or not input_root
            or not isinstance(chunking, dict)
        ):
            raise ManifestError("Pipeline configuration is incomplete.")
        config = cls(
            input_root=input_root,
            sentence_processor=_required_string(chunking, "sentence_processor"),
            sentence_model=_required_string(chunking, "sentence_model"),
            target_characters=_required_integer(chunking, "target_characters"),
            max_characters=_required_integer(chunking, "max_characters"),
            overlap_sentences=_required_integer(chunking, "overlap_sentences"),
        )
        _validate_config(config)
        return config


@dataclass(frozen=True, slots=True)
class SynchronizationSummary:
    """Non-content results from one complete mirror synchronization."""

    input_root: str
    scanned_paths: int
    unique_sources: int
    added_sources: int
    removed_sources: int
    retained_sources: int
    extracted_sources: int
    reused_extractions: int
    repointed_extractions: int
    chunked_sources: int
    reused_chunks: int
    failed_sources: int
    pruned_extraction_artifacts: int
    pruned_chunk_artifacts: int
    pruned_semantic_vectors: int
    pruned_analysis_records: int
    exact_index_rebuilt: bool
    indexed_chunks: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_root": self.input_root,
            "scan": {
                "paths": self.scanned_paths,
                "unique_sources": self.unique_sources,
            },
            "sources": {
                "added": self.added_sources,
                "removed": self.removed_sources,
                "retained": self.retained_sources,
                "failed": self.failed_sources,
            },
            "extraction": {
                "processed": self.extracted_sources,
                "reused": self.reused_extractions,
                "repointed": self.repointed_extractions,
                "pruned": self.pruned_extraction_artifacts,
            },
            "chunking": {
                "processed": self.chunked_sources,
                "reused": self.reused_chunks,
                "pruned": self.pruned_chunk_artifacts,
            },
            "semantic_vectors_pruned": self.pruned_semantic_vectors,
            "analysis_records_pruned": self.pruned_analysis_records,
            "exact_index": {
                "rebuilt": self.exact_index_rebuilt,
                "chunks": self.indexed_chunks,
            },
        }


def pipeline_config_path_for(project_root: Path | str) -> Path:
    return (
        Path(project_root).expanduser().resolve()
        / STATE_DIRECTORY_NAME
        / PIPELINE_CONFIG_FILE_NAME
    )


def configure_input_mirror(
    project_root: Path | str,
    input_root: Path | str,
    *,
    sentence_processor: str = "sat",
    sentence_model: str = DEFAULT_SENTENCE_MODEL,
    target_characters: int = DEFAULT_TARGET_CHARACTERS,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> PipelineConfig:
    """Persist an authoritative local input directory without scanning it."""

    root = Path(input_root).expanduser()
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ManifestError(
            f"Could not resolve input mirror '{input_root}': {error}."
        ) from error
    if not resolved.is_dir():
        raise ManifestError(f"Input mirror is not a directory: '{input_root}'.")
    config = PipelineConfig(
        input_root=str(resolved),
        sentence_processor=sentence_processor,
        sentence_model=sentence_model,
        target_characters=target_characters,
        max_characters=max_characters,
        overlap_sentences=overlap_sentences,
    )
    _validate_config(config)
    _write_config(pipeline_config_path_for(project_root), config)
    return config


def load_pipeline_config(project_root: Path | str) -> PipelineConfig | None:
    """Load resume settings, returning ``None`` for an unconfigured project."""

    path = pipeline_config_path_for(project_root)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as config_file:
            payload = json.load(config_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ManifestError(
            f"Could not read pipeline configuration: {error}."
        ) from error
    return PipelineConfig.from_dict(payload)


def synchronize_input_mirror(
    project_root: Path | str,
    config: PipelineConfig,
    *,
    progress: ProgressCallback | None = None,
) -> SynchronizationSummary:
    """Reconcile, process only missing work, prune absences, and refresh indexes."""

    _validate_config(config)
    root = Path(project_root).expanduser().resolve()
    source_files = discover_mirror_files(config.input_root)
    store = ManifestStore(root)
    reconciliation = store.reconcile_mirror(source_files)
    current_ids = frozenset(reconciliation.manifest.sources)
    pruned_extractions = _prune_orphan_artifacts(
        root / STATE_DIRECTORY_NAME / "extracted", current_ids
    )
    pruned_chunks = _prune_orphan_artifacts(
        root / STATE_DIRECTORY_NAME / "chunks", current_ids
    )

    processor_name, processor_version, processor_model = sentence_processor_identity(
        config.sentence_processor,
        model_name=config.sentence_model,
    )
    processor = None
    extracted = 0
    reused_extractions = 0
    repointed = 0
    chunked = 0
    reused_chunks = 0
    failed_sources = 0
    artifacts_changed = bool(pruned_extractions or pruned_chunks)
    sources = tuple(
        reconciliation.manifest.sources[source_id]
        for source_id in sorted(reconciliation.manifest.sources)
    )
    for ordinal, source in enumerate(sources, start=1):
        if progress is not None:
            progress("ingest", ordinal - 1, len(sources))
        source_path = _available_source_path(source)
        extraction = _load_extraction_for_reuse(root, source.source_id)
        extraction_rebuilt = not extraction_artifact_is_current(extraction, source)
        if extraction_rebuilt:
            artifact = extract_source(source, source_path)
            write_extraction_artifact(root, artifact)
            extraction = artifact.to_dict()
            extracted += 1
            artifacts_changed = True
        else:
            assert isinstance(extraction, dict)
            if extraction.get("source_path") not in source.original_paths:
                repoint_extraction_artifact(root, extraction, source, source_path)
                extraction = {**extraction, "source_path": str(source_path)}
                repointed += 1
                artifacts_changed = True
            else:
                reused_extractions += 1

        if extraction.get("status") == "failed":
            failed_sources += 1
        chunks = _load_chunks_for_reuse(root, source.source_id)
        chunks_current = not extraction_rebuilt and chunk_artifact_is_current(
            chunks,
            source,
            sentence_processor_name=processor_name,
            sentence_processor_version=processor_version,
            sentence_model=processor_model,
            target_characters=config.target_characters,
            max_characters=config.max_characters,
            overlap_sentences=config.overlap_sentences,
        )
        if chunks_current:
            reused_chunks += 1
            continue
        if processor is None:
            processor = sentence_processor_from(
                config.sentence_processor,
                model_name=config.sentence_model,
            )
        chunk_artifact = chunk_extraction_artifact(
            extraction,
            processor,
            target_characters=config.target_characters,
            max_characters=config.max_characters,
            overlap_sentences=config.overlap_sentences,
        )
        write_chunk_artifact(root, chunk_artifact)
        chunked += 1
        artifacts_changed = True
    if progress is not None:
        progress("ingest", len(sources), len(sources))

    index_report = index_status_report(root)
    rebuild_index = (
        reconciliation.changed
        or artifacts_changed
        or index_report.get("status") != "ready"
    )
    if rebuild_index:
        index_summary = build_search_index(root)
        indexed_chunks = index_summary.chunks
    else:
        indexed_chunks = int(index_report.get("chunks", 0))

    # These two operations retain work for evidence that is still present while
    # removing derived state for sources no longer in the authoritative mirror.
    pruned_semantic = prune_semantic_index_cache(root)
    pruned_analysis = reconcile_analysis_database(root)
    return SynchronizationSummary(
        input_root=config.input_root,
        scanned_paths=reconciliation.scanned_paths,
        unique_sources=reconciliation.unique_sources,
        added_sources=len(reconciliation.added_source_ids),
        removed_sources=len(reconciliation.removed_source_ids),
        retained_sources=len(reconciliation.retained_source_ids),
        extracted_sources=extracted,
        reused_extractions=reused_extractions,
        repointed_extractions=repointed,
        chunked_sources=chunked,
        reused_chunks=reused_chunks,
        failed_sources=failed_sources,
        pruned_extraction_artifacts=pruned_extractions,
        pruned_chunk_artifacts=pruned_chunks,
        pruned_semantic_vectors=pruned_semantic,
        pruned_analysis_records=pruned_analysis,
        exact_index_rebuilt=rebuild_index,
        indexed_chunks=indexed_chunks,
    )


def _validate_config(config: PipelineConfig) -> None:
    sentence_processor_identity(
        config.sentence_processor,
        model_name=config.sentence_model,
    )
    if config.target_characters < 1 or config.max_characters < config.target_characters:
        raise ManifestError(
            "Chunk target must be positive and no larger than the maximum."
        )
    if config.overlap_sentences < 0:
        raise ManifestError("Sentence overlap cannot be negative.")


def _available_source_path(source: SourceRecord) -> Path:
    for original_path in source.original_paths:
        path = Path(original_path)
        if path.is_file():
            return path
    raise ManifestError(
        f"No registered path is currently available for source '{source.source_id}'."
    )


def _load_extraction_for_reuse(root: Path, source_id: str) -> dict[str, Any] | None:
    try:
        return load_extraction_artifact(root, source_id)
    except ExtractionError:
        return None


def _load_chunks_for_reuse(root: Path, source_id: str) -> dict[str, Any] | None:
    try:
        return load_chunk_artifact(root, source_id)
    except ChunkingError:
        return None


def _prune_orphan_artifacts(directory: Path, current_ids: frozenset[str]) -> int:
    if not directory.is_dir():
        return 0
    removed = 0
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if (
            path.is_file()
            and path.suffix == ".json"
            and _SOURCE_ID_PATTERN.fullmatch(path.stem) is not None
            and path.stem not in current_ids
        ):
            try:
                path.unlink()
            except OSError as error:
                raise ManifestError(
                    f"Could not prune derived artifact '{path}': {error}."
                ) from error
            removed += 1
    return removed


def _write_config(path: Path, config: PipelineConfig) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as config_file:
            json.dump(config.to_dict(), config_file, indent=2, sort_keys=True)
            config_file.write("\n")
            config_file.flush()
            os.fsync(config_file.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ManifestError(
            f"Could not write pipeline configuration: {error}."
        ) from error


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ManifestError(f"Pipeline field '{key}' must be a non-empty string.")
    return value


def _required_integer(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ManifestError(f"Pipeline field '{key}' must be an integer.")
    return value
