"""Anchor-aware, fully local sentence processing and chunk generation."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import re
from typing import Any, Literal, Protocol
from uuid import uuid4

from corpusdock import __version__
from corpusdock.extraction import EXTRACTION_SCHEMA_VERSION
from corpusdock.manifest import STATE_DIRECTORY_NAME, SourceRecord, utc_now


CHUNK_SCHEMA_VERSION = 2
CHUNK_DIRECTORY_NAME = "chunks"
DEFAULT_SENTENCE_MODEL = "sat-12l-sm"
DEFAULT_TARGET_CHARACTERS = 1_200
DEFAULT_MAX_CHARACTERS = 1_800
DEFAULT_OVERLAP_SENTENCES = 1

ChunkingStatus = Literal["complete", "partial", "failed"]


class ChunkingError(Exception):
    """An expected sentence-processing or chunk-artifact failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class SentenceProcessor(Protocol):
    """A local sentence splitter that must preserve every input character."""

    name: str
    version: str
    model_name: str

    def split_many(self, texts: Sequence[str]) -> tuple[tuple[str, ...], ...]: ...


class RuleSentenceProcessor:
    """Dependency-free fallback sentence boundary disambiguation."""

    name = "corpusdock.rule_sentence"
    version = __version__
    model_name = "none"
    _ABBREVIATIONS = {
        "dr.",
        "mr.",
        "mrs.",
        "ms.",
        "prof.",
        "sr.",
        "jr.",
        "st.",
        "vs.",
        "etc.",
        "e.g.",
        "i.e.",
        "fig.",
        "no.",
    }
    _BOUNDARY = re.compile(r"[.!?]+[\"'’”\)\]]*(?:\s+|$)")

    def split_many(self, texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        return tuple(self._split(text) for text in texts)

    def _split(self, text: str) -> tuple[str, ...]:
        if not text:
            return ()
        segments: list[str] = []
        start = 0
        for match in self._BOUNDARY.finditer(text):
            punctuation_end = match.start()
            prefix = text[start : punctuation_end + 1]
            final_token = (
                prefix.rstrip().rsplit(maxsplit=1)[-1].lower() if prefix.strip() else ""
            )
            if final_token in self._ABBREVIATIONS:
                continue
            end = match.end()
            segments.append(text[start:end])
            start = end
        if start < len(text):
            segments.append(text[start:])
        return tuple(segments)


class SaTSentenceProcessor:
    """State-of-the-art SaT segmentation using a local ONNX model."""

    name = "wtpsplit-lite.SaT"

    def __init__(self, model_name: str = DEFAULT_SENTENCE_MODEL) -> None:
        try:
            from wtpsplit_lite import SaT
        except ImportError as error:
            raise ChunkingError(
                "sentence_model_unavailable",
                "SaT requires the local-models extra: run 'uv sync --extra local-models'.",
            ) from error
        self.model_name = model_name
        try:
            self.version = package_version("wtpsplit-lite")
        except PackageNotFoundError:
            self.version = "unknown"
        try:
            self._model = SaT(model_name, ort_providers=["CPUExecutionProvider"])
        except Exception as error:
            raise ChunkingError(
                "sentence_model_load_failed",
                f"Could not load local SaT model '{model_name}': {error}",
            ) from error

    def split_many(self, texts: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        if not texts:
            return ()
        try:
            raw_results = self._model.split(
                list(texts),
                strip_whitespace=False,
                treat_newline_as_space=True,
                weighting="hat",
                stride=128,
                block_size=512,
                batch_size=32,
            )
            results = tuple(tuple(segments) for segments in raw_results)
        except Exception as error:
            raise ChunkingError(
                "sentence_processing_failed", f"Local SaT inference failed: {error}"
            ) from error
        if len(results) != len(texts):
            raise ChunkingError(
                "sentence_reconstruction_failed",
                "Sentence processor returned a different number of documents than it received.",
            )
        for text, segments in zip(texts, results, strict=True):
            if "".join(segments) != text:
                raise ChunkingError(
                    "sentence_reconstruction_failed",
                    "Sentence processor did not preserve the exact input text and offsets.",
                )
        return results


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A contiguous exact excerpt mapped to one or more source anchors."""

    chunk_id: str
    source_id: str
    text: str
    start_offset: int
    end_offset: int
    anchor_ids: tuple[str, ...]
    locators: tuple[dict[str, Any], ...]
    sentence_count: int
    lexical_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "anchor_ids": list(self.anchor_ids),
            "locators": list(self.locators),
            "sentence_count": self.sentence_count,
            "lexical_token_count": self.lexical_token_count,
        }


@dataclass(frozen=True, slots=True)
class ChunkArtifact:
    """Versioned chunk output derived from one extraction artifact."""

    source_id: str
    source_sha256: str
    chunked_at: str
    status: ChunkingStatus
    sentence_processor_name: str
    sentence_processor_version: str
    sentence_model: str
    target_characters: int
    max_characters: int
    overlap_sentences: int
    chunks: tuple[ChunkRecord, ...]
    warnings: tuple[str, ...]
    schema_version: int = CHUNK_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "chunked_at": self.chunked_at,
            "status": self.status,
            "chunker": {
                "name": "corpusdock.anchor_sentence",
                "version": __version__,
                "sentence_processor": self.sentence_processor_name,
                "sentence_processor_version": self.sentence_processor_version,
                "sentence_model": self.sentence_model,
                "target_characters": self.target_characters,
                "max_characters": self.max_characters,
                "overlap_sentences": self.overlap_sentences,
            },
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "warnings": list(self.warnings),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "status": self.status,
            "chunk_count": len(self.chunks),
            "warning_count": len(self.warnings),
            "sentence_processor": self.sentence_processor_name,
            "sentence_model": self.sentence_model,
        }


@dataclass(frozen=True, slots=True)
class _AnchorSpan:
    anchor_id: str
    start: int
    end: int
    locator: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _StructuralGroup:
    start: int
    end: int
    anchors: tuple[_AnchorSpan, ...]


@dataclass(frozen=True, slots=True)
class _SentenceSpan:
    start: int
    end: int


def sentence_processor_from(
    name: str, *, model_name: str = DEFAULT_SENTENCE_MODEL
) -> SentenceProcessor:
    if name == "rule":
        return RuleSentenceProcessor()
    if name == "sat":
        return SaTSentenceProcessor(model_name)
    raise ChunkingError(
        "sentence_processor_unknown", f"Unknown sentence processor '{name}'."
    )


def chunk_artifact_path_for(project_root: Path | str, source_id: str) -> Path:
    return (
        Path(project_root).expanduser().resolve()
        / STATE_DIRECTORY_NAME
        / CHUNK_DIRECTORY_NAME
        / f"{source_id}.json"
    )


def chunk_id_for(
    *,
    source_id: str,
    start_offset: int,
    end_offset: int,
    sentence_processor_name: str,
    sentence_model: str,
    target_characters: int,
    max_characters: int,
    overlap_sentences: int,
    text: str,
) -> str:
    """Return the stable ID for an exact chunk and its derivation settings."""

    fingerprint = "\0".join(
        (
            source_id,
            str(start_offset),
            str(end_offset),
            sentence_processor_name,
            sentence_model,
            str(target_characters),
            str(max_characters),
            str(overlap_sentences),
            text,
        )
    )
    return f"chk-{sha256(fingerprint.encode('utf-8')).hexdigest()}"


def chunk_extraction_artifact(
    extraction: dict[str, Any],
    sentence_processor: SentenceProcessor,
    *,
    target_characters: int = DEFAULT_TARGET_CHARACTERS,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    now: Callable[[], str] = utc_now,
) -> ChunkArtifact:
    """Build exact, structure-bounded chunks from an extraction artifact."""

    if target_characters <= 0 or max_characters < target_characters:
        raise ChunkingError(
            "chunk_size_invalid",
            "Chunk target must be positive and no larger than the maximum.",
        )
    if overlap_sentences < 0:
        raise ChunkingError(
            "chunk_overlap_invalid", "Sentence overlap cannot be negative."
        )
    if extraction.get("schema_version") != EXTRACTION_SCHEMA_VERSION:
        raise ChunkingError(
            "extraction_schema_invalid", "Unsupported extraction artifact schema."
        )
    source_id = _required_string(extraction, "source_id")
    source_sha256 = _required_string(extraction, "source_sha256")
    extraction_status = extraction.get("status")
    if extraction_status == "failed":
        return _failed_chunk_artifact(
            source_id,
            source_sha256,
            sentence_processor,
            target_characters,
            max_characters,
            overlap_sentences,
            "source_extraction_failed: Source extraction failed; no text was chunked.",
            now(),
        )
    text = extraction.get("text")
    if not isinstance(text, str):
        raise ChunkingError(
            "extraction_text_invalid", "Extraction text must be a string."
        )
    anchors = _read_anchor_spans(extraction.get("anchors"), text_length=len(text))
    groups = _structural_groups(anchors)
    group_texts = [text[group.start : group.end] for group in groups]
    sentence_groups = sentence_processor.split_many(group_texts)

    chunk_records: list[ChunkRecord] = []
    for group, group_sentences in zip(groups, sentence_groups, strict=True):
        sentence_spans = _sentence_spans(group, group_sentences)
        for packed_sentences in _pack_sentences(
            sentence_spans,
            target_characters=target_characters,
            max_characters=max_characters,
            overlap_sentences=overlap_sentences,
            text=text,
        ):
            start = packed_sentences[0].start
            end = packed_sentences[-1].end
            chunk_text = text[start:end]
            if not chunk_text.strip():
                continue
            intersecting = tuple(
                anchor
                for anchor in group.anchors
                if anchor.start < end and anchor.end > start
            )
            locators = tuple(
                _clamped_locator(anchor, start=start, end=end)
                for anchor in intersecting
            )
            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id_for(
                        source_id=source_id,
                        start_offset=start,
                        end_offset=end,
                        sentence_processor_name=sentence_processor.name,
                        sentence_model=sentence_processor.model_name,
                        target_characters=target_characters,
                        max_characters=max_characters,
                        overlap_sentences=overlap_sentences,
                        text=chunk_text,
                    ),
                    source_id=source_id,
                    text=chunk_text,
                    start_offset=start,
                    end_offset=end,
                    anchor_ids=tuple(anchor.anchor_id for anchor in intersecting),
                    locators=locators,
                    sentence_count=len(packed_sentences),
                    lexical_token_count=len(
                        re.findall(r"\w+|[^\w\s]", chunk_text, flags=re.UNICODE)
                    ),
                )
            )

    warnings: list[str] = []
    status: ChunkingStatus = "complete"
    if extraction_status == "partial":
        status = "partial"
        warnings.append(
            "source_extraction_partial: Chunks cover only the text available from extraction."
        )
    if not chunk_records:
        status = "partial"
        warnings.append(
            "no_extractable_text: No non-whitespace extracted text was available to chunk."
        )
    return ChunkArtifact(
        source_id=source_id,
        source_sha256=source_sha256,
        chunked_at=now(),
        status=status,
        sentence_processor_name=sentence_processor.name,
        sentence_processor_version=sentence_processor.version,
        sentence_model=sentence_processor.model_name,
        target_characters=target_characters,
        max_characters=max_characters,
        overlap_sentences=overlap_sentences,
        chunks=tuple(chunk_records),
        warnings=tuple(warnings),
    )


def write_chunk_artifact(project_root: Path | str, artifact: ChunkArtifact) -> Path:
    path = chunk_artifact_path_for(project_root, artifact.source_id)
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as artifact_file:
            json.dump(
                artifact.to_dict(),
                artifact_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            artifact_file.write("\n")
            artifact_file.flush()
            os.fsync(artifact_file.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise ChunkingError(
            "chunk_artifact_write_failed", f"Could not write chunk artifact: {error}."
        ) from error
    return path


def load_chunk_artifact(
    project_root: Path | str, source_id: str
) -> dict[str, Any] | None:
    path = chunk_artifact_path_for(project_root, source_id)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as artifact_file:
            payload = json.load(artifact_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChunkingError(
            "chunk_artifact_read_failed", f"Could not read chunk artifact: {error}."
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHUNK_SCHEMA_VERSION
    ):
        raise ChunkingError(
            "chunk_artifact_schema_invalid",
            f"Chunk artifact at '{path}' has an unsupported schema.",
        )
    return payload


def chunk_artifact_is_current(
    artifact: object,
    source: SourceRecord,
    *,
    sentence_processor_name: str,
    sentence_processor_version: str,
    sentence_model: str,
    target_characters: int = DEFAULT_TARGET_CHARACTERS,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> bool:
    """Return whether a chunk artifact exactly matches its derivation settings."""

    if not isinstance(artifact, dict):
        return False
    chunker = artifact.get("chunker")
    if not isinstance(chunker, dict):
        return False
    return (
        artifact.get("schema_version") == CHUNK_SCHEMA_VERSION
        and artifact.get("source_id") == source.source_id
        and artifact.get("source_sha256") == source.sha256
        and artifact.get("status") in {"complete", "partial", "failed"}
        and chunker.get("name") == "corpusdock.anchor_sentence"
        and chunker.get("version") == __version__
        and chunker.get("sentence_processor") == sentence_processor_name
        and chunker.get("sentence_processor_version") == sentence_processor_version
        and chunker.get("sentence_model") == sentence_model
        and chunker.get("target_characters") == target_characters
        and chunker.get("max_characters") == max_characters
        and chunker.get("overlap_sentences") == overlap_sentences
    )


def sentence_processor_identity(
    name: str, *, model_name: str = DEFAULT_SENTENCE_MODEL
) -> tuple[str, str, str]:
    """Return requested splitter provenance without loading its inference model."""

    if name == "rule":
        return RuleSentenceProcessor.name, RuleSentenceProcessor.version, "none"
    if name == "sat":
        try:
            processor_version = package_version("wtpsplit-lite")
        except PackageNotFoundError:
            processor_version = "unknown"
        return SaTSentenceProcessor.name, processor_version, model_name
    raise ChunkingError(
        "sentence_processor_unknown", f"Unknown sentence processor '{name}'."
    )


def chunk_coverage_report(
    project_root: Path | str, sources: Iterable[SourceRecord]
) -> dict[str, Any]:
    source_records = tuple(sources)
    statuses = {"complete": 0, "partial": 0, "failed": 0, "pending": 0, "stale": 0}
    total_chunks = 0
    for source in source_records:
        artifact = load_chunk_artifact(project_root, source.source_id)
        if artifact is None:
            status = "pending"
        elif artifact.get("source_sha256") != source.sha256:
            status = "stale"
        else:
            status = artifact.get("status")
            if status not in {"complete", "partial", "failed"}:
                status = "failed"
            chunks = artifact.get("chunks", [])
            if isinstance(chunks, list):
                total_chunks += len(chunks)
        statuses[status] += 1
    return {
        "registered_sources": len(source_records),
        "artifacts": len(source_records) - statuses["pending"],
        "statuses": statuses,
        "chunks": total_chunks,
    }


def _read_anchor_spans(value: object, *, text_length: int) -> tuple[_AnchorSpan, ...]:
    if not isinstance(value, list):
        raise ChunkingError(
            "extraction_anchors_invalid", "Extraction anchors must be an array."
        )
    anchors: list[_AnchorSpan] = []
    for raw_anchor in value:
        if not isinstance(raw_anchor, dict):
            raise ChunkingError(
                "extraction_anchor_invalid", "Each extraction anchor must be an object."
            )
        anchor_id = _required_string(raw_anchor, "anchor_id")
        locator = raw_anchor.get("locator")
        if not isinstance(locator, dict):
            raise ChunkingError(
                "extraction_locator_invalid",
                f"Anchor '{anchor_id}' has no locator object.",
            )
        start = locator.get("start_offset")
        end = locator.get("end_offset")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end < start
            or end > text_length
        ):
            raise ChunkingError(
                "extraction_offsets_invalid",
                f"Anchor '{anchor_id}' has invalid offsets.",
            )
        anchors.append(
            _AnchorSpan(
                anchor_id=anchor_id, start=start, end=end, locator=dict(locator)
            )
        )
    return tuple(
        sorted(anchors, key=lambda anchor: (anchor.start, anchor.end, anchor.anchor_id))
    )


def _structural_groups(anchors: Sequence[_AnchorSpan]) -> tuple[_StructuralGroup, ...]:
    groups: list[_StructuralGroup] = []
    current: list[_AnchorSpan] = []
    current_key: tuple[object, ...] | None = None
    for anchor in anchors:
        if anchor.start == anchor.end:
            continue
        key = _structural_key(anchor)
        if current and key != current_key:
            groups.append(
                _StructuralGroup(
                    start=current[0].start, end=current[-1].end, anchors=tuple(current)
                )
            )
            current = []
        current.append(anchor)
        current_key = key
    if current:
        groups.append(
            _StructuralGroup(
                start=current[0].start, end=current[-1].end, anchors=tuple(current)
            )
        )
    return tuple(groups)


def _structural_key(anchor: _AnchorSpan) -> tuple[object, ...]:
    locator = anchor.locator
    locator_type = locator.get("locator_type")
    if locator_type == "pdf_page":
        return (locator_type, locator.get("page"))
    if locator_type == "epub_spine":
        return (locator_type, locator.get("spine_item"), locator.get("heading"))
    if locator_type in {"docx_paragraph", "mobi_section"}:
        return (locator_type, locator.get("heading"))
    return (locator_type,)


def _sentence_spans(
    group: _StructuralGroup, sentences: Sequence[str]
) -> tuple[_SentenceSpan, ...]:
    cursor = group.start
    spans: list[_SentenceSpan] = []
    for sentence in sentences:
        start = cursor
        cursor += len(sentence)
        if sentence.strip():
            spans.append(_SentenceSpan(start=start, end=cursor))
    if cursor != group.end:
        raise ChunkingError(
            "sentence_reconstruction_failed",
            "Sentence offsets did not reconstruct the complete structural group.",
        )
    return tuple(spans)


def _pack_sentences(
    sentences: Sequence[_SentenceSpan],
    *,
    target_characters: int,
    max_characters: int,
    overlap_sentences: int,
    text: str,
) -> tuple[tuple[_SentenceSpan, ...], ...]:
    expanded: list[_SentenceSpan] = []
    for sentence in sentences:
        expanded.extend(
            _split_oversized_sentence(
                sentence, max_characters=max_characters, text=text
            )
        )
    if not expanded:
        return ()

    chunks: list[tuple[_SentenceSpan, ...]] = []
    current: list[_SentenceSpan] = []
    for sentence in expanded:
        if current and current[-1].end - current[0].start >= target_characters:
            chunks.append(tuple(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
        if current and sentence.end - current[0].start > max_characters:
            if not chunks or tuple(current) != chunks[-1]:
                chunks.append(tuple(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            while current and sentence.end - current[0].start > max_characters:
                current.pop(0)
        current.append(sentence)
    if current and (not chunks or tuple(current) != chunks[-1]):
        chunks.append(tuple(current))
    return tuple(chunks)


def _split_oversized_sentence(
    sentence: _SentenceSpan,
    *,
    max_characters: int,
    text: str,
) -> tuple[_SentenceSpan, ...]:
    if sentence.end - sentence.start <= max_characters:
        return (sentence,)
    spans: list[_SentenceSpan] = []
    start = sentence.start
    while sentence.end - start > max_characters:
        hard_end = start + max_characters
        search_start = start + max_characters // 2
        boundary = max(
            text.rfind("\n", search_start, hard_end),
            text.rfind(" ", search_start, hard_end),
        )
        end = boundary + 1 if boundary >= search_start else hard_end
        spans.append(_SentenceSpan(start=start, end=end))
        start = end
    if start < sentence.end:
        spans.append(_SentenceSpan(start=start, end=sentence.end))
    return tuple(spans)


def _clamped_locator(anchor: _AnchorSpan, *, start: int, end: int) -> dict[str, Any]:
    locator = dict(anchor.locator)
    locator["start_offset"] = max(anchor.start, start)
    locator["end_offset"] = min(anchor.end, end)
    return locator


def _failed_chunk_artifact(
    source_id: str,
    source_sha256: str,
    sentence_processor: SentenceProcessor,
    target_characters: int,
    max_characters: int,
    overlap_sentences: int,
    warning: str,
    chunked_at: str,
) -> ChunkArtifact:
    return ChunkArtifact(
        source_id=source_id,
        source_sha256=source_sha256,
        chunked_at=chunked_at,
        status="failed",
        sentence_processor_name=sentence_processor.name,
        sentence_processor_version=sentence_processor.version,
        sentence_model=sentence_processor.model_name,
        target_characters=target_characters,
        max_characters=max_characters,
        overlap_sentences=overlap_sentences,
        chunks=(),
        warnings=(warning,),
    )


def _required_string(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ChunkingError(
            "chunk_input_invalid", f"Required chunk input field '{key}' is missing."
        )
    return result
