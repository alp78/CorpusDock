"""Local extraction of registered sources into durable, ignored derived artifacts.

Original documents remain the source of truth. This module only writes derived JSON
artifacts under ``.corpusdock/extracted`` and never uploads source contents.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
from typing import Any, Literal
from uuid import uuid4
import zipfile
from xml.etree import ElementTree as ET

from pypdf import PdfReader

from corpusdock import __version__
from corpusdock.contracts import CitationLocator
from corpusdock.manifest import (
    STATE_DIRECTORY_NAME,
    SourceRecord,
    hash_source_file,
    utc_now,
)
from corpusdock.mobi_huffcdic import HuffCdicDecoder, HuffCdicError


EXTRACTION_SCHEMA_VERSION = 2
EXTRACTED_DIRECTORY_NAME = "extracted"
TEXT_OFFSET_UNIT = "unicode_codepoint"
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 256 * 1024 * 1024
MAX_MOBI_TEXT_BYTES = 256 * 1024 * 1024
MOBI_HUFF_CDIC_COMPRESSION = 0x4448

ExtractionStatus = Literal["complete", "partial", "failed"]


class ExtractionError(Exception):
    """An expected, user-actionable extraction failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    """Exact derived text with a durable locator back to an immutable source."""

    anchor_id: str
    locator: CitationLocator
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "locator": self.locator.to_dict(),
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class ExtractionArtifact:
    """An auditable extraction result for one registered source revision."""

    source_id: str
    source_sha256: str
    source_format: str
    source_path: str
    extracted_at: str
    parser_name: str
    parser_version: str
    status: ExtractionStatus
    text: str
    anchors: tuple[SourceAnchor, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any]
    text_offset_unit: str = TEXT_OFFSET_UNIT
    schema_version: int = EXTRACTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
            "source_path": self.source_path,
            "extracted_at": self.extracted_at,
            "parser": {
                "name": self.parser_name,
                "version": self.parser_version,
            },
            "status": self.status,
            "text_offset_unit": self.text_offset_unit,
            "text": self.text,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }

    def summary(self) -> dict[str, Any]:
        """Return non-content metadata suitable for CLI status output."""

        summary: dict[str, Any] = {
            "source_id": self.source_id,
            "source_format": self.source_format,
            "status": self.status,
            "anchor_count": len(self.anchors),
            "text_characters": len(self.text),
            "warning_count": len(self.warnings),
            "parser": {
                "name": self.parser_name,
                "version": self.parser_version,
            },
        }
        if self.source_format == "pdf":
            summary.update(
                {
                    "unresolved_pdf_pages": len(self.metadata.get("empty_pages", [])),
                }
            )
        return summary


@dataclass(frozen=True, slots=True)
class _AnchorDraft:
    key: str
    text: str
    locator: CitationLocator


@dataclass(frozen=True, slots=True)
class _ParserResult:
    parser_name: str
    parser_version: str
    text: str
    anchors: tuple[SourceAnchor, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any]
    status: ExtractionStatus


@dataclass(frozen=True, slots=True)
class _HtmlBlock:
    text: str
    tag: str
    heading: str | None


def artifact_path_for(project_root: Path | str, source_id: str) -> Path:
    """Return the stable local path for a source's derived extraction artifact."""

    return (
        Path(project_root).expanduser().resolve()
        / STATE_DIRECTORY_NAME
        / EXTRACTED_DIRECTORY_NAME
        / f"{source_id}.json"
    )


def extract_source(
    source: SourceRecord,
    source_path: Path | str,
    *,
    now: Callable[[], str] = utc_now,
) -> ExtractionArtifact:
    """Extract one registered source, returning failure metadata instead of guessing.

    The source is re-hashed immediately before parsing. If it differs from the
    registered immutable bytes, no extraction text is trusted or retained.
    """

    parser_name, parser_version = _parser_identity(source.source_format)
    display_path = str(Path(source_path).expanduser())
    try:
        resolved_path = Path(source_path).expanduser().resolve(strict=True)
        if not resolved_path.is_file():
            raise ExtractionError(
                "source_not_file", "Registered source path is not a regular file."
            )
        digest, size_bytes = hash_source_file(resolved_path)
        if digest != source.sha256 or size_bytes != source.size_bytes:
            raise ExtractionError(
                "source_changed",
                "Current file bytes do not match the registered SHA-256 and size.",
            )

        extractor = _EXTRACTORS.get(source.source_format)
        if extractor is None:
            raise ExtractionError(
                "unsupported_format", f"No extractor for '{source.source_format}'."
            )
        result = extractor(source, resolved_path)
        return ExtractionArtifact(
            source_id=source.source_id,
            source_sha256=source.sha256,
            source_format=source.source_format,
            source_path=str(resolved_path),
            extracted_at=now(),
            parser_name=result.parser_name,
            parser_version=result.parser_version,
            status=result.status,
            text=result.text,
            anchors=result.anchors,
            warnings=result.warnings,
            metadata=result.metadata,
        )
    except ExtractionError as error:
        return _failed_artifact(
            source,
            display_path,
            parser_name,
            parser_version,
            error.code,
            str(error),
            now(),
        )
    except (
        OSError,
        ValueError,
        struct.error,
        zipfile.BadZipFile,
        ET.ParseError,
    ) as error:
        return _failed_artifact(
            source,
            display_path,
            parser_name,
            parser_version,
            "parser_error",
            str(error),
            now(),
        )
    except (
        Exception
    ) as error:  # Defensive: malformed local documents must not crash a batch.
        return _failed_artifact(
            source,
            display_path,
            parser_name,
            parser_version,
            "unexpected_parser_error",
            f"{type(error).__name__}: {error}",
            now(),
        )


def write_extraction_artifact(
    project_root: Path | str, artifact: ExtractionArtifact
) -> Path:
    """Atomically write a content-bearing derived artifact outside version control."""

    return _write_extraction_payload(
        project_root, artifact.to_dict(), artifact.source_id
    )


def repoint_extraction_artifact(
    project_root: Path | str,
    artifact: dict[str, Any],
    source: SourceRecord,
    source_path: Path | str,
) -> Path:
    """Update only path provenance when immutable source bytes moved locally."""

    if not extraction_artifact_is_current(artifact, source):
        raise ExtractionError(
            "artifact_stale",
            "Only a current extraction artifact can be repointed.",
        )
    resolved_path = Path(source_path).expanduser().resolve(strict=True)
    if str(resolved_path) not in source.original_paths:
        raise ExtractionError(
            "artifact_source_path_invalid",
            "The replacement extraction path is not registered for this source.",
        )
    payload = dict(artifact)
    payload["source_path"] = str(resolved_path)
    return _write_extraction_payload(project_root, payload, source.source_id)


def _write_extraction_payload(
    project_root: Path | str,
    payload: dict[str, Any],
    source_id: str,
) -> Path:
    """Atomically persist one already-validated extraction payload."""

    path = artifact_path_for(project_root, source_id)
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as artifact_file:
            json.dump(
                payload,
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
        raise ExtractionError(
            "artifact_write_failed", f"Could not write extraction artifact: {error}."
        ) from error
    return path


def load_extraction_artifact(
    project_root: Path | str, source_id: str
) -> dict[str, Any] | None:
    """Load an artifact's JSON for status inspection without exposing its text by default."""

    path = artifact_path_for(project_root, source_id)
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as artifact_file:
            payload = json.load(artifact_file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExtractionError(
            "artifact_read_failed", f"Could not read extraction artifact: {error}."
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != EXTRACTION_SCHEMA_VERSION
    ):
        raise ExtractionError(
            "artifact_schema_invalid",
            f"Extraction artifact at '{path}' has an unsupported schema.",
        )
    return payload


def extraction_artifact_is_current(artifact: object, source: SourceRecord) -> bool:
    """Return whether an extraction can be reused for immutable source bytes."""

    if not isinstance(artifact, dict):
        return False
    parser = artifact.get("parser")
    if not isinstance(parser, dict):
        return False
    parser_name, parser_version = _parser_identity(source.source_format)
    return (
        artifact.get("schema_version") == EXTRACTION_SCHEMA_VERSION
        and artifact.get("source_id") == source.source_id
        and artifact.get("source_sha256") == source.sha256
        and artifact.get("source_format") == source.source_format
        and parser.get("name") == parser_name
        and parser.get("version") == parser_version
        and artifact.get("status") in {"complete", "partial", "failed"}
    )


def extraction_coverage_report(
    project_root: Path | str,
    sources: Iterable[SourceRecord],
) -> dict[str, Any]:
    """Summarize local extraction coverage without loading document text into CLI output."""

    source_records = tuple(sources)
    statuses = {"complete": 0, "partial": 0, "failed": 0, "pending": 0, "stale": 0}
    by_format: dict[str, dict[str, int]] = {}
    total_anchors = 0
    total_text_characters = 0
    unresolved_pdf_pages = 0
    pdf_sources_with_unresolved_pages = 0

    for source in source_records:
        format_statuses = by_format.setdefault(
            source.source_format,
            {"complete": 0, "partial": 0, "failed": 0, "pending": 0, "stale": 0},
        )
        artifact = load_extraction_artifact(project_root, source.source_id)
        if artifact is None:
            status = "pending"
        elif artifact.get("source_sha256") != source.sha256:
            status = "stale"
        else:
            status = artifact.get("status")
            if status not in {"complete", "partial", "failed"}:
                status = "failed"
            total_anchors += len(artifact.get("anchors", []))
            text = artifact.get("text", "")
            if isinstance(text, str):
                total_text_characters += len(text)
            metadata = artifact.get("metadata", {})
            if isinstance(metadata, dict):
                empty_pages = metadata.get("empty_pages", [])
                if isinstance(empty_pages, list):
                    unresolved_pdf_pages += len(empty_pages)
                    if empty_pages:
                        pdf_sources_with_unresolved_pages += 1
        statuses[status] += 1
        format_statuses[status] += 1

    return {
        "registered_sources": len(source_records),
        "artifacts": len(source_records) - statuses["pending"],
        "statuses": statuses,
        "by_format": {
            source_format: by_format[source_format]
            for source_format in sorted(by_format)
        },
        "anchors": total_anchors,
        "text_characters": total_text_characters,
        "pdf_text_layers": {
            "unresolved_pdf_pages": unresolved_pdf_pages,
            "sources_with_unresolved_pages": pdf_sources_with_unresolved_pages,
        },
    }


def _failed_artifact(
    source: SourceRecord,
    source_path: str,
    parser_name: str,
    parser_version: str,
    code: str,
    message: str,
    extracted_at: str,
) -> ExtractionArtifact:
    return ExtractionArtifact(
        source_id=source.source_id,
        source_sha256=source.sha256,
        source_format=source.source_format,
        source_path=source_path,
        extracted_at=extracted_at,
        parser_name=parser_name,
        parser_version=parser_version,
        status="failed",
        text="",
        anchors=(),
        warnings=(f"{code}: {message}",),
        metadata={"failure_code": code},
    )


def _parser_identity(source_format: str) -> tuple[str, str]:
    if source_format == "pdf":
        import pypdf

        return "pypdf", pypdf.__version__
    return f"corpusdock.{source_format}", __version__


def _extract_txt(source: SourceRecord, path: Path) -> _ParserResult:
    raw_bytes = path.read_bytes()
    text, encoding, warnings = _decode_plain_text(raw_bytes)
    anchors: list[SourceAnchor] = []
    cursor = 0
    lines = text.splitlines(keepends=True)
    for line_number, line_text in enumerate(lines, start=1):
        start_offset = cursor
        cursor += len(line_text)
        locator = CitationLocator(
            source_id=source.source_id,
            locator_type="text_line",
            label=f"line {line_number}",
            line_start=line_number,
            line_end=line_number,
            start_offset=start_offset,
            end_offset=cursor,
        )
        anchors.append(
            SourceAnchor(
                anchor_id=f"{source.source_id}:text-line:{line_number:06d}",
                locator=locator,
                text=line_text,
            )
        )

    status: ExtractionStatus = "complete" if anchors else "partial"
    warning_list = list(warnings)
    if not anchors:
        warning_list.append("txt_empty: The source contains no text lines.")
    return _ParserResult(
        parser_name="corpusdock.txt",
        parser_version=__version__,
        text=text,
        anchors=tuple(anchors),
        warnings=tuple(warning_list),
        metadata={"encoding": encoding, "line_count": len(lines)},
        status=status,
    )


def _extract_pdf(
    source: SourceRecord,
    path: Path,
) -> _ParserResult:
    reader = PdfReader(path, strict=False)
    if reader.is_encrypted:
        raise ExtractionError(
            "pdf_encrypted",
            "Encrypted PDFs are not extracted without an explicit password workflow.",
        )

    try:
        page_labels = list(reader.page_labels)
    except Exception:
        page_labels = []

    warnings: list[str] = []
    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as error:
            page_text = ""
            warnings.append(
                f"pdf_page_{page_number}_error: {type(error).__name__}: {error}"
            )
        page_texts.append(page_text)

    native_text_pages = [
        page_index + 1
        for page_index, page_text in enumerate(page_texts)
        if page_text.strip()
    ]
    empty_pages = [
        page_index + 1
        for page_index, page_text in enumerate(page_texts)
        if not page_text.strip()
    ]

    drafts: list[_AnchorDraft] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        page_label = (
            page_labels[page_number - 1] if page_number <= len(page_labels) else None
        )
        label = f"p. {page_label}" if page_label else f"PDF p. {page_number}"
        extraction_method = "pdf_text_layer" if page_text.strip() else "no_text"
        drafts.append(
            _AnchorDraft(
                key=f"pdf-page:{page_number:06d}",
                text=page_text,
                locator=CitationLocator(
                    source_id=source.source_id,
                    locator_type="pdf_page",
                    label=label,
                    page=page_number,
                    page_label=str(page_label) if page_label is not None else None,
                    extraction_method=extraction_method,
                ),
            )
        )

    text, anchors = _assemble_anchors(source, drafts, separator="\n\f\n")
    if empty_pages:
        warnings.append(
            "pdf_no_text_pages: "
            + ", ".join(str(page_number) for page_number in empty_pages)
            + ". No embedded text was found; text-only extraction skipped these pages."
        )
    status: ExtractionStatus = "complete" if anchors and not empty_pages else "partial"
    if not anchors:
        warnings.append("pdf_empty: No PDF pages were available for extraction.")
    parser_name = "pypdf"
    parser_version = _parser_identity("pdf")[1]
    return _ParserResult(
        parser_name=parser_name,
        parser_version=parser_version,
        text=text,
        anchors=anchors,
        warnings=tuple(warnings),
        metadata={
            "page_count": len(reader.pages),
            "pages_with_text": len(page_texts) - len(empty_pages),
            "native_text_pages": native_text_pages,
            "empty_pages": empty_pages,
        },
        status=status,
    )


def _extract_epub(source: SourceRecord, path: Path) -> _ParserResult:
    with _safe_zip_file(path) as archive:
        rootfile = _epub_rootfile(archive)
        package = ET.fromstring(_read_archive_entry(archive, rootfile))
        package_directory = str(PurePosixPath(rootfile).parent)
        manifest_items = _epub_manifest_items(package)
        spine = _epub_spine(package)
        title = _first_element_text(package, "title")
        encryption_metadata = "META-INF/encryption.xml" in archive.namelist()

        drafts: list[_AnchorDraft] = []
        warnings: list[str] = []
        processed_items = 0
        empty_items = 0
        coverage_complete = True
        for spine_index, (item_id, linear) in enumerate(spine, start=1):
            item = manifest_items.get(item_id)
            if item is None:
                warnings.append(
                    f"epub_missing_manifest_item: spine item '{item_id}' was skipped."
                )
                coverage_complete = False
                continue
            href, media_type = item
            if not _is_html_spine_item(href, media_type):
                warnings.append(f"epub_non_html_spine_item: '{href}' was skipped.")
                coverage_complete = False
                continue
            item_path = _resolve_archive_path(package_directory, href)
            try:
                item_bytes = _read_archive_entry(archive, item_path)
            except ExtractionError as error:
                warnings.append(f"{error.code}: {error}")
                coverage_complete = False
                continue
            blocks, decode_warnings = _parse_html_blocks(item_bytes)
            warnings.extend(f"epub_{item_id}: {warning}" for warning in decode_warnings)
            if decode_warnings:
                coverage_complete = False
            if not blocks:
                warnings.append(
                    f"epub_empty_spine_item: '{href}' contains no extractable text blocks."
                )
                empty_items += 1
                continue
            processed_items += 1
            spine_item = f"{spine_index}:{item_id}:{href}"
            for block_index, block in enumerate(blocks, start=1):
                heading = block.heading
                label = heading or f"spine item {spine_index}, block {block_index}"
                drafts.append(
                    _AnchorDraft(
                        key=f"epub-spine:{spine_index:04d}:block:{block_index:06d}",
                        text=block.text,
                        locator=CitationLocator(
                            source_id=source.source_id,
                            locator_type="epub_spine",
                            label=label,
                            chapter=heading,
                            heading=heading,
                            spine_item=spine_item,
                        ),
                    )
                )
            if not linear:
                warnings.append(
                    f"epub_non_linear_spine_item: '{href}' was included but marked non-linear."
                )

    text, anchors = _assemble_anchors(source, drafts)
    if encryption_metadata:
        warnings.append(
            "epub_encryption_metadata_present: Readable content was extracted without decryption; "
            "unreadable encrypted content is not bypassed."
        )
    status: ExtractionStatus = (
        "complete" if anchors and coverage_complete else "partial"
    )
    if not anchors:
        warnings.append("epub_empty: No extractable EPUB spine content was found.")
    return _ParserResult(
        parser_name="corpusdock.epub",
        parser_version=__version__,
        text=text,
        anchors=anchors,
        warnings=tuple(warnings),
        metadata={
            "title": title,
            "package_path": rootfile,
            "spine_item_count": len(spine),
            "processed_spine_item_count": processed_items,
            "empty_spine_item_count": empty_items,
            "encryption_metadata_present": encryption_metadata,
        },
        status=status,
    )


def _extract_docx(source: SourceRecord, path: Path) -> _ParserResult:
    with _safe_zip_file(path) as archive:
        document = ET.fromstring(_read_archive_entry(archive, "word/document.xml"))

    drafts: list[_AnchorDraft] = []
    warnings: list[str] = []
    heading_levels: list[str | None] = [None] * 9
    paragraphs = list(document.iter(_word_tag("p")))
    for ordinal, paragraph in enumerate(paragraphs, start=1):
        text = _docx_paragraph_text(paragraph)
        if not text:
            continue
        paragraph_id = paragraph.attrib.get(_word14_tag("paraId")) or f"p-{ordinal:06d}"
        heading_level = _docx_heading_level(paragraph)
        if heading_level is not None:
            heading_levels[heading_level - 1] = text
            for index in range(heading_level, len(heading_levels)):
                heading_levels[index] = None
        heading_path = " > ".join(item for item in heading_levels if item)
        label = f"paragraph {paragraph_id}"
        if heading_path:
            label = f"{heading_path}, {label}"
        drafts.append(
            _AnchorDraft(
                key=f"docx-paragraph:{paragraph_id}",
                text=text,
                locator=CitationLocator(
                    source_id=source.source_id,
                    locator_type="docx_paragraph",
                    label=label,
                    heading=heading_path or None,
                    paragraph_id=paragraph_id,
                ),
            )
        )

    text, anchors = _assemble_anchors(source, drafts)
    if not anchors:
        warnings.append("docx_empty: No non-empty paragraphs were found.")
    return _ParserResult(
        parser_name="corpusdock.docx",
        parser_version=__version__,
        text=text,
        anchors=anchors,
        warnings=tuple(warnings),
        metadata={
            "paragraph_count": len(paragraphs),
            "anchored_paragraph_count": len(anchors),
        },
        status="complete" if anchors else "partial",
    )


def _extract_mobi(source: SourceRecord, path: Path) -> _ParserResult:
    records = _mobi_records(path.read_bytes())
    palm_doc = records[0]
    if len(palm_doc) < 16:
        raise ExtractionError(
            "mobi_header_invalid", "MOBI PalmDOC header is truncated."
        )
    compression, _, text_length, text_record_count, _, encryption_type = struct.unpack(
        ">HHIHHH", palm_doc[:14]
    )
    if encryption_type != 0:
        raise ExtractionError(
            "mobi_encrypted",
            "MOBI encryption is present. CorpusDock will not decrypt or bypass DRM.",
        )
    if palm_doc[16:20] != b"MOBI":
        raise ExtractionError(
            "mobi_header_invalid", "The PalmDOC record does not contain a MOBI header."
        )
    encoding_code, mobi_version, drm_count, trailer_count, multibyte_trailer = (
        _mobi_header_fields(palm_doc)
    )
    if drm_count:
        raise ExtractionError(
            "mobi_drm_present",
            "MOBI DRM metadata is present. CorpusDock will not decrypt or bypass DRM.",
        )
    if 1 + text_record_count > len(records):
        raise ExtractionError(
            "mobi_records_invalid",
            "MOBI text-record count exceeds the file record table.",
        )
    if text_length > MAX_MOBI_TEXT_BYTES:
        raise ExtractionError(
            "mobi_text_too_large",
            "Declared MOBI text exceeds the safe local extraction limit.",
        )

    text_records = tuple(
        _trim_mobi_record(
            record, trailer_count=trailer_count, multibyte_trailer=multibyte_trailer
        )
        for record in records[1 : 1 + text_record_count]
    )
    if compression == 1:
        raw_text = b"".join(text_records)
        extraction_method = "native_uncompressed_extraction"
    elif compression == 2:
        # PalmDOC compression state resets at each PalmDB text-record boundary.
        raw_text = b"".join(_decompress_palmdoc(record) for record in text_records)
        extraction_method = "native_palmdoc_extraction"
    elif compression == MOBI_HUFF_CDIC_COMPRESSION:
        huff_record, cdic_records = _mobi_huff_dictionary_records(
            palm_doc,
            records,
            text_record_count=text_record_count,
        )
        try:
            raw_text = HuffCdicDecoder(
                huff_record,
                cdic_records,
                max_output_bytes=max(text_length, 1),
            ).decode_records(text_records)
        except HuffCdicError as error:
            raise ExtractionError(
                "mobi_compression_invalid", f"Invalid MOBI HUFF/CDIC data: {error}"
            ) from error
        extraction_method = "native_huffcdic_extraction"
    else:
        raise ExtractionError(
            "mobi_compression_unsupported",
            f"MOBI compression {compression} is not supported by the bundled parser. "
            "No external converter or DRM bypass was attempted.",
        )

    if len(raw_text) > MAX_MOBI_TEXT_BYTES:
        raise ExtractionError(
            "mobi_text_too_large", "Decoded MOBI text exceeds the safe local limit."
        )

    text, encoding, decode_warnings = _decode_mobi_text(raw_text, encoding_code)
    blocks, html_warnings = _parse_html_blocks(text.encode("utf-8"))
    if not blocks:
        blocks = tuple(
            _HtmlBlock(text=line, tag="p", heading=None)
            for line in text.splitlines()
            if line.strip()
        )
    drafts: list[_AnchorDraft] = []
    for section_number, block in enumerate(blocks, start=1):
        heading = block.heading
        label = heading or f"MOBI section {section_number}"
        drafts.append(
            _AnchorDraft(
                key=f"mobi-section:{section_number:06d}",
                text=block.text,
                locator=CitationLocator(
                    source_id=source.source_id,
                    locator_type="mobi_section",
                    label=label,
                    chapter=heading,
                    heading=heading,
                ),
            )
        )
    assembled_text, anchors = _assemble_anchors(source, drafts)
    warnings = list(decode_warnings)
    warnings.extend(f"mobi_html: {warning}" for warning in html_warnings)
    if len(raw_text) != text_length:
        warnings.append(
            f"mobi_text_length_mismatch: Header reports {text_length} bytes; decoded records contain {len(raw_text)} bytes."
        )
    if not anchors:
        warnings.append("mobi_empty: No extractable text sections were found.")
    return _ParserResult(
        parser_name="corpusdock.mobi",
        parser_version=__version__,
        text=assembled_text,
        anchors=anchors,
        warnings=tuple(warnings),
        metadata={
            "record_count": len(records),
            "text_record_count": text_record_count,
            "compression": compression,
            "mobi_version": mobi_version,
            "encoding": encoding,
            "trailing_data_entries": trailer_count,
            "multibyte_trailer": multibyte_trailer,
            "conversion": {"method": extraction_method, "converted": False},
        },
        status="complete" if anchors and not warnings else "partial",
    )


def _assemble_anchors(
    source: SourceRecord,
    drafts: Iterable[_AnchorDraft],
    *,
    separator: str = "\n\n",
) -> tuple[str, tuple[SourceAnchor, ...]]:
    text_parts: list[str] = []
    anchors: list[SourceAnchor] = []
    offset = 0
    for draft in drafts:
        if draft.text:
            if text_parts:
                text_parts.append(separator)
                offset += len(separator)
            start_offset = offset
            text_parts.append(draft.text)
            offset += len(draft.text)
            end_offset = offset
        else:
            start_offset = offset
            end_offset = offset
        locator = replace(
            draft.locator,
            source_id=source.source_id,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        anchors.append(
            SourceAnchor(
                anchor_id=f"{source.source_id}:{draft.key}",
                locator=locator,
                text=draft.text,
            )
        )
    return "".join(text_parts), tuple(anchors)


def _decode_plain_text(raw_bytes: bytes) -> tuple[str, str, tuple[str, ...]]:
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return raw_bytes.decode("utf-8-sig"), "utf-8-sig", ()
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw_bytes.decode("utf-16"), "utf-16", ()
    if raw_bytes.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return raw_bytes.decode("utf-32"), "utf-32", ()
    try:
        return raw_bytes.decode("utf-8"), "utf-8", ()
    except UnicodeDecodeError:
        return (
            raw_bytes.decode("cp1252"),
            "cp1252",
            (
                "txt_encoding_fallback: UTF-8 decoding failed; CP-1252 was used and should be reviewed.",
            ),
        )


def _decode_mobi_text(
    raw_bytes: bytes, encoding_code: int
) -> tuple[str, str, tuple[str, ...]]:
    encodings = {65001: "utf-8", 1252: "cp1252", 1200: "utf-16", 65000: "utf-7"}
    encoding = encodings.get(encoding_code, "cp1252")
    try:
        return raw_bytes.decode(encoding), encoding, ()
    except UnicodeDecodeError:
        return (
            raw_bytes.decode(encoding, errors="replace"),
            encoding,
            (
                "mobi_decode_replacement: Invalid encoded bytes were replaced and should be visually verified.",
            ),
        )


def _safe_zip_file(path: Path) -> zipfile.ZipFile:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ExtractionError(
            "archive_invalid", f"Could not open ZIP-based document: {error}."
        ) from error
    try:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise ExtractionError(
                "archive_member_limit",
                "Archive has too many members to extract safely.",
            )
        total_bytes = 0
        for info in infos:
            member_path = PurePosixPath(info.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ExtractionError(
                    "archive_path_invalid", "Archive contains an unsafe member path."
                )
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ExtractionError(
                    "archive_member_limit", "Archive contains an oversized member."
                )
            total_bytes += info.file_size
            if total_bytes > MAX_ARCHIVE_TOTAL_BYTES:
                raise ExtractionError(
                    "archive_size_limit",
                    "Archive expands beyond the safe extraction limit.",
                )
            if info.flag_bits & 0x1:
                raise ExtractionError(
                    "archive_encrypted",
                    "Archive encryption is present; CorpusDock will not decrypt or bypass it.",
                )
    except Exception:
        archive.close()
        raise
    return archive


def _read_archive_entry(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except KeyError as error:
        raise ExtractionError(
            "archive_member_missing", f"Required archive member '{name}' is missing."
        ) from error
    except RuntimeError as error:
        raise ExtractionError(
            "archive_member_encrypted",
            f"Could not read archive member '{name}': {error}.",
        ) from error


def _epub_rootfile(archive: zipfile.ZipFile) -> str:
    container = ET.fromstring(_read_archive_entry(archive, "META-INF/container.xml"))
    for element in container.iter():
        if _local_name(element.tag) == "rootfile":
            rootfile = element.attrib.get("full-path")
            if rootfile:
                return _resolve_archive_path("", rootfile)
    raise ExtractionError(
        "epub_rootfile_missing",
        "EPUB container.xml does not declare a package rootfile.",
    )


def _epub_manifest_items(package: ET.Element) -> dict[str, tuple[str, str]]:
    items: dict[str, tuple[str, str]] = {}
    for element in package.iter():
        if _local_name(element.tag) != "item":
            continue
        item_id = element.attrib.get("id")
        href = element.attrib.get("href")
        media_type = element.attrib.get("media-type", "")
        if item_id and href:
            items[item_id] = (href, media_type)
    return items


def _epub_spine(package: ET.Element) -> tuple[tuple[str, bool], ...]:
    result: list[tuple[str, bool]] = []
    for element in package.iter():
        if _local_name(element.tag) != "itemref":
            continue
        item_id = element.attrib.get("idref")
        if item_id:
            result.append(
                (item_id, element.attrib.get("linear", "yes").lower() != "no")
            )
    if not result:
        raise ExtractionError(
            "epub_spine_missing", "EPUB package does not contain a spine."
        )
    return tuple(result)


def _first_element_text(element: ET.Element, name: str) -> str | None:
    for child in element.iter():
        if _local_name(child.tag) == name and child.text and child.text.strip():
            return child.text.strip()
    return None


def _is_html_spine_item(href: str, media_type: str) -> bool:
    return media_type in {
        "application/xhtml+xml",
        "text/html",
    } or href.lower().endswith((".xhtml", ".html", ".htm"))


def _resolve_archive_path(base_directory: str, href: str) -> str:
    raw_path = href.split("#", maxsplit=1)[0]
    components: list[str] = []
    for component in (
        *PurePosixPath(base_directory).parts,
        *PurePosixPath(raw_path).parts,
    ):
        if component in {"", "."}:
            continue
        if component == "..":
            if not components:
                raise ExtractionError(
                    "archive_path_invalid",
                    f"Archive reference '{href}' escapes its root.",
                )
            components.pop()
            continue
        components.append(component)
    if not components:
        raise ExtractionError(
            "archive_path_invalid", f"Archive reference '{href}' is empty."
        )
    return "/".join(components)


def _parse_html_blocks(
    payload: bytes,
) -> tuple[tuple[_HtmlBlock, ...], tuple[str, ...]]:
    text, encoding, warnings = _decode_markup(payload)
    parser = _HtmlBlockParser()
    parser.feed(text)
    parser.close()
    result_warnings = list(warnings)
    if encoding != "utf-8":
        result_warnings.append(f"markup_encoding: Decoded markup as {encoding}.")
    return tuple(parser.blocks), tuple(result_warnings)


def _decode_markup(payload: bytes) -> tuple[str, str, tuple[str, ...]]:
    for encoding in ("utf-8", "utf-16"):
        try:
            return payload.decode(encoding), encoding, ()
        except UnicodeDecodeError:
            continue
    return (
        payload.decode("cp1252", errors="replace"),
        "cp1252",
        ("markup_decode_replacement: Markup bytes were decoded with replacements.",),
    )


class _HtmlBlockParser(HTMLParser):
    _BLOCK_TAGS = {
        "p",
        "li",
        "blockquote",
        "pre",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "dt",
        "dd",
        "td",
        "th",
    }
    _SKIP_TAGS = {"head", "script", "style", "title", "nav"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[_HtmlBlock] = []
        self._open_tags: list[str] = []
        self._body_depth = 0
        self._seen_body = False
        self._skip_depth = 0
        self._active_tag: str | None = None
        self._active_depth: int | None = None
        self._active_chunks: list[str] = []
        self._loose_chunks: list[str] = []
        self._heading: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized_tag = tag.lower()
        self._open_tags.append(normalized_tag)
        if normalized_tag == "body":
            self._seen_body = True
            self._body_depth += 1
        if normalized_tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if not self._in_content():
            return
        if normalized_tag == "br":
            self._append_text("\n")
        elif normalized_tag in self._BLOCK_TAGS and self._active_tag is None:
            self._active_tag = normalized_tag
            self._active_depth = len(self._open_tags)
            self._active_chunks = []

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()
        if self._active_tag == normalized_tag and self._active_depth == len(
            self._open_tags
        ):
            self._finish_active_block()
        if normalized_tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if normalized_tag == "body" and self._body_depth:
            self._body_depth -= 1
        if self._open_tags:
            self._open_tags.pop()

    def handle_data(self, data: str) -> None:
        if self._in_content():
            self._append_text(data)

    def close(self) -> None:
        super().close()
        if self._active_tag is not None:
            self._finish_active_block()
        loose_text = _clean_html_text("".join(self._loose_chunks), tag="p")
        if loose_text:
            self.blocks.append(
                _HtmlBlock(text=loose_text, tag="p", heading=self._heading)
            )

    def _in_content(self) -> bool:
        return self._skip_depth == 0 and (not self._seen_body or self._body_depth > 0)

    def _append_text(self, text: str) -> None:
        if self._active_tag is None:
            self._loose_chunks.append(text)
        else:
            self._active_chunks.append(text)

    def _finish_active_block(self) -> None:
        assert self._active_tag is not None
        text = _clean_html_text("".join(self._active_chunks), tag=self._active_tag)
        if text:
            heading = self._heading
            if self._active_tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                heading = text
                self._heading = text
            self.blocks.append(
                _HtmlBlock(text=text, tag=self._active_tag, heading=heading)
            )
        self._active_tag = None
        self._active_depth = None
        self._active_chunks = []


def _clean_html_text(value: str, *, tag: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if tag == "pre":
        return normalized.strip("\n")
    return " ".join(normalized.split())


def _word_tag(name: str) -> str:
    return f"{{http://schemas.openxmlformats.org/wordprocessingml/2006/main}}{name}"


def _word14_tag(name: str) -> str:
    return f"{{http://schemas.microsoft.com/office/word/2010/wordml}}{name}"


def _docx_paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for element in paragraph.iter():
        if element.tag == _word_tag("t"):
            parts.append(element.text or "")
        elif element.tag == _word_tag("tab"):
            parts.append("\t")
        elif element.tag in {_word_tag("br"), _word_tag("cr")}:
            parts.append("\n")
    return "".join(parts).strip()


def _docx_heading_level(paragraph: ET.Element) -> int | None:
    style = paragraph.find(f"./{_word_tag('pPr')}/{_word_tag('pStyle')}")
    if style is not None:
        style_name = style.attrib.get(_word_tag("val"), "")
        match = re.search(r"heading\s*([1-9])", style_name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    outline = paragraph.find(f"./{_word_tag('pPr')}/{_word_tag('outlineLvl')}")
    if outline is not None:
        value = outline.attrib.get(_word_tag("val"))
        if value is not None and value.isdigit() and int(value) < 9:
            return int(value) + 1
    return None


def _mobi_records(payload: bytes) -> tuple[bytes, ...]:
    if len(payload) < 78 or payload[60:68] != b"BOOKMOBI":
        raise ExtractionError(
            "mobi_header_invalid", "File is not a PalmDB MOBI container."
        )
    record_count = struct.unpack(">H", payload[76:78])[0]
    table_end = 78 + record_count * 8
    if record_count < 2 or table_end > len(payload):
        raise ExtractionError("mobi_records_invalid", "MOBI record table is invalid.")
    offsets = [
        struct.unpack(">I", payload[78 + index * 8 : 82 + index * 8])[0]
        for index in range(record_count)
    ]
    if any(
        offset < table_end or offset > len(payload) for offset in offsets
    ) or offsets != sorted(offsets):
        raise ExtractionError(
            "mobi_records_invalid", "MOBI record offsets are invalid."
        )
    return tuple(
        payload[
            offset : offsets[index + 1] if index + 1 < len(offsets) else len(payload)
        ]
        for index, offset in enumerate(offsets)
    )


def _mobi_header_fields(palm_doc: bytes) -> tuple[int, int, int, int, bool]:
    mobi_start = 16
    if len(palm_doc) < mobi_start + 24:
        raise ExtractionError("mobi_header_invalid", "MOBI header is truncated.")
    header_length = struct.unpack(">I", palm_doc[mobi_start + 4 : mobi_start + 8])[0]
    if header_length < 24 or len(palm_doc) < mobi_start + header_length:
        raise ExtractionError("mobi_header_invalid", "MOBI header length is invalid.")
    encoding_code = struct.unpack(">I", palm_doc[mobi_start + 12 : mobi_start + 16])[0]
    mobi_version = struct.unpack(">I", palm_doc[mobi_start + 20 : mobi_start + 24])[0]
    drm_count = 0
    if header_length >= 0xB0:
        drm_count = struct.unpack(
            ">I", palm_doc[mobi_start + 0xAC : mobi_start + 0xB0]
        )[0]
    extra_data_flags = 0
    if len(palm_doc) >= 0xF4:
        extra_data_flags = struct.unpack(">H", palm_doc[0xF2:0xF4])[0]
    multibyte_trailer = bool(extra_data_flags & 1)
    trailer_count = 0
    while extra_data_flags > 1:
        if extra_data_flags & 2:
            trailer_count += 1
        extra_data_flags >>= 1
    return encoding_code, mobi_version, drm_count, trailer_count, multibyte_trailer


def _mobi_huff_dictionary_records(
    palm_doc: bytes,
    records: tuple[bytes, ...],
    *,
    text_record_count: int,
) -> tuple[bytes, tuple[bytes, ...]]:
    """Resolve the bundled HUFF table and CDIC dictionaries from MOBI records."""

    mobi_start = 16
    if len(palm_doc) < mobi_start + 8:
        raise ExtractionError("mobi_header_invalid", "MOBI header is truncated.")
    header_length = struct.unpack_from(">I", palm_doc, mobi_start + 4)[0]
    huff_fields_end = 0x68
    if header_length < huff_fields_end or len(palm_doc) < mobi_start + huff_fields_end:
        raise ExtractionError(
            "mobi_huff_dictionary_invalid",
            "MOBI HUFF/CDIC dictionary fields are missing from the header.",
        )
    huff_record_index, huff_record_count = struct.unpack_from(">II", palm_doc, 0x70)
    if huff_record_count < 2:
        raise ExtractionError(
            "mobi_huff_dictionary_invalid",
            "MOBI HUFF/CDIC data must contain a HUFF record and at least one CDIC record.",
        )
    huff_end = huff_record_index + huff_record_count
    if (
        huff_record_index <= text_record_count
        or huff_record_index >= len(records)
        or huff_end > len(records)
    ):
        raise ExtractionError(
            "mobi_huff_dictionary_invalid",
            "MOBI HUFF/CDIC record range lies outside the container.",
        )
    return records[huff_record_index], records[huff_record_index + 1 : huff_end]


def _trim_mobi_record(
    payload: bytes,
    *,
    trailer_count: int,
    multibyte_trailer: bool,
) -> bytes:
    """Remove declared PalmDOC record trailers before local decompression."""

    result = payload
    for _ in range(trailer_count):
        trailer_size = _mobi_trailer_size(result)
        if trailer_size <= 0 or trailer_size > len(result):
            raise ExtractionError(
                "mobi_trailer_invalid", "MOBI trailing-data entry has an invalid size."
            )
        result = result[:-trailer_size]
    if multibyte_trailer:
        if not result:
            raise ExtractionError(
                "mobi_trailer_invalid", "MOBI multibyte trailer is missing."
            )
        trailer_size = (result[-1] & 0x03) + 1
        if trailer_size > len(result):
            raise ExtractionError(
                "mobi_trailer_invalid", "MOBI multibyte trailer has an invalid size."
            )
        result = result[:-trailer_size]
    return result


def _mobi_trailer_size(payload: bytes) -> int:
    size = 0
    for value in payload[-4:]:
        if value & 0x80:
            size = 0
        size = (size << 7) | (value & 0x7F)
    return size


def _decompress_palmdoc(payload: bytes) -> bytes:
    result = bytearray()
    cursor = 0
    while cursor < len(payload):
        value = payload[cursor]
        cursor += 1
        if 1 <= value <= 8:
            if cursor + value > len(payload):
                raise ExtractionError(
                    "mobi_compression_invalid", "PalmDOC literal run is truncated."
                )
            result.extend(payload[cursor : cursor + value])
            cursor += value
        elif value == 0 or value <= 0x7F:
            result.append(value)
        elif value >= 0xC0:
            result.extend((0x20, value ^ 0x80))
        else:
            if cursor >= len(payload):
                raise ExtractionError(
                    "mobi_compression_invalid", "PalmDOC back-reference is truncated."
                )
            distance_and_length = ((value & 0x3F) << 8) | payload[cursor]
            cursor += 1
            distance = distance_and_length >> 3
            length = (distance_and_length & 0x07) + 3
            if distance <= 0 or distance > len(result):
                raise ExtractionError(
                    "mobi_compression_invalid",
                    "PalmDOC back-reference points outside its record.",
                )
            if distance > length:
                result.extend(result[-distance : length - distance])
            else:
                for _ in range(length):
                    result.append(result[-distance])
    return bytes(result)


_EXTRACTORS = {
    "txt": _extract_txt,
    "pdf": _extract_pdf,
    "epub": _extract_epub,
    "docx": _extract_docx,
    "mobi": _extract_mobi,
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]
