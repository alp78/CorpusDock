from __future__ import annotations

import json
import os
from pathlib import Path
import re

import pytest

from corpusdock.chunking import (
    RuleSentenceProcessor,
    chunk_artifact_path_for,
    chunk_extraction_artifact,
    write_chunk_artifact,
)
from corpusdock.contracts import CitationLocator
from corpusdock.extraction import (
    ExtractionArtifact,
    SourceAnchor,
    write_extraction_artifact,
)
from corpusdock.manifest import ManifestStore
from corpusdock.retrieval import (
    RetrievalError,
    SQLiteSearchBackend,
    build_search_index,
    index_path_for,
)


TIMESTAMP = "2026-08-10T12:00:00Z"


def _add_derived_source(
    project_root: Path,
    documents: Path,
    *,
    source_format: str,
    text: str = "Uniquely retrievable local evidence.",
    partial: bool = False,
):  # type: ignore[no-untyped-def]
    source_path = documents / f"fixture-{source_format}.{source_format}"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(f"immutable-{source_format}-source".encode())
    source = (
        ManifestStore(project_root, now=lambda: TIMESTAMP)
        .register([source_path])[0]
        .source
    )
    locator_values: dict[str, object] = {
        "source_id": source.source_id,
        "locator_type": {
            "txt": "text_line",
            "pdf": "pdf_page",
            "epub": "epub_spine",
            "docx": "docx_paragraph",
            "mobi": "mobi_section",
        }[source_format],
        "label": {
            "txt": "line 1",
            "pdf": "PDF p. 1",
            "epub": "Chapter One",
            "docx": "paragraph B2",
            "mobi": "MOBI chapter",
        }[source_format],
        "start_offset": 0,
        "end_offset": len(text),
    }
    if source_format == "txt":
        locator_values.update(line_start=1, line_end=1)
    elif source_format == "pdf":
        locator_values.update(page=1, extraction_method="pdf_text_layer")
    elif source_format == "epub":
        locator_values.update(
            chapter="Chapter One", heading="Chapter One", spine_item="1:chapter.xhtml"
        )
    elif source_format == "docx":
        locator_values.update(heading="Chapter One", paragraph_id="B2")
    elif source_format == "mobi":
        locator_values.update(chapter="MOBI chapter", heading="MOBI chapter")
    locator = CitationLocator(**locator_values)  # type: ignore[arg-type]
    anchors = [
        SourceAnchor(
            anchor_id=f"{source.source_id}:anchor:000001", locator=locator, text=text
        )
    ]
    metadata: dict[str, object] = {"title": f"Stored {source_format.upper()} title"}
    warnings: tuple[str, ...] = ()
    if source_format == "pdf" and partial:
        anchors.append(
            SourceAnchor(
                anchor_id=f"{source.source_id}:anchor:000002",
                locator=CitationLocator(
                    source_id=source.source_id,
                    locator_type="pdf_page",
                    label="PDF p. 2",
                    page=2,
                    start_offset=len(text),
                    end_offset=len(text),
                    extraction_method="no_text",
                ),
                text="",
            )
        )
        metadata["empty_pages"] = [2]
        warnings = ("pdf_no_text_pages: 2. No embedded text was found.",)
    artifact = ExtractionArtifact(
        source_id=source.source_id,
        source_sha256=source.sha256,
        source_format=source.source_format,
        source_path=str(source_path.resolve()),
        extracted_at=TIMESTAMP,
        parser_name=f"fixture.{source_format}",
        parser_version="1",
        status="partial" if partial else "complete",
        text=text,
        anchors=tuple(anchors),
        warnings=warnings,
        metadata=metadata,
    )
    write_extraction_artifact(project_root, artifact)
    chunk_artifact = chunk_extraction_artifact(
        artifact.to_dict(),
        RuleSentenceProcessor(),
        target_characters=100,
        max_characters=200,
        overlap_sentences=0,
        now=lambda: TIMESTAMP,
    )
    write_chunk_artifact(project_root, chunk_artifact)
    return source_path, source, chunk_artifact.chunks[0]


@pytest.mark.parametrize("source_format", ["txt", "pdf", "epub", "docx", "mobi"])
def test_search_and_verify_preserve_every_format_locator(
    tmp_path: Path, source_format: str
) -> None:
    project_root = tmp_path / "corpus"
    source_path, source, chunk = _add_derived_source(
        project_root,
        tmp_path / "documents",
        source_format=source_format,
        partial=source_format == "pdf",
    )

    summary = build_search_index(project_root, now=lambda: TIMESTAMP)
    response = SQLiteSearchBackend(project_root).search(
        "uniquely retrievable", source_id=source.source_id
    )

    assert summary.sources == 1
    assert summary.chunks == 1
    assert index_path_for(project_root).is_file()
    assert len(response.results) == 1
    result = response.results[0]
    assert result.excerpt == chunk.text
    assert result.chunk_id == chunk.chunk_id
    assert result.anchor_ids == chunk.anchor_ids
    assert (
        result.locators[0].locator_type
        == {
            "txt": "text_line",
            "pdf": "pdf_page",
            "epub": "epub_spine",
            "docx": "docx_paragraph",
            "mobi": "mobi_section",
        }[source_format]
    )
    assert source.source_id in result.citation
    assert result.verification_status == "artifact-anchor-confirmed"
    assert result.source_coverage is not None
    if source_format == "pdf":
        assert result.source_coverage.extraction_status == "partial"
        assert result.source_coverage.unresolved_pdf_pages == (2,)

    report = SQLiteSearchBackend(project_root).verify(result.evidence_id)
    assert report.evidence.excerpt == chunk.text
    assert report.evidence.source_path == str(source_path.resolve())
    assert report.evidence.verification_status == "source-anchor-confirmed"
    assert "source-sha256-confirmed" in report.checks


def test_all_any_and_phrase_matching_are_literal_and_predictable(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "corpus"
    _add_derived_source(
        project_root,
        tmp_path / "documents",
        source_format="txt",
        text="Alpha beta appears here. Gamma appears later.",
    )
    build_search_index(project_root)
    backend = SQLiteSearchBackend(project_root)

    assert len(backend.search("beta gamma", match_mode="all").results) == 1
    assert len(backend.search("missing gamma", match_mode="any").results) == 1
    assert backend.search("beta gamma", match_mode="phrase").results == ()
    assert len(backend.search('"alpha beta"', match_mode="all").results) == 1


def test_corpus_snapshot_preserves_exact_evidence_for_derived_retrievers(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "corpus"
    _, source, chunk = _add_derived_source(
        project_root,
        tmp_path / "documents",
        source_format="txt",
        text="Exact local evidence for a semantic snapshot.",
    )
    build_search_index(project_root, now=lambda: TIMESTAMP)

    snapshot = SQLiteSearchBackend(project_root).corpus_snapshot()

    assert snapshot.index_built_at == TIMESTAMP
    assert snapshot.indexed_sources == 1
    assert snapshot.indexed_chunks == 1
    assert snapshot.partial_sources == 0
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", snapshot.index_fingerprint)
    assert snapshot.source_ids == (source.source_id,)
    assert len(snapshot.evidence) == 1
    assert snapshot.evidence[0].chunk_id == chunk.chunk_id
    assert snapshot.evidence[0].excerpt == chunk.text
    assert snapshot.evidence[0].locator.source_id == source.source_id


def test_index_build_rejects_a_tampered_chunk_slice(tmp_path: Path) -> None:
    project_root = tmp_path / "corpus"
    _, source, _ = _add_derived_source(
        project_root, tmp_path / "documents", source_format="txt"
    )
    path = chunk_artifact_path_for(project_root, source.source_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["chunks"][0]["text"] = "Tampered text."
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RetrievalError, match="does not match its extraction offsets"):
        build_search_index(project_root)


def test_search_rejects_an_index_after_an_artifact_changes(tmp_path: Path) -> None:
    project_root = tmp_path / "corpus"
    _, source, _ = _add_derived_source(
        project_root, tmp_path / "documents", source_format="txt"
    )
    build_search_index(project_root)
    path = chunk_artifact_path_for(project_root, source.source_id)
    state = path.stat()
    os.utime(
        path,
        ns=(state.st_atime_ns, state.st_mtime_ns + 2_000_000_000),
    )

    with pytest.raises(RetrievalError, match="changed after indexing"):
        SQLiteSearchBackend(project_root).search("retrievable")


def test_live_verification_rejects_a_changed_original(tmp_path: Path) -> None:
    project_root = tmp_path / "corpus"
    source_path, _, _ = _add_derived_source(
        project_root, tmp_path / "documents", source_format="txt"
    )
    build_search_index(project_root)
    backend = SQLiteSearchBackend(project_root)
    evidence_id = backend.search("retrievable").results[0].evidence_id
    source_path.write_bytes(b"changed-original-bytes")

    with pytest.raises(RetrievalError, match="do not match source"):
        backend.verify(evidence_id)
