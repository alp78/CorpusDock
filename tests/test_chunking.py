from __future__ import annotations

from pathlib import Path

import pytest

from corpusdock.chunking import (
    CHUNK_SCHEMA_VERSION,
    ChunkingError,
    RuleSentenceProcessor,
    chunk_artifact_path_for,
    chunk_artifact_is_current,
    chunk_coverage_report,
    chunk_extraction_artifact,
    load_chunk_artifact,
    write_chunk_artifact,
)
from corpusdock.extraction import EXTRACTION_SCHEMA_VERSION
from corpusdock.manifest import SourceRecord


TIMESTAMP = "2026-08-10T12:00:00Z"
SOURCE_SHA256 = "a" * 64
SOURCE_ID = f"src-{SOURCE_SHA256}"


def _extraction(
    text: str,
    anchors: list[dict[str, object]],
    *,
    status: str = "complete",
) -> dict[str, object]:
    return {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "source_sha256": SOURCE_SHA256,
        "status": status,
        "text": text,
        "anchors": anchors,
    }


def _anchor(
    anchor_id: str,
    start: int,
    end: int,
    *,
    locator_type: str = "text_line",
    page: int | None = None,
    heading: str | None = None,
) -> dict[str, object]:
    return {
        "anchor_id": anchor_id,
        "text": "",
        "locator": {
            "source_id": SOURCE_ID,
            "locator_type": locator_type,
            "label": anchor_id,
            "page": page,
            "heading": heading,
            "start_offset": start,
            "end_offset": end,
        },
    }


def test_rule_chunker_preserves_exact_offsets_sentence_overlap_and_limits() -> None:
    text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."
    extraction = _extraction(text, [_anchor("line-1", 0, len(text))])

    artifact = chunk_extraction_artifact(
        extraction,
        RuleSentenceProcessor(),
        target_characters=25,
        max_characters=36,
        overlap_sentences=1,
        now=lambda: TIMESTAMP,
    )

    assert artifact.status == "complete"
    assert len(artifact.chunks) > 1
    for chunk in artifact.chunks:
        assert chunk.text == text[chunk.start_offset : chunk.end_offset]
        assert len(chunk.text) <= 36
        assert chunk.locators[0]["start_offset"] == chunk.start_offset
        assert chunk.locators[0]["end_offset"] == chunk.end_offset
    assert artifact.chunks[0].text.split()[-2:] == artifact.chunks[1].text.split()[:2]


def test_pdf_chunks_never_cross_page_boundaries() -> None:
    page_one = "Page one sentence. Another page one sentence."
    separator = "\n\f\n"
    page_two = "Page two sentence. Another page two sentence."
    text = page_one + separator + page_two
    extraction = _extraction(
        text,
        [
            _anchor("page-1", 0, len(page_one), locator_type="pdf_page", page=1),
            _anchor(
                "page-2",
                len(page_one) + len(separator),
                len(text),
                locator_type="pdf_page",
                page=2,
            ),
        ],
    )

    artifact = chunk_extraction_artifact(
        extraction,
        RuleSentenceProcessor(),
        target_characters=500,
        max_characters=600,
        overlap_sentences=0,
        now=lambda: TIMESTAMP,
    )

    assert len(artifact.chunks) == 2
    assert [
        {locator["page"] for locator in chunk.locators} for chunk in artifact.chunks
    ] == [{1}, {2}]
    assert all(separator not in chunk.text for chunk in artifact.chunks)


def test_oversized_unpunctuated_text_is_split_without_losing_characters() -> None:
    text = "word " * 90
    extraction = _extraction(text, [_anchor("line-1", 0, len(text))])

    artifact = chunk_extraction_artifact(
        extraction,
        RuleSentenceProcessor(),
        target_characters=80,
        max_characters=100,
        overlap_sentences=0,
        now=lambda: TIMESTAMP,
    )

    assert all(
        chunk.text == text[chunk.start_offset : chunk.end_offset]
        for chunk in artifact.chunks
    )
    assert all(len(chunk.text) <= 100 for chunk in artifact.chunks)
    assert "".join(chunk.text for chunk in artifact.chunks) == text


def test_partial_empty_extraction_produces_auditable_partial_artifact() -> None:
    artifact = chunk_extraction_artifact(
        _extraction("", [], status="partial"),
        RuleSentenceProcessor(),
        now=lambda: TIMESTAMP,
    )

    assert artifact.status == "partial"
    assert artifact.chunks == ()
    assert artifact.warnings == (
        "source_extraction_partial: Chunks cover only the text available from extraction.",
        "no_extractable_text: No non-whitespace extracted text was available to chunk.",
    )


def test_chunk_configuration_is_validated() -> None:
    with pytest.raises(ChunkingError, match="no larger than the maximum"):
        chunk_extraction_artifact(
            _extraction("Text.", [_anchor("line-1", 0, 5)]),
            RuleSentenceProcessor(),
            target_characters=20,
            max_characters=10,
        )


def test_chunk_artifact_write_load_and_coverage(tmp_path: Path) -> None:
    text = "Stored chunk."
    artifact = chunk_extraction_artifact(
        _extraction(text, [_anchor("line-1", 0, len(text))]),
        RuleSentenceProcessor(),
        now=lambda: TIMESTAMP,
    )
    source = SourceRecord(
        source_id=SOURCE_ID,
        sha256=SOURCE_SHA256,
        source_format="txt",
        size_bytes=len(text),
        original_paths=("/local/source.txt",),
        registered_at=TIMESTAMP,
        registration_tool_version="0.1.0",
    )

    written = write_chunk_artifact(tmp_path, artifact)
    loaded = load_chunk_artifact(tmp_path, SOURCE_ID)
    report = chunk_coverage_report(tmp_path, [source])

    assert written == chunk_artifact_path_for(tmp_path, SOURCE_ID)
    assert loaded is not None
    assert loaded["schema_version"] == CHUNK_SCHEMA_VERSION
    assert loaded["chunks"][0]["text"] == text
    assert report["statuses"]["complete"] == 1
    assert report["chunks"] == 1

    assert chunk_artifact_is_current(
        loaded,
        source,
        sentence_processor_name=artifact.sentence_processor_name,
        sentence_processor_version=artifact.sentence_processor_version,
        sentence_model=artifact.sentence_model,
    )
    assert not chunk_artifact_is_current(
        loaded,
        source,
        sentence_processor_name=artifact.sentence_processor_name,
        sentence_processor_version=artifact.sentence_processor_version,
        sentence_model=artifact.sentence_model,
        target_characters=999,
    )
