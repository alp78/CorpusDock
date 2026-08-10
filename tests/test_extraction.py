from __future__ import annotations

import json
from pathlib import Path
import struct
import zipfile

from corpusdock.extraction import (
    EXTRACTION_SCHEMA_VERSION,
    MOBI_HUFF_CDIC_COMPRESSION,
    artifact_path_for,
    extract_source,
    extraction_coverage_report,
    write_extraction_artifact,
)
from corpusdock.manifest import ManifestStore
from corpusdock.mobi_huffcdic import CDIC_MAGIC, HUFF_MAGIC


TIMESTAMP = "2026-08-10T12:00:00Z"


def _register(project_root: Path, source_path: Path):
    store = ManifestStore(project_root, now=lambda: TIMESTAMP)
    registration = store.register([source_path])[0]
    return store, registration.source


def _write_minimal_pdf(path: Path, text: str) -> None:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, object_value in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_number} 0 obj\n".encode("ascii"))
        payload.extend(object_value)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(payload))


def _write_minimal_epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED
        )
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml" /></rootfiles>
</container>""",
        )
        archive.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Fixture EPUB</dc:title></metadata>
  <manifest><item id="chapter-one" href="text/chapter-one.xhtml" media-type="application/xhtml+xml" /></manifest>
  <spine><itemref idref="chapter-one" /></spine>
</package>""",
        )
        archive.writestr(
            "OEBPS/text/chapter-one.xhtml",
            """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>Chapter One</h1><p>Exact EPUB fixture text.</p>
</body></html>""",
        )


def _write_minimal_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types" />""",
        )
        archive.writestr(
            "word/document.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml">
  <w:body>
    <w:p w14:paraId="A1"><w:pPr><w:pStyle w:val="Heading1" /></w:pPr><w:r><w:t>Chapter One</w:t></w:r></w:p>
    <w:p w14:paraId="B2"><w:r><w:t>Exact DOCX fixture text.</w:t></w:r></w:p>
  </w:body>
</w:document>""",
        )


def _minimal_mobi_header(
    *,
    text_length: int,
    compression: int,
    text_record_count: int = 1,
    encryption_type: int = 0,
) -> bytearray:
    palm_doc = bytearray(16 + 0xE4)
    struct.pack_into(">H", palm_doc, 0, compression)
    struct.pack_into(">I", palm_doc, 4, text_length)
    struct.pack_into(">H", palm_doc, 8, text_record_count)
    struct.pack_into(">H", palm_doc, 10, 4096)
    struct.pack_into(">H", palm_doc, 12, encryption_type)
    palm_doc[16:20] = b"MOBI"
    struct.pack_into(">I", palm_doc, 20, 0xE4)
    struct.pack_into(">I", palm_doc, 28, 65001)  # UTF-8
    struct.pack_into(">I", palm_doc, 36, 6)
    struct.pack_into(">I", palm_doc, 16 + 0xAC, 0)  # DRM count
    return palm_doc


def _write_mobi_records(path: Path, records: tuple[bytes, ...]) -> None:
    record_count = len(records)
    next_offset = 78 + record_count * 8
    record_table = bytearray()
    for record_number, record in enumerate(records):
        record_table.extend(
            struct.pack(">I", next_offset) + record_number.to_bytes(4, "big")
        )
        next_offset += len(record)

    header = bytearray(78)
    header[60:68] = b"BOOKMOBI"
    struct.pack_into(">H", header, 76, record_count)
    path.write_bytes(bytes(header + record_table) + b"".join(records))


def _write_minimal_mobi(
    path: Path,
    text: str,
    *,
    encryption_type: int = 0,
    compression: int = 2,
) -> None:
    text_bytes = text.encode("utf-8")
    palm_doc = _minimal_mobi_header(
        text_length=len(text_bytes),
        compression=compression,
        encryption_type=encryption_type,
    )
    _write_mobi_records(path, (bytes(palm_doc), text_bytes))


def _write_huffcdic_mobi(path: Path) -> str:
    # One-bit codes map 0 to dictionary phrase 1 and 1 to literal phrase 0.
    # Phrase 1 recursively expands eight 1-bits, exercising CDIC expansion.
    expected_text = "B" * 64
    primary_offset = 24
    secondary_offset = primary_offset + 256 * 4
    huff = (
        HUFF_MAGIC
        + struct.pack(">II", primary_offset, secondary_offset)
        + bytes(8)
        + struct.pack(">256I", *([0x181] * 256))
        + bytes(64 * 4)
    )
    cdic = (
        CDIC_MAGIC
        + struct.pack(">II", 2, 1)
        + struct.pack(">2H", 4, 7)
        + struct.pack(">H", 0x8001)
        + b"B"
        + struct.pack(">H", 1)
        + b"\xff"
    )
    palm_doc = _minimal_mobi_header(
        text_length=len(expected_text),
        compression=MOBI_HUFF_CDIC_COMPRESSION,
    )
    struct.pack_into(">II", palm_doc, 0x70, 2, 2)
    _write_mobi_records(path, (bytes(palm_doc), b"\x00", huff, cdic))
    return expected_text


def test_txt_extraction_preserves_line_anchors_and_offsets(tmp_path: Path) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "notes.txt"
    source_path.write_bytes(b"First line\r\nSecond line\n")
    _, source = _register(project_root, source_path)

    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)

    assert artifact.status == "complete"
    assert artifact.text == "First line\r\nSecond line\n"
    assert artifact.anchors[0].locator.line_start == 1
    assert artifact.anchors[0].locator.start_offset == 0
    assert artifact.anchors[1].locator.line_end == 2
    assert artifact.anchors[1].locator.start_offset == len("First line\r\n")


def test_pdf_extraction_creates_a_page_level_anchor(tmp_path: Path) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "fixture.pdf"
    _write_minimal_pdf(source_path, "Exact PDF fixture text.")
    _, source = _register(project_root, source_path)

    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)

    assert artifact.status == "complete"
    assert "Exact PDF fixture text." in artifact.text
    assert artifact.anchors[0].locator.locator_type == "pdf_page"
    assert artifact.anchors[0].locator.page == 1
    assert artifact.anchors[0].locator.extraction_method == "pdf_text_layer"
    assert artifact.metadata["native_text_pages"] == [1]
    assert artifact.schema_version == EXTRACTION_SCHEMA_VERSION


def test_pdf_missing_text_is_reported_and_never_inferred(tmp_path: Path) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "scan.pdf"
    _write_minimal_pdf(source_path, "")
    store, source = _register(project_root, source_path)

    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)
    write_extraction_artifact(project_root, artifact)
    coverage = extraction_coverage_report(project_root, store.load().sources.values())

    assert artifact.status == "partial"
    assert artifact.text == ""
    assert artifact.parser_name == "pypdf"
    assert artifact.anchors[0].text == ""
    assert artifact.anchors[0].locator.extraction_method == "no_text"
    assert artifact.metadata["pages_with_text"] == 0
    assert artifact.metadata["empty_pages"] == [1]
    assert artifact.warnings[0].startswith("pdf_no_text_pages:")
    assert "text-only extraction skipped" in artifact.warnings[0]
    assert coverage["pdf_text_layers"] == {
        "unresolved_pdf_pages": 1,
        "sources_with_unresolved_pages": 1,
    }


def test_epub_extraction_tracks_spine_item_heading_and_offsets(tmp_path: Path) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "fixture.epub"
    _write_minimal_epub(source_path)
    _, source = _register(project_root, source_path)

    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)

    assert artifact.status == "complete"
    assert artifact.metadata["title"] == "Fixture EPUB"
    assert artifact.anchors[1].locator.locator_type == "epub_spine"
    assert (
        artifact.anchors[1].locator.spine_item == "1:chapter-one:text/chapter-one.xhtml"
    )
    assert artifact.anchors[1].locator.heading == "Chapter One"
    assert artifact.anchors[1].locator.start_offset is not None


def test_docx_extraction_tracks_heading_and_paragraph_id(tmp_path: Path) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "fixture.docx"
    _write_minimal_docx(source_path)
    _, source = _register(project_root, source_path)

    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)

    assert artifact.status == "complete"
    assert artifact.anchors[1].locator.locator_type == "docx_paragraph"
    assert artifact.anchors[1].locator.paragraph_id == "B2"
    assert artifact.anchors[1].locator.heading == "Chapter One"


def test_unencrypted_mobi_extraction_records_native_conversion_provenance(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "fixture.mobi"
    _write_minimal_mobi(
        source_path,
        "<html><body><h1>MOBI chapter</h1><p>Exact MOBI text.</p></body></html>",
    )
    _, source = _register(project_root, source_path)

    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)

    assert artifact.status == "complete"
    assert "Exact MOBI text." in artifact.text
    assert artifact.anchors[1].locator.locator_type == "mobi_section"
    assert artifact.metadata["conversion"] == {
        "method": "native_palmdoc_extraction",
        "converted": False,
    }


def test_huffcdic_mobi_extraction_is_bundled_and_dependency_free(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "huffcdic.mobi"
    expected_text = _write_huffcdic_mobi(source_path)
    _, source = _register(project_root, source_path)

    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)

    assert artifact.status == "complete"
    assert artifact.text == expected_text
    assert artifact.parser_name == "corpusdock.mobi"
    assert artifact.metadata["compression"] == MOBI_HUFF_CDIC_COMPRESSION
    assert artifact.metadata["conversion"] == {
        "method": "native_huffcdic_extraction",
        "converted": False,
    }


def test_unknown_mobi_compression_fails_without_external_converter(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "unsupported.mobi"
    _write_minimal_mobi(source_path, "Unsupported compression", compression=99)
    _, source = _register(project_root, source_path)

    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)

    assert artifact.status == "failed"
    assert artifact.metadata["failure_code"] == "mobi_compression_unsupported"
    assert "No external converter" in artifact.warnings[0]


def test_encrypted_mobi_is_recorded_as_failed_without_decryption(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "encrypted.mobi"
    _write_minimal_mobi(source_path, "Encrypted source", encryption_type=2)
    _, source = _register(project_root, source_path)

    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)

    assert artifact.status == "failed"
    assert artifact.metadata["failure_code"] == "mobi_encrypted"
    assert "will not decrypt or bypass DRM" in artifact.warnings[0]


def test_artifact_write_and_coverage_report_are_local_and_non_content_summary(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "notes.txt"
    source_path.write_bytes(b"Coverage text.\n")
    store, source = _register(project_root, source_path)
    artifact = extract_source(source, source_path, now=lambda: TIMESTAMP)

    written_path = write_extraction_artifact(project_root, artifact)
    report = extraction_coverage_report(project_root, store.load().sources.values())

    assert written_path == artifact_path_for(project_root, source.source_id)
    assert (
        json.loads(written_path.read_text(encoding="utf-8"))["text"]
        == "Coverage text.\n"
    )
    assert report["statuses"]["complete"] == 1
    assert report["anchors"] == 1
    assert report["text_characters"] == len("Coverage text.\n")
