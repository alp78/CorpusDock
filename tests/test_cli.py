from __future__ import annotations

from hashlib import sha256
import json

import pytest

from corpusdock.cli import build_parser, main


def test_search_parser_accepts_json_output() -> None:
    args = build_parser().parse_args(
        ["search", "citation anchors", "--json", "--limit", "4", "--match", "any"]
    )

    assert args.command == "search"
    assert args.query == "citation anchors"
    assert args.json is True
    assert args.limit == 4
    assert args.match == "any"


def test_ingest_parser_accepts_multiple_text_sources() -> None:
    args = build_parser().parse_args(["ingest", "one.pdf", "two.epub"])

    assert args.path == ["one.pdf", "two.epub"]
    assert not hasattr(args, "ocr")


def test_removed_ocr_option_is_rejected() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ingest", "one.pdf", "--ocr", "missing"])


def test_init_ingest_and_source_commands_register_and_extract(tmp_path, capsys) -> None:  # type: ignore[no-untyped-def]
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "documents" / "notes.txt"
    source_path.parent.mkdir()
    source_path.write_bytes(b"A local source, not an index.\n")
    source_id = f"src-{sha256(source_path.read_bytes()).hexdigest()}"

    assert main(["init", str(project_root)]) == 0
    assert "Initialized CorpusDock project" in capsys.readouterr().out

    assert (
        main(
            [
                "ingest",
                str(source_path),
                "--project",
                str(project_root),
                "--sentence-processor",
                "rule",
            ]
        )
        == 0
    )
    ingest_output = capsys.readouterr().out
    assert f"registered: {source_id} (txt)" in ingest_output
    assert "extraction: complete; 1 anchors" in ingest_output
    assert "chunking: complete; 1 chunks" in ingest_output
    assert (project_root / ".corpusdock" / "extracted" / f"{source_id}.json").is_file()
    assert (project_root / ".corpusdock" / "chunks" / f"{source_id}.json").is_file()

    assert main(["source", source_id, "--project", str(project_root)]) == 0
    source_payload = json.loads(capsys.readouterr().out)
    assert source_payload["source_id"] == source_id
    assert source_payload["original_paths"] == [str(source_path.resolve())]

    assert main(["doctor", "--project", str(project_root), "--json"]) == 1
    coverage = json.loads(capsys.readouterr().out)
    assert coverage["extraction"]["statuses"]["complete"] == 1
    assert coverage["chunking"]["statuses"]["complete"] == 1
    assert coverage["index"]["status"] == "missing"

    assert main(["index", "--project", str(project_root), "--json"]) == 0
    index_summary = json.loads(capsys.readouterr().out)
    assert index_summary["sources"] == 1
    assert index_summary["chunks"] == 1

    assert (
        main(
            [
                "search",
                "local source",
                "--project",
                str(project_root),
                "--json",
            ]
        )
        == 0
    )
    search_payload = json.loads(capsys.readouterr().out)
    assert search_payload["result_count"] == 1
    evidence = search_payload["results"][0]
    assert evidence["excerpt"] == "A local source, not an index.\n"
    assert evidence["locator"]["line_start"] == 1
    assert evidence["verification_status"] == "artifact-anchor-confirmed"

    assert (
        main(
            [
                "verify",
                evidence["evidence_id"],
                "--project",
                str(project_root),
                "--json",
            ]
        )
        == 0
    )
    verification = json.loads(capsys.readouterr().out)
    assert verification["verification_status"] == "source-anchor-confirmed"
    assert verification["evidence"]["excerpt"] == evidence["excerpt"]

    assert main(["doctor", "--project", str(project_root), "--json"]) == 0
    coverage = json.loads(capsys.readouterr().out)
    assert coverage["index"]["status"] == "ready"
