"""Command-line interface for local CorpusDock project state."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys

from corpusdock import __version__
from corpusdock.chunking import (
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_OVERLAP_SENTENCES,
    DEFAULT_SENTENCE_MODEL,
    DEFAULT_TARGET_CHARACTERS,
    ChunkingError,
    chunk_coverage_report,
    chunk_extraction_artifact,
    sentence_processor_from,
    write_chunk_artifact,
)
from corpusdock.extraction import (
    ExtractionError,
    extract_source,
    extraction_coverage_report,
    write_extraction_artifact,
)
from corpusdock.manifest import (
    ManifestError,
    ManifestStore,
    discover_source_files,
    find_project_root,
)
from corpusdock.retrieval import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    RetrievalError,
    SQLiteSearchBackend,
    build_search_index,
    index_status_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpusdock",
        description="Citation-aware local document ingestion and retrieval.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init_parser = commands.add_parser("init", help="Create CorpusDock project state.")
    init_parser.add_argument("path", nargs="?", default=".", help="Project directory.")
    init_parser.set_defaults(handler=_initialize_project)

    ingest_parser = commands.add_parser(
        "ingest",
        help="Register, extract, and chunk supported local documents.",
    )
    ingest_parser.add_argument(
        "path", nargs="+", help="One or more files or directories to ingest."
    )
    ingest_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    ingest_parser.add_argument(
        "--register-only",
        action="store_true",
        help="Register source hashes and paths without extracting or chunking text.",
    )
    ingest_parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Register and extract text without creating chunks.",
    )
    ingest_parser.add_argument(
        "--sentence-processor",
        choices=("sat", "rule"),
        default="sat",
        help=(
            "Local sentence processor: quality-first SaT (default) or the "
            "dependency-free rule fallback."
        ),
    )
    ingest_parser.add_argument(
        "--sentence-model",
        default=DEFAULT_SENTENCE_MODEL,
        help=f"Local SaT model name or directory (default: {DEFAULT_SENTENCE_MODEL}).",
    )
    ingest_parser.add_argument(
        "--target-characters",
        type=int,
        default=DEFAULT_TARGET_CHARACTERS,
        help=f"Soft chunk target in Unicode code points (default: {DEFAULT_TARGET_CHARACTERS}).",
    )
    ingest_parser.add_argument(
        "--max-characters",
        type=int,
        default=DEFAULT_MAX_CHARACTERS,
        help=f"Hard chunk maximum in Unicode code points (default: {DEFAULT_MAX_CHARACTERS}).",
    )
    ingest_parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=DEFAULT_OVERLAP_SENTENCES,
        help=f"Whole-sentence overlap between chunks (default: {DEFAULT_OVERLAP_SENTENCES}).",
    )
    ingest_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit registration, extraction, and chunk summaries as JSON.",
    )
    ingest_parser.set_defaults(handler=_register_sources)

    index_parser = commands.add_parser(
        "index", help="Build the embedded full-text index from persisted chunks."
    )
    index_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    index_parser.add_argument(
        "--json", action="store_true", help="Emit non-content index metadata as JSON."
    )
    index_parser.set_defaults(handler=_build_index)

    search_parser = commands.add_parser(
        "search", help="Search exact text in the local index."
    )
    search_parser.add_argument("query", help="Literal words or quoted phrases.")
    search_parser.add_argument(
        "--json", action="store_true", help="Emit the evidence contract as JSON."
    )
    search_parser.add_argument("--source", help="Restrict results to one source ID.")
    search_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Maximum results from 1 to {MAX_SEARCH_LIMIT} (default: {DEFAULT_SEARCH_LIMIT}).",
    )
    search_parser.add_argument(
        "--match",
        choices=("all", "any", "phrase"),
        default="all",
        help="Match all terms (default), any term, or the complete phrase.",
    )
    search_parser.set_defaults(handler=_search_sources)

    source_parser = commands.add_parser("source", help="Inspect a registered source.")
    source_parser.add_argument("source_id", help="Stable CorpusDock source ID.")
    source_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    source_parser.set_defaults(handler=_show_source)

    verify_parser = commands.add_parser(
        "verify", help="Open or validate an evidence anchor."
    )
    verify_parser.add_argument("evidence_id", help="Stable CorpusDock evidence ID.")
    verify_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    verify_parser.add_argument(
        "--json", action="store_true", help="Emit the verification report as JSON."
    )
    verify_parser.set_defaults(handler=_verify_evidence)

    mcp_parser = commands.add_parser("mcp", help="Run the optional local MCP adapter.")
    mcp_parser.set_defaults(handler=_not_implemented)

    doctor_parser = commands.add_parser(
        "doctor", help="Report local extraction and chunk coverage."
    )
    doctor_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    doctor_parser.add_argument(
        "--json", action="store_true", help="Emit the coverage report as JSON."
    )
    doctor_parser.set_defaults(handler=_report_coverage)

    return parser


def _not_implemented(args: argparse.Namespace) -> int:
    print(
        f"The '{args.command}' command is part of the CorpusDock scaffold and is not "
        "implemented yet."
    )
    return 2


def _build_index(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    summary = build_search_index(project_root)
    if args.json:
        print(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        )
    else:
        print(
            f"Indexed {summary.chunks} chunks and {summary.anchors} anchors "
            f"from {summary.sources} sources."
        )
        print(
            "Coverage: "
            f"{summary.complete_sources} complete, {summary.partial_sources} partial, "
            f"{summary.failed_sources} failed; "
            f"{summary.unresolved_pdf_pages} PDF pages without embedded text"
        )
        print(f"Index: {summary.path}")
    return 0


def _search_sources(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    response = SQLiteSearchBackend(project_root).search(
        args.query,
        limit=args.limit,
        source_id=args.source,
        match_mode=args.match,
    )
    if args.json:
        print(
            json.dumps(response.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        )
        return 0

    if not response.results:
        print("No exact full-text matches.")
        return 0
    for position, result in enumerate(response.results, start=1):
        print(f"[{position}] {result.citation}")
        print(f"Evidence: {result.evidence_id}")
        if result.score is not None:
            print(f"Score: {result.score:.12g}")
        print(f"Verification: {result.verification_status}")
        if result.source_coverage is not None:
            coverage = result.source_coverage
            coverage_line = f"Coverage: {coverage.extraction_status}"
            if coverage.unresolved_pdf_pages:
                coverage_line += f"; {len(coverage.unresolved_pdf_pages)} PDF pages without embedded text"
            print(coverage_line)
        print(f"Source: {result.source_path}")
        print("Excerpt:")
        print(result.excerpt)
        print()
    return 0


def _verify_evidence(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    report = SQLiteSearchBackend(project_root).verify(args.evidence_id)
    if args.json:
        print(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        )
    else:
        print(f"Verified: {report.evidence.evidence_id}")
        print(f"Status: {report.evidence.verification_status}")
        print(f"Citation: {report.evidence.citation}")
        print(f"Source: {report.evidence.source_path}")
        print("Checks: " + ", ".join(report.checks))
        print("Excerpt:")
        print(report.evidence.excerpt)
    return 0


def _initialize_project(args: argparse.Namespace) -> int:
    project_root = Path(args.path).expanduser()
    if project_root.exists() and not project_root.is_dir():
        raise ManifestError(f"Project path is not a directory: '{project_root}'.")

    store = ManifestStore(project_root)
    _, created = store.initialize()
    action = "Initialized" if created else "CorpusDock project already initialized at"
    if created:
        print(f"{action} CorpusDock project at '{store.path}'.")
    else:
        print(f"{action} '{store.path}'.")
    return 0


def _register_sources(args: argparse.Namespace) -> int:
    if args.register_only and args.extract_only:
        raise ManifestError(
            "--register-only and --extract-only cannot be used together."
        )

    project_root = _resolve_project_root(args.project)
    source_files = tuple(
        dict.fromkeys(
            source_file
            for source_path in args.path
            for source_file in discover_source_files(source_path)
        )
    )
    sentence_processor = None
    if not args.register_only and not args.extract_only:
        sentence_processor = sentence_processor_from(
            args.sentence_processor,
            model_name=args.sentence_model,
        )
    store = ManifestStore(project_root)
    registrations = store.register(source_files)

    processed_sources: dict[str, dict[str, object]] = {}
    results: list[dict[str, object]] = []
    has_failure = False
    for registration in registrations:
        result: dict[str, object] = {
            "registration": {
                "status": registration.status,
                "source_id": registration.source.source_id,
                "source_format": registration.source.source_format,
                "source_path": registration.source_path,
            }
        }
        if not args.register_only:
            processed = processed_sources.get(registration.source.source_id)
            if processed is None:
                artifact = extract_source(
                    registration.source,
                    registration.source_path,
                )
                try:
                    artifact_path = write_extraction_artifact(project_root, artifact)
                    extraction = artifact.summary()
                except ExtractionError as error:
                    extraction = {
                        "source_id": registration.source.source_id,
                        "source_format": registration.source.source_format,
                        "status": "failed",
                        "anchor_count": 0,
                        "text_characters": 0,
                        "warning_count": 1,
                        "error": str(error),
                    }
                    artifact_path = None
                if extraction["status"] == "failed":
                    has_failure = True
                processed = {
                    "extraction": extraction,
                }
                if artifact_path is not None:
                    processed["extraction_artifact_path"] = str(artifact_path)

                if sentence_processor is not None and artifact_path is not None:
                    try:
                        chunk_artifact = chunk_extraction_artifact(
                            artifact.to_dict(),
                            sentence_processor,
                            target_characters=args.target_characters,
                            max_characters=args.max_characters,
                            overlap_sentences=args.overlap_sentences,
                        )
                        chunk_path = write_chunk_artifact(project_root, chunk_artifact)
                        chunking = chunk_artifact.summary()
                        processed["chunking"] = chunking
                        processed["chunk_artifact_path"] = str(chunk_path)
                        if chunking["status"] == "failed":
                            has_failure = True
                    except ChunkingError as error:
                        processed["chunking"] = {
                            "source_id": registration.source.source_id,
                            "status": "failed",
                            "chunk_count": 0,
                            "warning_count": 1,
                            "sentence_processor": sentence_processor.name,
                            "sentence_model": sentence_processor.model_name,
                            "error": str(error),
                        }
                        has_failure = True

                processed_sources[registration.source.source_id] = processed
            else:
                processed = {
                    key: (
                        {**value, "reused_for_duplicate_content": True}
                        if isinstance(value, dict)
                        else value
                    )
                    for key, value in processed.items()
                }
            result.update(processed)
        results.append(result)

    if args.json:
        print(
            json.dumps(
                {
                    "manifest": str(store.path),
                    "register_only": args.register_only,
                    "extract_only": args.extract_only,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for result in results:
            registration = result["registration"]
            assert isinstance(registration, dict)
            print(
                f"{registration['status']}: {registration['source_id']} "
                f"({registration['source_format']}) {registration['source_path']}"
            )
            extraction = result.get("extraction")
            if isinstance(extraction, dict):
                print(
                    f"  extraction: {extraction['status']}; "
                    f"{extraction['anchor_count']} anchors; "
                    f"{extraction['text_characters']} characters"
                )
                unresolved_pages = extraction.get("unresolved_pdf_pages")
                if isinstance(unresolved_pages, int) and unresolved_pages:
                    print(
                        f"  PDF text layer: {unresolved_pages} pages without embedded text"
                    )
            chunking = result.get("chunking")
            if isinstance(chunking, dict):
                print(
                    f"  chunking: {chunking['status']}; "
                    f"{chunking['chunk_count']} chunks; "
                    f"{chunking['sentence_processor']} ({chunking['sentence_model']})"
                )
        print(f"Manifest: {store.path}")
    return 1 if has_failure else 0


def _show_source(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    source = ManifestStore(project_root).get_source(args.source_id)
    if source is None:
        raise ManifestError(
            f"No source with ID '{args.source_id}' is registered in this project."
        )

    print(json.dumps(source.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _report_coverage(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    manifest = ManifestStore(project_root).load()
    source_records = manifest.sources.values()
    extraction_report = extraction_coverage_report(project_root, source_records)
    chunk_report = chunk_coverage_report(project_root, manifest.sources.values())
    search_index_report = index_status_report(project_root)
    report = {
        "extraction": extraction_report,
        "chunking": chunk_report,
        "index": search_index_report,
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        statuses = extraction_report["statuses"]
        print(f"Registered sources: {extraction_report['registered_sources']}")
        print(
            "Extraction: "
            f"{statuses['complete']} complete, {statuses['partial']} partial, "
            f"{statuses['failed']} failed, {statuses['pending']} pending, {statuses['stale']} stale"
        )
        print(f"Anchors: {extraction_report['anchors']}")
        print(
            f"Extracted text: {extraction_report['text_characters']} Unicode code points"
        )
        pdf_text_layers = extraction_report["pdf_text_layers"]
        print(
            "PDF text layers: "
            f"{pdf_text_layers['unresolved_pdf_pages']} pages without embedded text "
            f"across {pdf_text_layers['sources_with_unresolved_pages']} sources"
        )
        chunk_statuses = chunk_report["statuses"]
        print(
            "Chunking: "
            f"{chunk_statuses['complete']} complete, {chunk_statuses['partial']} partial, "
            f"{chunk_statuses['failed']} failed, {chunk_statuses['pending']} pending, "
            f"{chunk_statuses['stale']} stale"
        )
        print(f"Chunks: {chunk_report['chunks']}")
        if search_index_report["status"] == "ready":
            print(
                "Search index: ready; "
                f"{search_index_report['sources']} sources; "
                f"{search_index_report['chunks']} chunks"
            )
        else:
            print(f"Search index: {search_index_report['status']}")
    unhealthy = ("failed", "pending", "stale")
    return (
        1
        if (
            search_index_report["status"] != "ready"
            or any(
                report_part["statuses"][status]
                for report_part in (extraction_report, chunk_report)
                for status in unhealthy
            )
        )
        else 0
    )


def _resolve_project_root(project: str | None) -> Path:
    if project is not None:
        project_root = Path(project).expanduser()
        if not project_root.is_dir():
            raise ManifestError(f"Project path is not a directory: '{project_root}'.")
        return project_root

    project_root = find_project_root(Path.cwd())
    if project_root is None:
        raise ManifestError(
            "No initialized CorpusDock project found. Run 'corpusdock init' or pass --project."
        )
    return project_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (ManifestError, ExtractionError, ChunkingError, RetrievalError) as error:
        print(f"corpusdock: error: {error}", file=sys.stderr)
        return 1
