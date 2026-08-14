"""Command-line interface for local CorpusDock project state."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

from corpusdock import __version__
from corpusdock.analysis_evaluation import (
    AnalysisEvaluationError,
    evaluate_analysis_benchmark,
    load_analysis_benchmark,
)
from corpusdock.analysis_models import (
    DEFAULT_ANALYSIS_BATCH_SIZE,
    DEFAULT_ANALYSIS_MAX_INPUT_TOKENS,
    DEFAULT_ANALYSIS_MAX_OUTPUT_TOKENS,
    DEFAULT_ANALYSIS_MODEL,
    MAX_ANALYSIS_BATCH_SIZE,
    AnalysisModelError,
    StructuredExtractionProvider,
    TransformersStructuredExtractionProvider,
)
from corpusdock.analysis_runner import run_corpus_analysis
from corpusdock.analysis_store import AnalysisStoreError, analysis_status_report
from corpusdock.analysis_vllm import (
    DEFAULT_VLLM_ANALYSIS_BATCH_SIZE,
    DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
    VLLMStructuredExtractionProvider,
    _configure_vllm_environment,
)
from corpusdock.chunking import (
    DEFAULT_MAX_CHARACTERS,
    DEFAULT_OVERLAP_SENTENCES,
    DEFAULT_SENTENCE_DEVICE,
    DEFAULT_SENTENCE_MODEL,
    DEFAULT_TARGET_CHARACTERS,
    ChunkingError,
    chunk_coverage_report,
    chunk_extraction_artifact,
    sentence_processor_from,
    write_chunk_artifact,
)
from corpusdock.concept_graph import (
    DEFAULT_GRAPH_EVIDENCE_LIMIT,
    DEFAULT_GRAPH_QUERY_LIMIT,
    MAX_GRAPH_EVIDENCE_LIMIT,
    MAX_GRAPH_QUERY_LIMIT,
    ConceptGraphError,
    build_concept_graph,
    concept_graph_status_report,
    query_concept_graph,
)
from corpusdock.extraction import (
    ExtractionError,
    extract_source,
    extraction_coverage_report,
    write_extraction_artifact,
)
from corpusdock.evaluation import (
    EvaluationError,
    evaluate_retrieval,
    load_evaluation_dataset,
)
from corpusdock.embeddings import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    MAX_EMBEDDING_BATCH_SIZE,
    EmbeddingError,
    InMemorySemanticSearchBackend,
    SentenceTransformersEmbeddingProvider,
    model_cache_dir_for,
)
from corpusdock.manifest import (
    ManifestError,
    ManifestStore,
    discover_source_files,
    find_project_root,
)
from corpusdock.hybrid import HybridSearchBackend
from corpusdock.retrieval import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    RetrievalError,
    SQLiteSearchBackend,
    build_search_index,
    index_status_report,
)
from corpusdock.semantic_index import (
    PersistentSemanticSearchBackend,
    SemanticIndexDescriptor,
    SemanticIndexError,
    build_semantic_index,
    read_current_semantic_index_descriptor,
    semantic_index_status_report,
)
from corpusdock.synchronization import (
    PipelineConfig,
    SynchronizationSummary,
    configure_input_mirror,
    load_pipeline_config,
    synchronize_input_mirror,
)


def _add_embedding_options(
    parser: argparse.ArgumentParser,
    *,
    default_model: str | None,
    include_truncation: bool,
    default_help: str | None = None,
) -> None:
    default_description = default_help or (
        str(default_model)
        if default_model is not None
        else "the model recorded in the semantic index"
    )
    model_help = (
        f"Local directory or namespace/model ID (default: {default_description})."
    )
    parser.add_argument(
        "--embedding-model",
        default=default_model,
        help=model_help,
    )
    parser.add_argument(
        "--model-revision",
        help="Optional model commit, tag, or branch; the resolved revision is reported.",
    )
    parser.add_argument(
        "--model-cache",
        help="Model cache directory (default: the project's ignored model cache).",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Explicitly permit downloading public model weights; document text is never uploaded.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Local inference device understood by the runtime (default: cpu).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help=(
            "Local embedding batch size from 1 to "
            f"{MAX_EMBEDDING_BATCH_SIZE} (default: {DEFAULT_EMBEDDING_BATCH_SIZE})."
        ),
    )
    if include_truncation:
        parser.add_argument(
            "--truncate-dimension",
            type=int,
            help="Optional output dimension for a model trained to support truncation.",
        )


def _add_analysis_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--analysis-runtime",
        choices=("transformers", "vllm"),
        default="transformers",
        help=(
            "Local inference runtime: portable Transformers (default) or "
            "high-throughput CUDA vLLM."
        ),
    )
    parser.add_argument(
        "--analysis-model",
        default=DEFAULT_ANALYSIS_MODEL,
        help=(
            "Local directory or namespace/model ID "
            f"(default: {DEFAULT_ANALYSIS_MODEL})."
        ),
    )
    parser.add_argument(
        "--model-revision",
        help="Optional model commit, tag, or branch; the resolved revision is reported.",
    )
    parser.add_argument(
        "--model-cache",
        help="Model cache directory (default: the project's ignored model cache).",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Explicitly permit downloading public model weights; document text is never uploaded.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Local inference device understood by PyTorch (default: cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Local model parameter dtype (default: model configuration).",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "bnb-4bit", "bnb-8bit"),
        default="none",
        help="Optional Transformers CUDA weight quantization (default: none).",
    )
    parser.add_argument(
        "--structured-output",
        choices=("json-schema", "prompt-only"),
        default="json-schema",
        help="Local output constraint mode (default: JSON Schema enforcement).",
    )
    parser.add_argument(
        "--support-unit-processor",
        choices=("sat", "rule"),
        default="sat",
        help=(
            "Local sentence processor for exact analysis support units: "
            "quality-first SaT (default) or the rule fallback."
        ),
    )
    parser.add_argument(
        "--support-unit-model",
        default=DEFAULT_SENTENCE_MODEL,
        help=f"Local SaT support-unit model (default: {DEFAULT_SENTENCE_MODEL}).",
    )
    parser.add_argument(
        "--analysis-batch-size",
        type=int,
        default=None,
        help=(
            f"Generation batch size from 1 to {MAX_ANALYSIS_BATCH_SIZE} "
            f"(default: {DEFAULT_ANALYSIS_BATCH_SIZE} for Transformers, "
            f"{DEFAULT_VLLM_ANALYSIS_BATCH_SIZE} for vLLM)."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        help=(
            "Fraction of local GPU memory available to vLLM "
            f"(default: {DEFAULT_VLLM_GPU_MEMORY_UTILIZATION})."
        ),
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=DEFAULT_ANALYSIS_MAX_INPUT_TOKENS,
        help=(
            "Maximum local prompt tokens per evidence chunk "
            f"(default: {DEFAULT_ANALYSIS_MAX_INPUT_TOKENS})."
        ),
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_ANALYSIS_MAX_OUTPUT_TOKENS,
        help=(
            "Maximum generated tokens per evidence chunk "
            f"(default: {DEFAULT_ANALYSIS_MAX_OUTPUT_TOKENS})."
        ),
    )
    parser.add_argument(
        "--prompt-style",
        choices=("auto", "chat", "nuextract3"),
        default="auto",
        help="Structured extraction prompt adapter (default: detect from model).",
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
        "--sentence-device",
        choices=("cpu", "cuda"),
        default=DEFAULT_SENTENCE_DEVICE,
        help="Local SaT inference device (default: cpu).",
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

    sync_parser = commands.add_parser(
        "sync",
        help="Mirror one local input directory into resumable project state.",
    )
    sync_parser.add_argument(
        "input",
        help="Authoritative directory scanned recursively for supported documents.",
    )
    sync_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    sync_parser.add_argument(
        "--configure-only",
        action="store_true",
        help="Remember the input directory for future resumes without scanning it now.",
    )
    sync_parser.add_argument(
        "--sentence-processor",
        choices=("sat", "rule"),
        default="sat",
        help=(
            "Local sentence processor: quality-first SaT (default) or the "
            "dependency-free rule fallback."
        ),
    )
    sync_parser.add_argument(
        "--sentence-model",
        default=DEFAULT_SENTENCE_MODEL,
        help=f"Local SaT model name or directory (default: {DEFAULT_SENTENCE_MODEL}).",
    )
    sync_parser.add_argument(
        "--sentence-device",
        choices=("cpu", "cuda"),
        default=DEFAULT_SENTENCE_DEVICE,
        help="Local SaT inference device (default: cpu).",
    )
    sync_parser.add_argument(
        "--target-characters",
        type=int,
        default=DEFAULT_TARGET_CHARACTERS,
        help=f"Soft chunk target in Unicode code points (default: {DEFAULT_TARGET_CHARACTERS}).",
    )
    sync_parser.add_argument(
        "--max-characters",
        type=int,
        default=DEFAULT_MAX_CHARACTERS,
        help=f"Hard chunk maximum in Unicode code points (default: {DEFAULT_MAX_CHARACTERS}).",
    )
    sync_parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=DEFAULT_OVERLAP_SENTENCES,
        help=f"Whole-sentence overlap between chunks (default: {DEFAULT_OVERLAP_SENTENCES}).",
    )
    sync_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a non-content synchronization summary as JSON.",
    )
    sync_parser.set_defaults(handler=_sync_sources)

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

    embed_parser = commands.add_parser(
        "embed",
        help="Build the persistent local semantic index from exact chunks.",
    )
    embed_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    _add_embedding_options(
        embed_parser,
        default_model=DEFAULT_EMBEDDING_MODEL,
        include_truncation=True,
    )
    embed_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit non-content semantic-index metadata as JSON.",
    )
    embed_parser.set_defaults(handler=_build_semantic_index)

    search_parser = commands.add_parser(
        "search", help="Search indexed local document evidence."
    )
    search_parser.add_argument("query", help="Search question, phrase, or terms.")
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
        help="Lexical matching mode; retained as metadata for semantic and hybrid search.",
    )
    search_parser.add_argument(
        "--retrieval",
        choices=("lexical", "semantic", "hybrid"),
        default="lexical",
        help="Retrieval path (default: lexical).",
    )
    _add_embedding_options(
        search_parser,
        default_model=None,
        include_truncation=False,
    )
    search_parser.set_defaults(handler=_search_sources)

    eval_parser = commands.add_parser(
        "eval", help="Measure local retrieval against relevance judgments."
    )
    eval_parser.add_argument("dataset", help="Versioned evaluation dataset JSON file.")
    eval_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    eval_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Results per case from 1 to {MAX_SEARCH_LIMIT} (default: {DEFAULT_SEARCH_LIMIT}).",
    )
    eval_parser.add_argument(
        "--retrieval",
        choices=("lexical", "semantic", "hybrid"),
        default="lexical",
        help="Retrieval path to evaluate (default: lexical).",
    )
    _add_embedding_options(
        eval_parser,
        default_model=None,
        include_truncation=True,
        default_help=(
            f"{DEFAULT_EMBEDDING_MODEL} for semantic evaluation; the indexed model "
            "for hybrid evaluation"
        ),
    )
    eval_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip live source-byte verification of returned evidence.",
    )
    eval_parser.add_argument(
        "--json", action="store_true", help="Emit the versioned evaluation report."
    )
    eval_parser.set_defaults(handler=_evaluate_retrieval)

    analyze_parser = commands.add_parser(
        "analyze",
        help="Extract grounded concept, claim, and relation candidates locally.",
    )
    analyze_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    analyze_parser.add_argument(
        "--input",
        help=(
            "Authoritative input directory to remember and synchronize before local "
            "model loading. A previously configured directory is scanned by default."
        ),
    )
    analyze_parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Restrict analysis to one source ID; repeat to select multiple sources.",
    )
    analyze_parser.add_argument(
        "--limit",
        type=int,
        help="Analyze at most this many exact evidence chunks in stable index order.",
    )
    analyze_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a new run instead of resuming a matching incomplete run.",
    )
    _add_analysis_model_options(analyze_parser)
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit non-content run provenance and aggregate counts as JSON.",
    )
    analyze_parser.set_defaults(handler=_analyze_sources)

    analysis_eval_parser = commands.add_parser(
        "analysis-eval",
        help="Benchmark a local structured-extraction model on public fixtures.",
    )
    analysis_eval_parser.add_argument(
        "dataset", help="Versioned analysis benchmark JSON file."
    )
    analysis_eval_parser.add_argument(
        "--project",
        help=(
            "Initialized CorpusDock project used only for its ignored model cache "
            "(defaults to the nearest project)."
        ),
    )
    _add_analysis_model_options(analysis_eval_parser)
    analysis_eval_parser.add_argument(
        "--json", action="store_true", help="Emit the non-content benchmark report."
    )
    analysis_eval_parser.set_defaults(handler=_evaluate_analysis)

    graph_parser = commands.add_parser(
        "graph",
        help="Build or query the evidence-linked corpus concept graph.",
    )
    graph_commands = graph_parser.add_subparsers(dest="graph_command", required=True)
    graph_build_parser = graph_commands.add_parser(
        "build",
        help="Resolve one completed analysis run into an atomic local graph.",
    )
    graph_build_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    graph_build_parser.add_argument(
        "--run",
        help=(
            "Completed analysis run ID. By default, select the run with the widest "
            "evidence coverage, then the newest completion time."
        ),
    )
    graph_build_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit non-content graph provenance and counts as JSON.",
    )
    graph_build_parser.set_defaults(handler=_build_concept_graph)

    graph_query_parser = graph_commands.add_parser(
        "query",
        help="Search resolved concepts and return exact citation-ready support.",
    )
    graph_query_parser.add_argument("query", help="Concept terms to find.")
    graph_query_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    graph_query_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_GRAPH_QUERY_LIMIT,
        help=(
            f"Maximum concepts from 1 to {MAX_GRAPH_QUERY_LIMIT} "
            f"(default: {DEFAULT_GRAPH_QUERY_LIMIT})."
        ),
    )
    graph_query_parser.add_argument(
        "--evidence-limit",
        type=int,
        default=DEFAULT_GRAPH_EVIDENCE_LIMIT,
        help=(
            f"Maximum source-diverse support records per concept from 1 to "
            f"{MAX_GRAPH_EVIDENCE_LIMIT} "
            f"(default: {DEFAULT_GRAPH_EVIDENCE_LIMIT})."
        ),
    )
    graph_query_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit resolved concepts and exact evidence as JSON.",
    )
    graph_query_parser.set_defaults(handler=_query_concept_graph)

    graph_status_parser = graph_commands.add_parser(
        "status", help="Report non-content concept graph health and coverage."
    )
    graph_status_parser.add_argument(
        "--project",
        help="Initialized CorpusDock project directory (defaults to the nearest project).",
    )
    graph_status_parser.add_argument(
        "--json", action="store_true", help="Emit graph health as JSON."
    )
    graph_status_parser.set_defaults(handler=_report_concept_graph)

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


def _build_concept_graph(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    descriptor = build_concept_graph(project_root, run_id=args.run)
    if args.json:
        print(
            json.dumps(
                descriptor.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
        )
        return 0
    print(
        f"Concept graph: {descriptor.concepts} resolved concepts from "
        f"{descriptor.concept_mentions} mentions"
    )
    print(
        f"Grounded candidates: {descriptor.claims} claims; "
        f"{descriptor.relations} relations; "
        f"{descriptor.represented_sources}/{descriptor.indexed_sources} sources represented"
    )
    print(
        f"Analysis run: {descriptor.analysis_run_id} "
        f"({descriptor.analysis_run_selection})"
    )
    print(f"Resolution policy: {descriptor.resolution_policy}")
    print(f"Concept graph: {descriptor.path}")
    return 0


def _query_concept_graph(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    response = query_concept_graph(
        project_root,
        args.query,
        limit=args.limit,
        evidence_limit=args.evidence_limit,
    )
    if args.json:
        print(
            json.dumps(response.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        )
        return 0
    if not response.results:
        print("No resolved concepts matched.")
        return 0
    for result in response.results:
        print(
            f"{result.rank}. {result.canonical_label} [{result.canonical_type}] — "
            f"{result.mentions} mentions, {result.sources} sources, "
            f"{result.claims} claims, {result.relations} relations"
        )
        for support in result.support:
            print(f"   {support.evidence.citation}")
            print(f"   {support.text}")
    return 0


def _report_concept_graph(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    report = concept_graph_status_report(project_root)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    elif report["status"] == "ready":
        counts = report["counts"]
        analysis = report["analysis"]
        print(
            "Concept graph: ready; "
            f"{counts['concepts']} concepts from "
            f"{counts['concept_mentions']} mentions; "
            f"{counts['claims']} claims; {counts['relations']} relations"
        )
        print(f"Analysis run: {analysis['run_id']}")
    else:
        print(f"Concept graph: {report['status']}")
        if "error" in report:
            print(str(report["error"]))
    return 0 if report["status"] == "ready" else 1


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


def _sync_sources(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    config = configure_input_mirror(
        project_root,
        args.input,
        sentence_processor=args.sentence_processor,
        sentence_model=args.sentence_model,
        sentence_device=args.sentence_device,
        target_characters=args.target_characters,
        max_characters=args.max_characters,
        overlap_sentences=args.overlap_sentences,
    )
    if args.configure_only:
        payload = {"configured": True, **config.to_dict()}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(f"Configured authoritative input mirror: {config.input_root}")
            print("No documents were scanned or processed.")
        return 0

    summary = synchronize_input_mirror(
        project_root,
        config,
        progress=_sync_progress_callback(enabled=not args.json),
    )
    if args.json:
        print(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        )
    else:
        _print_sync_summary(summary)
    return 1 if summary.failed_sources else 0


def _build_semantic_index(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    if not SQLiteSearchBackend(project_root).corpus_snapshot().evidence:
        raise SemanticIndexError(
            "semantic_corpus_empty",
            "The exact index has no chunks to embed for semantic retrieval.",
        )
    provider = SentenceTransformersEmbeddingProvider(
        args.embedding_model,
        revision=args.model_revision,
        cache_dir=args.model_cache or model_cache_dir_for(project_root),
        allow_download=args.allow_model_download,
        device=args.device,
        batch_size=args.embedding_batch_size,
        truncate_dimension=args.truncate_dimension,
        show_progress=not args.json,
    )
    descriptor = build_semantic_index(project_root, provider)
    if args.json:
        print(
            json.dumps(
                descriptor.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
        )
        return 0

    embedding = descriptor.embedding
    print(
        f"Semantic index: {descriptor.indexed_chunks} chunks at "
        f"{descriptor.dimension} dimensions; "
        f"{int(descriptor.build['embedded_documents'])} embedded, "
        f"{int(descriptor.build['reused_documents'])} reused."
    )
    print(f"Model: {descriptor.model_id} @ {descriptor.model_revision}")
    print(
        f"Build: {descriptor.build['document_embedding_ms']:.3f} ms; "
        f"{descriptor.build['documents_per_second']:.3f} chunks/second"
    )
    print(
        f"Vectors: {descriptor.vector_size_bytes} bytes; "
        f"database: {descriptor.index_size_bytes} bytes"
    )
    accelerator_memory = embedding["accelerator_peak_memory_allocated_bytes"]
    if accelerator_memory is not None:
        print(f"Accelerator peak allocated memory: {accelerator_memory} bytes")
    print(f"Semantic index: {descriptor.path}")
    return 0


def _search_sources(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    exact_backend = SQLiteSearchBackend(project_root)
    if args.retrieval in {"semantic", "hybrid"}:
        descriptor = read_current_semantic_index_descriptor(project_root)
        provider = _semantic_query_provider(project_root, descriptor, args)
        semantic_backend = PersistentSemanticSearchBackend(exact_backend, provider)
        backend = (
            semantic_backend
            if args.retrieval == "semantic"
            else HybridSearchBackend(exact_backend, semantic_backend)
        )
    else:
        backend = exact_backend
    response = backend.search(
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
        print("No indexed matches.")
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


def _semantic_query_provider(
    project_root: Path,
    descriptor: SemanticIndexDescriptor,
    args: argparse.Namespace,
) -> SentenceTransformersEmbeddingProvider:
    model = args.embedding_model
    if model is None:
        if descriptor.model_id.startswith("local:"):
            raise SemanticIndexError(
                "semantic_model_path_required",
                "This semantic index used a local model directory; pass "
                "--embedding-model with that directory.",
            )
        model = descriptor.model_id
    revision = args.model_revision
    if revision is None and not descriptor.model_id.startswith("local:"):
        revision = descriptor.model_revision
    return SentenceTransformersEmbeddingProvider(
        model,
        revision=revision,
        cache_dir=args.model_cache or model_cache_dir_for(project_root),
        allow_download=args.allow_model_download,
        device=args.device,
        batch_size=args.embedding_batch_size,
        truncate_dimension=descriptor.dimension,
        show_progress=False,
    )


def _evaluate_retrieval(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    dataset = load_evaluation_dataset(args.dataset)
    exact_backend = SQLiteSearchBackend(project_root)
    if args.retrieval == "semantic":
        provider = SentenceTransformersEmbeddingProvider(
            args.embedding_model or DEFAULT_EMBEDDING_MODEL,
            revision=args.model_revision,
            cache_dir=args.model_cache or model_cache_dir_for(project_root),
            allow_download=args.allow_model_download,
            device=args.device,
            batch_size=args.embedding_batch_size,
            truncate_dimension=args.truncate_dimension,
            show_progress=not args.json,
        )
        backend = InMemorySemanticSearchBackend(exact_backend, provider)
        backend_name = "in_memory_dense"
        retrieval_metadata = backend.evaluation_metadata()
        index_size_bytes = (
            exact_backend.path.stat().st_size if exact_backend.path.is_file() else None
        )
    elif args.retrieval == "hybrid":
        descriptor = read_current_semantic_index_descriptor(project_root)
        if (
            args.truncate_dimension is not None
            and args.truncate_dimension != descriptor.dimension
        ):
            raise SemanticIndexError(
                "semantic_model_mismatch",
                "Hybrid evaluation must use the persisted semantic index dimension.",
            )
        provider = _semantic_query_provider(project_root, descriptor, args)
        semantic_backend = PersistentSemanticSearchBackend(exact_backend, provider)
        backend = HybridSearchBackend(exact_backend, semantic_backend)
        backend_name = "sqlite_fts5+persistent_dense_rrf"
        retrieval_metadata = backend.evaluation_metadata()
        index_size_bytes = (
            exact_backend.path.stat().st_size + descriptor.index_size_bytes
        )
    else:
        backend = exact_backend
        backend_name = "sqlite_fts5"
        retrieval_metadata = None
        index_size_bytes = (
            exact_backend.path.stat().st_size if exact_backend.path.is_file() else None
        )
    report = evaluate_retrieval(
        dataset,
        backend,
        limit=args.limit,
        verify=not args.no_verify,
        backend_name=backend_name,
        retrieval_mode=args.retrieval,
        index_size_bytes=index_size_bytes,
        retrieval_metadata=retrieval_metadata,
    )
    if retrieval_metadata is not None:
        retrieval_metadata = backend.evaluation_metadata()
        report = replace(report, retrieval_metadata=retrieval_metadata)
    if args.json:
        print(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        )
        return 0

    summary = report.summary
    print(f"Dataset: {dataset.name} ({summary.cases} cases)")
    print(
        f"Retrieval: {report.retrieval_mode} via {report.backend_name}; limit {report.limit}"
    )
    print(f"Recall@{report.limit}: {summary.recall_at_k:.3f}")
    print(f"MRR@{report.limit}: {summary.mean_reciprocal_rank_at_k:.3f}")
    print(f"Locator accuracy: {_format_optional_rate(summary.locator_accuracy)}")
    print(f"Source verification: {_format_optional_rate(summary.verification_rate)}")
    print(
        "Search latency: "
        f"p50 {summary.latency_ms.p50:.3f} ms; "
        f"p95 {summary.latency_ms.p95:.3f} ms"
    )
    if report.index_size_bytes is not None:
        print(f"Index size: {report.index_size_bytes} bytes")
    if report.process_peak_rss_bytes is not None:
        print(f"Process peak RSS: {report.process_peak_rss_bytes} bytes")
    if retrieval_metadata is not None:
        embedding = retrieval_metadata["embedding"]
        semantic_index = retrieval_metadata["semantic_index"]
        print(
            "Embedding: "
            f"{embedding['model_id']} @ {embedding['model_revision']}; "
            f"{embedding['dimension']} dimensions on {embedding['device']}"
        )
        print(
            "Semantic build: "
            f"{semantic_index['documents']} documents in "
            f"{semantic_index['document_embedding_ms']:.3f} ms; "
            f"{semantic_index['vector_size_bytes']} vector bytes"
        )
        accelerator_memory = embedding["accelerator_peak_memory_allocated_bytes"]
        if accelerator_memory is not None:
            print(f"Accelerator peak allocated memory: {accelerator_memory} bytes")
        fusion = retrieval_metadata.get("fusion")
        if fusion is not None:
            print(
                "Fusion: "
                f"{fusion['algorithm']}; {fusion['candidate_limit']} candidates; "
                f"RRF k={fusion['rrf_k']}"
            )
    print("Categories:")
    for category, metrics in report.by_category:
        print(
            f"  {category}: {metrics.cases} cases; "
            f"recall {metrics.recall_at_k:.3f}; "
            f"MRR {metrics.mean_reciprocal_rank_at_k:.3f}"
        )
    return 0


def _analysis_provider(
    project_root: Path, args: argparse.Namespace
) -> StructuredExtractionProvider:
    if not args.allow_model_download:
        # The command-line flag is the sole download authority. An inherited
        # environment value must never silently broaden it before SaT is loaded.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.analysis_runtime == "vllm":
        _configure_vllm_environment(offline=not args.allow_model_download)
    support_unit_processor = sentence_processor_from(
        args.support_unit_processor,
        model_name=args.support_unit_model,
    )
    if args.analysis_runtime == "vllm":
        if args.quantization != "none":
            raise AnalysisModelError(
                "analysis_vllm_quantization_invalid",
                "The vLLM runtime does not use Transformers bitsandbytes options; "
                "set --quantization none.",
            )
        return VLLMStructuredExtractionProvider(
            args.analysis_model,
            revision=args.model_revision,
            cache_dir=args.model_cache or model_cache_dir_for(project_root),
            allow_download=args.allow_model_download,
            device=args.device,
            dtype=args.dtype,
            structured_output=args.structured_output,
            batch_size=(
                args.analysis_batch_size
                if args.analysis_batch_size is not None
                else DEFAULT_VLLM_ANALYSIS_BATCH_SIZE
            ),
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            prompt_style=args.prompt_style,
            support_unit_processor=support_unit_processor,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        )
    return TransformersStructuredExtractionProvider(
        args.analysis_model,
        revision=args.model_revision,
        cache_dir=args.model_cache or model_cache_dir_for(project_root),
        allow_download=args.allow_model_download,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        structured_output=args.structured_output,
        batch_size=(
            args.analysis_batch_size
            if args.analysis_batch_size is not None
            else DEFAULT_ANALYSIS_BATCH_SIZE
        ),
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        prompt_style=args.prompt_style,
        support_unit_processor=support_unit_processor,
    )


def _analyze_sources(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    if not args.allow_model_download:
        # Automatic resume scans can load SaT before the analysis provider is
        # constructed. Enforce the same cache-only policy for that earlier stage.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    _synchronize_before_analysis(project_root, args)
    if not SQLiteSearchBackend(project_root).corpus_snapshot().evidence:
        raise AnalysisStoreError(
            "analysis_scope_empty",
            "The synchronized exact index contains no evidence chunks to analyze.",
        )
    provider = _analysis_provider(project_root, args)
    progress = None
    if not args.json:
        last_reported = [-1]

        def report_progress(done: int, total: int, descriptor: object) -> None:
            interval = max(1, total // 20)
            if done in {0, total} or done // interval > last_reported[0] // interval:
                print(
                    f"Analysis progress: {done}/{total} evidence chunks",
                    file=sys.stderr,
                )
                last_reported[0] = done

        progress = report_progress
    descriptor = run_corpus_analysis(
        str(project_root),
        provider,
        source_ids=args.source,
        limit=args.limit,
        resume=not args.no_resume,
        progress=progress,
    )
    if args.json:
        print(
            json.dumps(
                descriptor.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
            )
        )
        return 0
    print(f"Analysis run: {descriptor.run_id} ({descriptor.status})")
    print(
        f"Evidence: {descriptor.analyzed_evidence}; "
        f"{descriptor.accepted_evidence} accepted, "
        f"{descriptor.partial_evidence} partial, "
        f"{descriptor.empty_evidence} empty, "
        f"{descriptor.rejected_evidence} rejected"
    )
    print(
        f"Candidates: {descriptor.concepts} concepts, "
        f"{descriptor.claims} claims, {descriptor.relations} relations; "
        f"{descriptor.rejected_candidates} rejected during validation"
    )
    print(
        f"Generation: {descriptor.output_tokens} tokens; "
        f"{descriptor.truncated_evidence} truncated responses"
    )
    print(f"Inference: {descriptor.inference_ms:.3f} ms")
    print(f"Analysis database: {descriptor.path}")
    return 0


def _evaluate_analysis(args: argparse.Namespace) -> int:
    project_root = _resolve_project_root(args.project)
    benchmark = load_analysis_benchmark(args.dataset)
    provider = _analysis_provider(project_root, args)
    report = evaluate_analysis_benchmark(benchmark, provider)
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    summary = payload["summary"]
    assert isinstance(summary, dict)
    print(f"Benchmark: {benchmark.benchmark_id} ({len(benchmark.cases)} cases)")
    print(
        f"Model: {provider.info.model_id} @ {provider.info.model_revision}; "
        f"{provider.info.device}; {provider.info.dtype}"
    )
    print(f"Valid responses: {float(summary['valid_response_rate']):.3f}")
    print(
        "Fully grounded responses: "
        f"{float(summary['fully_grounded_response_rate']):.3f}"
    )
    print(f"Exact cases: {float(summary['exact_case_rate']):.3f}")
    print(f"Macro candidate F1: {float(summary['macro_candidate_f1']):.3f}")
    for category in ("concepts", "claims", "relations"):
        metrics = summary[category]
        assert isinstance(metrics, dict)
        print(
            f"{category.title()}: precision {float(metrics['precision']):.3f}; "
            f"recall {float(metrics['recall']):.3f}; "
            f"F1 {float(metrics['f1']):.3f}"
        )
    latency = summary["latency_ms"]
    assert isinstance(latency, dict)
    print(
        f"Inference latency: p50 {float(latency['p50']):.3f} ms; "
        f"p95 {float(latency['p95']):.3f} ms"
    )
    peak = provider.info.accelerator_peak_memory_allocated_bytes
    if peak is not None:
        print(f"Accelerator peak allocated memory: {peak} bytes")
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
            device=args.sentence_device,
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
    semantic_index_report = semantic_index_status_report(project_root)
    derived_analysis_report = analysis_status_report(project_root)
    concept_graph_report = concept_graph_status_report(project_root)
    report = {
        "extraction": extraction_report,
        "chunking": chunk_report,
        "index": search_index_report,
        "semantic_index": semantic_index_report,
        "analysis": derived_analysis_report,
        "concept_graph": concept_graph_report,
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
        if semantic_index_report["status"] == "ready":
            print(
                "Semantic index: ready; "
                f"{semantic_index_report['chunks']} chunks; "
                f"{semantic_index_report['dimension']} dimensions; "
                f"{semantic_index_report['model_id']}"
            )
        else:
            print(f"Semantic index: {semantic_index_report['status']}")
        if derived_analysis_report["status"] == "ready":
            print(
                "Derived analysis: ready; "
                f"{derived_analysis_report['analyzed_evidence_records']} evidence records; "
                f"{derived_analysis_report['claims']} claims; "
                f"{derived_analysis_report['relations']} relations"
            )
            latest_run = derived_analysis_report.get("latest_run")
            if isinstance(latest_run, dict):
                counts = latest_run.get("counts", {})
                scope = latest_run.get("scope", {})
                if isinstance(counts, dict) and isinstance(scope, dict):
                    print(
                        "Latest analysis run: "
                        f"{latest_run.get('status')}; "
                        f"{counts.get('analyzed_evidence', 0)}/"
                        f"{scope.get('selected_evidence', '?')} evidence chunks; "
                        f"{latest_run.get('run_id')}"
                    )
        else:
            print(f"Derived analysis: {derived_analysis_report['status']}")
        if concept_graph_report["status"] == "ready":
            counts = concept_graph_report["counts"]
            print(
                "Concept graph: ready; "
                f"{counts['concepts']} resolved concepts; "
                f"{counts['claims']} claims; {counts['relations']} relations"
            )
        else:
            print(f"Concept graph: {concept_graph_report['status']}")
    unhealthy = ("failed", "pending", "stale")
    return (
        1
        if (
            search_index_report["status"] != "ready"
            or semantic_index_report["status"] not in {"missing", "ready"}
            or derived_analysis_report["status"] not in {"missing", "ready"}
            or concept_graph_report["status"] not in {"missing", "ready"}
            or any(
                report_part["statuses"][status]
                for report_part in (extraction_report, chunk_report)
                for status in unhealthy
            )
        )
        else 0
    )


def _synchronize_before_analysis(
    project_root: Path, args: argparse.Namespace
) -> SynchronizationSummary | None:
    config = load_pipeline_config(project_root)
    if args.input is not None:
        settings = config or PipelineConfig(input_root=str(args.input))
        config = configure_input_mirror(
            project_root,
            args.input,
            sentence_processor=settings.sentence_processor,
            sentence_model=settings.sentence_model,
            sentence_device=settings.sentence_device,
            target_characters=settings.target_characters,
            max_characters=settings.max_characters,
            overlap_sentences=settings.overlap_sentences,
        )
    if config is None:
        return None
    summary = synchronize_input_mirror(
        project_root,
        config,
        progress=_sync_progress_callback(enabled=not args.json),
    )
    if not args.json:
        _print_sync_summary(summary, file=sys.stderr)
    return summary


def _sync_progress_callback(*, enabled: bool) -> Callable[[str, int, int], None] | None:
    if not enabled:
        return None
    last_reported = [-1]

    def report(stage: str, done: int, total: int) -> None:
        interval = max(1, total // 20)
        if done in {0, total} or done // interval > last_reported[0] // interval:
            print(f"Input {stage}: {done}/{total} sources", file=sys.stderr)
            last_reported[0] = done

    return report


def _print_sync_summary(
    summary: SynchronizationSummary, *, file: object | None = None
) -> None:
    output = file or sys.stdout
    print(
        f"Input scan: {summary.scanned_paths} files, "
        f"{summary.unique_sources} unique sources; "
        f"{summary.added_sources} added, {summary.removed_sources} removed.",
        file=output,
    )
    print(
        f"Extraction: {summary.extracted_sources} processed, "
        f"{summary.reused_extractions} reused; "
        f"chunking: {summary.chunked_sources} processed, "
        f"{summary.reused_chunks} reused.",
        file=output,
    )
    print(
        f"Exact index: {'rebuilt' if summary.exact_index_rebuilt else 'reused'}; "
        f"{summary.indexed_chunks} chunks.",
        file=output,
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


def _format_optional_rate(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (
        ManifestError,
        ExtractionError,
        ChunkingError,
        RetrievalError,
        EvaluationError,
        EmbeddingError,
        SemanticIndexError,
        AnalysisModelError,
        AnalysisStoreError,
        AnalysisEvaluationError,
        ConceptGraphError,
    ) as error:
        print(f"corpusdock: error: {error}", file=sys.stderr)
        return 1
