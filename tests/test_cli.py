from __future__ import annotations

from hashlib import sha256
import json

import pytest

from corpusdock.analysis_models import AnalysisModelError
from corpusdock.cli import _analysis_provider, build_parser, main
from corpusdock.embeddings import DEFAULT_EMBEDDING_MODEL
from corpusdock.manifest import ManifestStore
from corpusdock.retrieval import index_status_report


def test_search_parser_accepts_json_output() -> None:
    args = build_parser().parse_args(
        ["search", "citation anchors", "--json", "--limit", "4", "--match", "any"]
    )

    assert args.command == "search"
    assert args.query == "citation anchors"
    assert args.json is True
    assert args.limit == 4
    assert args.match == "any"
    assert args.retrieval == "lexical"
    assert args.embedding_model is None


def test_embed_parser_accepts_cuda_model_options() -> None:
    args = build_parser().parse_args(
        [
            "embed",
            "--embedding-model",
            "Qwen/Qwen3-Embedding-0.6B",
            "--model-revision",
            "a" * 40,
            "--device",
            "cuda",
            "--embedding-batch-size",
            "32",
            "--allow-model-download",
            "--json",
        ]
    )

    assert args.command == "embed"
    assert args.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert args.model_revision == "a" * 40
    assert args.device == "cuda"
    assert args.embedding_batch_size == 32
    assert args.allow_model_download is True
    assert args.json is True


def test_eval_parser_accepts_local_baseline_options() -> None:
    args = build_parser().parse_args(
        ["eval", "judgments.json", "--json", "--limit", "5", "--no-verify"]
    )

    assert args.command == "eval"
    assert args.dataset == "judgments.json"
    assert args.json is True
    assert args.limit == 5
    assert args.no_verify is True
    assert args.embedding_model is None


def test_eval_parser_accepts_explicit_local_semantic_options() -> None:
    args = build_parser().parse_args(
        [
            "eval",
            "judgments.json",
            "--retrieval",
            "semantic",
            "--embedding-model",
            "Qwen/Qwen3-Embedding-0.6B",
            "--model-revision",
            "0123456789abcdef",
            "--allow-model-download",
            "--device",
            "cuda",
            "--embedding-batch-size",
            "8",
            "--truncate-dimension",
            "256",
        ]
    )

    assert args.retrieval == "semantic"
    assert args.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert args.model_revision == "0123456789abcdef"
    assert args.allow_model_download is True
    assert args.device == "cuda"
    assert args.embedding_batch_size == 8
    assert args.truncate_dimension == 256


def test_search_and_eval_parsers_accept_hybrid_retrieval() -> None:
    search = build_parser().parse_args(
        ["search", "operational knowledge", "--retrieval", "hybrid"]
    )
    evaluate = build_parser().parse_args(
        ["eval", "judgments.json", "--retrieval", "hybrid"]
    )

    assert search.retrieval == "hybrid"
    assert search.embedding_model is None
    assert evaluate.retrieval == "hybrid"
    assert evaluate.embedding_model is None


def test_analysis_parsers_accept_local_cuda_options() -> None:
    analyze = build_parser().parse_args(
        [
            "analyze",
            "--analysis-model",
            "numind/NuExtract3",
            "--model-revision",
            "a" * 40,
            "--source",
            "src-" + "b" * 64,
            "--limit",
            "12",
            "--device",
            "cuda",
            "--dtype",
            "bfloat16",
            "--quantization",
            "bnb-4bit",
            "--structured-output",
            "json-schema",
            "--analysis-batch-size",
            "2",
            "--no-resume",
            "--json",
        ]
    )
    evaluate = build_parser().parse_args(
        ["analysis-eval", "cases.json", "--device", "cuda", "--json"]
    )

    assert analyze.analysis_model == "numind/NuExtract3"
    assert analyze.model_revision == "a" * 40
    assert analyze.limit == 12
    assert analyze.device == "cuda"
    assert analyze.dtype == "bfloat16"
    assert analyze.quantization == "bnb-4bit"
    assert analyze.structured_output == "json-schema"
    assert analyze.support_unit_processor == "sat"
    assert analyze.support_unit_model == "sat-12l-sm"
    assert analyze.analysis_batch_size == 2
    assert analyze.no_resume is True
    assert evaluate.dataset == "cases.json"
    assert evaluate.device == "cuda"


def test_analysis_parser_accepts_high_throughput_vllm_runtime() -> None:
    args = build_parser().parse_args(
        [
            "analyze",
            "--analysis-runtime",
            "vllm",
            "--device",
            "cuda",
            "--vllm-gpu-memory-utilization",
            "0.85",
        ]
    )

    assert args.analysis_runtime == "vllm"
    assert args.analysis_batch_size is None
    assert args.vllm_gpu_memory_utilization == 0.85


def test_vllm_cli_forces_offline_mode_and_uses_cache_safe_default(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    args = build_parser().parse_args(
        ["analyze", "--analysis-runtime", "vllm", "--device", "cuda"]
    )
    captured = {}
    unit_processor = object()

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setattr(
        "corpusdock.cli.sentence_processor_from",
        lambda *_args, **_kwargs: unit_processor,
    )

    def fake_provider(*provider_args, **provider_kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = provider_args
        captured["kwargs"] = provider_kwargs
        return object()

    monkeypatch.setattr(
        "corpusdock.cli.VLLMStructuredExtractionProvider", fake_provider
    )

    provider = _analysis_provider(tmp_path, args)

    assert provider is not None
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
    assert captured["kwargs"]["batch_size"] == 16
    assert captured["kwargs"]["support_unit_processor"] is unit_processor


def test_ingest_parser_accepts_multiple_text_sources() -> None:
    args = build_parser().parse_args(["ingest", "one.pdf", "two.epub"])

    assert args.path == ["one.pdf", "two.epub"]
    assert args.sentence_device == "cpu"
    assert not hasattr(args, "ocr")


def test_sync_parser_and_analysis_input_option_are_authoritative() -> None:
    sync = build_parser().parse_args(
        [
            "sync",
            "books",
            "--sentence-processor",
            "sat",
            "--sentence-device",
            "cuda",
            "--configure-only",
        ]
    )
    analyze = build_parser().parse_args(["analyze", "--input", "books"])

    assert sync.input == "books"
    assert sync.sentence_processor == "sat"
    assert sync.sentence_device == "cuda"
    assert sync.configure_only is True
    assert analyze.input == "books"


def test_analysis_scans_configured_input_before_loading_the_model(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    project = tmp_path / "project"
    mirror = tmp_path / "input"
    mirror.mkdir()
    (mirror / "first.txt").write_text("First local source.\n", encoding="utf-8")
    assert main(["init", str(project)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "sync",
                str(mirror),
                "--project",
                str(project),
                "--sentence-processor",
                "rule",
                "--configure-only",
            ]
        )
        == 0
    )
    capsys.readouterr()
    (mirror / "second.txt").write_text("Second local source.\n", encoding="utf-8")

    def assert_synced_then_stop(project_root, args):  # type: ignore[no-untyped-def]
        assert len(ManifestStore(project_root).load().sources) == 2
        assert index_status_report(project_root)["status"] == "ready"
        assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
        assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
        assert __import__("os").environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
        raise AnalysisModelError("fixture_stop", "Model loading stopped by fixture.")

    monkeypatch.setattr("corpusdock.cli._analysis_provider", assert_synced_then_stop)
    assert main(["analyze", "--project", str(project), "--json"]) == 1
    assert "Model loading stopped by fixture" in capsys.readouterr().err


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
