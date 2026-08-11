from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpusdock.analysis_models import (
    AnalysisModelError,
    ModelExtraction,
    StructuredExtractionModelInfo,
)
from corpusdock.analysis_runner import run_corpus_analysis
from corpusdock.analysis_store import AnalysisStore, analysis_status_report
from corpusdock.chunking import (
    RuleSentenceProcessor,
    chunk_extraction_artifact,
    write_chunk_artifact,
)
from corpusdock.extraction import extract_source, write_extraction_artifact
from corpusdock.manifest import ManifestStore
from corpusdock.retrieval import build_search_index


TIMESTAMP = "2026-08-11T12:00:00Z"
EMPTY_OUTPUT = json.dumps(
    {"schema_version": 3, "concepts": [], "claims": [], "relations": []}
)


class _ResumableProvider:
    def __init__(
        self, *, fail_on_call: int | None = None, load_ms: float = 1.0
    ) -> None:
        self._fail_on_call = fail_on_call
        self._calls = 0
        self.info = StructuredExtractionModelInfo(
            provider="fixture_local",
            runtime="fixture-runtime",
            runtime_version="1",
            model_id="project-authored-fixture",
            model_revision="fixture-v1",
            model_fingerprint="sha256:" + "a" * 64,
            model_size_bytes=1,
            prompt_style="chat",
            prompt_version="analysis-extraction-v1",
            prompt_sha256="b" * 64,
            max_input_tokens=2048,
            max_output_tokens=512,
            batch_size=1,
            device="cpu",
            dtype="float32",
            quantization="none",
            quantization_runtime=None,
            quantization_runtime_version=None,
            structured_output="json-schema",
            structured_output_runtime_version="fixture-1",
            support_unit_processor="corpusdock.rule_sentence",
            support_unit_processor_version="fixture-1",
            support_unit_model="none",
            remote_code_trusted=False,
            download_allowed=False,
            deterministic=True,
            thinking_enabled=False,
            load_ms=load_ms,
        )

    def extract(self, texts):  # type: ignore[no-untyped-def]
        self._calls += 1
        if self._fail_on_call == self._calls:
            raise AnalysisModelError("fixture_failure", "Synthetic local failure.")
        return tuple(ModelExtraction(EMPTY_OUTPUT, 1.0) for _ in texts)


def _build_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "project"
    project_root.mkdir()
    store = ManifestStore(project_root, now=lambda: TIMESTAMP)
    for number in (1, 2):
        path = tmp_path / f"source-{number}.txt"
        path.write_text(
            f"Project-authored source sentence {number}.\n", encoding="utf-8"
        )
        registration = store.register([path])[0]
        extraction = extract_source(
            registration.source, registration.source_path, now=lambda: TIMESTAMP
        )
        write_extraction_artifact(project_root, extraction)
        chunks = chunk_extraction_artifact(
            extraction.to_dict(), RuleSentenceProcessor(), now=lambda: TIMESTAMP
        )
        write_chunk_artifact(project_root, chunks)
    build_search_index(project_root, now=lambda: TIMESTAMP)
    return project_root


def test_analysis_runner_resumes_after_committed_batch_without_reprocessing(
    tmp_path: Path,
) -> None:
    project_root = _build_project(tmp_path)

    with pytest.raises(AnalysisModelError, match="Synthetic"):
        run_corpus_analysis(
            str(project_root), _ResumableProvider(fail_on_call=2, load_ms=1.0)
        )

    interrupted_status = analysis_status_report(project_root)
    assert interrupted_status["runs"]["running"] == 1
    assert interrupted_status["analyzed_evidence_records"] == 1

    completed = run_corpus_analysis(str(project_root), _ResumableProvider(load_ms=99.0))

    assert completed.status == "completed"
    assert completed.analyzed_evidence == 2
    assert completed.empty_evidence == 2
    assert completed.inference_ms == 2.0
    assert completed.extractor["load_ms"] == 99.0
    assert analysis_status_report(project_root)["runs"] == {
        "running": 0,
        "completed": 1,
        "failed": 0,
    }


def test_analysis_runner_validates_source_scope_and_limit(tmp_path: Path) -> None:
    project_root = _build_project(tmp_path)
    source_id = AnalysisStore(project_root).snapshot.source_ids[0]

    completed = run_corpus_analysis(
        str(project_root),
        _ResumableProvider(),
        source_ids=[source_id, source_id],
        limit=1,
        resume=False,
    )

    assert completed.analyzed_evidence == 1
    assert completed.scope["source_ids"] == [source_id]
    assert completed.scope["limit"] == 1


def test_analysis_resume_reuses_stable_evidence_after_corpus_growth(
    tmp_path: Path,
) -> None:
    project_root = _build_project(tmp_path)
    first_provider = _ResumableProvider(fail_on_call=2)
    with pytest.raises(AnalysisModelError):
        run_corpus_analysis(str(project_root), first_provider)
    interrupted = analysis_status_report(project_root)["latest_run"]
    assert interrupted["counts"]["analyzed_evidence"] == 1

    third = tmp_path / "source-3.txt"
    third.write_text("Project-authored source sentence 3.\n", encoding="utf-8")
    registration = ManifestStore(project_root).register([third])[0]
    extraction = extract_source(registration.source, registration.source_path)
    write_extraction_artifact(project_root, extraction)
    chunks = chunk_extraction_artifact(extraction.to_dict(), RuleSentenceProcessor())
    write_chunk_artifact(project_root, chunks)
    build_search_index(project_root)

    resumed_provider = _ResumableProvider()
    completed = run_corpus_analysis(str(project_root), resumed_provider)

    assert completed.run_id == interrupted["run_id"]
    assert completed.analyzed_evidence == 3
    assert completed.empty_evidence == 3
    assert resumed_provider._calls == 2
