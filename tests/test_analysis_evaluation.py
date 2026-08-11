from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from corpusdock.analysis_contracts import ANALYSIS_PROMPT_VERSION
from corpusdock.analysis_evaluation import (
    evaluate_analysis_benchmark,
    load_analysis_benchmark,
)
from corpusdock.analysis_models import (
    DEFAULT_ANALYSIS_MODEL,
    ModelExtraction,
    StructuredExtractionModelInfo,
    analysis_prompt_sha256,
)
from corpusdock.analysis_vllm import DEFAULT_VLLM_ANALYSIS_BATCH_SIZE


BENCHMARK_ROOT = Path(__file__).parents[1] / "benchmarks" / "analysis-v1"


class _FixtureProvider:
    def __init__(self, raw_output: str) -> None:
        self._raw_output = raw_output
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
            load_ms=1.0,
        )

    def extract(self, texts):  # type: ignore[no-untyped-def]
        return tuple(ModelExtraction(self._raw_output, 4.0) for _ in texts)


def test_analysis_baseline_is_bound_to_dataset_prompt_and_default() -> None:
    baseline = json.loads(
        (BENCHMARK_ROOT / "expected-results.json").read_text(encoding="utf-8")
    )
    dataset_digest = sha256((BENCHMARK_ROOT / "cases.json").read_bytes()).hexdigest()
    profile = baseline["selected_profile"]

    assert baseline["dataset_sha256"] == dataset_digest
    assert profile["model_id"] == DEFAULT_ANALYSIS_MODEL
    assert profile["prompt_version"] == ANALYSIS_PROMPT_VERSION
    assert profile["prompt_sha256"] == analysis_prompt_sha256("chat")
    assert profile["batch_size"] == DEFAULT_VLLM_ANALYSIS_BATCH_SIZE
    for metric, minimum in baseline["quality_gates"].items():
        assert baseline["observed"][metric] >= minimum


def _write_one_case(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark_id": "fixture-analysis",
                "description": "A tiny project-authored evaluator fixture.",
                "cases": [
                    {
                        "case_id": "fixture-causal",
                        "language": "en",
                        "text": "The zircon clasp reduces crate vibration.",
                        "expected": {
                            "concepts": [
                                {
                                    "gold_id": "clasp",
                                    "support_contains": "zircon clasp",
                                    "label_aliases": ["zircon clasp"],
                                },
                                {
                                    "gold_id": "vibration",
                                    "support_contains": "crate vibration",
                                    "label_aliases": ["crate vibration"],
                                },
                            ],
                            "claims": [
                                {
                                    "gold_id": "claim",
                                    "support_contains": "reduces crate vibration",
                                    "claim_type": "causal",
                                    "polarity": "affirmed",
                                    "certainty": "asserted",
                                    "conditional": False,
                                    "attribution": "source",
                                    "normative_force": "none",
                                }
                            ],
                            "relations": [
                                {
                                    "gold_id": "relation",
                                    "support_contains": "reduces crate vibration",
                                    "subject_gold_id": "clasp",
                                    "relation_type": "inhibits",
                                    "object_gold_id": "vibration",
                                    "polarity": "affirmed",
                                    "certainty": "asserted",
                                    "conditional": False,
                                    "attribution": "source",
                                    "normative_force": "none",
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _perfect_output() -> str:
    return json.dumps(
        {
            "schema_version": 3,
            "concepts": [
                {
                    "local_id": "c1",
                    "label": "zircon clasp",
                    "description": "A clasp.",
                    "concept_type": "component",
                    "confidence": 0.9,
                    "support": {"unit_ids": ["u1"]},
                },
                {
                    "local_id": "c2",
                    "label": "crate vibration",
                    "description": "Movement of a crate.",
                    "concept_type": "effect",
                    "confidence": 0.9,
                    "support": {"unit_ids": ["u1"]},
                },
            ],
            "claims": [
                {
                    "local_id": "q1",
                    "statement": "The clasp reduces vibration.",
                    "claim_type": "causal",
                    "polarity": "affirmed",
                    "certainty": "asserted",
                    "conditional": False,
                    "attribution": "source",
                    "normative_force": "none",
                    "confidence": 0.9,
                    "support": {"unit_ids": ["u1"]},
                    "concept_ids": ["c1", "c2"],
                }
            ],
            "relations": [
                {
                    "local_id": "r1",
                    "subject_concept_id": "c1",
                    "relation_type": "inhibits",
                    "predicate": "reduces",
                    "object_concept_id": "c2",
                    "claim_local_id": "q1",
                    "confidence": 0.9,
                }
            ],
        }
    )


def test_public_analysis_benchmark_loads_strict_generic_cases() -> None:
    benchmark = load_analysis_benchmark(BENCHMARK_ROOT / "cases.json")

    assert benchmark.benchmark_id == "corpusdock-analysis-v1"
    assert len(benchmark.cases) == 9
    assert {case.language for case in benchmark.cases} == {"en", "de", "es"}
    assert len(benchmark.sha256) == 64


def test_analysis_evaluator_scores_candidates_without_leaking_content(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "cases.json"
    _write_one_case(dataset_path)
    report = evaluate_analysis_benchmark(
        load_analysis_benchmark(dataset_path), _FixtureProvider(_perfect_output())
    )
    payload = report.to_dict()

    assert payload["summary"]["valid_response_rate"] == 1.0
    assert payload["summary"]["fully_grounded_response_rate"] == 1.0
    assert payload["summary"]["exact_case_rate"] == 1.0
    assert payload["summary"]["macro_candidate_f1"] == 1.0
    assert payload["summary"]["latency_ms"]["p50"] == 4.0
    serialized = json.dumps(payload)
    assert "zircon clasp" not in serialized
    assert "raw_output" not in serialized
    assert "source_path" not in serialized
    assert "prompt" not in payload["extractor"]


def test_analysis_evaluator_counts_a_malformed_response_without_echoing_it(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "cases.json"
    _write_one_case(dataset_path)

    payload = evaluate_analysis_benchmark(
        load_analysis_benchmark(dataset_path), _FixtureProvider("READY")
    ).to_dict()

    assert payload["summary"]["valid_response_rate"] == 0.0
    assert payload["summary"]["accepted_candidate_rate"] == 0.0
    assert payload["cases"][0]["rejection_codes"] == ["analysis_json_invalid"]
    assert "READY" not in json.dumps(payload)
