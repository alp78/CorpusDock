from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from corpusdock.analysis_models import AnalysisModelError
from corpusdock.analysis_vllm import (
    DEFAULT_VLLM_ANALYSIS_BATCH_SIZE,
    VLLMStructuredExtractionProvider,
    _configure_vllm_environment,
    _xgrammar_analysis_schema,
)


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def test_vllm_default_batch_is_safe_for_a_sixteen_gigabyte_gpu() -> None:
    assert DEFAULT_VLLM_ANALYSIS_BATCH_SIZE == 16


def test_xgrammar_schema_removes_only_decoder_unsupported_uniqueness_hint() -> None:
    schema = _xgrammar_analysis_schema()

    assert not _contains_key(schema, "uniqueItems")
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["schema_version", "concepts", "claims", "relations"]


def test_vllm_environment_is_local_and_respects_explicit_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_NO_USAGE_STATS", "0")
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")

    _configure_vllm_environment()

    assert __import__("os").environ["VLLM_NO_USAGE_STATS"] == "1"
    assert __import__("os").environ["DO_NOT_TRACK"] == "1"
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
    assert __import__("os").environ["VLLM_USE_FLASHINFER_SAMPLER"] == "1"


def test_vllm_rejects_non_cuda_before_loading_dependencies(tmp_path: Path) -> None:
    with pytest.raises(AnalysisModelError) as captured:
        VLLMStructuredExtractionProvider(tmp_path, device="cpu")

    assert captured.value.code == "analysis_vllm_device_invalid"


def test_vllm_provider_extracts_batched_grounding_units_without_remote_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    captured: dict[str, Any] = {}

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def get_device_name(_device: int) -> str:
            return "Fixture GPU"

        @staticmethod
        def synchronize() -> None:
            captured["synchronized"] = True

    class FakeTokenizer:
        def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
            assert messages
            assert kwargs["tokenize"] is True
            assert kwargs["enable_thinking"] is False
            return [1, 2, 3]

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            captured["engine"] = kwargs
            self.llm_engine = SimpleNamespace(
                model_config=SimpleNamespace(dtype="torch.bfloat16")
            )

        def get_tokenizer(self) -> FakeTokenizer:
            return FakeTokenizer()

        def chat(
            self, messages: list[object], sampling: object, **kwargs: object
        ) -> list[object]:
            captured["messages"] = messages
            captured["sampling"] = sampling
            captured["chat"] = kwargs
            raw = json.dumps(
                {"schema_version": 3, "concepts": [], "claims": [], "relations": []}
            )
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text=raw,
                            token_ids=[7, 8, 9],
                            finish_reason="stop",
                        )
                    ]
                )
                for _ in messages
            ]

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.options = kwargs

    class FakeStructuredOutputsParams:
        def __init__(self, **kwargs: object) -> None:
            self.options = kwargs

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="13.0"),
        __version__="2.13.0+cu130",
    )
    monkeypatch.setattr(
        "corpusdock.analysis_vllm._vllm_dependencies",
        lambda: {
            "torch": fake_torch,
            "LLM": FakeLLM,
            "SamplingParams": FakeSamplingParams,
            "StructuredOutputsParams": FakeStructuredOutputsParams,
            "model_info": lambda **_kwargs: None,
            "snapshot_download": lambda **_kwargs: str(tmp_path),
        },
    )
    times = iter((1.0, 2.0, 3.0, 5.0))
    provider = VLLMStructuredExtractionProvider(
        tmp_path,
        device="cuda",
        dtype="bfloat16",
        batch_size=2,
        max_input_tokens=4096,
        max_output_tokens=512,
        clock=lambda: next(times),
    )

    outputs = provider.extract(("First sentence. Second sentence.", "Another source."))

    assert provider.info.provider == "local_vllm"
    assert provider.info.runtime == "vllm"
    assert provider.info.dtype == "bfloat16"
    assert provider.info.remote_code_trusted is False
    assert provider.info.download_allowed is False
    assert provider.info.thinking_enabled is False
    assert provider.info.engine_performance_mode == "throughput"
    assert provider.info.prefix_caching_enabled is True
    assert provider.info.gpu_memory_utilization == 0.9
    assert provider.info.structured_output_backend == "xgrammar"
    assert provider.info.sampling_backend == "vllm-native"
    assert captured["engine"] == {
        "model": str(tmp_path.resolve()),
        "dtype": "bfloat16",
        "max_model_len": 4608,
        "gpu_memory_utilization": 0.9,
        "language_model_only": True,
        "trust_remote_code": False,
        "max_num_seqs": 2,
        "enable_prefix_caching": True,
        "disable_log_stats": True,
        "performance_mode": "throughput",
        "seed": 0,
        "structured_outputs_config": {"backend": "xgrammar"},
    }
    assert captured["chat"] == {
        "use_tqdm": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    assert captured["synchronized"] is True
    assert len(outputs) == 2
    assert all(output.output_tokens == 3 for output in outputs)
    assert all(output.truncated is False for output in outputs)
    assert all(output.evidence_units for output in outputs)
    assert outputs[0].inference_ms == 1000.0
