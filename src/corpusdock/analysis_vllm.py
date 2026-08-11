"""Optional high-throughput local CUDA extraction through vLLM."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Literal

from corpusdock.analysis_contracts import (
    ANALYSIS_PROMPT_VERSION,
    analysis_json_schema,
    evidence_units_for,
)
from corpusdock.analysis_models import (
    DEFAULT_ANALYSIS_MAX_INPUT_TOKENS,
    DEFAULT_ANALYSIS_MAX_OUTPUT_TOKENS,
    DEFAULT_ANALYSIS_MODEL,
    MAX_ANALYSIS_BATCH_SIZE,
    MAX_ANALYSIS_INPUT_TOKENS,
    MAX_ANALYSIS_OUTPUT_TOKENS,
    AnalysisModelError,
    ModelExtraction,
    StructuredExtractionModelInfo,
    StructuredOutput,
    _REVISION_PATTERN,
    _analysis_texts,
    _bounded_integer,
    _reject_repository_code,
    analysis_prompt_sha256,
    chat_messages_for,
)
from corpusdock.chunking import RuleSentenceProcessor, SentenceProcessor
from corpusdock.embeddings import EmbeddingError, _directory_size, _resolve_model


# Sixteen keeps the configured 6,144-token request ceiling inside the KV-cache
# concurrency available on common 16 GB NVIDIA cards. Larger GPUs can opt into 32.
DEFAULT_VLLM_ANALYSIS_BATCH_SIZE = 16
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION = 0.9
MIN_VLLM_GPU_MEMORY_UTILIZATION = 0.1
MAX_VLLM_GPU_MEMORY_UTILIZATION = 0.99

_DTYPES = {"auto", "float32", "float16", "bfloat16"}
_STRUCTURED_OUTPUTS = {"json-schema", "prompt-only"}


class VLLMStructuredExtractionProvider:
    """Local-only batched CUDA adapter with application-level strict validation."""

    def __init__(
        self,
        model: str | Path = DEFAULT_ANALYSIS_MODEL,
        *,
        revision: str | None = None,
        cache_dir: Path | str | None = None,
        allow_download: bool = False,
        device: str = "cuda",
        dtype: str = "auto",
        structured_output: StructuredOutput = "json-schema",
        batch_size: int = DEFAULT_VLLM_ANALYSIS_BATCH_SIZE,
        max_input_tokens: int = DEFAULT_ANALYSIS_MAX_INPUT_TOKENS,
        max_output_tokens: int = DEFAULT_ANALYSIS_MAX_OUTPUT_TOKENS,
        prompt_style: Literal["auto", "chat", "nuextract3"] = "auto",
        support_unit_processor: SentenceProcessor | None = None,
        gpu_memory_utilization: float = DEFAULT_VLLM_GPU_MEMORY_UTILIZATION,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        model_value = str(model)
        if not model_value.strip():
            raise AnalysisModelError(
                "analysis_model_invalid", "Analysis model cannot be empty."
            )
        if revision is not None and _REVISION_PATTERN.fullmatch(revision) is None:
            raise AnalysisModelError(
                "analysis_revision_invalid",
                "Model revision contains unsupported characters.",
            )
        if not isinstance(device, str) or not device.startswith("cuda"):
            raise AnalysisModelError(
                "analysis_vllm_device_invalid",
                "The vLLM analysis runtime requires a local CUDA device.",
            )
        if dtype not in _DTYPES:
            raise AnalysisModelError(
                "analysis_dtype_invalid",
                f"Analysis dtype must be one of: {', '.join(sorted(_DTYPES))}.",
            )
        if structured_output not in _STRUCTURED_OUTPUTS:
            raise AnalysisModelError(
                "analysis_structured_output_invalid",
                "Structured output must be one of: "
                f"{', '.join(sorted(_STRUCTURED_OUTPUTS))}.",
            )
        if prompt_style not in {"auto", "chat"}:
            raise AnalysisModelError(
                "analysis_vllm_prompt_style_invalid",
                "The vLLM analysis runtime currently supports chat prompts only.",
            )
        _bounded_integer(
            batch_size, 1, MAX_ANALYSIS_BATCH_SIZE, "analysis_batch_size_invalid"
        )
        _bounded_integer(
            max_input_tokens,
            256,
            MAX_ANALYSIS_INPUT_TOKENS,
            "analysis_input_tokens_invalid",
        )
        _bounded_integer(
            max_output_tokens,
            64,
            MAX_ANALYSIS_OUTPUT_TOKENS,
            "analysis_output_tokens_invalid",
        )
        if (
            not isinstance(gpu_memory_utilization, (int, float))
            or isinstance(gpu_memory_utilization, bool)
            or not MIN_VLLM_GPU_MEMORY_UTILIZATION
            <= float(gpu_memory_utilization)
            <= MAX_VLLM_GPU_MEMORY_UTILIZATION
        ):
            raise AnalysisModelError(
                "analysis_vllm_gpu_memory_invalid",
                "vLLM GPU memory utilization must be between 0.1 and 0.99.",
            )
        unit_processor = support_unit_processor or RuleSentenceProcessor()
        if any(
            not isinstance(getattr(unit_processor, field, None), str)
            or not getattr(unit_processor, field)
            for field in ("name", "version", "model_name")
        ):
            raise AnalysisModelError(
                "analysis_support_unit_processor_invalid",
                "Support-unit processor provenance is incomplete.",
            )

        _configure_vllm_environment(offline=not allow_download)
        with _silence_vllm_stdout():
            dependencies = _vllm_dependencies()
        torch = dependencies["torch"]
        if not torch.cuda.is_available():
            raise AnalysisModelError(
                "analysis_device_unavailable",
                "CUDA was requested but the local PyTorch runtime cannot access it.",
            )
        started = clock()
        try:
            resolved_path, public_model_id, resolved_revision, fingerprint = (
                _resolve_model(
                    model_value,
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_download=allow_download,
                    model_info=dependencies["model_info"],
                    snapshot_download=dependencies["snapshot_download"],
                )
            )
        except EmbeddingError as error:
            raise AnalysisModelError(
                "analysis_model_unavailable",
                str(error).replace("Embedding model", "Analysis model"),
            ) from error
        _reject_repository_code(resolved_path)

        engine_options: dict[str, Any] = {
            "model": str(resolved_path),
            "dtype": dtype,
            "max_model_len": max_input_tokens + max_output_tokens,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "language_model_only": True,
            "trust_remote_code": False,
            "max_num_seqs": batch_size,
            "enable_prefix_caching": True,
            "disable_log_stats": True,
            "performance_mode": "throughput",
            "seed": 0,
        }
        if structured_output == "json-schema":
            engine_options["structured_outputs_config"] = {"backend": "xgrammar"}
        try:
            with _silence_vllm_stdout():
                engine = dependencies["LLM"](**engine_options)
            structured = (
                dependencies["StructuredOutputsParams"](
                    json=_xgrammar_analysis_schema()
                )
                if structured_output == "json-schema"
                else None
            )
            sampling = dependencies["SamplingParams"](
                temperature=0,
                seed=0,
                max_tokens=max_output_tokens,
                structured_outputs=structured,
            )
        except Exception as error:
            raise AnalysisModelError(
                "analysis_model_load_failed",
                "Could not initialize the local vLLM analysis runtime. "
                "See the chained exception for local diagnostics.",
            ) from error

        runtime_dtype = dtype
        model_config = getattr(
            getattr(engine, "llm_engine", None), "model_config", None
        )
        configured_dtype = getattr(model_config, "dtype", None)
        if configured_dtype is not None:
            runtime_dtype = str(configured_dtype).removeprefix("torch.")
        runtime_version = _installed_version("vllm")
        framework_version = str(getattr(torch, "__version__", "unknown"))
        accelerator_runtime = str(getattr(torch.version, "cuda", None) or "unknown")
        try:
            accelerator_name = str(torch.cuda.get_device_name(0))
        except (RuntimeError, ValueError):
            accelerator_name = None

        self._torch = torch
        self._engine = engine
        self._sampling = sampling
        self._batch_size = batch_size
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._support_unit_processor = unit_processor
        self._clock = clock
        self._info = StructuredExtractionModelInfo(
            provider="local_vllm",
            runtime="vllm",
            runtime_version=runtime_version,
            model_id=public_model_id,
            model_revision=resolved_revision,
            model_fingerprint=fingerprint,
            model_size_bytes=_directory_size(resolved_path),
            prompt_style="chat",
            prompt_version=ANALYSIS_PROMPT_VERSION,
            prompt_sha256=analysis_prompt_sha256("chat"),
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            batch_size=batch_size,
            device=device,
            dtype=runtime_dtype,
            quantization="none",
            quantization_runtime=None,
            quantization_runtime_version=None,
            structured_output=structured_output,
            structured_output_runtime_version=(
                _installed_version("xgrammar")
                if structured_output == "json-schema"
                else None
            ),
            support_unit_processor=unit_processor.name,
            support_unit_processor_version=unit_processor.version,
            support_unit_model=unit_processor.model_name,
            remote_code_trusted=False,
            download_allowed=allow_download,
            deterministic=True,
            thinking_enabled=False,
            load_ms=round(max(0.0, clock() - started) * 1_000, 6),
            framework_version=framework_version,
            accelerator_runtime_version=accelerator_runtime,
            accelerator_name=accelerator_name,
            engine_performance_mode="throughput",
            prefix_caching_enabled=True,
            gpu_memory_utilization=float(gpu_memory_utilization),
            structured_output_backend=(
                "xgrammar" if structured_output == "json-schema" else None
            ),
            sampling_backend=(
                "vllm-flashinfer"
                if os.environ.get("VLLM_USE_FLASHINFER_SAMPLER") == "1"
                else "vllm-native"
            ),
        )

    @property
    def info(self) -> StructuredExtractionModelInfo:
        return self._info

    def extract(self, texts: Sequence[str]) -> tuple[ModelExtraction, ...]:
        clean_texts = _analysis_texts(texts)
        results: list[ModelExtraction] = []
        for start in range(0, len(clean_texts), self._batch_size):
            results.extend(
                self._extract_batch(clean_texts[start : start + self._batch_size])
            )
        return tuple(results)

    def _extract_batch(self, texts: Sequence[str]) -> tuple[ModelExtraction, ...]:
        try:
            segments_by_text = self._support_unit_processor.split_many(texts)
            if len(segments_by_text) != len(texts):
                raise ValueError("sentence processor returned the wrong batch length")
            units_by_text = tuple(
                evidence_units_for(text, segments)
                for text, segments in zip(texts, segments_by_text, strict=True)
            )
            messages = [
                chat_messages_for(text, evidence_units=units)
                for text, units in zip(texts, units_by_text, strict=True)
            ]
            self._ensure_prompt_limits(messages)
        except AnalysisModelError:
            raise
        except Exception as error:
            raise AnalysisModelError(
                "analysis_support_unit_processing_failed",
                "Could not derive exact local evidence units for vLLM extraction.",
            ) from error

        try:
            started = self._clock()
            outputs = self._engine.chat(
                messages,
                self._sampling,
                use_tqdm=False,
                chat_template_kwargs={"enable_thinking": False},
            )
            self._torch.cuda.synchronize()
            elapsed_ms = round(max(0.0, self._clock() - started) * 1_000, 6)
        except Exception as error:
            raise AnalysisModelError(
                "analysis_inference_failed",
                "Local vLLM structured extraction failed. "
                "See the chained exception for local diagnostics.",
            ) from error
        if len(outputs) != len(texts):
            raise AnalysisModelError(
                "analysis_output_invalid",
                "The local vLLM runtime returned the wrong number of results.",
            )

        per_item_ms = round(elapsed_ms / len(texts), 6)
        results: list[ModelExtraction] = []
        for output, evidence_units in zip(outputs, units_by_text, strict=True):
            candidates = getattr(output, "outputs", ())
            if len(candidates) != 1:
                raise AnalysisModelError(
                    "analysis_output_invalid",
                    "The local vLLM runtime returned an invalid result shape.",
                )
            candidate = candidates[0]
            finish_reason = str(getattr(candidate, "finish_reason", ""))
            if finish_reason == "error":
                raise AnalysisModelError(
                    "analysis_inference_failed",
                    "The local vLLM runtime rejected a structured extraction request.",
                )
            raw_output = getattr(candidate, "text", None)
            token_ids = getattr(candidate, "token_ids", None)
            if not isinstance(raw_output, str) or not isinstance(token_ids, Sequence):
                raise AnalysisModelError(
                    "analysis_output_invalid",
                    "The local vLLM runtime returned invalid generation data.",
                )
            results.append(
                ModelExtraction(
                    raw_output=raw_output.strip(),
                    inference_ms=per_item_ms,
                    output_tokens=len(token_ids),
                    truncated=finish_reason == "length",
                    evidence_units=evidence_units,
                )
            )
        return tuple(results)

    def _ensure_prompt_limits(
        self, messages: Sequence[Sequence[Mapping[str, str]]]
    ) -> None:
        try:
            tokenizer = self._engine.get_tokenizer()
            for request in messages:
                token_ids = tokenizer.apply_chat_template(
                    request,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                if not isinstance(token_ids, Sequence):
                    raise TypeError("tokenizer did not return a token sequence")
                if len(token_ids) > self._max_input_tokens:
                    raise AnalysisModelError(
                        "analysis_input_too_large",
                        "An analysis prompt exceeds the configured token limit; "
                        "increase --max-input-tokens to preserve the complete evidence.",
                    )
        except AnalysisModelError:
            raise
        except Exception as error:
            raise AnalysisModelError(
                "analysis_prompt_render_failed",
                "Could not measure the local vLLM analysis prompt safely.",
            ) from error


def _xgrammar_analysis_schema() -> dict[str, object]:
    """Return the strict schema minus one hint unsupported by xgrammar 0.x.

    ``uniqueItems`` is still enforced by CorpusDock's mandatory validator after
    generation; removing it here only makes the decoder grammar a safe superset.
    """

    schema = analysis_json_schema()
    pending: list[object] = [schema]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            current.pop("uniqueItems", None)
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return schema


def _configure_vllm_environment(*, offline: bool = True) -> None:
    """Select a no-telemetry, compiler-free greedy local default."""

    forced = {
        "DO_NOT_TRACK": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1",
    }
    if offline:
        forced.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
    os.environ.update(forced)

    defaults = {
        "TRANSFORMERS_VERBOSITY": "critical",
        "VLLM_LOGGING_LEVEL": "ERROR",
        # Greedy extraction does not benefit materially from FlashInfer sampling.
        # The native sampler avoids requiring nvcc or any external CUDA toolkit.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


@contextmanager
def _silence_vllm_stdout() -> Iterator[None]:
    """Keep optional-runtime startup chatter out of the CLI JSON channel."""

    sys.stdout.flush()
    saved_stdout = os.dup(1)
    null_stdout = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null_stdout, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)
        os.close(null_stdout)


def _vllm_dependencies() -> dict[str, Any]:
    try:
        import torch
        from huggingface_hub import HfApi, snapshot_download
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:
        raise AnalysisModelError(
            "analysis_vllm_dependency_missing",
            "The high-throughput CUDA runtime requires the 'analysis-vllm' extra: "
            "install with 'pip install corpusdock[analysis-vllm]'.",
        ) from error
    return {
        "torch": torch,
        "LLM": LLM,
        "SamplingParams": SamplingParams,
        "StructuredOutputsParams": StructuredOutputsParams,
        "model_info": HfApi().model_info,
        "snapshot_download": snapshot_download,
    }


def _installed_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:  # pragma: no cover - imports guarantee metadata
        return "unknown"
