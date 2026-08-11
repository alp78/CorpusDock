"""Provider-neutral local embeddings and an ephemeral semantic benchmark backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Protocol

from corpusdock.retrieval import (
    DEFAULT_SEARCH_LIMIT,
    MAX_SEARCH_LIMIT,
    MatchMode,
    RetrievalError,
    SQLiteSearchBackend,
    SearchResponse,
    VerificationReport,
)


DEFAULT_EMBEDDING_BATCH_SIZE = 16
MAX_EMBEDDING_BATCH_SIZE = 1_024
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

_HUB_MODEL_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
)
_REVISION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
_MATCH_MODES = {"all", "any", "phrase"}


class EmbeddingError(Exception):
    """A local embedding configuration, model, or inference failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class EmbeddingModelInfo:
    """Non-content provenance for one loaded embedding model."""

    provider: str
    runtime: str
    runtime_version: str
    model_id: str
    model_revision: str
    model_fingerprint: str
    model_size_bytes: int
    dimension: int
    max_sequence_tokens: int | None
    device: str
    dtype: str
    normalized: bool
    remote_code_trusted: bool
    download_allowed: bool
    query_prompt_name: str | None
    document_prompt_name: str | None
    batch_size: int
    load_ms: float
    framework_version: str | None = None
    accelerator_runtime_version: str | None = None
    accelerator_name: str | None = None
    accelerator_peak_memory_allocated_bytes: int | None = None
    accelerator_peak_memory_reserved_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "runtime": self.runtime,
            "runtime_version": self.runtime_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_fingerprint": self.model_fingerprint,
            "model_size_bytes": self.model_size_bytes,
            "dimension": self.dimension,
            "max_sequence_tokens": self.max_sequence_tokens,
            "device": self.device,
            "dtype": self.dtype,
            "normalized": self.normalized,
            "remote_code_trusted": self.remote_code_trusted,
            "download_allowed": self.download_allowed,
            "query_prompt_name": self.query_prompt_name,
            "document_prompt_name": self.document_prompt_name,
            "batch_size": self.batch_size,
            "load_ms": self.load_ms,
            "framework_version": self.framework_version,
            "accelerator_runtime_version": self.accelerator_runtime_version,
            "accelerator_name": self.accelerator_name,
            "accelerator_peak_memory_allocated_bytes": (
                self.accelerator_peak_memory_allocated_bytes
            ),
            "accelerator_peak_memory_reserved_bytes": (
                self.accelerator_peak_memory_reserved_bytes
            ),
        }


@dataclass(frozen=True, slots=True)
class SemanticBuildStats:
    """Resource measurements for an ephemeral dense benchmark index."""

    documents: int
    dimension: int
    vector_size_bytes: int
    document_embedding_ms: float
    documents_per_second: float
    source_index_fingerprint: str

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "documents": self.documents,
            "dimension": self.dimension,
            "vector_size_bytes": self.vector_size_bytes,
            "document_embedding_ms": self.document_embedding_ms,
            "documents_per_second": self.documents_per_second,
            "source_index_fingerprint": self.source_index_fingerprint,
        }


class EmbeddingProvider(Protocol):
    """Backend-neutral asymmetric text embedding operations."""

    @property
    def info(self) -> EmbeddingModelInfo: ...

    def embed_documents(self, texts: Sequence[str]) -> object: ...

    def embed_queries(self, texts: Sequence[str]) -> object: ...


def model_cache_dir_for(project_root: Path | str) -> Path:
    """Return the ignored, project-local model cache directory."""

    return Path(project_root).expanduser().resolve() / ".corpusdock" / "models"


class SentenceTransformersEmbeddingProvider:
    """Local-only-by-default adapter for standard Sentence Transformers models."""

    def __init__(
        self,
        model: str | Path,
        *,
        revision: str | None = None,
        cache_dir: Path | str | None = None,
        allow_download: bool = False,
        device: str = "cpu",
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        truncate_dimension: int | None = None,
        show_progress: bool = False,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        model_value = str(model)
        if not model_value.strip():
            raise EmbeddingError(
                "embedding_model_invalid", "Embedding model cannot be empty."
            )
        if revision is not None and _REVISION_PATTERN.fullmatch(revision) is None:
            raise EmbeddingError(
                "embedding_revision_invalid",
                "Model revision contains unsupported characters.",
            )
        _validate_batch_size(batch_size)
        if truncate_dimension is not None and (
            not isinstance(truncate_dimension, int)
            or isinstance(truncate_dimension, bool)
            or truncate_dimension < 1
        ):
            raise EmbeddingError(
                "embedding_dimension_invalid",
                "Truncated embedding dimension must be a positive integer.",
            )
        if not isinstance(device, str) or not device.strip() or len(device) > 100:
            raise EmbeddingError(
                "embedding_device_invalid", "Embedding device must be a short string."
            )

        SentenceTransformer, model_info, snapshot_download = _embedding_dependencies()
        started = clock()
        resolved_path, public_model_id, resolved_revision, fingerprint = _resolve_model(
            model_value,
            revision=revision,
            cache_dir=cache_dir,
            allow_download=allow_download,
            model_info=model_info,
            snapshot_download=snapshot_download,
        )
        _reject_custom_model_code(resolved_path)
        try:
            loaded_model = SentenceTransformer(
                str(resolved_path),
                device=device,
                trust_remote_code=False,
                local_files_only=True,
                truncate_dim=truncate_dimension,
            )
        except Exception as error:
            raise EmbeddingError(
                "embedding_model_load_failed",
                f"Could not load the local embedding model without remote code: {error}.",
            ) from error

        get_dimension = getattr(loaded_model, "get_embedding_dimension", None)
        if not callable(get_dimension):
            get_dimension = loaded_model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if not isinstance(dimension, int) or dimension < 1:
            raise EmbeddingError(
                "embedding_dimension_invalid",
                "The embedding runtime did not report a valid output dimension.",
            )
        prompts = getattr(loaded_model, "prompts", {})
        prompt_names = set(prompts) if isinstance(prompts, Mapping) else set()
        max_tokens = loaded_model.get_max_seq_length()
        load_ms = _rounded_ms(clock() - started)
        dtype = str(getattr(loaded_model, "dtype", "unknown")).removeprefix("torch.")
        try:
            runtime_version = package_version("sentence-transformers")
        except PackageNotFoundError:  # pragma: no cover - import guarantees metadata
            runtime_version = "unknown"
        framework_version, accelerator_runtime, accelerator_name = _runtime_versions(
            loaded_model.device
        )

        self._model = loaded_model
        self._batch_size = batch_size
        self._show_progress = show_progress
        self._info = EmbeddingModelInfo(
            provider="local_sentence_transformers",
            runtime="sentence-transformers",
            runtime_version=runtime_version,
            model_id=public_model_id,
            model_revision=resolved_revision,
            model_fingerprint=fingerprint,
            model_size_bytes=_directory_size(resolved_path),
            dimension=dimension,
            max_sequence_tokens=max_tokens if isinstance(max_tokens, int) else None,
            device=str(loaded_model.device),
            dtype=dtype,
            normalized=True,
            remote_code_trusted=False,
            download_allowed=allow_download,
            query_prompt_name="query" if "query" in prompt_names else None,
            document_prompt_name=next(
                (
                    name
                    for name in ("document", "passage", "corpus")
                    if name in prompt_names
                ),
                None,
            ),
            batch_size=batch_size,
            load_ms=load_ms,
            framework_version=framework_version,
            accelerator_runtime_version=accelerator_runtime,
            accelerator_name=accelerator_name,
            accelerator_peak_memory_allocated_bytes=None,
            accelerator_peak_memory_reserved_bytes=None,
        )

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def embed_documents(self, texts: Sequence[str]) -> object:
        return self._embed(texts, query=False)

    def embed_queries(self, texts: Sequence[str]) -> object:
        return self._embed(texts, query=True)

    def _embed(self, texts: Sequence[str], *, query: bool) -> object:
        clean_texts = _embedding_texts(texts)
        method = self._model.encode_query if query else self._model.encode_document
        try:
            vectors = method(
                list(clean_texts),
                batch_size=self._batch_size,
                show_progress_bar=self._show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            self._record_accelerator_memory()
            return vectors
        except Exception as error:
            kind = "query" if query else "document"
            raise EmbeddingError(
                "embedding_inference_failed",
                f"Local {kind} embedding failed: {error}.",
            ) from error

    def _record_accelerator_memory(self) -> None:
        device = getattr(self._model, "device", None)
        if getattr(device, "type", None) != "cuda":
            return
        try:
            import torch

            torch.cuda.synchronize(device)
            allocated = int(torch.cuda.max_memory_allocated(device))
            reserved = int(torch.cuda.max_memory_reserved(device))
        except (ImportError, RuntimeError, ValueError):
            return
        self._info = replace(
            self._info,
            accelerator_peak_memory_allocated_bytes=allocated,
            accelerator_peak_memory_reserved_bytes=reserved,
        )


class InMemorySemanticSearchBackend:
    """Exact-evidence semantic retrieval used to compare local embedding models."""

    def __init__(
        self,
        exact_backend: SQLiteSearchBackend,
        provider: EmbeddingProvider,
        *,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        np = _numpy()
        snapshot = exact_backend.corpus_snapshot()
        if not snapshot.evidence:
            raise EmbeddingError(
                "semantic_corpus_empty",
                "The exact index has no chunks to embed for semantic retrieval.",
            )
        started = clock()
        raw_vectors = provider.embed_documents(
            tuple(evidence.excerpt for evidence in snapshot.evidence)
        )
        matrix = _normalized_matrix(
            raw_vectors,
            rows=len(snapshot.evidence),
            dimension=provider.info.dimension,
            label="document",
            np=np,
        )
        elapsed = max(0.0, clock() - started)
        elapsed_ms = _rounded_ms(elapsed)
        self._np = np
        self._exact_backend = exact_backend
        self._provider = provider
        self._snapshot = snapshot
        self._matrix = matrix
        self._source_ids = frozenset(snapshot.source_ids)
        self.build_stats = SemanticBuildStats(
            documents=len(snapshot.evidence),
            dimension=provider.info.dimension,
            vector_size_bytes=int(matrix.nbytes),
            document_embedding_ms=elapsed_ms,
            documents_per_second=round(
                len(snapshot.evidence) / elapsed if elapsed else 0.0, 6
            ),
            source_index_fingerprint=snapshot.index_fingerprint,
        )

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._provider.info

    def evaluation_metadata(self) -> dict[str, Any]:
        return {
            "embedding": self.info.to_dict(),
            "semantic_index": self.build_stats.to_dict(),
        }

    def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_SEARCH_LIMIT,
        source_id: str | None = None,
        match_mode: MatchMode = "all",
    ) -> SearchResponse:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_SEARCH_LIMIT
        ):
            raise RetrievalError(
                "search_limit_invalid",
                f"Search limit must be between 1 and {MAX_SEARCH_LIMIT}.",
            )
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query_empty", "Search query cannot be empty.")
        if not isinstance(match_mode, str) or match_mode not in _MATCH_MODES:
            raise RetrievalError(
                "match_mode_invalid", f"Unknown search match mode '{match_mode}'."
            )
        if source_id is not None and source_id not in self._source_ids:
            raise RetrievalError(
                "source_not_indexed", f"No indexed source has ID '{source_id}'."
            )

        self._exact_backend.assert_snapshot_current(self._snapshot)
        raw_query = self._provider.embed_queries((query,))
        query_matrix = _normalized_matrix(
            raw_query,
            rows=1,
            dimension=self.info.dimension,
            label="query",
            np=self._np,
        )
        scores = self._matrix @ query_matrix[0]
        candidates = (
            index
            for index, evidence in enumerate(self._snapshot.evidence)
            if source_id is None or evidence.locator.source_id == source_id
        )
        ranked = sorted(
            candidates,
            key=lambda index: (
                -float(scores[index]),
                self._snapshot.evidence[index].chunk_id or "",
            ),
        )[:limit]
        results = tuple(
            replace(
                self._snapshot.evidence[index],
                score=round(float(scores[index]), 12),
            )
            for index in ranked
        )
        return SearchResponse(
            query=query,
            match_mode=match_mode,
            results=results,
            index_built_at=self._snapshot.index_built_at,
            indexed_sources=self._snapshot.indexed_sources,
            indexed_chunks=self._snapshot.indexed_chunks,
            partial_sources=self._snapshot.partial_sources,
        )

    def verify(self, evidence_id: str) -> VerificationReport:
        return self._exact_backend.verify(evidence_id)


def _embedding_dependencies() -> tuple[
    type[Any], Callable[..., Any], Callable[..., str]
]:
    try:
        from huggingface_hub import HfApi, snapshot_download
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise EmbeddingError(
            "embedding_dependency_missing",
            "Local semantic models require the 'semantic' extra: "
            "install with 'pip install corpusdock[semantic]'.",
        ) from error
    return SentenceTransformer, HfApi().model_info, snapshot_download


def _runtime_versions(device: object) -> tuple[str | None, str | None, str | None]:
    try:
        import torch
    except ImportError:  # pragma: no cover - Sentence Transformers imports torch
        return None, None, None
    framework_version = str(getattr(torch, "__version__", "unknown"))
    if getattr(device, "type", None) != "cuda":
        return framework_version, None, None
    try:
        accelerator_runtime = str(torch.version.cuda or "unknown")
        accelerator_name = str(torch.cuda.get_device_name(device))
    except (RuntimeError, ValueError):
        return framework_version, None, None
    return framework_version, accelerator_runtime, accelerator_name


def _resolve_model(
    model: str,
    *,
    revision: str | None,
    cache_dir: Path | str | None,
    allow_download: bool,
    model_info: Callable[..., Any],
    snapshot_download: Callable[..., str],
) -> tuple[Path, str, str, str]:
    candidate = Path(model).expanduser()
    if candidate.exists():
        if not candidate.is_dir():
            raise EmbeddingError(
                "embedding_model_invalid",
                "A local embedding model must be a directory.",
            )
        resolved = candidate.resolve()
        digest = _directory_digest(resolved)
        return resolved, f"local:{digest[:16]}", "local", f"sha256:{digest}"

    if _HUB_MODEL_PATTERN.fullmatch(model) is None:
        raise EmbeddingError(
            "embedding_model_invalid",
            "Embedding model must be a local directory or a 'namespace/model' ID.",
        )
    if cache_dir is None:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_root = Path(HF_HUB_CACHE).expanduser().resolve()
    else:
        cache_root = Path(cache_dir).expanduser().resolve()
    resolved_cache = str(cache_root)
    selected_revision = revision
    allow_patterns: tuple[str, ...] | None = None
    try:
        if allow_download:
            info = model_info(
                repo_id=model,
                revision=revision,
                files_metadata=False,
                token=False,
            )
            info_revision = getattr(info, "sha", None)
            siblings = getattr(info, "siblings", None)
            if not isinstance(info_revision, str) or not info_revision:
                raise ValueError(
                    "the model registry did not return an immutable revision"
                )
            if not isinstance(siblings, Sequence):
                raise ValueError("the model registry did not return a file manifest")
            filenames = tuple(
                filename
                for sibling in siblings
                if isinstance((filename := getattr(sibling, "rfilename", None)), str)
            )
            allow_patterns = _download_file_allowlist(filenames)
            selected_revision = info_revision
        try:
            snapshot = snapshot_download(
                repo_id=model,
                revision=selected_revision,
                cache_dir=resolved_cache,
                local_files_only=not allow_download,
                token=False,
                allow_patterns=allow_patterns,
            )
        except Exception:
            if allow_download:
                raise
            cached_snapshot = _cached_model_snapshot(
                cache_root, model=model, revision=revision
            )
            if cached_snapshot is None:
                raise
            snapshot = str(cached_snapshot)
    except Exception as error:
        action = (
            "could not be downloaded"
            if allow_download
            else "is not available in the local model cache"
        )
        raise EmbeddingError(
            "embedding_model_unavailable",
            f"Embedding model '{model}' {action}: {error}.",
        ) from error
    resolved = Path(snapshot).resolve()
    resolved_revision = (
        resolved.name if resolved.parent.name == "snapshots" else revision or "unknown"
    )
    return (
        resolved,
        model,
        resolved_revision,
        f"hf-revision:{resolved_revision}",
    )


def _download_file_allowlist(filenames: Sequence[str]) -> tuple[str, ...]:
    """Select one safe Transformers weight format plus required local metadata."""

    excluded_prefixes = (".eval_results/", "onnx/", "openvino/")
    usable = tuple(
        filename
        for filename in filenames
        if filename
        and not filename.startswith(excluded_prefixes)
        and not filename.endswith((".py", ".pyc"))
    )
    safetensors = tuple(
        filename for filename in usable if filename.endswith(".safetensors")
    )
    if safetensors:
        weights = safetensors
    else:
        weights = tuple(filename for filename in usable if filename.endswith(".bin"))
    if not weights:
        raise ValueError(
            "no supported safetensors or PyTorch weight files were advertised"
        )

    metadata_suffixes = (
        ".json",
        ".jinja",
        ".model",
        ".spm",
        ".txt",
        ".vocab",
    )
    metadata = tuple(
        filename for filename in usable if filename.endswith(metadata_suffixes)
    )
    return tuple(dict.fromkeys((*metadata, *weights)))


def _cached_model_snapshot(
    cache_root: Path, *, model: str, revision: str | None
) -> Path | None:
    """Resolve an already-downloaded immutable snapshot without network access."""

    repository = cache_root / f"models--{model.replace('/', '--')}"
    snapshots = repository / "snapshots"
    if not snapshots.is_dir():
        return None

    commit: str | None = None
    if revision is not None and re.fullmatch(r"[0-9a-f]{40,64}", revision):
        commit = revision
    else:
        reference = revision or "main"
        reference_root = (repository / "refs").resolve()
        reference_path = (reference_root / reference).resolve()
        if reference_path.is_relative_to(reference_root) and reference_path.is_file():
            try:
                candidate = reference_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeDecodeError):
                candidate = ""
            if re.fullmatch(r"[0-9a-f]{40,64}", candidate):
                commit = candidate

    if commit is None and revision is None:
        candidates = tuple(
            path
            for path in snapshots.iterdir()
            if path.is_dir() and re.fullmatch(r"[0-9a-f]{40,64}", path.name)
        )
        if len(candidates) == 1:
            commit = candidates[0].name
        elif len(candidates) > 1:
            raise ValueError(
                "multiple revisions are cached; pass --model-revision with the reported commit"
            )

    if commit is None:
        return None
    snapshot = (snapshots / commit).resolve()
    resolved_snapshots = snapshots.resolve()
    if not snapshot.is_relative_to(resolved_snapshots) or not snapshot.is_dir():
        return None
    return snapshot


def _reject_custom_model_code(model_path: Path) -> None:
    for config_name in ("config.json", "tokenizer_config.json"):
        config_path = model_path / config_name
        if not config_path.is_file():
            continue
        config = _read_model_json(config_path)
        if not isinstance(config, Mapping):
            raise EmbeddingError(
                "embedding_model_invalid",
                f"Embedding model {config_name} must be a JSON object.",
            )
        if config.get("auto_map"):
            raise EmbeddingError(
                "embedding_remote_code_required",
                "The selected model declares custom repository code; CorpusDock will not execute it.",
            )
    modules_path = model_path / "modules.json"
    if not modules_path.is_file():
        return
    modules = _read_model_json(modules_path)
    if not isinstance(modules, list):
        raise EmbeddingError(
            "embedding_model_invalid", "Embedding model modules.json must be an array."
        )
    for module in modules:
        if not isinstance(module, dict):
            raise EmbeddingError(
                "embedding_model_invalid",
                "Embedding model modules.json contains a non-object module.",
            )
        module_type = module.get("type")
        if not isinstance(module_type, str) or not module_type.startswith(
            "sentence_transformers."
        ):
            raise EmbeddingError(
                "embedding_remote_code_required",
                "The selected model declares a custom Sentence Transformers module; "
                "CorpusDock will not execute it.",
            )


def _read_model_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EmbeddingError(
            "embedding_model_invalid", f"Could not read model configuration: {error}."
        ) from error


def _embedding_texts(texts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(texts, (str, bytes)):
        raise EmbeddingError(
            "embedding_input_invalid", "Embedding input must be a sequence of strings."
        )
    result = tuple(texts)
    if not result or any(
        not isinstance(text, str) or not text.strip() for text in result
    ):
        raise EmbeddingError(
            "embedding_input_invalid",
            "Embedding input must contain at least one non-empty string.",
        )
    return result


def _validate_batch_size(batch_size: int) -> None:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or not 1 <= batch_size <= MAX_EMBEDDING_BATCH_SIZE
    ):
        raise EmbeddingError(
            "embedding_batch_size_invalid",
            f"Embedding batch size must be between 1 and {MAX_EMBEDDING_BATCH_SIZE}.",
        )


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:
        raise EmbeddingError(
            "embedding_dependency_missing",
            "Local semantic retrieval requires NumPy from the 'semantic' extra.",
        ) from error
    return np


def _normalized_matrix(
    values: object,
    *,
    rows: int,
    dimension: int,
    label: str,
    np: Any,
) -> Any:
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise EmbeddingError(
            "embedding_output_invalid",
            f"The {label} embedding output is not a numeric matrix.",
        ) from error
    if matrix.ndim != 2 or matrix.shape != (rows, dimension):
        raise EmbeddingError(
            "embedding_output_invalid",
            f"The {label} embedding matrix must have shape ({rows}, {dimension}).",
        )
    if not bool(np.isfinite(matrix).all()):
        raise EmbeddingError(
            "embedding_output_invalid",
            f"The {label} embedding matrix contains a non-finite value.",
        )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if bool((norms <= 0).any()):
        raise EmbeddingError(
            "embedding_output_invalid",
            f"The {label} embedding matrix contains a zero-length vector.",
        )
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def _directory_size(root: Path) -> int:
    total = 0
    try:
        for path in root.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
    except OSError as error:
        raise EmbeddingError(
            "embedding_model_invalid", f"Could not measure local model files: {error}."
        ) from error
    return total


def _directory_digest(root: Path) -> str:
    digest = sha256()
    try:
        files = sorted(path for path in root.rglob("*") if path.is_file())
        if not files:
            raise EmbeddingError(
                "embedding_model_invalid", "Local embedding model directory is empty."
            )
        for path in files:
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with path.open("rb") as model_file:
                while block := model_file.read(1024 * 1024):
                    digest.update(block)
    except EmbeddingError:
        raise
    except OSError as error:
        raise EmbeddingError(
            "embedding_model_invalid", f"Could not hash local model files: {error}."
        ) from error
    return digest.hexdigest()


def _rounded_ms(seconds: float) -> float:
    return round(max(0.0, seconds) * 1_000, 6)
