"""Local, versioned source-registration manifests.

The manifest records immutable source-file identities before extraction or indexing.
It is intentionally small and uses only the Python standard library so registering a
document remains a local-first operation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from corpusdock import __version__


MANIFEST_SCHEMA_VERSION = 1
STATE_DIRECTORY_NAME = ".corpusdock"
MANIFEST_FILE_NAME = "manifest.json"
HASH_BLOCK_SIZE = 1024 * 1024

SUPPORTED_FORMATS = {
    ".docx": "docx",
    ".epub": "epub",
    ".mobi": "mobi",
    ".pdf": "pdf",
    ".txt": "txt",
}


class ManifestError(Exception):
    """Raised when local CorpusDock state cannot be read or written safely."""


class UnsupportedSourceError(ManifestError):
    """Raised when a file is not one of the version-one source formats."""


class SourceRegistrationError(ManifestError):
    """Raised when a source file cannot be registered faithfully."""


RegistrationStatus = Literal[
    "registered", "additional_path_registered", "already_registered"
]


def utc_now() -> str:
    """Return a portable, unambiguous timestamp for manifest records."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def manifest_path_for(project_root: Path | str) -> Path:
    """Return the standard manifest location for a CorpusDock project."""

    return (
        Path(project_root).expanduser().resolve()
        / STATE_DIRECTORY_NAME
        / MANIFEST_FILE_NAME
    )


def source_format_for(path: Path | str) -> str | None:
    """Identify a supported source format from its case-insensitive extension."""

    return SUPPORTED_FORMATS.get(Path(path).suffix.lower())


def identify_source_format(path: Path | str) -> str:
    """Identify a supported source format or raise a useful registration error."""

    source_format = source_format_for(path)
    if source_format is None:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise UnsupportedSourceError(
            f"Unsupported source format for '{path}'. Supported formats: {supported}."
        )
    return source_format


def source_id_for(sha256_digest: str) -> str:
    """Derive a stable content-addressed source ID from a SHA-256 digest."""

    return f"src-{sha256_digest}"


def discover_source_files(path: Path | str) -> tuple[Path, ...]:
    """Return supported files at ``path``, recursively when it is a directory."""

    candidate = Path(path).expanduser()
    if candidate.is_file():
        identify_source_format(candidate)
        return (candidate,)
    if not candidate.is_dir():
        raise SourceRegistrationError(
            f"Source path does not exist or is not a file or directory: '{path}'."
        )

    files = tuple(
        sorted(
            (
                item
                for item in candidate.rglob("*")
                if item.is_file() and source_format_for(item) is not None
            ),
            key=lambda item: str(item).casefold(),
        )
    )
    if not files:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise SourceRegistrationError(
            f"No supported documents found in '{path}'. Supported formats: {supported}."
        )
    return files


def discover_mirror_files(path: Path | str) -> tuple[Path, ...]:
    """Return every supported file in an authoritative directory, including none."""

    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        raise SourceRegistrationError(
            f"Input mirror is not an existing directory: '{path}'."
        )
    return tuple(
        sorted(
            (
                item
                for item in candidate.rglob("*")
                if item.is_file() and source_format_for(item) is not None
            ),
            key=lambda item: str(item).casefold(),
        )
    )


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """The durable identity and local provenance of one immutable source file."""

    source_id: str
    sha256: str
    source_format: str
    size_bytes: int
    original_paths: tuple[str, ...]
    registered_at: str
    registration_tool_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "sha256": self.sha256,
            "format": self.source_format,
            "size_bytes": self.size_bytes,
            "original_paths": list(self.original_paths),
            "registered_at": self.registered_at,
            "registration_tool_version": self.registration_tool_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> SourceRecord:
        """Construct a source record while checking the persisted manifest shape."""

        if not isinstance(value, dict):
            raise ManifestError("Each manifest source must be a JSON object.")

        source_id = _required_string(value, "source_id")
        sha256_digest = _required_string(value, "sha256")
        source_format = _required_string(value, "format")
        registered_at = _required_string(value, "registered_at")
        registration_tool_version = _required_string(value, "registration_tool_version")
        size_bytes = value.get("size_bytes")
        original_paths = value.get("original_paths")

        if len(sha256_digest) != 64 or any(
            character not in "0123456789abcdef" for character in sha256_digest
        ):
            raise ManifestError(f"Source '{source_id}' has an invalid SHA-256 digest.")
        if source_id != source_id_for(sha256_digest):
            raise ManifestError(
                f"Source '{source_id}' does not match its SHA-256 digest."
            )
        if source_format not in SUPPORTED_FORMATS.values():
            raise ManifestError(
                f"Source '{source_id}' has an unsupported format '{source_format}'."
            )
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise ManifestError(
                f"Source '{source_id}' has an invalid size_bytes value."
            )
        if (
            not isinstance(original_paths, list)
            or not original_paths
            or any(not isinstance(item, str) or not item for item in original_paths)
        ):
            raise ManifestError(
                f"Source '{source_id}' must contain one or more original_paths."
            )

        return cls(
            source_id=source_id,
            sha256=sha256_digest,
            source_format=source_format,
            size_bytes=size_bytes,
            original_paths=tuple(original_paths),
            registered_at=registered_at,
            registration_tool_version=registration_tool_version,
        )


@dataclass(frozen=True, slots=True)
class CorpusManifest:
    """The versioned, backend-independent registration state for a corpus."""

    schema_version: int
    created_at: str
    updated_at: str
    sources: dict[str, SourceRecord]

    @classmethod
    def empty(cls, timestamp: str) -> CorpusManifest:
        return cls(
            schema_version=MANIFEST_SCHEMA_VERSION,
            created_at=timestamp,
            updated_at=timestamp,
            sources={},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": [
                self.sources[source_id].to_dict() for source_id in sorted(self.sources)
            ],
        }

    @classmethod
    def from_dict(cls, value: object) -> CorpusManifest:
        """Read schema version one without accepting ambiguous or damaged records."""

        if not isinstance(value, dict):
            raise ManifestError("The manifest root must be a JSON object.")

        schema_version = value.get("schema_version")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ManifestError("Manifest schema_version must be an integer.")
        if schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"Unsupported manifest schema version {schema_version}; "
                f"this version of CorpusDock supports {MANIFEST_SCHEMA_VERSION}."
            )

        created_at = _required_string(value, "created_at")
        updated_at = _required_string(value, "updated_at")
        raw_sources = value.get("sources")
        if not isinstance(raw_sources, list):
            raise ManifestError("Manifest sources must be a JSON array.")

        sources: dict[str, SourceRecord] = {}
        for raw_source in raw_sources:
            source = SourceRecord.from_dict(raw_source)
            if source.source_id in sources:
                raise ManifestError(
                    f"Manifest contains duplicate source ID '{source.source_id}'."
                )
            sources[source.source_id] = source

        return cls(
            schema_version=schema_version,
            created_at=created_at,
            updated_at=updated_at,
            sources=sources,
        )


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    """The outcome of registering one path in a manifest."""

    source: SourceRecord
    source_path: str
    status: RegistrationStatus


@dataclass(frozen=True, slots=True)
class MirrorReconciliation:
    """One atomic reconciliation of an authoritative input-folder snapshot."""

    manifest: CorpusManifest
    scanned_paths: int
    unique_sources: int
    added_source_ids: tuple[str, ...]
    removed_source_ids: tuple[str, ...]
    retained_source_ids: tuple[str, ...]
    added_paths: int
    removed_paths: int
    changed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_paths": self.scanned_paths,
            "unique_sources": self.unique_sources,
            "changed": self.changed,
            "sources": {
                "added": len(self.added_source_ids),
                "removed": len(self.removed_source_ids),
                "retained": len(self.retained_source_ids),
            },
            "paths": {
                "added": self.added_paths,
                "removed": self.removed_paths,
            },
            "added_source_ids": list(self.added_source_ids),
            "removed_source_ids": list(self.removed_source_ids),
        }


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    path: Path
    source_format: str
    sha256: str
    size_bytes: int

    @property
    def source_id(self) -> str:
        return source_id_for(self.sha256)


class ManifestStore:
    """Read and atomically update the local manifest for one CorpusDock project."""

    def __init__(
        self, project_root: Path | str, *, now: Callable[[], str] = utc_now
    ) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.path = manifest_path_for(self.project_root)
        self._now = now

    def initialize(self) -> tuple[CorpusManifest, bool]:
        """Create an empty manifest unless the project already has one."""

        if self.path.exists():
            return self.load(), False

        manifest = CorpusManifest.empty(self._now())
        self._write(manifest)
        return manifest, True

    def load(self) -> CorpusManifest:
        """Load and validate the current manifest."""

        if not self.path.is_file():
            raise ManifestError(
                f"No CorpusDock manifest found at '{self.path}'. Run 'corpusdock init' first."
            )
        try:
            with self.path.open(encoding="utf-8") as manifest_file:
                raw_manifest = json.load(manifest_file)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ManifestError(
                f"Manifest at '{self.path}' is not valid JSON."
            ) from error
        except OSError as error:
            raise ManifestError(
                f"Could not read manifest at '{self.path}': {error.strerror or error}."
            ) from error

        return CorpusManifest.from_dict(raw_manifest)

    def register(self, paths: Iterable[Path | str]) -> tuple[RegistrationResult, ...]:
        """Register files by hash without extracting, copying, or indexing their content."""

        prepared_sources = tuple(_prepare_source(path) for path in paths)
        if not prepared_sources:
            raise SourceRegistrationError(
                "At least one supported source file is required for registration."
            )

        if self.path.exists():
            manifest = self.load()
        else:
            manifest = CorpusManifest.empty(self._now())

        sources = dict(manifest.sources)
        registration_time = self._now()
        changed = not self.path.exists()
        results: list[RegistrationResult] = []

        for prepared in prepared_sources:
            original_path = str(prepared.path)
            existing = sources.get(prepared.source_id)
            if existing is None:
                source = SourceRecord(
                    source_id=prepared.source_id,
                    sha256=prepared.sha256,
                    source_format=prepared.source_format,
                    size_bytes=prepared.size_bytes,
                    original_paths=(original_path,),
                    registered_at=registration_time,
                    registration_tool_version=__version__,
                )
                sources[source.source_id] = source
                results.append(
                    RegistrationResult(
                        source=source,
                        source_path=original_path,
                        status="registered",
                    )
                )
                changed = True
                continue

            if existing.source_format != prepared.source_format:
                raise ManifestError(
                    f"Source '{existing.source_id}' is already registered as {existing.source_format}, "
                    f"not {prepared.source_format}."
                )
            if existing.size_bytes != prepared.size_bytes:
                raise ManifestError(
                    f"Source '{existing.source_id}' has a SHA-256 match but a different recorded size."
                )
            if original_path in existing.original_paths:
                results.append(
                    RegistrationResult(
                        source=existing,
                        source_path=original_path,
                        status="already_registered",
                    )
                )
                continue

            source = replace(
                existing, original_paths=(*existing.original_paths, original_path)
            )
            sources[source.source_id] = source
            results.append(
                RegistrationResult(
                    source=source,
                    source_path=original_path,
                    status="additional_path_registered",
                )
            )
            changed = True

        if changed:
            manifest = replace(manifest, updated_at=registration_time, sources=sources)
            self._write(manifest)

        return tuple(results)

    def reconcile_mirror(self, paths: Iterable[Path | str]) -> MirrorReconciliation:
        """Make the manifest exactly match one authoritative set of local files.

        Content identity, rather than path, controls reuse. Moving or renaming a file
        therefore updates provenance without creating a new source, while changed
        bytes create a new source ID and absent content leaves the manifest.
        """

        prepared_sources = tuple(_prepare_source(path) for path in paths)
        manifest = (
            self.load() if self.path.exists() else CorpusManifest.empty(self._now())
        )
        prior_sources = manifest.sources
        grouped: dict[str, list[_PreparedSource]] = {}
        for prepared in prepared_sources:
            grouped.setdefault(prepared.source_id, []).append(prepared)

        reconciliation_time = self._now()
        sources: dict[str, SourceRecord] = {}
        for source_id in sorted(grouped):
            group = grouped[source_id]
            formats = {prepared.source_format for prepared in group}
            sizes = {prepared.size_bytes for prepared in group}
            if len(formats) != 1 or len(sizes) != 1:
                raise ManifestError(
                    f"Content-identical paths for source '{source_id}' have "
                    "inconsistent formats or sizes."
                )
            original_paths = tuple(
                sorted({str(prepared.path) for prepared in group}, key=str.casefold)
            )
            existing = prior_sources.get(source_id)
            if existing is not None:
                source_format = next(iter(formats))
                size_bytes = next(iter(sizes))
                if (
                    existing.source_format != source_format
                    or existing.size_bytes != size_bytes
                ):
                    raise ManifestError(
                        f"Source '{source_id}' conflicts with its registered metadata."
                    )
                sources[source_id] = replace(existing, original_paths=original_paths)
                continue
            first = group[0]
            sources[source_id] = SourceRecord(
                source_id=source_id,
                sha256=first.sha256,
                source_format=first.source_format,
                size_bytes=first.size_bytes,
                original_paths=original_paths,
                registered_at=reconciliation_time,
                registration_tool_version=__version__,
            )

        prior_ids = set(prior_sources)
        current_ids = set(sources)
        prior_paths = {
            path for source in prior_sources.values() for path in source.original_paths
        }
        current_paths = {
            path for source in sources.values() for path in source.original_paths
        }
        changed = sources != prior_sources or not self.path.exists()
        if changed:
            manifest = replace(
                manifest,
                updated_at=reconciliation_time,
                sources=sources,
            )
            self._write(manifest)

        return MirrorReconciliation(
            manifest=manifest,
            scanned_paths=len(prepared_sources),
            unique_sources=len(sources),
            added_source_ids=tuple(sorted(current_ids - prior_ids)),
            removed_source_ids=tuple(sorted(prior_ids - current_ids)),
            retained_source_ids=tuple(sorted(prior_ids & current_ids)),
            added_paths=len(current_paths - prior_paths),
            removed_paths=len(prior_paths - current_paths),
            changed=changed,
        )

    def get_source(self, source_id: str) -> SourceRecord | None:
        """Return a registered source by its stable ID."""

        return self.load().sources.get(source_id)

    def _write(self, manifest: CorpusManifest) -> None:
        """Write a complete manifest atomically to avoid partial registration state."""

        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            with temporary_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as manifest_file:
                json.dump(
                    manifest.to_dict(),
                    manifest_file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                manifest_file.write("\n")
                manifest_file.flush()
                os.fsync(manifest_file.fileno())
            os.replace(temporary_path, self.path)
        except OSError as error:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise ManifestError(
                f"Could not write manifest at '{self.path}': {error.strerror or error}."
            ) from error


def find_project_root(start: Path | str) -> Path | None:
    """Find the nearest ancestor that contains an initialized CorpusDock manifest."""

    current = Path(start).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if manifest_path_for(candidate).is_file():
            return candidate
    return None


def _prepare_source(path: Path | str) -> _PreparedSource:
    candidate = Path(path).expanduser()
    try:
        resolved_path = candidate.resolve(strict=True)
    except OSError as error:
        raise SourceRegistrationError(
            f"Could not resolve source path '{path}': {error.strerror or error}."
        ) from error

    if not resolved_path.is_file():
        raise SourceRegistrationError(f"Source path is not a regular file: '{path}'.")

    source_format = identify_source_format(resolved_path)
    digest, size_bytes = hash_source_file(resolved_path)
    return _PreparedSource(
        path=resolved_path,
        source_format=source_format,
        sha256=digest,
        size_bytes=size_bytes,
    )


def hash_source_file(path: Path) -> tuple[str, int]:
    """Hash a file and reject it if it changes while being registered."""

    try:
        before = path.stat()
        digest = sha256()
        with path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(HASH_BLOCK_SIZE), b""):
                digest.update(block)
        after = path.stat()
    except OSError as error:
        raise SourceRegistrationError(
            f"Could not read source file '{path}': {error.strerror or error}."
        ) from error

    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise SourceRegistrationError(
            f"Source file changed while it was being registered: '{path}'."
        )
    return digest.hexdigest(), after.st_size


def _required_string(value: dict[str, object], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ManifestError(f"Manifest field '{key}' must be a non-empty string.")
    return result
