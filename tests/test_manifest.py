from __future__ import annotations

from hashlib import sha256
import json

import pytest

from corpusdock import __version__
from corpusdock.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestStore,
    UnsupportedSourceError,
    discover_source_files,
    identify_source_format,
    source_id_for,
)


TIMESTAMP = "2026-08-10T12:00:00Z"


def _store(project_root) -> ManifestStore:  # type: ignore[no-untyped-def]
    return ManifestStore(project_root, now=lambda: TIMESTAMP)


def test_registration_writes_a_versioned_content_addressed_manifest(tmp_path) -> None:  # type: ignore[no-untyped-def]
    project_root = tmp_path / "corpus"
    source_path = tmp_path / "documents" / "Notes.TXT"
    source_path.parent.mkdir()
    source_path.write_bytes(b"A locally registered source.\n")
    digest = sha256(source_path.read_bytes()).hexdigest()

    store = _store(project_root)
    _, created = store.initialize()
    registrations = store.register([source_path])

    assert created is True
    assert registrations[0].status == "registered"
    assert registrations[0].source.source_id == source_id_for(digest)

    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
        "sources": [
            {
                "source_id": source_id_for(digest),
                "sha256": digest,
                "format": "txt",
                "size_bytes": source_path.stat().st_size,
                "original_paths": [str(source_path.resolve())],
                "registered_at": TIMESTAMP,
                "registration_tool_version": __version__,
            }
        ],
    }


def test_duplicate_content_keeps_one_source_id_and_all_registered_paths(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    project_root = tmp_path / "corpus"
    first_path = tmp_path / "documents" / "first.txt"
    second_path = tmp_path / "documents" / "copies" / "second.txt"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text("Same immutable content.\n", encoding="utf-8")
    second_path.write_text("Same immutable content.\n", encoding="utf-8")

    store = _store(project_root)
    first = store.register([first_path])
    second = store.register([second_path])
    repeated = store.register([second_path])
    manifest = store.load()

    assert first[0].status == "registered"
    assert second[0].status == "additional_path_registered"
    assert repeated[0].status == "already_registered"
    assert list(manifest.sources) == [first[0].source.source_id]
    assert manifest.sources[first[0].source.source_id].original_paths == (
        str(first_path.resolve()),
        str(second_path.resolve()),
    )


def test_mirror_reconciliation_is_content_idempotent_and_prunes_absent_sources(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    project_root = tmp_path / "corpus"
    mirror = tmp_path / "documents"
    mirror.mkdir()
    retained = mirror / "retained.txt"
    removed = mirror / "removed.txt"
    retained.write_text("Stable content.\n", encoding="utf-8")
    removed.write_text("Content to remove.\n", encoding="utf-8")
    store = _store(project_root)

    first = store.reconcile_mirror(discover_source_files(mirror))
    retained_id = source_id_for(sha256(retained.read_bytes()).hexdigest())
    removed_id = source_id_for(sha256(removed.read_bytes()).hexdigest())
    assert set(first.added_source_ids) == {retained_id, removed_id}

    moved = mirror / "nested" / "renamed.txt"
    moved.parent.mkdir()
    retained.rename(moved)
    removed.unlink()
    added = mirror / "added.txt"
    added.write_text("New content.\n", encoding="utf-8")
    second = store.reconcile_mirror(discover_source_files(mirror))
    repeated = store.reconcile_mirror(discover_source_files(mirror))

    assert second.removed_source_ids == (removed_id,)
    assert second.added_source_ids == (
        source_id_for(sha256(added.read_bytes()).hexdigest()),
    )
    assert retained_id in second.retained_source_ids
    assert store.load().sources[retained_id].original_paths == (str(moved.resolve()),)
    assert repeated.changed is False
    assert repeated.added_source_ids == ()
    assert repeated.removed_source_ids == ()


@pytest.mark.parametrize(
    ("filename", "expected_format"),
    [
        ("notes.txt", "txt"),
        ("paper.PDF", "pdf"),
        ("book.epub", "epub"),
        ("legacy.MOBI", "mobi"),
        ("report.docx", "docx"),
    ],
)
def test_identifies_every_supported_registration_format(
    filename: str, expected_format: str
) -> None:
    assert identify_source_format(filename) == expected_format


def test_directory_discovery_skips_unsupported_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    documents = tmp_path / "documents"
    nested = documents / "nested"
    nested.mkdir(parents=True)
    (documents / "notes.txt").write_text("Notes", encoding="utf-8")
    (nested / "report.PDF").write_bytes(b"not parsed during registration")
    (documents / "cover.jpg").write_bytes(b"not a supported source")

    discovered = discover_source_files(documents)

    assert [path.name for path in discovered] == ["report.PDF", "notes.txt"]


def test_single_unsupported_source_is_rejected(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source_path = tmp_path / "cover.jpg"
    source_path.write_bytes(b"not a supported source")

    with pytest.raises(UnsupportedSourceError, match="Unsupported source format"):
        discover_source_files(source_path)
