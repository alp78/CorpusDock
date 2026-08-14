from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from corpusdock.extraction import artifact_path_for
from corpusdock.manifest import ManifestStore, source_id_for
from corpusdock.retrieval import index_status_report
from corpusdock.synchronization import (
    configure_input_mirror,
    load_pipeline_config,
    synchronize_input_mirror,
)


def _source_id(text: str) -> str:
    return source_id_for(sha256(text.encode()).hexdigest())


def test_input_mirror_sync_is_idempotent_and_tracks_add_move_remove(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    mirror = tmp_path / "input"
    mirror.mkdir()
    ManifestStore(project).initialize()
    first_text = "A copper pin aligns the reusable fixture.\n"
    second_text = "A paper ledger records each local inspection.\n"
    first = mirror / "first.txt"
    second = mirror / "second.txt"
    first.write_bytes(first_text.encode())
    second.write_bytes(second_text.encode())
    config = configure_input_mirror(project, mirror, sentence_processor="rule")

    initial = synchronize_input_mirror(project, config)
    extraction_path = artifact_path_for(project, _source_id(first_text))
    initial_artifact = extraction_path.read_bytes()
    repeated = synchronize_input_mirror(project, config)

    assert initial.added_sources == 2
    assert initial.extracted_sources == 2
    assert initial.chunked_sources == 2
    assert initial.exact_index_rebuilt
    assert repeated.added_sources == 0
    assert repeated.removed_sources == 0
    assert repeated.extracted_sources == 0
    assert repeated.reused_extractions == 2
    assert repeated.chunked_sources == 0
    assert repeated.reused_chunks == 2
    assert not repeated.exact_index_rebuilt
    assert extraction_path.read_bytes() == initial_artifact

    moved = mirror / "renamed.txt"
    first.rename(moved)
    renamed = synchronize_input_mirror(project, config)
    repointed = json.loads(extraction_path.read_text(encoding="utf-8"))

    assert renamed.added_sources == 0
    assert renamed.removed_sources == 0
    assert renamed.repointed_extractions == 1
    assert renamed.extracted_sources == 0
    assert repointed["source_path"] == str(moved.resolve())

    moved.unlink()
    third_text = "A ceramic guide keeps the assembly square.\n"
    (mirror / "third.txt").write_bytes(third_text.encode())
    changed = synchronize_input_mirror(project, config)

    assert changed.added_sources == 1
    assert changed.removed_sources == 1
    assert changed.extracted_sources == 1
    assert changed.chunked_sources == 1
    assert changed.pruned_extraction_artifacts == 1
    assert not extraction_path.exists()
    assert set(ManifestStore(project).load().sources) == {
        _source_id(second_text),
        _source_id(third_text),
    }
    assert index_status_report(project)["status"] == "ready"
    assert load_pipeline_config(project) == config


def test_empty_input_mirror_removes_all_registered_sources(tmp_path: Path) -> None:
    project = tmp_path / "project"
    mirror = tmp_path / "input"
    mirror.mkdir()
    ManifestStore(project).initialize()
    source = mirror / "source.txt"
    source.write_text("Temporary source text.\n", encoding="utf-8")
    config = configure_input_mirror(project, mirror, sentence_processor="rule")
    synchronize_input_mirror(project, config)

    source.unlink()
    emptied = synchronize_input_mirror(project, config)

    assert emptied.unique_sources == 0
    assert emptied.removed_sources == 1
    assert emptied.indexed_chunks == 0
    assert ManifestStore(project).load().sources == {}
    assert index_status_report(project)["status"] == "ready"


def test_pipeline_configuration_persists_sentence_device(tmp_path: Path) -> None:
    mirror = tmp_path / "input"
    mirror.mkdir()
    project = tmp_path / "project"

    config = configure_input_mirror(project, mirror, sentence_device="cuda")

    assert config.sentence_device == "cuda"
    assert load_pipeline_config(project) == config
