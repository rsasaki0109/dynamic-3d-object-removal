import json
from pathlib import Path

import pytest

from scripts import validate_frozen_nuscenes_suite as suite


def test_streaming_metadata_handles_nested_objects_unicode_and_chunk_boundaries(tmp_path):
    rows = [{"token": "日本語", "data": [1, {"quoted": 'a } ] \\" b'}]},
            {"token": "next", "active": True, "value": None}]
    path = tmp_path / "table.json"
    path.write_text(json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8")
    assert list(suite.iter_table(path, chunk_size=3)) == rows
    path.write_text("[ ]\n")
    assert list(suite.iter_table(path, chunk_size=1)) == []


@pytest.mark.parametrize("contents", ['[{"a": 1}', '[{"a":', '[{},]', '[{}] extra', '{"a":1}', '[1]'])
def test_invalid_or_truncated_metadata_never_becomes_a_complete_table(tmp_path, contents):
    path = tmp_path / "table.json"
    path.write_text(contents)
    with pytest.raises(ValueError):
        list(suite.iter_table(path, chunk_size=2))


def test_selection_excludes_mini_and_unavailable_logs_without_gt_screening():
    scenes = [{"name": name, "log_token": token} for name, token in
              [("scene-0003", "a"), ("scene-0002", "a"), ("scene-0001", "b"), ("scene-0061", "a")]]
    logs = {"a": {"logfile": "available"}, "b": {"logfile": "other_archive"}}
    assert suite.select_scenes(scenes, logs, ["available"], ["scene-0061"], 2) == ["scene-0002", "scene-0003"]
    with pytest.raises(ValueError, match="not enough"):
        suite.select_scenes(scenes, logs, ["available"], ["scene-0061"], 3)


def test_changed_plan_is_rejected_before_metadata_or_cloud_access(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="plan digest"):
        suite.verify({}, path, "previously recorded digest", tmp_path / "missing")


def test_inventory_cannot_accept_a_mini_plan_with_a_matching_digest(tmp_path):
    path = tmp_path / "plan.json"
    plan = {"schema_version": 1, "dataset": "nuscenes-mini", "metadata_version": "v1.0-mini", "modes": list(suite.MODES)}
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="frozen trainval"):
        suite.verify(plan, path, suite.sha256(path), tmp_path / "missing")


def test_export_existing_output_is_rejected_before_any_scene_is_written(tmp_path):
    output = tmp_path / "manifests"
    output.mkdir()
    with pytest.raises(ValueError, match="root exists"):
        suite.export_manifests(tmp_path, output, {}, "digest", {}, {})
