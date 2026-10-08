from scripts.visualize_nuscenes_errors import moving_instances


def test_moving_track_anchor_follows_selected_time_order_not_metadata_order():
    # Anchoring to the middle position would miss this >2 m displacement.
    annotations = {
        "middle": [{"instance_token": "car", "translation": [1.1, 0, 0]}],
        "last": [{"instance_token": "car", "translation": [2.1, 0, 0]}],
        "first": [{"instance_token": "car", "translation": [0, 0, 0]}],
    }
    assert moving_instances(annotations, ["first", "middle", "last"]) == ["car"]
    assert moving_instances(annotations, ["middle", "last", "first"]) == []


def test_motion_cutoff_is_strict_and_single_observation_is_excluded():
    annotations = {
        "first": [{"instance_token": "boundary", "translation": [0, 0, 0]},
                  {"instance_token": "single", "translation": [100, 0, 0]}],
        "last": [{"instance_token": "boundary", "translation": [2, 0, 0]}],
    }
    assert moving_instances(annotations, ["first", "last"]) == []
