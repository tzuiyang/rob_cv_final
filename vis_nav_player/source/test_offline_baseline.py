"""
Focused tests for offline navigation using the uploaded exploration_data dataset.

Run:
    conda activate game
    cd vis_nav_player
    python source/test_offline_baseline.py
"""

from __future__ import annotations

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "exploration_data")

PASSED = 0
FAILED = 0


def run_test(name, func):
    global PASSED, FAILED
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print(f"{'=' * 60}")
    try:
        func()
        PASSED += 1
        print("  >> PASSED")
    except Exception as e:
        FAILED += 1
        print(f"  >> FAILED: {e}")


def _import_baseline():
    source_dir = os.path.join(ROOT, "source")
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    import baseline  # noqa: WPS433
    return baseline


def test_uploaded_dataset_exists():
    assert os.path.isdir(DATASET_DIR), f"Missing dataset dir: {DATASET_DIR}"
    assert os.path.exists(os.path.join(DATASET_DIR, "data_info.json")), "Missing data_info.json"
    assert os.path.isdir(os.path.join(DATASET_DIR, "images")), "Missing images/"
    print(f"  Dataset OK: {DATASET_DIR}")


def test_resolve_dataset_paths_legacy():
    baseline = _import_baseline()
    info_path, image_dir = baseline.resolve_dataset_paths(DATASET_DIR)
    assert info_path.endswith("data_info.json")
    assert image_dir.endswith("images")
    assert os.path.exists(info_path), f"Missing resolved info path: {info_path}"
    assert os.path.isdir(image_dir), f"Missing resolved image dir: {image_dir}"
    print(f"  info_path={info_path}")
    print(f"  image_dir={image_dir}")


def test_load_motion_frames_filters_idle_and_multi_action():
    baseline = _import_baseline()
    motion_frames, _, _ = baseline.load_motion_frames(DATASET_DIR, subsample_rate=1)
    assert len(motion_frames) > 0, "No motion frames loaded"
    actions = {m["action"] for m in motion_frames}
    assert "IDLE" not in actions, "IDLE frames should be filtered"
    for frame in motion_frames[:20]:
        assert frame["action"] in {"FORWARD", "LEFT", "RIGHT", "BACKWARD"}
    print(f"  motion_frames={len(motion_frames)}")
    print(f"  actions={sorted(actions)}")


def test_load_motion_frames_subsamples_correctly():
    baseline = _import_baseline()
    full_motion, _, _ = baseline.load_motion_frames(DATASET_DIR, subsample_rate=1)
    subsampled, _, _ = baseline.load_motion_frames(DATASET_DIR, subsample_rate=5)
    expected = len(full_motion[::5])
    assert len(subsampled) == expected, f"{len(subsampled)} != {expected}"
    assert subsampled[0]["step"] == full_motion[0]["step"]
    print(f"  full={len(full_motion)} subsampled={len(subsampled)}")


def test_load_motion_frames_returns_legacy_boundary():
    baseline = _import_baseline()
    motion_frames, _, traj_boundaries = baseline.load_motion_frames(DATASET_DIR, subsample_rate=5)
    assert len(traj_boundaries) == 1, f"Expected 1 legacy boundary, got {traj_boundaries}"
    assert traj_boundaries[0] == (0, len(motion_frames))
    print(f"  traj_boundaries={traj_boundaries}")


def test_keyboard_player_uses_selected_data_dir():
    baseline = _import_baseline()
    player = baseline.KeyboardPlayerPyGame(data_dir=DATASET_DIR, subsample_rate=5)
    assert player.data_dir == DATASET_DIR
    assert player.image_dir == os.path.join(DATASET_DIR, "images")
    assert len(player.motion_frames) > 0, "Player loaded zero motion frames"
    print(f"  loaded_motion_frames={len(player.motion_frames)}")


def test_pre_navigation_defers_without_targets():
    baseline = _import_baseline()
    player = baseline.KeyboardPlayerPyGame(data_dir=DATASET_DIR, subsample_rate=5)
    player.get_target_images = lambda: []
    player.pre_navigation()
    assert player.database is None, "Database should not build before targets exist"
    assert player.goal_node is None, "Goal should remain unset without targets"
    print("  pre_navigation deferred as expected")


def test_goal_matching_uses_all_targets():
    baseline = _import_baseline()
    player = baseline.KeyboardPlayerPyGame(data_dir=DATASET_DIR, subsample_rate=20, n_clusters=8, top_k_shortcuts=5)
    player.database = None
    player._build_database()
    player.G = None
    player._build_graph()
    calls = []

    def fake_extract(_img):
        idx = len(calls)
        calls.append(idx)
        vec = player.database[idx % len(player.database)].copy()
        return vec

    player.extractor.extract = fake_extract
    player.get_target_images = lambda: [object(), object(), object(), object()]
    player._setup_goal()
    assert len(calls) == 4, f"Expected 4 target extracts, got {len(calls)}"
    assert player.goal_node is not None
    print(f"  goal_node={player.goal_node}")


def test_temporal_edges_respect_trajectory_boundaries():
    baseline = _import_baseline()
    synthetic_motion = [
        {"step": 0, "image": "a.jpg", "action": "FORWARD", "traj_id": "traj_0", "image_root": "r0"},
        {"step": 1, "image": "b.jpg", "action": "FORWARD", "traj_id": "traj_0", "image_root": "r0"},
        {"step": 2, "image": "c.jpg", "action": "FORWARD", "traj_id": "traj_1", "image_root": "r1"},
    ]
    player = baseline.KeyboardPlayerPyGame(data_dir=DATASET_DIR, subsample_rate=5)
    player.motion_frames = synthetic_motion
    player.file_list = [m["image"] for m in synthetic_motion]
    player.traj_boundaries = [(0, 2), (2, 3)]
    player.database = __import__("numpy").zeros((3, 8))
    player.top_k_shortcuts = 0
    player._build_graph()
    assert player.G.has_edge(0, 1), "Expected temporal edge within traj_0"
    assert not player.G.has_edge(1, 2), "Unexpected temporal edge across trajectory boundary"
    print(f"  edges={sorted(player.G.edges())}")


def test_load_img_uses_dataset_image_root():
    baseline = _import_baseline()
    player = baseline.KeyboardPlayerPyGame(data_dir=DATASET_DIR, subsample_rate=5)
    img = player._load_img(0)
    assert img is not None, "Failed to load first dataset image"
    assert len(img.shape) == 3, f"Unexpected image shape: {img.shape}"
    print(f"  img_shape={img.shape}")


if __name__ == "__main__":
    run_test("dataset exists", test_uploaded_dataset_exists)
    run_test("resolve dataset paths", test_resolve_dataset_paths_legacy)
    run_test("motion filtering", test_load_motion_frames_filters_idle_and_multi_action)
    run_test("subsampling", test_load_motion_frames_subsamples_correctly)
    run_test("legacy boundary", test_load_motion_frames_returns_legacy_boundary)
    run_test("player uses selected data dir", test_keyboard_player_uses_selected_data_dir)
    run_test("pre_navigation defers without targets", test_pre_navigation_defers_without_targets)
    run_test("goal matching uses all targets", test_goal_matching_uses_all_targets)
    run_test("temporal edges respect trajectory boundaries", test_temporal_edges_respect_trajectory_boundaries)
    run_test("load_img uses dataset image root", test_load_img_uses_dataset_image_root)

    print(f"\nPASSED: {PASSED}")
    print(f"FAILED: {FAILED}")
    raise SystemExit(1 if FAILED else 0)
