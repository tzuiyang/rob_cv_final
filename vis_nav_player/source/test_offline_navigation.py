"""
Offline navigation verification for the uploaded exploration dataset.

Run:
  conda activate game
  cd vis_nav_player
  python source/test_offline_navigation.py
"""

import json
import os
import tempfile
from pathlib import Path

import cv2
import networkx as nx
import numpy as np

from player import Action, KeyboardPlayerPyGame


PASSED = 0
FAILED = 0
ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "exploration_data"
DATA_INFO = DATASET_DIR / "data_info.json"
IMAGES_DIR = DATASET_DIR / "images"


def run_test(name, func):
    global PASSED, FAILED
    print(f"\n{'=' * 72}")
    print(f"TEST: {name}")
    print(f"{'=' * 72}")
    try:
        func()
        PASSED += 1
        print("  >> PASSED")
    except Exception as exc:
        FAILED += 1
        print(f"  >> FAILED: {exc}")


def _load_raw():
    with open(DATA_INFO) as f:
        return json.load(f)


def _motion_frames(raw):
    pure = {"FORWARD", "LEFT", "RIGHT", "BACKWARD"}
    return [d for d in raw if len(d["action"]) == 1 and d["action"][0] in pure]


def _build_small_player(subsample=100, n_clusters=8, top_k=10):
    player = KeyboardPlayerPyGame(
        n_clusters=n_clusters,
        subsample_rate=subsample,
        top_k_shortcuts=top_k,
        data_dir=str(DATASET_DIR),
        offline_navigation=True,
    )
    return player


def test_offline_dataset_exists():
    assert DATASET_DIR.is_dir(), f"Missing dataset dir: {DATASET_DIR}"
    assert DATA_INFO.is_file(), f"Missing metadata file: {DATA_INFO}"
    assert IMAGES_DIR.is_dir(), f"Missing images dir: {IMAGES_DIR}"
    print(f"  Dataset dir: {DATASET_DIR}")


def test_offline_dataset_metadata_keys():
    raw = _load_raw()
    assert len(raw) > 0, "Dataset metadata is empty"
    for item in raw[:20]:
        assert "step" in item, "Missing key: step"
        assert "image" in item, "Missing key: image"
        assert "action" in item, "Missing key: action"
    print(f"  Metadata records: {len(raw)}")


def test_offline_dataset_sample_images_exist():
    raw = _load_raw()
    sample_indices = [0, 1, 10, 100, len(raw) // 2, len(raw) - 1]
    for idx in sample_indices:
        img_name = raw[idx]["image"]
        img_path = IMAGES_DIR / img_name
        assert img_path.exists(), f"Missing image: {img_path}"
    print(f"  Verified sample images at indices: {sample_indices}")


def test_offline_dataset_has_motion_frames():
    raw = _load_raw()
    motion = _motion_frames(raw)
    assert len(motion) > 0, "No motion frames found"
    actions = {m["action"][0] for m in motion}
    assert "IDLE" not in actions, "IDLE frames should have been filtered out"
    print(f"  Motion frames: {len(motion)}")
    print(f"  Motion actions: {sorted(actions)}")


def test_player_offline_paths_resolve_correctly():
    player = _build_small_player()
    assert player.offline_navigation is True
    assert Path(player.data_dir) == DATASET_DIR.resolve()
    assert Path(player.data_info_path) == DATA_INFO.resolve()
    assert Path(player.image_dir) == IMAGES_DIR.resolve()
    print(f"  data_dir={player.data_dir}")
    print(f"  data_info_path={player.data_info_path}")
    print(f"  image_dir={player.image_dir}")


def test_cache_tag_depends_on_data_dir():
    a = KeyboardPlayerPyGame(data_dir=str(DATASET_DIR), offline_navigation=True)
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "images"))
        with open(os.path.join(tmp, "data_info.json"), "w") as f:
            json.dump([], f)
        b = KeyboardPlayerPyGame(data_dir=tmp, offline_navigation=True)
    assert a.cache_tag != b.cache_tag, "cache_tag should vary by data_dir"
    print(f"  cache_tag(dataset)={a.cache_tag}")
    print(f"  cache_tag(temp)={b.cache_tag}")


def test_load_trajectory_data_filters_idle():
    raw = _load_raw()
    expected_motion = _motion_frames(raw)
    player = _build_small_player(subsample=1)
    player._load_trajectory_data()
    assert len(player.motion_frames) == len(expected_motion), (
        f"Expected {len(expected_motion)} motion frames, got {len(player.motion_frames)}"
    )
    assert all(m["action"] in {"FORWARD", "LEFT", "RIGHT", "BACKWARD"} for m in player.motion_frames)
    print(f"  Filtered motion frames: {len(player.motion_frames)}")


def test_load_trajectory_data_subsamples():
    raw = _load_raw()
    motion = _motion_frames(raw)
    subsample = 100
    player = _build_small_player(subsample=subsample)
    player._load_trajectory_data()
    expected = motion[::subsample]
    assert len(player.motion_frames) == len(expected)
    assert player.file_list[0] == expected[0]["image"]
    assert player.file_list[-1] == expected[-1]["image"]
    print(f"  Subsample {subsample} -> {len(player.motion_frames)} frames")


def test_offline_mode_pre_navigation_defers_without_targets():
    player = _build_small_player(subsample=100)
    player.pre_navigation()
    assert player.pipeline_ready is False
    assert player.exploring_in_nav is False, "Offline mode should not schedule live exploration"
    print("  pre_navigation() deferred cleanly without targets")


def test_offline_mode_act_is_idle_before_pipeline():
    player = _build_small_player(subsample=100)
    player.pipeline_ready = False
    player.offline_navigation = True
    player.last_act = Action.IDLE
    action = player.act()
    assert action == Action.IDLE, f"Expected IDLE, got {action}"
    print("  act() returned IDLE while waiting for offline pipeline")


def test_small_offline_pipeline_smoke():
    player = _build_small_player(subsample=100, n_clusters=8, top_k=10)
    player._load_trajectory_data()
    player._build_database()
    player._build_graph()
    assert player.database is not None and player.database.shape[0] > 0
    assert player.G is not None and player.G.number_of_nodes() == player.database.shape[0]
    assert player.G.number_of_edges() >= max(0, player.G.number_of_nodes() - 1)
    print(f"  database_shape={player.database.shape}")
    print(f"  graph_nodes={player.G.number_of_nodes()} edges={player.G.number_of_edges()}")


def test_cache_files_created_for_offline_dataset():
    player = _build_small_player(subsample=100, n_clusters=8, top_k=10)
    player._load_trajectory_data()
    player._build_database()
    cache_dir = ROOT / "cache"
    sift_cache = cache_dir / f"sift_{player.cache_tag}_ss{player.subsample_rate}.pkl"
    codebook_cache = cache_dir / f"codebook_{player.cache_tag}_k{player.extractor.n_clusters}.pkl"
    assert sift_cache.exists(), f"Missing SIFT cache: {sift_cache}"
    assert codebook_cache.exists(), f"Missing codebook cache: {codebook_cache}"
    print(f"  sift_cache={sift_cache.name}")
    print(f"  codebook_cache={codebook_cache.name}")


def test_small_offline_goal_setup_smoke():
    player = _build_small_player(subsample=100, n_clusters=8, top_k=10)
    player._load_trajectory_data()
    player._build_database()
    player._build_graph()
    img = cv2.imread(str(IMAGES_DIR / player.file_list[min(10, len(player.file_list) - 1)]))
    assert img is not None, "Failed to load target image for smoke test"
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: -1
    player.set_target_images([img, img, img, img])
    player._setup_goal()
    assert player.goal_node is not None
    assert 0 <= player.goal_node < len(player.database)
    print(f"  goal_node={player.goal_node}")


def test_goal_smoothing_prefers_consensus_region():
    player = _build_small_player()
    sims = np.array([0.1, 0.2, 0.9, 0.3, 0.2], dtype=np.float32)
    smoothed = player._smooth_similarity_curve(sims, radius=1)
    assert int(np.argmax(smoothed)) in {2, 3}
    assert smoothed[2] > smoothed[1] > smoothed[0]
    assert smoothed[3] > smoothed[4]
    print(f"  smoothed={smoothed.tolist()}")


def test_goal_confidence_uses_similarity_and_consensus():
    player = _build_small_player()
    assert player._is_low_confidence_goal(0.20, 4) is True
    assert player._is_low_confidence_goal(0.40, 2) is True
    assert player._is_low_confidence_goal(0.40, 4) is False
    print("  goal confidence classification behaves as expected")


def test_exact_goal_match_is_not_marked_low_confidence():
    player = _build_small_player(subsample=100, n_clusters=8, top_k=10)
    player._load_trajectory_data()
    player._build_database()
    player._build_graph()
    img = cv2.imread(str(IMAGES_DIR / player.file_list[min(10, len(player.file_list) - 1)]))
    assert img is not None, "Failed to load target image for exact-match test"
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: -1
    player.set_target_images([img, img, img, img])
    player._setup_goal()
    assert player.low_confidence_goal is False
    assert player.goal_avg_sim > 0.9
    assert player.goal_consensus == 4
    assert player.goal_candidates[0] == player.goal_node
    print(f"  avg_sim={player.goal_avg_sim:.4f} consensus={player.goal_consensus}")


def test_localization_rejects_large_unnecessary_jump():
    player = _build_small_player()
    player.database = np.zeros((300, 4), dtype=np.float32)
    player.fpv = np.zeros((20, 20, 3), dtype=np.uint8)
    player._prev_node = 50
    player._prev_sims = None

    feat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    player.extractor.extract = lambda img: feat
    player.database[50] = np.array([0.94, 0.0, 0.0, 0.0], dtype=np.float32)
    player.database[260] = np.array([0.95, 0.0, 0.0, 0.0], dtype=np.float32)

    cur = player._get_current_node()
    assert cur == 50, f"Expected local node 50, got {cur}"
    print(f"  localized_node={cur}")


def test_goal_candidates_are_separated_and_ranked():
    player = _build_small_player()
    scores = np.array([0.1, 0.9, 0.85, 0.2, 0.8, 0.1, 0.75], dtype=np.float32)
    candidates = player._select_goal_candidates(scores, count=3, separation=2)
    assert candidates == [1, 4, 6]
    print(f"  candidates={candidates}")


def test_active_goal_node_uses_candidate_index():
    player = _build_small_player()
    player.goal_candidates = [4, 8, 12]
    player.goal_candidate_index = 1
    assert player._active_goal_node() == 8
    print(f"  active_goal={player._active_goal_node()}")


def test_advance_goal_candidate_switches_target():
    player = _build_small_player()
    player.goal_candidates = [4, 8, 12]
    player.goal_candidate_index = 0
    player.goal_node = 4
    moved = player._advance_goal_candidate()
    assert moved is True
    assert player.goal_candidate_index == 1
    assert player.goal_node == 8
    print(f"  new_goal={player.goal_node}")


def test_get_path_uses_active_goal_candidate():
    player = _build_small_player()
    player.G = nx.Graph()
    player.G.add_edge(0, 1, weight=1.0)
    player.G.add_edge(1, 2, weight=1.0)
    player.G.add_edge(0, 4, weight=1.0)
    player.goal_node = 2
    player.goal_candidates = [4, 2]
    player.goal_candidate_index = 1
    path = player._get_path(0)
    assert path == [0, 1, 2], f"Expected path to active candidate, got {path}"
    print(f"  path={path}")


def test_low_confidence_stability_streak_increments_on_consistent_progress():
    player = _build_small_player()
    player.low_confidence_goal = True
    player.prev_avg_hops = None
    player._update_low_confidence_stability(cur=100, hops=8, avg_hops=8)
    player.prev_avg_hops = 8
    player._update_low_confidence_stability(cur=120, hops=6, avg_hops=7)
    assert player.low_confidence_streak == 2
    print(f"  streak={player.low_confidence_streak}")


def test_low_confidence_stability_streak_resets_on_large_regression():
    player = _build_small_player()
    player.low_confidence_goal = True
    player.prev_avg_hops = None
    player._update_low_confidence_stability(cur=100, hops=8, avg_hops=8)
    player.prev_avg_hops = 8
    player._update_low_confidence_stability(cur=260, hops=30, avg_hops=20)
    assert player.low_confidence_streak == 0
    print(f"  streak={player.low_confidence_streak}")


def test_low_confidence_stability_streak_resets_on_escape():
    player = _build_small_player()
    player.low_confidence_goal = True
    player.low_confidence_streak = 3
    player._last_gradient_node = 123
    player._reset_low_confidence_stability()
    assert player.low_confidence_streak == 0
    assert player._last_gradient_node is None
    print("  stability reset cleared low-confidence state")


def test_low_confidence_does_not_enter_approach_too_early():
    player = _build_small_player()
    player.low_confidence_goal = True
    player.low_confidence_streak = 1
    assert player._should_enter_approach(5) is False
    print("  approach gate held under low confidence")


def test_low_confidence_enters_approach_after_stable_gradient_checks():
    player = _build_small_player()
    player.low_confidence_goal = True
    player.low_confidence_streak = 2
    assert player._should_enter_approach(5) is True
    print("  approach gate opened after stable checks")


def test_low_confidence_does_not_enter_search_without_strong_near_goal_evidence():
    player = _build_small_player()
    player.low_confidence_goal = True
    player.low_confidence_streak = 2
    assert player._should_enter_search(1) is False
    player.low_confidence_streak = 3
    assert player._should_enter_search(4) is False
    print("  search gate remains closed without strong evidence")


def test_backup_checkin_is_stricter_under_low_confidence():
    player = _build_small_player()
    player.low_confidence_goal = False
    high_conf_threshold = player._backup_checkin_threshold()
    player.low_confidence_goal = True
    low_conf_threshold = player._backup_checkin_threshold()
    assert low_conf_threshold > high_conf_threshold
    print(f"  high_conf={high_conf_threshold} low_conf={low_conf_threshold}")


def test_high_confidence_behavior_still_allows_normal_approach_transition():
    player = _build_small_player()
    player.low_confidence_goal = False
    assert player._should_enter_approach(10) is True
    assert player._should_enter_search(3) is True
    print("  high-confidence thresholds remain permissive")


def test_missing_data_info_graceful():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "images"))
        player = KeyboardPlayerPyGame(data_dir=tmp, offline_navigation=True)
        player._load_trajectory_data()
        assert player.motion_frames == []
        assert player.file_list == []
    print("  Missing data_info.json handled without crash")


def test_empty_dataset_directory_graceful():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "images"))
        with open(os.path.join(tmp, "data_info.json"), "w") as f:
            json.dump([], f)
        player = KeyboardPlayerPyGame(data_dir=tmp, offline_navigation=True)
        player._load_trajectory_data()
        player._build_database()
        player._build_graph()
        assert player.database.shape[0] == 0
        assert player.G.number_of_nodes() == 0
    print("  Empty dataset handled without crash")


def test_missing_image_is_skipped_not_crash():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = os.path.join(tmp, "images")
        os.makedirs(img_dir)
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(img_dir, "ok.jpg"), img)
        with open(os.path.join(tmp, "data_info.json"), "w") as f:
            json.dump([
                {"step": 0, "image": "missing.jpg", "action": ["FORWARD"]},
                {"step": 1, "image": "ok.jpg", "action": ["FORWARD"]},
            ], f)
        player = KeyboardPlayerPyGame(
            data_dir=tmp, offline_navigation=True, subsample_rate=1, n_clusters=4, top_k_shortcuts=2
        )
        player._load_trajectory_data()
        player._build_database()
        assert player.database.shape[0] in {0, 2}
        if player.database.shape[0] == 0:
            print("  No descriptors found; graceful empty database")
        else:
            print("  Valid image retained despite one missing image")
    print("  Missing image skipped without pipeline crash")


if __name__ == "__main__":
    tests = [
        ("offline_dataset_exists", test_offline_dataset_exists),
        ("offline_dataset_metadata_keys", test_offline_dataset_metadata_keys),
        ("offline_dataset_sample_images_exist", test_offline_dataset_sample_images_exist),
        ("offline_dataset_has_motion_frames", test_offline_dataset_has_motion_frames),
        ("player_offline_paths_resolve_correctly", test_player_offline_paths_resolve_correctly),
        ("cache_tag_depends_on_data_dir", test_cache_tag_depends_on_data_dir),
        ("load_trajectory_data_filters_idle", test_load_trajectory_data_filters_idle),
        ("load_trajectory_data_subsamples", test_load_trajectory_data_subsamples),
        ("offline_mode_pre_navigation_defers_without_targets", test_offline_mode_pre_navigation_defers_without_targets),
        ("offline_mode_act_is_idle_before_pipeline", test_offline_mode_act_is_idle_before_pipeline),
        ("small_offline_pipeline_smoke", test_small_offline_pipeline_smoke),
        ("cache_files_created_for_offline_dataset", test_cache_files_created_for_offline_dataset),
        ("small_offline_goal_setup_smoke", test_small_offline_goal_setup_smoke),
        ("goal_smoothing_prefers_consensus_region", test_goal_smoothing_prefers_consensus_region),
        ("goal_confidence_uses_similarity_and_consensus", test_goal_confidence_uses_similarity_and_consensus),
        ("exact_goal_match_is_not_marked_low_confidence", test_exact_goal_match_is_not_marked_low_confidence),
        ("localization_rejects_large_unnecessary_jump", test_localization_rejects_large_unnecessary_jump),
        ("goal_candidates_are_separated_and_ranked", test_goal_candidates_are_separated_and_ranked),
        ("active_goal_node_uses_candidate_index", test_active_goal_node_uses_candidate_index),
        ("advance_goal_candidate_switches_target", test_advance_goal_candidate_switches_target),
        ("get_path_uses_active_goal_candidate", test_get_path_uses_active_goal_candidate),
        ("low_confidence_stability_streak_increments_on_consistent_progress", test_low_confidence_stability_streak_increments_on_consistent_progress),
        ("low_confidence_stability_streak_resets_on_large_regression", test_low_confidence_stability_streak_resets_on_large_regression),
        ("low_confidence_stability_streak_resets_on_escape", test_low_confidence_stability_streak_resets_on_escape),
        ("low_confidence_does_not_enter_approach_too_early", test_low_confidence_does_not_enter_approach_too_early),
        ("low_confidence_enters_approach_after_stable_gradient_checks", test_low_confidence_enters_approach_after_stable_gradient_checks),
        ("low_confidence_does_not_enter_search_without_strong_near_goal_evidence", test_low_confidence_does_not_enter_search_without_strong_near_goal_evidence),
        ("backup_checkin_is_stricter_under_low_confidence", test_backup_checkin_is_stricter_under_low_confidence),
        ("high_confidence_behavior_still_allows_normal_approach_transition", test_high_confidence_behavior_still_allows_normal_approach_transition),
        ("missing_data_info_graceful", test_missing_data_info_graceful),
        ("empty_dataset_directory_graceful", test_empty_dataset_directory_graceful),
        ("missing_image_is_skipped_not_crash", test_missing_image_is_skipped_not_crash),
    ]

    for name, fn in tests:
        run_test(name, fn)

    print(f"\nSUMMARY: {PASSED} passed, {FAILED} failed")
    raise SystemExit(1 if FAILED else 0)
