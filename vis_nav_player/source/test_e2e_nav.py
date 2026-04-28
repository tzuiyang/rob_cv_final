"""End-to-end navigation simulation test.

Builds pipeline from exploration_data, simulates navigation by feeding
real images along the shortest path, and verifies:
1. Pipeline builds successfully
2. Localization tracks movement (hops decrease)
3. CHECKIN triggers when at goal frame
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from player import (
    KeyboardPlayerPyGame, Action, NavState, ExploreState,
    SEARCH_HOPS_ENTER, SEARCH_SIM_THRESHOLD,
    STUCK_MSE_THRESHOLD, STUCK_FRAME_SIZE,
)
import numpy as np
import cv2
import networkx as nx

PASSED = 0
FAILED = 0

def run_test(name, func):
    global PASSED, FAILED
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    try:
        func()
        PASSED += 1
        print("  >> PASSED")
    except Exception as e:
        FAILED += 1
        print(f"  >> FAILED: {e}")
        import traceback
        traceback.print_exc()

# Build pipeline once
print("Building pipeline...")
p = KeyboardPlayerPyGame(
    n_clusters=64, subsample_rate=5, top_k_shortcuts=100,
    data_dir="exploration_data", offline_navigation=True
)
p._load_trajectory_data()
p._build_database()
p._build_graph()
print("Pipeline ready.\n")


def test_pipeline_build():
    """Verify database, graph, and features are built correctly."""
    assert p.database is not None, "Database is None"
    assert p.database.shape[0] > 2000, f"Database too small: {p.database.shape[0]}"
    assert p.database.shape[1] == 8192, f"Wrong VLAD dim: {p.database.shape[1]}"
    print(f"  Database: {p.database.shape}")

    assert p.G is not None, "Graph is None"
    assert p.G.number_of_nodes() > 2000, f"Too few nodes: {p.G.number_of_nodes()}"
    assert p.G.number_of_edges() > 2000, f"Too few edges: {p.G.number_of_edges()}"
    print(f"  Graph: {p.G.number_of_nodes()} nodes, {p.G.number_of_edges()} edges")

    n_comp = nx.number_connected_components(p.G)
    assert n_comp == 1, f"Graph has {n_comp} components (should be 1)"
    print(f"  Connected components: {n_comp}")


def test_sift_feature_quality():
    """Verify SIFT extracts enough features with new params."""
    cache = p.extractor._sift_cache
    assert len(cache) == len(p.file_list), f"Cache incomplete: {len(cache)}/{len(p.file_list)}"
    counts = [len(v) for v in cache.values()]
    low = sum(1 for c in counts if c < 10)
    median = sorted(counts)[len(counts) // 2]
    print(f"  Feature counts: min={min(counts)}, max={max(counts)}, median={median}")
    print(f"  Low-feature images (<10): {low}/{len(counts)}")
    assert low == 0, f"{low} images have <10 features"
    assert median > 100, f"Median features {median} too low"


def test_vlad_discriminability():
    """Adjacent frames should be more similar than distant frames."""
    n = len(p.database)
    adj_sims = [float(p.database[i] @ p.database[i + 1]) for i in range(min(200, n - 1))]
    dist_sims = [float(p.database[i] @ p.database[i + 200]) for i in range(min(200, n - 200))]
    adj_mean = np.mean(adj_sims)
    dist_mean = np.mean(dist_sims)
    print(f"  Adjacent similarity: mean={adj_mean:.4f}")
    print(f"  Distant similarity:  mean={dist_mean:.4f}")
    assert adj_mean > dist_mean, f"Adjacent ({adj_mean:.4f}) not > distant ({dist_mean:.4f})"
    print(f"  Ratio: {adj_mean / max(abs(dist_mean), 1e-6):.1f}x")


def test_localization_tracks_movement():
    """Feed frames along a path, verify localization follows."""
    n = len(p.database)
    start = n // 4
    goal = n * 3 // 4
    path = nx.shortest_path(p.G, start, goal, weight="weight")
    print(f"  Path: node {start} -> {goal}, {len(path) - 1} hops")

    # Reset localization state
    p._prev_sims = None
    p._prev_node = start

    correct = 0
    total = 0
    for i, node_idx in enumerate(path[:50]):
        fname = p.file_list[node_idx]
        img = cv2.imread(os.path.join(p.image_dir, fname))
        if img is None:
            continue
        p.fpv = img
        cur = p._get_current_node()
        total += 1
        if abs(cur - node_idx) <= 10:
            correct += 1
        if i % 10 == 0:
            print(f"    Node {node_idx} -> localized {cur} (err={abs(cur - node_idx)})")

    accuracy = correct / total if total > 0 else 0
    print(f"  Accuracy (within ±10): {correct}/{total} = {accuracy:.1%}")
    assert accuracy > 0.5, f"Localization accuracy {accuracy:.1%} too low"


def test_hops_decrease_along_path():
    """Hops should decrease as we walk along shortest path toward goal."""
    n = len(p.database)
    start = n // 4
    goal = n * 3 // 4
    p.goal_node = goal
    p.goal_candidates = [goal]
    p.goal_candidate_index = 0

    path = nx.shortest_path(p.G, start, goal, weight="weight")
    initial_hops = len(path) - 1
    print(f"  Walking from node {start} to {goal} ({initial_hops} hops)")

    p._prev_sims = None
    p._prev_node = start

    hops_log = []
    for node_idx in path[:50]:
        fname = p.file_list[node_idx]
        img = cv2.imread(os.path.join(p.image_dir, fname))
        if img is None:
            continue
        p.fpv = img
        cur = p._get_current_node()
        nav_path = p._get_path(cur)
        hops = len(nav_path) - 1
        hops_log.append(hops)

    print(f"  Hops: {hops_log[0]} -> {hops_log[-1]}")
    improving = "YES" if hops_log[-1] < hops_log[0] else "NO"
    print(f"  Improving: {improving}")
    assert hops_log[-1] < hops_log[0], f"Hops not decreasing: {hops_log[0]} -> {hops_log[-1]}"


def test_checkin_at_goal():
    """When at the goal frame, visual similarity should trigger CHECKIN."""
    n = len(p.database)
    goal = n * 3 // 4
    p.goal_node = goal
    p._target_vlads = [p.database[goal].copy() for _ in range(4)]

    fname = p.file_list[goal]
    img = cv2.imread(os.path.join(p.image_dir, fname))
    assert img is not None, f"Cannot read goal image: {fname}"

    vlad = p.extractor.extract(img)
    sims = [float(vlad @ tv) for tv in p._target_vlads]
    max_sim = max(sims)
    print(f"  Goal frame visual similarity: {max_sim:.4f}")
    print(f"  SEARCH_SIM_THRESHOLD: {SEARCH_SIM_THRESHOLD}")
    assert max_sim > SEARCH_SIM_THRESHOLD, f"Similarity {max_sim:.4f} below threshold {SEARCH_SIM_THRESHOLD}"

    # Also check localization places us at/near goal
    p._prev_sims = None
    p._prev_node = goal - 5
    p.fpv = img
    cur = p._get_current_node()
    nav_path = p._get_path(cur)
    hops = len(nav_path) - 1
    print(f"  Localized to node {cur}, hops to goal = {hops}")
    assert hops <= 5, f"At goal but hops = {hops}"


def test_stuck_detection_with_real_frames():
    """Verify stuck detection works with real maze textures."""
    n = len(p.database)
    # Same frame twice = stuck
    fname = p.file_list[n // 2]
    img = cv2.imread(os.path.join(p.image_dir, fname))
    assert img is not None
    stuck = p._is_stuck(img, img.copy())
    print(f"  Same frame: stuck={stuck} (should be True)")
    assert stuck, "Same frame should be detected as stuck"

    # Different frames = not stuck (adjacent frames should differ enough)
    fname2 = p.file_list[n // 2 + 5]
    img2 = cv2.imread(os.path.join(p.image_dir, fname2))
    assert img2 is not None
    stuck2 = p._is_stuck(img, img2)
    # Measure MSE for diagnostics
    g1 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), STUCK_FRAME_SIZE)
    g2 = cv2.resize(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), STUCK_FRAME_SIZE)
    mse = float(np.mean((g1.astype(np.float32) - g2.astype(np.float32)) ** 2))
    print(f"  5-apart frames: stuck={stuck2}, MSE={mse:.1f} (threshold={STUCK_MSE_THRESHOLD})")
    assert not stuck2, f"5-apart frames detected as stuck (MSE={mse:.1f})"


def test_graph_path_no_disconnect():
    """_get_path should return valid path for any node pair in connected graph."""
    n = len(p.database)
    p.goal_node = n - 1
    p.goal_candidates = [n - 1]
    p.goal_candidate_index = 0

    for start in [0, n // 4, n // 2, n * 3 // 4]:
        path = p._get_path(start)
        hops = len(path) - 1
        print(f"  Node {start} -> {n - 1}: {hops} hops")
        assert hops < 9999, f"No path from {start} to {n - 1}"
        assert hops > 0, f"Zero hops from {start} to {n - 1}"


# Run all tests
run_test("Pipeline Build", test_pipeline_build)
run_test("SIFT Feature Quality", test_sift_feature_quality)
run_test("VLAD Discriminability", test_vlad_discriminability)
run_test("Localization Tracks Movement", test_localization_tracks_movement)
run_test("Hops Decrease Along Path", test_hops_decrease_along_path)
run_test("CHECKIN At Goal", test_checkin_at_goal)
run_test("Stuck Detection With Real Frames", test_stuck_detection_with_real_frames)
run_test("Graph Path No Disconnect", test_graph_path_no_disconnect)

print(f"\n{'#'*60}")
print(f"  RESULTS: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED}")
if FAILED == 0:
    print("  ALL TESTS PASSED")
print(f"{'#'*60}")
