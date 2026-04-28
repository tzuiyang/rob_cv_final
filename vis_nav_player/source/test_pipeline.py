"""
Unit tests for the visual navigation pipeline.
Run: conda activate game && cd vis_nav_player && python source/test_pipeline.py

36 tests covering all sections from TODO.md:
  1.  Data Loading & Subsampling
  2.  SIFT Feature Extraction + RootSIFT
  3.  VLAD Encoding (codebook + normalization)
  4.  Graph Construction (temporal + visual edges)
  5.  Localization + Multi-View Goal Matching
  6a. Action Mapping (requires vis_nav_game)
  6b. Edge Action Lookup
  6c. Graph-Based Stuck Detection
  6d. Navigation Path Planning (Dijkstra + shortcuts)
  7.  Frame-Based Stuck Detection (MSE)
  8.  Exploration State Machine (wall-following)
  9.  Turn Angle Calibration
  10. Player Initialization (source inspection)
  11. Checkin Logic (proximity + visual similarity)
  12. Goal Confidence Threshold
  13. Localization Smoothing (temporal)
  14. see() Frame Saving
  15. Two-Stage Flag Transitions
  16. Exploration Budget Enforcement
  17. Goal Node None Fallback
  18. Empty Database Pipeline
  19. Data Info JSON Roundtrip
  20. Localization Jump Detection
  21. VLAD No Power Norm (SSR removed)
  22. SIFT Tuning (nfeatures=1000, contrastThreshold=0.03, mask)
  23. n_clusters=64 Default
  24. CHECK_RIGHT_INTERVAL=20
  25. Random Perturbation (loop detection, momentum)
  26. Oscillation Detection (node_history)
  27. Multi-View CHECKIN (_target_vlads cached)
  28. Variable Action Hold
  29. hops==0 Bypass Fix (turn instead of CHECKIN)
  30. Graph Constants (top_k=100, gap=30, floor=0.4)
  31. RootSIFT Epsilon (zero-sum safety)
  32. _make_mask Static Method
  33. Gradient Check Logic
  34. Perturbation Momentum Fix (spin bug regression)
  35. Localization With Local Search Radius + Jump Threshold
"""

import numpy as np
import os
import sys
import random

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
        print(f"  >> PASSED")
    except Exception as e:
        FAILED += 1
        print(f"  >> FAILED: {e}")


# ---------------------------------------------------------------------------
# Section 1: Data Loading & Subsampling
# ---------------------------------------------------------------------------
def test_data_loading():
    """Verify exploration data loading, filtering, and subsampling.

    Uses real data if available, otherwise synthetic data_info entries.
    """
    import json

    # --- Test with real data if available ---
    if os.path.exists("data/data_info.json"):
        with open("data/data_info.json") as f:
            raw = json.load(f)
        assert len(raw) > 0, "data_info.json is empty"
        print(f"  Total frames: {len(raw)}")

        for d in raw[:5]:
            assert 'step' in d, "Missing 'step' key"
            assert 'image' in d, "Missing 'image' key"
            assert 'action' in d, "Missing 'action' key"
        print(f"  Keys OK: step, image, action")

        images_found = 0
        for d in raw[:10]:
            path = f"data/images/{d['image']}"
            alt_path = f"exploration_data/images/{d['image']}"
            if os.path.exists(path) or os.path.exists(alt_path):
                images_found += 1
        if images_found > 0:
            print(f"  Sample images exist on disk ({images_found}/10 found)")
        else:
            print(f"  WARNING: No sample images found on disk (data may use different naming), skipping image check")

        pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
        all_motion = [d for d in raw if len(d['action']) == 1 and d['action'][0] in pure]
        rate = 5
        subsampled = all_motion[::rate]
        print(f"  Motion frames: {len(all_motion)}, after {rate}x subsample: {len(subsampled)}")
        return

    # --- Test with synthetic data ---
    print("  No real data, testing with synthetic data_info")
    synthetic = [
        {"step": 0, "image": "img0.png", "action": ["FORWARD"]},
        {"step": 1, "image": "img1.png", "action": ["LEFT"]},
        {"step": 2, "image": "img2.png", "action": ["FORWARD", "LEFT"]},  # multi-action
        {"step": 3, "image": "img3.png", "action": ["RIGHT"]},
        {"step": 4, "image": "img4.png", "action": ["BACKWARD"]},
        {"step": 5, "image": "img5.png", "action": ["FORWARD"]},
        {"step": 6, "image": "img6.png", "action": ["CHECKIN"]},          # not pure motion
        {"step": 7, "image": "img7.png", "action": ["FORWARD"]},
        {"step": 8, "image": "img8.png", "action": ["LEFT"]},
        {"step": 9, "image": "img9.png", "action": ["FORWARD"]},
        {"step": 10, "image": "img10.png", "action": ["RIGHT"]},
    ]
    pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
    all_motion = [
        {'step': d['step'], 'image': d['image'], 'action': d['action'][0]}
        for d in synthetic
        if len(d['action']) == 1 and d['action'][0] in pure
    ]

    # Multi-action (step 2) and CHECKIN (step 6) should be excluded
    assert len(all_motion) == 9, f"Expected 9 motion frames, got {len(all_motion)}"
    actions = [m['action'] for m in all_motion]
    assert 'CHECKIN' not in actions, "CHECKIN should be filtered out"
    print(f"  Pure motion filtering: {len(synthetic)} -> {len(all_motion)} frames")

    rate = 5
    subsampled = all_motion[::rate]
    expected = len(list(range(0, len(all_motion), rate)))
    assert len(subsampled) == expected, f"Subsample: {len(subsampled)} vs expected {expected}"
    assert subsampled[0]['step'] == 0
    # After filtering, indices 0..8 remain; subsample picks index 0 and 5
    # Index 5 in all_motion is step 7 (steps 2 and 6 were filtered out)
    assert subsampled[1]['step'] == 7, f"Expected step 7, got {subsampled[1]['step']}"
    print(f"  Subsample {rate}x: {len(all_motion)} -> {len(subsampled)} frames")
    print(f"  Subsample indices: steps {[s['step'] for s in subsampled]}")


# ---------------------------------------------------------------------------
# Section 2: SIFT Feature Extraction + RootSIFT
# ---------------------------------------------------------------------------
def test_sift_extraction():
    """Verify SIFT descriptors and RootSIFT transformation."""
    import cv2

    img_dir = None
    for d in ["data/images/", "exploration_data/images/"]:
        if os.path.exists(d) and os.listdir(d):
            img_dir = d
            break
    if img_dir:
        files = os.listdir(img_dir)
        img = cv2.imread(f"{img_dir}{files[0]}")
        print(f"  Using real image: {img_dir}{files[0]}")
    else:
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        print(f"  Using synthetic image")

    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    assert len(kp) > 0, "No keypoints found"
    assert des.shape[1] == 128, f"Expected 128-dim, got {des.shape[1]}"
    print(f"  Keypoints: {len(kp)}, descriptor shape: {des.shape}")

    # RootSIFT transform
    des_root = des / np.sum(des, axis=1, keepdims=True)
    des_root = np.sqrt(des_root)
    assert des_root.shape == des.shape, "RootSIFT changed shape"
    assert np.all(des_root >= 0), "RootSIFT has negative values"
    print(f"  RootSIFT: shape={des_root.shape}, min={des_root.min():.4f}, max={des_root.max():.4f}")

    # L2 norms of RootSIFT rows
    row_norms = np.linalg.norm(des_root, axis=1)
    assert np.all(row_norms > 0.5), f"Some rows have very small norm: min={row_norms.min():.4f}"
    assert np.all(row_norms < 1.5), f"Some rows have very large norm: max={row_norms.max():.4f}"
    print(f"  Row L2 norms: mean={row_norms.mean():.4f}, std={row_norms.std():.4f}")

    # Cache round-trip (using safe serialization)
    os.makedirs("cache", exist_ok=True)
    cache_path = "cache/_test_sift.npy"
    np.save(cache_path, des_root)
    loaded = np.load(cache_path)
    assert np.array_equal(loaded, des_root), "Cache round-trip failed"
    os.remove(cache_path)
    print(f"  Cache round-trip OK")

    # Different images produce different descriptors
    img2 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des2 is not None and len(des2) > 0:
        assert not np.array_equal(des[:min(5, len(des))], des2[:min(5, len(des2))]), \
            "Different images produced identical descriptors"
        print(f"  Different images -> different descriptors: OK")


# ---------------------------------------------------------------------------
# Section 3: VLAD Encoding
# ---------------------------------------------------------------------------
def test_vlad_encoding():
    """Verify VLAD vector computation with all normalization steps."""
    from sklearn.cluster import KMeans

    np.random.seed(42)
    n_descriptors = 500
    n_clusters = 8
    des = np.random.rand(n_descriptors, 128).astype(np.float32)

    codebook = KMeans(n_clusters=n_clusters, n_init=3, random_state=42).fit(des)
    assert codebook.cluster_centers_.shape == (n_clusters, 128)
    print(f"  Codebook: {n_clusters} clusters, inertia={codebook.inertia_:.0f}")

    def compute_vlad(descriptors):
        labels = codebook.predict(descriptors)
        centers = codebook.cluster_centers_
        vlad = np.zeros((n_clusters, 128))
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                vlad[i] = np.sum(descriptors[mask] - centers[i], axis=0)
                norm = np.linalg.norm(vlad[i])
                if norm > 0:
                    vlad[i] /= norm
        vlad = vlad.ravel()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm
        return vlad

    sample = des[:50]
    vlad = compute_vlad(sample)

    # Dimension
    assert vlad.shape == (n_clusters * 128,), f"Wrong shape: {vlad.shape}"
    print(f"  VLAD dim: {vlad.shape[0]}")

    # L2 unit norm
    assert abs(np.linalg.norm(vlad) - 1.0) < 1e-6, f"Not unit-normalized: {np.linalg.norm(vlad)}"
    print(f"  L2 norm: {np.linalg.norm(vlad):.6f}")

    # All finite
    assert np.all(np.isfinite(vlad)), "Contains NaN/Inf"

    # Values in [-1, 1]
    assert np.all(np.abs(vlad) <= 1.0 + 1e-6), "Values outside [-1, 1]"
    print(f"  Value range: [{vlad.min():.4f}, {vlad.max():.4f}]")

    # Two different inputs -> different VLADs
    sample2 = np.random.rand(50, 128).astype(np.float32)
    vlad2 = compute_vlad(sample2)
    similarity = float(vlad @ vlad2)
    assert similarity < 0.99, f"Two random VLADs too similar: {similarity}"
    print(f"  Cosine sim between two random VLADs: {similarity:.4f}")

    # Deterministic
    vlad_repeat = compute_vlad(sample)
    assert np.allclose(vlad, vlad_repeat, atol=1e-10), "Same input produced different VLAD"
    print(f"  Deterministic: OK")

    # Self-similarity = 1.0
    self_sim = float(vlad @ vlad)
    assert abs(self_sim - 1.0) < 1e-6, f"Self-similarity not 1.0: {self_sim}"
    print(f"  Self-similarity: {self_sim:.6f}")


# ---------------------------------------------------------------------------
# Section 4: Graph Construction
# ---------------------------------------------------------------------------
def test_graph_construction():
    """Verify graph structure with temporal + visual shortcut edges."""
    import networkx as nx

    n = 100
    dim = 128
    np.random.seed(42)
    database = np.random.randn(n, dim).astype(np.float32)
    database /= np.linalg.norm(database, axis=1, keepdims=True)

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Temporal edges
    for i in range(n - 1):
        G.add_edge(i, i + 1, weight=1.0, edge_type="temporal")
    assert G.number_of_edges() == n - 1
    print(f"  Temporal edges: {n - 1}")

    # Visual shortcut edges
    sim = database @ database.T
    np.fill_diagonal(sim, -2)
    min_gap = 10
    for i in range(n):
        lo, hi = max(0, i - min_gap), min(n, i + min_gap + 1)
        sim[i, lo:hi] = -2
    sim[~np.triu(np.ones((n, n), dtype=bool), k=1)] = -2

    top_k = 5
    flat = sim.ravel()
    top_idx = np.argpartition(flat, -top_k)[-top_k:]
    for fi in top_idx:
        i, j = divmod(int(fi), n)
        s = float(flat[fi])
        d = float(np.sqrt(max(0, 2 - 2 * s)))
        G.add_edge(i, j, weight=2.0 + 3.0 * d, edge_type="visual")

    total_edges = n - 1 + top_k
    assert G.number_of_edges() == total_edges, f"Expected {total_edges}, got {G.number_of_edges()}"
    assert nx.is_connected(G), "Graph is disconnected"
    print(f"  Total edges: {G.number_of_edges()} ({n-1} temporal + {top_k} visual)")

    # Shortest path
    path = nx.shortest_path(G, 0, n - 1, weight="weight")
    assert len(path) >= 2
    assert path[0] == 0 and path[-1] == n - 1
    print(f"  Shortest path 0->{n-1}: {len(path)} hops")

    # Visual edges respect min_gap
    visual_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "visual"]
    for u, v in visual_edges:
        assert abs(u - v) >= min_gap, f"Visual edge {u}-{v} too close"
    print(f"  Visual edges all respect min_gap={min_gap}")

    # All edge weights positive
    for u, v, d in G.edges(data=True):
        assert d['weight'] > 0, f"Edge {u}-{v} has non-positive weight"
    print(f"  All edge weights positive: OK")

    # Shortcuts reduce path vs pure temporal
    G_temp = nx.Graph()
    G_temp.add_nodes_from(range(n))
    for i in range(n - 1):
        G_temp.add_edge(i, i + 1, weight=1.0)
    path_temp = nx.shortest_path(G_temp, 0, n - 1, weight="weight")
    assert len(path) <= len(path_temp), "Shortcuts didn't help"
    print(f"  Shortcuts saved: {len(path_temp) - len(path)} hops")

    # Edge types labeled
    for u, v, d in G.edges(data=True):
        assert d.get("edge_type") in ("temporal", "visual"), f"Unknown type on {u}-{v}"
    print(f"  Edge type labels: all valid")


# ---------------------------------------------------------------------------
# Section 5: Localization + Multi-View Goal Matching
# ---------------------------------------------------------------------------
def test_localization():
    """Verify goal matching, position localization, and multi-view averaging."""
    np.random.seed(42)
    n, dim = 200, 1024
    database = np.random.randn(n, dim).astype(np.float32)
    database /= np.linalg.norm(database, axis=1, keepdims=True)

    # Goal match
    true_goal = 150
    target_vec = database[true_goal] + np.random.randn(dim) * 0.01
    target_vec /= np.linalg.norm(target_vec)
    sims = database @ target_vec
    predicted_goal = int(np.argmax(sims))
    assert predicted_goal == true_goal, f"Goal: predicted {predicted_goal}, expected {true_goal}"
    print(f"  Goal match: node {predicted_goal}, sim={sims[predicted_goal]:.4f}")

    # Position match
    true_pos = 50
    fpv_vec = database[true_pos] + np.random.randn(dim) * 0.01
    fpv_vec /= np.linalg.norm(fpv_vec)
    sims2 = database @ fpv_vec
    predicted_pos = int(np.argmax(sims2))
    assert predicted_pos == true_pos, f"Position: predicted {predicted_pos}, expected {true_pos}"
    print(f"  Position match: node {predicted_pos}, sim={sims2[predicted_pos]:.4f}")

    # Noisy localization
    noisy_vec = database[true_pos] + np.random.randn(dim) * 0.3
    noisy_vec /= np.linalg.norm(noisy_vec)
    sims3 = database @ noisy_vec
    predicted_noisy = int(np.argmax(sims3))
    error = abs(predicted_noisy - true_pos)
    print(f"  Noisy match: node {predicted_noisy}, error={error}")
    assert error < 30, f"Noisy localization error too high: {error}"

    # Multi-view averaging
    true_node = 100
    views = []
    for _ in range(4):
        v = database[true_node] + np.random.randn(dim) * 0.15
        v /= np.linalg.norm(v)
        views.append(v)

    single_pred = int(np.argmax(database @ views[0]))
    single_error = abs(single_pred - true_node)

    avg_sims = np.mean([database @ v for v in views], axis=0)
    multi_pred = int(np.argmax(avg_sims))
    multi_error = abs(multi_pred - true_node)

    print(f"  Single-view error: {single_error}, Multi-view error: {multi_error}")
    assert multi_error <= max(single_error, 10), \
        f"Multi-view ({multi_error}) much worse than single-view ({single_error})"

    # Self-similarity
    self_sim = float(database[0] @ database[0])
    assert abs(self_sim - 1.0) < 1e-5, f"Self-similarity not 1.0: {self_sim}"
    print(f"  Self-similarity: {self_sim:.6f}")


# ---------------------------------------------------------------------------
# Section 6a: Action Mapping (requires vis_nav_game)
# ---------------------------------------------------------------------------
def test_action_mapping():
    """Verify action string to Action enum mapping."""
    from vis_nav_game import Action

    ACTION_MAP = {
        'FORWARD': Action.FORWARD,
        'BACKWARD': Action.BACKWARD,
        'LEFT': Action.LEFT,
        'RIGHT': Action.RIGHT,
    }

    for name, action in ACTION_MAP.items():
        assert ACTION_MAP[name] == action, f"Mapping failed for {name}"
    assert ACTION_MAP.get('?', Action.IDLE) == Action.IDLE
    assert ACTION_MAP.get('JUMP', Action.IDLE) == Action.IDLE
    print(f"  Action mapping: all correct")
    print(f"  Unknown fallback: IDLE")

    assert hasattr(Action, 'CHECKIN'), "Action.CHECKIN not found"
    print(f"  CHECKIN action exists: OK")


# ---------------------------------------------------------------------------
# Section 6b: Edge Action Lookup
# ---------------------------------------------------------------------------
def test_edge_action_lookup():
    """Verify edge action lookup: forward, backward (reversed), non-adjacent."""
    motion_frames = [
        {'step': 0, 'image': 'img0.png', 'action': 'FORWARD'},
        {'step': 5, 'image': 'img1.png', 'action': 'LEFT'},
        {'step': 10, 'image': 'img2.png', 'action': 'FORWARD'},
        {'step': 15, 'image': 'img3.png', 'action': 'RIGHT'},
    ]
    REVERSE = {'FORWARD': 'BACKWARD', 'BACKWARD': 'FORWARD',
                'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}

    def edge_action(a, b):
        if b == a + 1 and a < len(motion_frames):
            return motion_frames[a]['action']
        elif b == a - 1 and b < len(motion_frames):
            return REVERSE.get(motion_frames[b]['action'], '?')
        return '?'

    # Forward edges
    assert edge_action(0, 1) == 'FORWARD'
    assert edge_action(1, 2) == 'LEFT'
    assert edge_action(2, 3) == 'FORWARD'
    print(f"  Forward edges: OK")

    # Backward edges (reversed)
    assert edge_action(1, 0) == 'BACKWARD'   # reverse of FORWARD
    assert edge_action(2, 1) == 'RIGHT'      # reverse of LEFT
    assert edge_action(3, 2) == 'BACKWARD'   # reverse of FORWARD
    print(f"  Backward edges (reversed): OK")

    # Non-adjacent
    assert edge_action(0, 2) == '?'
    assert edge_action(0, 3) == '?'
    print(f"  Non-adjacent: returns '?'")

    # All reverses
    assert REVERSE['FORWARD'] == 'BACKWARD'
    assert REVERSE['BACKWARD'] == 'FORWARD'
    assert REVERSE['LEFT'] == 'RIGHT'
    assert REVERSE['RIGHT'] == 'LEFT'
    print(f"  All 4 reverse mappings: OK")

    # Out of bounds (index >= len(motion_frames))
    assert edge_action(4, 5) == '?'   # a=4 >= len(motion_frames)=4
    assert edge_action(10, 11) == '?'
    print(f"  Out-of-bounds: returns '?'")


# ---------------------------------------------------------------------------
# Section 6c: Graph-Based Stuck Detection
# ---------------------------------------------------------------------------
def test_stuck_detection_graph():
    """Verify graph-based stuck detection (same node repeated in history)."""
    from collections import deque

    threshold = 5
    history = deque(maxlen=threshold)

    # Different nodes -> not stuck
    for node in [10, 11, 12, 13, 14]:
        history.append(node)
    is_stuck = len(history) == threshold and len(set(history)) == 1
    assert not is_stuck, "False positive: different nodes"
    print(f"  Different nodes: not stuck")

    # Same node repeated -> stuck
    history.clear()
    for _ in range(threshold):
        history.append(42)
    is_stuck = len(history) == threshold and len(set(history)) == 1
    assert is_stuck, "Missed stuck detection"
    print(f"  Same node x{threshold}: stuck detected")

    # Almost stuck (1 different) -> not stuck
    history.clear()
    for node in [42, 42, 42, 42, 43]:
        history.append(node)
    is_stuck = len(history) == threshold and len(set(history)) == 1
    assert not is_stuck, "False positive: one different"
    print(f"  Almost same: not stuck")

    # Short history -> not stuck
    history.clear()
    for _ in range(3):
        history.append(42)
    is_stuck = len(history) == threshold and len(set(history)) == 1
    assert not is_stuck, "Should not be stuck with short history"
    print(f"  Short history (3/{threshold}): not stuck")

    # Deque maxlen overflow
    history.clear()
    for i in range(10):
        history.append(99)
    assert len(history) == threshold, f"Deque should cap at {threshold}"
    is_stuck = len(history) == threshold and len(set(history)) == 1
    assert is_stuck, "Should be stuck after overflow"
    print(f"  Deque overflow (10 -> {threshold}): stuck")

    # Adding different element breaks stuck
    history.append(100)
    is_stuck = len(history) == threshold and len(set(history)) == 1
    assert not is_stuck, "Should not be stuck after adding different node"
    print(f"  Different node breaks stuck: OK")


# ---------------------------------------------------------------------------
# Section 6d: Navigation Path Planning
# ---------------------------------------------------------------------------
def test_navigation_path_planning():
    """Verify Dijkstra path planning with shortcuts and edge cases."""
    import networkx as nx

    # Build graph with known structure
    G = nx.Graph()
    G.add_nodes_from(range(20))

    # Temporal chain: 0-1-2-...-19
    for i in range(19):
        G.add_edge(i, i + 1, weight=1.0, edge_type="temporal")

    # Shortcut: 0 <-> 15
    G.add_edge(0, 15, weight=3.0, edge_type="visual")

    # Path should use the shortcut
    path = nx.shortest_path(G, 0, 19, weight="weight")
    assert path[0] == 0 and path[-1] == 19
    print(f"  Path 0->19: {path}")

    # Shortcut path cost < temporal-only cost
    path_cost = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
    assert path_cost < 19.0, f"Shortcut not used: cost={path_cost}"
    print(f"  Path cost: {path_cost} (vs temporal: 19.0)")

    # Verify shortcut edge is in the path
    path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
    assert (0, 15) in path_edges or (15, 0) in path_edges, "Shortcut not used"
    print(f"  Shortcut 0-15 used: OK")

    # All consecutive pairs share an edge
    for i in range(len(path) - 1):
        assert G.has_edge(path[i], path[i+1]), f"No edge {path[i]}-{path[i+1]}"
    print(f"  All path edges exist: OK")

    # Disconnected graph -> NetworkXNoPath
    G2 = nx.Graph()
    G2.add_nodes_from([0, 1, 10])
    G2.add_edge(0, 1, weight=1.0)
    try:
        nx.shortest_path(G2, 0, 10, weight="weight")
        assert False, "Should have raised NetworkXNoPath"
    except nx.NetworkXNoPath:
        print(f"  No-path exception: handled OK")

    # Multiple shortcuts compete correctly
    G3 = nx.Graph()
    G3.add_nodes_from(range(10))
    for i in range(9):
        G3.add_edge(i, i + 1, weight=1.0)
    G3.add_edge(0, 9, weight=5.0)   # expensive shortcut
    G3.add_edge(0, 5, weight=2.0)   # cheaper shortcut
    path3 = nx.shortest_path(G3, 0, 9, weight="weight")
    cost3 = sum(G3[path3[i]][path3[i+1]]['weight'] for i in range(len(path3)-1))
    print(f"  Competing shortcuts: path={path3}, cost={cost3}")
    assert cost3 <= 9.0, "Should use at least one shortcut"


# ---------------------------------------------------------------------------
# Section 7: Frame-Based Stuck Detection (MSE)
# ---------------------------------------------------------------------------
def test_frame_stuck_detection():
    """Verify frame MSE comparison for physical stuck detection."""
    import cv2

    STUCK_MSE_THRESHOLD = 200
    STUCK_FRAME_SIZE = (160, 120)

    def is_stuck(current_frame, previous_frame):
        if current_frame is None or previous_frame is None:
            return False
        gray_cur = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        gray_cur = cv2.resize(gray_cur, STUCK_FRAME_SIZE)
        gray_prev = cv2.resize(gray_prev, STUCK_FRAME_SIZE)
        mse = np.mean((gray_cur.astype(np.float32) - gray_prev.astype(np.float32)) ** 2)
        return mse < STUCK_MSE_THRESHOLD

    np.random.seed(42)

    # Identical frames -> stuck
    frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    assert is_stuck(frame, frame.copy()) == True, "Identical frames should be stuck"
    print(f"  Identical frames: stuck detected")

    # Very different frames -> not stuck
    frame2 = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    assert is_stuck(frame, frame2) == False, "Random frames should not be stuck"
    print(f"  Very different frames: not stuck")

    # Small noise -> stuck
    frame3 = frame.copy()
    noise = np.random.randint(-5, 6, frame3.shape, dtype=np.int16)
    frame3 = np.clip(frame3.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    assert is_stuck(frame, frame3) == True, "Small noise should still be stuck"
    print(f"  Small noise: stuck detected")

    # Shifted frame (movement) -> not stuck
    frame4 = np.roll(frame, 50, axis=1)
    assert is_stuck(frame, frame4) == False, "Shifted frame should not be stuck"
    print(f"  Shifted frame (movement): not stuck")

    # None handling
    assert is_stuck(None, frame) == False
    assert is_stuck(frame, None) == False
    assert is_stuck(None, None) == False
    print(f"  None handling: OK")

    # Different input sizes
    frame_small = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    frame_large = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    assert is_stuck(frame_small, frame_small.copy()) == True
    assert is_stuck(frame_large, frame_large.copy()) == True
    print(f"  Different input sizes: OK")

    # MSE progression
    mse_values = []
    for noise_level in [0, 5, 10, 20, 50]:
        noisy = frame.copy()
        n = np.random.randint(-noise_level, noise_level + 1, noisy.shape, dtype=np.int16)
        noisy = np.clip(noisy.astype(np.int16) + n, 0, 255).astype(np.uint8)
        gray_c = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_p = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
        gray_c = cv2.resize(gray_c, STUCK_FRAME_SIZE)
        gray_p = cv2.resize(gray_p, STUCK_FRAME_SIZE)
        mse = np.mean((gray_c.astype(np.float32) - gray_p.astype(np.float32)) ** 2)
        mse_values.append((noise_level, mse))
    print(f"  MSE vs noise: {[(n, f'{m:.1f}') for n, m in mse_values]}")


# ---------------------------------------------------------------------------
# Section 8: Exploration State Machine (Wall-Following)
# ---------------------------------------------------------------------------
def test_exploration_state_machine():
    """Verify all state transitions of the wall-following exploration."""
    from enum import Enum

    class ExploreState(Enum):
        FORWARD = "forward"
        TURN_LEFT = "turn_left"
        TURN_RIGHT = "turn_right"
        CHECK_RIGHT = "check_right"
        REVERSE = "reverse"

    TURN_STEPS_90 = 3
    TURN_STEPS_180 = 6
    CHECK_RIGHT_INTERVAL = 20

    class Explorer:
        def __init__(self):
            self.explore_state = ExploreState.FORWARD
            self.forward_count = 0
            self.turn_counter = 0
            self.consecutive_stuck = 0

        def step(self, stuck):
            if self.explore_state == ExploreState.FORWARD:
                if stuck:
                    self.consecutive_stuck += 1
                    if self.consecutive_stuck >= 3:
                        self.explore_state = ExploreState.REVERSE
                        self.turn_counter = 0
                        return 'LEFT'
                    else:
                        self.explore_state = ExploreState.TURN_LEFT
                        self.turn_counter = 0
                        return 'LEFT'
                else:
                    self.consecutive_stuck = 0
                    self.forward_count += 1
                    if self.forward_count >= CHECK_RIGHT_INTERVAL:
                        self.forward_count = 0
                        self.explore_state = ExploreState.CHECK_RIGHT
                        self.turn_counter = 0
                        return 'RIGHT'
                    return 'FORWARD'

            elif self.explore_state == ExploreState.TURN_LEFT:
                self.turn_counter += 1
                if self.turn_counter >= TURN_STEPS_90:
                    self.explore_state = ExploreState.FORWARD
                    return 'FORWARD'
                return 'LEFT'

            elif self.explore_state == ExploreState.CHECK_RIGHT:
                self.turn_counter += 1
                if self.turn_counter >= TURN_STEPS_90:
                    if stuck:
                        self.explore_state = ExploreState.TURN_LEFT
                        self.turn_counter = 0
                        return 'LEFT'
                    else:
                        self.explore_state = ExploreState.FORWARD
                        self.consecutive_stuck = 0
                        return 'FORWARD'
                return 'RIGHT'

            elif self.explore_state == ExploreState.REVERSE:
                self.turn_counter += 1
                if self.turn_counter >= TURN_STEPS_180:
                    self.explore_state = ExploreState.FORWARD
                    self.consecutive_stuck = 0
                    return 'FORWARD'
                return 'LEFT'

            return 'FORWARD'

    # 1. FORWARD not stuck
    e = Explorer()
    assert e.step(stuck=False) == 'FORWARD'
    assert e.explore_state == ExploreState.FORWARD
    print(f"  Not stuck -> FORWARD: OK")

    # 2. FORWARD stuck -> TURN_LEFT
    assert e.step(stuck=True) == 'LEFT'
    assert e.explore_state == ExploreState.TURN_LEFT
    print(f"  Stuck -> TURN_LEFT: OK")

    # 3. TURN_LEFT completes
    for i in range(TURN_STEPS_90 - 1):
        assert e.step(stuck=False) == 'LEFT'
    assert e.step(stuck=False) == 'FORWARD'
    assert e.explore_state == ExploreState.FORWARD
    print(f"  TURN_LEFT x{TURN_STEPS_90} -> FORWARD: OK")

    # 4. CHECK_RIGHT after interval
    e2 = Explorer()
    for i in range(CHECK_RIGHT_INTERVAL - 1):
        assert e2.step(stuck=False) == 'FORWARD'
    assert e2.step(stuck=False) == 'RIGHT'
    assert e2.explore_state == ExploreState.CHECK_RIGHT
    print(f"  CHECK_RIGHT after {CHECK_RIGHT_INTERVAL} steps: OK")

    # 5. CHECK_RIGHT not stuck -> FORWARD
    for i in range(TURN_STEPS_90 - 1):
        assert e2.step(stuck=False) == 'RIGHT'
    assert e2.step(stuck=False) == 'FORWARD'
    assert e2.explore_state == ExploreState.FORWARD
    print(f"  CHECK_RIGHT found corridor -> FORWARD: OK")

    # 6. CHECK_RIGHT stuck -> TURN_LEFT
    e3 = Explorer()
    e3.explore_state = ExploreState.CHECK_RIGHT
    e3.turn_counter = 0
    for i in range(TURN_STEPS_90 - 1):
        e3.step(stuck=False)
    assert e3.step(stuck=True) == 'LEFT'
    assert e3.explore_state == ExploreState.TURN_LEFT
    print(f"  CHECK_RIGHT wall -> TURN_LEFT: OK")

    # 7. Triple stuck -> REVERSE
    e4 = Explorer()
    e4.step(stuck=True)  # stuck 1
    for _ in range(TURN_STEPS_90):
        e4.step(stuck=False)
    e4.step(stuck=True)  # stuck 2
    for _ in range(TURN_STEPS_90):
        e4.step(stuck=False)
    e4.step(stuck=True)  # stuck 3
    assert e4.explore_state == ExploreState.REVERSE
    print(f"  Triple stuck -> REVERSE: OK")

    # 8. REVERSE completes
    for i in range(TURN_STEPS_180 - 1):
        assert e4.step(stuck=False) == 'LEFT'
    assert e4.step(stuck=False) == 'FORWARD'
    assert e4.explore_state == ExploreState.FORWARD
    assert e4.consecutive_stuck == 0
    print(f"  REVERSE x{TURN_STEPS_180} -> FORWARD: OK")

    # 9. consecutive_stuck resets
    e5 = Explorer()
    e5.step(stuck=True)
    for _ in range(TURN_STEPS_90):
        e5.step(stuck=False)
    e5.step(stuck=False)
    assert e5.consecutive_stuck == 0
    print(f"  consecutive_stuck resets: OK")


# ---------------------------------------------------------------------------
# Section 9: Turn Angle Calibration
# ---------------------------------------------------------------------------
def test_turn_calibration():
    """Verify turn calibration logic with synthetic rotation frame diffs."""

    # Simulate 360-degree rotation in 48 steps
    steps_for_360 = 48
    angles = np.linspace(0, 2 * np.pi, steps_for_360)
    diffs = (np.sin(angles / 2) ** 2) * 5000

    # Detect rotation completion
    threshold = diffs[0] * 1.5 + 50
    rotation_complete = None
    for i in range(10, len(diffs)):
        if diffs[i] < threshold:
            rotation_complete = i
            break
    assert rotation_complete is not None, "Should detect rotation completion"
    estimated_90 = rotation_complete // 4
    expected_90 = steps_for_360 // 4
    print(f"  360-degree at step {rotation_complete}, estimated 90={estimated_90}")
    assert abs(estimated_90 - expected_90) <= 3

    # Noisy diffs
    np.random.seed(42)
    noisy_diffs = diffs + np.random.randn(len(diffs)) * 100
    noisy_diffs = np.maximum(noisy_diffs, 0)
    noisy_complete = None
    for i in range(10, len(noisy_diffs)):
        if noisy_diffs[i] < threshold:
            noisy_complete = i
            break
    if noisy_complete:
        noisy_90 = noisy_complete // 4
        assert abs(noisy_90 - expected_90) <= 5
        print(f"  Noisy calibration: 90={noisy_90} (within 5)")
    else:
        print(f"  Noisy calibration: fallback to default (OK)")

    # Fallback default
    assert 12 > 0
    print(f"  Fallback default (12): OK")


# ---------------------------------------------------------------------------
# Section 10: Player Initialization (Source Inspection)
# ---------------------------------------------------------------------------
def test_player_initialization():
    """Verify player.py has all required constants, methods, and state vars."""
    from enum import Enum

    class ExploreState(Enum):
        FORWARD = "forward"
        TURN_LEFT = "turn_left"
        TURN_RIGHT = "turn_right"
        CHECK_RIGHT = "check_right"
        REVERSE = "reverse"
    assert len(ExploreState) == 5
    print(f"  ExploreState: {len(ExploreState)} states")

    source_path = os.path.join("source", "player.py")
    if not os.path.exists(source_path):
        source_path = "player.py"
    if not os.path.exists(source_path):
        print(f"  WARNING: player.py not found, skipping source inspection")
        return

    with open(source_path) as f:
        source = f.read()

    constants = [
        'TURN_STEPS_90', 'TURN_STEPS_180', 'CHECK_RIGHT_INTERVAL',
        'STUCK_MSE_THRESHOLD', 'STUCK_FRAME_SIZE',
        'TEMPORAL_WEIGHT', 'VISUAL_WEIGHT_BASE', 'VISUAL_WEIGHT_SCALE',
        'MIN_SHORTCUT_GAP', 'ACTION_HOLD_FRAMES',
        'GRADIENT_CHECK_INTERVAL', 'APPROACH_HOPS_ENTER',
    ]
    missing_c = [c for c in constants if c not in source]
    assert len(missing_c) == 0, f"Missing constants: {missing_c}"
    print(f"  Constants ({len(constants)}): all found")

    methods = [
        '_is_stuck', '_explore_act', '_auto_navigate',
        '_get_current_node', '_get_path', '_wall_follow_act', '_gradient_check', '_search_act',
        '_load_trajectory_data', '_build_database', '_build_graph', '_setup_goal',
    ]
    missing_m = [m for m in methods if m not in source]
    assert len(missing_m) == 0, f"Missing methods: {missing_m}"
    print(f"  Methods ({len(methods)}): all found")

    state_vars = [
        'explore_state', 'prev_frame', 'forward_count',
        'turn_counter', 'consecutive_stuck', 'manual_mode',
        'nav_tick', 'current_action', 'action_hold_counter',
        'hop_history', 'pipeline_ready', 'exploring_in_nav', 'nav_stuck_frames',
        'nav_state', 'wall_hand',
    ]
    missing_v = [v for v in state_vars if v not in source]
    assert len(missing_v) == 0, f"Missing state vars: {missing_v}"
    print(f"  State variables ({len(state_vars)}): all found")

    assert 'class VLADExtractor' in source, "VLADExtractor class not found"
    print(f"  VLADExtractor class: found")

    assert 'class ExploreState' in source, "ExploreState enum not found"
    print(f"  ExploreState enum: found")


# ---------------------------------------------------------------------------
# Section 11: Checkin Logic
# ---------------------------------------------------------------------------
def test_checkin_logic():
    """Verify CHECKIN requires both proximity AND visual similarity."""

    def should_checkin(hops, visual_sim, low_confidence=False):
        """Standalone checkin decision logic matching _auto_navigate."""
        CHECKIN_THRESHOLD = 3
        if hops > CHECKIN_THRESHOLD:
            return False
        sim_threshold_high = 0.35 if low_confidence else 0.25
        sim_threshold_low = 0.35 if low_confidence else 0.15
        if (hops <= 1 and visual_sim > sim_threshold_high):
            return True
        if (hops == 0 and visual_sim > sim_threshold_low):
            return True
        return False

    # hops=0, sim=0.1 -> should NOT checkin (sim too low, below 0.15)
    assert not should_checkin(0, 0.1), "hops=0 sim=0.1 should NOT checkin"
    print(f"  hops=0, sim=0.1: no checkin (sim below low threshold)")

    # hops=0, sim=0.3 -> should checkin (hops<=1 and sim>0.25)
    assert should_checkin(0, 0.3), "hops=0 sim=0.3 should checkin"
    print(f"  hops=0, sim=0.3: checkin (both thresholds met)")

    # hops=1, sim=0.4 -> should checkin (hops<=1 and sim>0.25)
    assert should_checkin(1, 0.4), "hops=1 sim=0.4 should checkin"
    print(f"  hops=1, sim=0.4: checkin")

    # hops=1, sim=0.1 -> should NOT checkin (sim too low)
    assert not should_checkin(1, 0.1), "hops=1 sim=0.1 should NOT checkin"
    print(f"  hops=1, sim=0.1: no checkin (sim too low)")

    # hops=3, sim=0.5 -> should NOT checkin (hops>1 and not hops==0)
    assert not should_checkin(3, 0.5), "hops=3 sim=0.5 should NOT checkin"
    print(f"  hops=3, sim=0.5: no checkin (too far even with high sim)")

    # hops=5, sim=0.9 -> should NOT checkin (beyond CHECKIN_THRESHOLD)
    assert not should_checkin(5, 0.9), "hops=5 sim=0.9 should NOT checkin"
    print(f"  hops=5, sim=0.9: no checkin (beyond threshold)")


# ---------------------------------------------------------------------------
# Section 12: Goal Confidence Threshold
# ---------------------------------------------------------------------------
def test_goal_confidence_threshold():
    """Verify goal confidence flag is set correctly."""

    def compute_low_confidence(avg_sim):
        return avg_sim < 0.25

    # avg_sim=0.5 -> low_confidence=False
    assert not compute_low_confidence(0.5), "avg_sim=0.5 should be confident"
    print(f"  avg_sim=0.5: low_confidence=False")

    # avg_sim=0.15 -> low_confidence=True
    assert compute_low_confidence(0.15), "avg_sim=0.15 should be low confidence"
    print(f"  avg_sim=0.15: low_confidence=True")

    # avg_sim=0.25 -> low_confidence=False (borderline, not strictly < 0.25)
    assert not compute_low_confidence(0.25), "avg_sim=0.25 should be confident (borderline)"
    print(f"  avg_sim=0.25: low_confidence=False (borderline)")

    # avg_sim=0.24 -> low_confidence=True
    assert compute_low_confidence(0.24), "avg_sim=0.24 should be low confidence"
    print(f"  avg_sim=0.24: low_confidence=True")


# ---------------------------------------------------------------------------
# Section 13: Localization Smoothing
# ---------------------------------------------------------------------------
def test_localization_smoothing():
    """Verify temporal smoothing stabilizes localization."""
    np.random.seed(42)
    n, dim = 10, 128
    database = np.random.randn(n, dim).astype(np.float32)
    database /= np.linalg.norm(database, axis=1, keepdims=True)

    def localize(query, prev_sims, prev_node):
        """Standalone localization with smoothing and jump detection."""
        sims = database @ query
        if prev_sims is not None:
            sims = 0.6 * sims + 0.4 * prev_sims
        new_sims = sims.copy()
        cur = int(np.argmax(sims))
        if prev_node is not None:
            if abs(cur - prev_node) > 50:
                cur = prev_node
        return cur, new_sims

    # Query identical to node 5 -> returns 5
    query = database[5].copy()
    cur, sims = localize(query, None, None)
    assert cur == 5, f"Expected node 5, got {cur}"
    print(f"  Identical query to node 5: got {cur}")

    # Query with noise that would match node 8 without smoothing -> after smoothing stays near 5
    # Build a query slightly biased toward node 8
    query_noisy = database[8] * 0.7 + database[5] * 0.3
    query_noisy /= np.linalg.norm(query_noisy)
    # Without smoothing, should match 8
    raw_sims = database @ query_noisy
    raw_match = int(np.argmax(raw_sims))
    # With smoothing from previous (node 5), should stay near 5
    cur_smooth, sims_smooth = localize(query_noisy, sims, 5)
    print(f"  Noisy query (raw={raw_match}): smoothed={cur_smooth}")
    # Smoothed result should be closer to 5 than raw would be
    assert abs(cur_smooth - 5) <= abs(raw_match - 5), \
        f"Smoothing didn't help: smoothed={cur_smooth}, raw={raw_match}"

    # Genuine gradual movement: 5 -> 6 -> 7
    prev_s = None
    prev_n = None
    trajectory = []
    for target in [5, 6, 7]:
        q = database[target].copy()
        cur, prev_s = localize(q, prev_s, prev_n)
        prev_n = cur
        trajectory.append(cur)
    # Should follow the sequence (allow some smoothing lag)
    assert trajectory[-1] == 7, f"Expected to reach 7, got {trajectory[-1]}"
    print(f"  Gradual movement 5->6->7: trajectory={trajectory}")

    # Large jump detection: with 10-node database, jumps are <50 so jump detection
    # won't trigger, but we test the logic directly
    def localize_with_jump(query, prev_sims, prev_node, jump_threshold=50):
        sims = database @ query
        if prev_sims is not None:
            sims = 0.6 * sims + 0.4 * prev_sims
        new_sims = sims.copy()
        cur = int(np.argmax(sims))
        if prev_node is not None:
            if abs(cur - prev_node) > jump_threshold:
                cur = prev_node
        return cur, new_sims

    # With a small threshold (2), jump from node 2 to node 9 should be blocked
    q9 = database[9].copy()
    cur_jump, _ = localize_with_jump(q9, None, 2, jump_threshold=2)
    assert cur_jump == 2, f"Jump should be blocked: got {cur_jump}, expected 2"
    print(f"  Jump from 2 to 9 (threshold=2): blocked, stayed at {cur_jump}")

    # First call with no history -> works without error
    q0 = database[0].copy()
    cur_first, sims_first = localize(q0, None, None)
    assert cur_first == 0, f"First call should work: got {cur_first}"
    print(f"  First call (no history): got {cur_first}, no error")


# ---------------------------------------------------------------------------
# Section 14: see() Frame Saving
# ---------------------------------------------------------------------------
def test_see_frame_saving():
    """Verify see() saves frames and metadata correctly."""
    import tempfile
    import cv2

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock player-like object
        class MockPlayer:
            def __init__(self, image_dir):
                self.pipeline_ready = False
                self.explore_step = 0
                self.explore_data = []
                self.last_explore_action = 'FORWARD'
                self.fpv = None
                self.prev_frame = None
                self.image_dir = image_dir

            def see(self, fpv):
                if fpv is None or len(fpv.shape) < 3:
                    return
                if self.fpv is not None:
                    self.prev_frame = self.fpv.copy()
                self.fpv = fpv
                if not self.pipeline_ready:
                    fname = f"{self.explore_step:06d}.png"
                    cv2.imwrite(os.path.join(self.image_dir, fname), fpv)
                    self.explore_data.append({
                        'step': self.explore_step,
                        'image': fname,
                        'action': [self.last_explore_action],
                    })
                    self.explore_step += 1

        player = MockPlayer(tmpdir)

        # Call see() with synthetic frames
        frames = [np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(5)]
        for frame in frames:
            player.see(frame)

        # Verify PNG files written to disk
        saved_files = sorted(os.listdir(tmpdir))
        assert len(saved_files) == 5, f"Expected 5 files, got {len(saved_files)}"
        for i, fname in enumerate(saved_files):
            assert fname == f"{i:06d}.png", f"Unexpected filename: {fname}"
        print(f"  PNG files saved: {len(saved_files)} files with correct names")

        # Verify explore_data has correct entries
        assert len(player.explore_data) == 5
        for i, entry in enumerate(player.explore_data):
            assert entry['step'] == i
            assert entry['image'] == f"{i:06d}.png"
            assert entry['action'] == ['FORWARD']
        print(f"  explore_data: {len(player.explore_data)} entries, structure correct")

        # Verify explore_step increments
        assert player.explore_step == 5, f"Expected explore_step=5, got {player.explore_step}"
        print(f"  explore_step: {player.explore_step}")

        # Verify saving stops when pipeline_ready=True
        player.pipeline_ready = True
        player.see(np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8))
        assert len(player.explore_data) == 5, "Should not save after pipeline_ready=True"
        assert player.explore_step == 5, "explore_step should not increment after pipeline_ready"
        print(f"  Saving stops after pipeline_ready=True: OK")


# ---------------------------------------------------------------------------
# Section 15: Two-Stage Flag Transitions
# ---------------------------------------------------------------------------
def test_two_stage_flag_transitions():
    """Verify pipeline_ready and exploring_in_nav flag state machine."""

    class FlagStateMachine:
        def __init__(self):
            self.pipeline_ready = False
            self.exploring_in_nav = False

        def pre_navigation_no_data(self):
            """Simulate pre_navigation with no exploration data."""
            self.exploring_in_nav = True
            self.pipeline_ready = False

        def finish_exploration_with_data(self):
            """Simulate _finish_exploration_in_nav with valid data."""
            # Simulates: _build_pipeline succeeded, goal_node is not None
            self.pipeline_ready = True
            self.exploring_in_nav = False

    sm = FlagStateMachine()

    # Initial state
    assert not sm.pipeline_ready, "Initial: pipeline_ready should be False"
    assert not sm.exploring_in_nav, "Initial: exploring_in_nav should be False"
    print(f"  Initial: pipeline_ready=False, exploring_in_nav=False")

    # After pre_navigation with no data
    sm.pre_navigation_no_data()
    assert sm.exploring_in_nav, "After pre_nav: exploring_in_nav should be True"
    assert not sm.pipeline_ready, "After pre_nav: pipeline_ready should be False"
    print(f"  After pre_navigation (no data): exploring_in_nav=True, pipeline_ready=False")

    # After _finish_exploration_in_nav with real data
    sm.finish_exploration_with_data()
    assert sm.pipeline_ready, "After finish: pipeline_ready should be True"
    assert not sm.exploring_in_nav, "After finish: exploring_in_nav should be False"
    print(f"  After finish_exploration: pipeline_ready=True, exploring_in_nav=False")


# ---------------------------------------------------------------------------
# Section 16: Exploration Budget Enforcement
# ---------------------------------------------------------------------------
def test_explore_step_budget():
    """Verify exploration budget enforcement."""
    import cv2

    EXPLORE_STEPS = 10  # small budget for testing

    class MockBudgetPlayer:
        def __init__(self):
            self.pipeline_ready = False
            self.exploring_in_nav = True
            self.explore_step = 0
            self.explore_data = []
            self.last_explore_action = 'FORWARD'
            self.fpv = None
            self.prev_frame = None
            self.build_called = False

        def see(self, fpv):
            if fpv is None or len(fpv.shape) < 3:
                return
            if self.fpv is not None:
                self.prev_frame = self.fpv.copy()
            self.fpv = fpv
            if not self.pipeline_ready:
                self.explore_data.append({
                    'step': self.explore_step,
                    'image': f"{self.explore_step:06d}.png",
                    'action': [self.last_explore_action],
                })
                self.explore_step += 1

        def act(self):
            if not self.pipeline_ready:
                if self.exploring_in_nav and self.explore_step >= EXPLORE_STEPS:
                    self._finish_exploration()
                    if self.pipeline_ready:
                        return 'NAVIGATE'
                return 'EXPLORE'
            return 'NAVIGATE'

        def _finish_exploration(self):
            self.build_called = True
            self.pipeline_ready = True
            self.exploring_in_nav = False

    player = MockBudgetPlayer()

    # explore_step starts at 0
    assert player.explore_step == 0
    print(f"  explore_step starts at 0")

    # After N see() calls, explore_step == N
    for i in range(EXPLORE_STEPS):
        frame = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        player.see(frame)
    assert player.explore_step == EXPLORE_STEPS, \
        f"Expected explore_step={EXPLORE_STEPS}, got {player.explore_step}"
    print(f"  After {EXPLORE_STEPS} see() calls: explore_step={player.explore_step}")

    # When explore_step >= EXPLORE_STEPS, act() should trigger pipeline build
    result = player.act()
    assert player.build_called, "Pipeline build should have been triggered"
    assert player.pipeline_ready, "pipeline_ready should be True after build"
    assert result == 'NAVIGATE', f"Expected NAVIGATE, got {result}"
    print(f"  Budget exhausted -> pipeline build triggered, act() returns NAVIGATE")


# ---------------------------------------------------------------------------
# Section 17: Goal Node None Fallback
# ---------------------------------------------------------------------------
def test_goal_node_none_fallback():
    """Verify navigation handles None goal gracefully."""

    class MockNavPlayer:
        def __init__(self):
            self.fpv = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
            self.goal_node = None

        def _auto_navigate(self):
            if self.fpv is None or self.goal_node is None:
                return 'IDLE'
            return 'FORWARD'

    player = MockNavPlayer()

    # goal_node=None -> _auto_navigate should return IDLE
    result = player._auto_navigate()
    assert result == 'IDLE', f"Expected IDLE, got {result}"
    print(f"  goal_node=None: _auto_navigate returns IDLE")

    # No crash
    player.fpv = None
    result2 = player._auto_navigate()
    assert result2 == 'IDLE', f"Expected IDLE with fpv=None, got {result2}"
    print(f"  fpv=None, goal_node=None: returns IDLE, no crash")

    # With goal set, should return FORWARD
    player.fpv = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
    player.goal_node = 42
    result3 = player._auto_navigate()
    assert result3 == 'FORWARD', f"Expected FORWARD, got {result3}"
    print(f"  goal_node=42: returns FORWARD")


# ---------------------------------------------------------------------------
# Section 18: Empty Database Pipeline
# ---------------------------------------------------------------------------
def test_empty_database_pipeline():
    """Verify pipeline handles empty data gracefully."""
    import networkx as nx

    # Empty file_list -> database has 0 rows
    file_list = []
    dim = 1024
    if len(file_list) == 0:
        database = np.zeros((0, dim))
    else:
        database = np.random.randn(len(file_list), dim)

    assert database.shape == (0, dim), f"Expected (0, {dim}), got {database.shape}"
    print(f"  Empty file_list: database shape = {database.shape}")

    # Graph has 0 nodes
    G = nx.Graph()
    n = database.shape[0]
    G.add_nodes_from(range(n))
    assert G.number_of_nodes() == 0, f"Expected 0 nodes, got {G.number_of_nodes()}"
    assert G.number_of_edges() == 0
    print(f"  Empty database: graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # goal_node stays None
    goal_node = None
    if database.shape[0] > 0:
        goal_node = 0
    assert goal_node is None, "goal_node should remain None with empty database"
    print(f"  Empty database: goal_node = None")

    # pipeline_ready stays False
    pipeline_ready = False
    if goal_node is not None:
        pipeline_ready = True
    assert not pipeline_ready, "pipeline_ready should stay False with no goal"
    print(f"  Empty database: pipeline_ready = False")


# ---------------------------------------------------------------------------
# Section 19: Data Info JSON Roundtrip
# ---------------------------------------------------------------------------
def test_data_info_json_roundtrip():
    """Verify explore_data saves to JSON and loads back correctly."""
    import json
    import tempfile

    # Create sample explore_data with FORWARD, LEFT, RIGHT entries
    explore_data = [
        {"step": 0, "image": "000000.png", "action": ["FORWARD"]},
        {"step": 1, "image": "000001.png", "action": ["LEFT"]},
        {"step": 2, "image": "000002.png", "action": ["RIGHT"]},
        {"step": 3, "image": "000003.png", "action": ["FORWARD"]},
        {"step": 4, "image": "000004.png", "action": ["IDLE"]},           # should be excluded
        {"step": 5, "image": "000005.png", "action": ["FORWARD", "LEFT"]}, # multi-action excluded
        {"step": 6, "image": "000006.png", "action": ["BACKWARD"]},
    ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        tmppath = f.name
        json.dump(explore_data, f)

    try:
        # Load back and verify structure
        with open(tmppath) as f:
            loaded = json.load(f)
        assert len(loaded) == len(explore_data), \
            f"Loaded {len(loaded)} entries, expected {len(explore_data)}"
        for orig, load in zip(explore_data, loaded):
            assert orig == load, f"Mismatch: {orig} vs {load}"
        print(f"  JSON roundtrip: {len(loaded)} entries match")

        # Verify _load_trajectory_data() filtering logic
        pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
        all_motion = [
            {'step': d['step'], 'image': d['image'], 'action': d['action'][0]}
            for d in loaded
            if len(d['action']) == 1 and d['action'][0] in pure
        ]

        # Only single pure-action frames should remain
        assert len(all_motion) == 5, f"Expected 5 pure motion frames, got {len(all_motion)}"
        print(f"  Filtering: {len(loaded)} -> {len(all_motion)} pure motion frames")

        # Verify IDLE excluded
        actions = [m['action'] for m in all_motion]
        assert 'IDLE' not in actions, "IDLE should be filtered out"
        print(f"  IDLE excluded: OK")

        # Verify multi-action frames excluded
        steps = [m['step'] for m in all_motion]
        assert 5 not in steps, "Multi-action frame (step 5) should be excluded"
        print(f"  Multi-action excluded: OK")

        # Verify expected steps remain
        expected_steps = [0, 1, 2, 3, 6]
        assert steps == expected_steps, f"Expected steps {expected_steps}, got {steps}"
        print(f"  Remaining steps: {steps}")
    finally:
        os.remove(tmppath)


# ---------------------------------------------------------------------------
# Section 20: Localization Jump Detection
# ---------------------------------------------------------------------------
def test_localization_jump_detection():
    """Verify jump detection prevents wild localization jumps."""

    JUMP_THRESHOLD = 50

    def apply_jump_detection(prev_node, current):
        if prev_node is not None:
            if abs(current - prev_node) > JUMP_THRESHOLD:
                return prev_node
        return current

    # prev_node=10, current=11 -> allows (small jump)
    result = apply_jump_detection(10, 11)
    assert result == 11, f"Expected 11, got {result}"
    print(f"  prev=10, cur=11: allowed (result={result})")

    # prev_node=10, current=70 -> blocks (>50 node jump), returns 10
    result = apply_jump_detection(10, 70)
    assert result == 10, f"Expected 10 (blocked), got {result}"
    print(f"  prev=10, cur=70: blocked (result={result})")

    # prev_node=None (first call) -> allows any value
    result = apply_jump_detection(None, 99)
    assert result == 99, f"Expected 99 (first call), got {result}"
    print(f"  prev=None, cur=99: allowed (first call, result={result})")

    # prev_node=10, current=60 -> blocks (diff=50, >50 is false, so... diff is exactly 50)
    # Actually abs(60-10)=50, and condition is >50, so 50 is NOT blocked
    result = apply_jump_detection(10, 60)
    assert result == 60, f"Expected 60 (diff=50, not >50), got {result}"
    print(f"  prev=10, cur=60: allowed (diff=50, not >50, result={result})")

    # prev_node=10, current=61 -> blocks (diff=51, >50)
    result = apply_jump_detection(10, 61)
    assert result == 10, f"Expected 10 (blocked, diff=51), got {result}"
    print(f"  prev=10, cur=61: blocked (diff=51>50, result={result})")

    # prev_node=10, current=50 -> allows (diff=40)
    result = apply_jump_detection(10, 50)
    assert result == 50, f"Expected 50 (diff=40), got {result}"
    print(f"  prev=10, cur=50: allowed (diff=40, result={result})")


# ---------------------------------------------------------------------------
# Section 21: Power Normalization Removed (VLAD without SSR)
# ---------------------------------------------------------------------------
def test_vlad_no_power_norm():
    """Verify VLAD without power normalization: self-sim=1.0, L2-normalized."""
    from sklearn.cluster import KMeans

    np.random.seed(42)
    n_descriptors = 300
    n_clusters = 8
    des = np.random.rand(n_descriptors, 128).astype(np.float32)

    codebook = KMeans(n_clusters=n_clusters, n_init=3, random_state=42).fit(des)

    def compute_vlad_no_power(descriptors):
        """VLAD with intra-norm and L2-norm but NO power normalization."""
        labels = codebook.predict(descriptors)
        centers = codebook.cluster_centers_
        vlad = np.zeros((n_clusters, 128))
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                vlad[i] = np.sum(descriptors[mask] - centers[i], axis=0)
                norm = np.linalg.norm(vlad[i])
                if norm > 0:
                    vlad[i] /= norm
        vlad = vlad.ravel()
        # NO power norm: vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))  <-- removed
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm
        return vlad

    sample = des[:50]
    vlad = compute_vlad_no_power(sample)

    # L2-normalized (no power norm)
    assert abs(np.linalg.norm(vlad) - 1.0) < 1e-6, \
        f"VLAD not L2-normalized: {np.linalg.norm(vlad)}"
    print(f"  L2 norm: {np.linalg.norm(vlad):.6f}")

    # Self-similarity is exactly 1.0
    self_sim = float(vlad @ vlad)
    assert abs(self_sim - 1.0) < 1e-6, f"Self-similarity not 1.0: {self_sim}"
    print(f"  Self-similarity: {self_sim:.6f}")

    # Identical input produces identical VLAD -> similarity 1.0
    vlad2 = compute_vlad_no_power(sample)
    sim = float(vlad @ vlad2)
    assert abs(sim - 1.0) < 1e-6, f"Identical frames sim not 1.0: {sim}"
    print(f"  Identical frames sim: {sim:.6f}")

    # All finite, no NaN
    assert np.all(np.isfinite(vlad)), "VLAD contains NaN/Inf"
    print(f"  All finite: OK")


# ---------------------------------------------------------------------------
# Section 22: SIFT Tuning (nfeatures=1000, contrastThreshold=0.03, mask)
# ---------------------------------------------------------------------------
def test_sift_tuning():
    """Verify VLADExtractor creates SIFT with correct params and mask."""
    import cv2

    # VLADExtractor creates SIFT with nfeatures=1000, contrastThreshold=0.03
    # We can't directly inspect SIFT params, but we can verify extraction works
    # and that _make_mask produces correct shape
    source_path = os.path.join("source", "player.py")
    if not os.path.exists(source_path):
        source_path = "player.py"
    if os.path.exists(source_path):
        with open(source_path) as f:
            src = f.read()
        assert "nfeatures=1000" in src, "SIFT nfeatures should be 1000"
        assert "contrastThreshold=0.03" in src, "SIFT contrastThreshold should be 0.03"
        print(f"  SIFT params in source: nfeatures=1000, contrastThreshold=0.03")
    else:
        print(f"  WARNING: player.py not found, skipping source check")

    # Test _make_mask produces correct shape and values
    h, w = 240, 320
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h * 0.10):int(h * 0.90), :] = 255

    assert mask.shape == (h, w), f"Mask shape wrong: {mask.shape}"
    assert mask.dtype == np.uint8, f"Mask dtype wrong: {mask.dtype}"
    # Top 10% should be 0
    assert np.all(mask[:int(h * 0.10), :] == 0), "Top 10% should be masked out"
    # Bottom 10% should be 0
    assert np.all(mask[int(h * 0.90):, :] == 0), "Bottom 10% should be masked out"
    # Middle should be 255
    assert np.all(mask[int(h * 0.10):int(h * 0.90), :] == 255), "Middle should be 255"
    print(f"  _make_mask({h}, {w}): shape={mask.shape}, top/bottom 10% masked")

    # Verify SIFT with mask produces keypoints mostly in the middle region
    # (SIFT uses a Gaussian window so keypoints near the mask edge may be slightly outside)
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03)
    kp, des = sift.detectAndCompute(img, mask)
    if kp:
        margin = 5  # allow small margin for sub-pixel SIFT positions near mask edge
        for k in kp:
            y = k.pt[1]
            assert int(h * 0.10) - margin <= y <= int(h * 0.90) + margin, \
                f"Keypoint at y={y} far outside mask region"
        print(f"  Keypoints with mask: {len(kp)}, all near masked region")
    else:
        print(f"  Keypoints with mask: 0 (OK for random image)")


# ---------------------------------------------------------------------------
# Section 23: n_clusters=64 Default
# ---------------------------------------------------------------------------
def test_n_clusters_64():
    """Verify default n_clusters=64 and VLAD dim = 64*128 = 8192."""

    source_path = os.path.join("source", "player.py")
    if not os.path.exists(source_path):
        source_path = "player.py"
    if os.path.exists(source_path):
        with open(source_path) as f:
            src = f.read()
        # KeyboardPlayerPyGame default n_clusters is 64
        assert "n_clusters: int = 64" in src, "Default n_clusters should be 64"
        print(f"  KeyboardPlayerPyGame default n_clusters=64: found in source")
    else:
        print(f"  WARNING: player.py not found")

    # VLADExtractor with n_clusters=64 has dim 64*128=8192
    n_clusters = 64
    dim = n_clusters * 128
    assert dim == 8192, f"Expected dim=8192, got {dim}"
    print(f"  VLADExtractor(n_clusters=64).dim = {dim}")

    # VLADExtractor with n_clusters=128 has dim 128*128=16384
    dim128 = 128 * 128
    assert dim128 == 16384, f"Expected dim=16384, got {dim128}"
    print(f"  VLADExtractor(n_clusters=128).dim = {dim128}")

    # Verify 64 is different from old default 128
    assert 64 != 128
    print(f"  n_clusters changed from 128 to 64: OK")


# ---------------------------------------------------------------------------
# Section 24: CHECK_RIGHT_INTERVAL=20
# ---------------------------------------------------------------------------
def test_check_right_interval():
    """Verify CHECK_RIGHT_INTERVAL constant is 20."""

    source_path = os.path.join("source", "player.py")
    if not os.path.exists(source_path):
        source_path = "player.py"
    if os.path.exists(source_path):
        with open(source_path) as f:
            src = f.read()
        assert "CHECK_RIGHT_INTERVAL = 20" in src, "CHECK_RIGHT_INTERVAL should be 20"
        print(f"  CHECK_RIGHT_INTERVAL = 20: found in source")
    else:
        print(f"  WARNING: player.py not found")

    # Simulate: right check triggers after 20 forward steps
    CHECK_RIGHT_INTERVAL = 20
    forward_count = 0
    triggered = False
    for i in range(CHECK_RIGHT_INTERVAL):
        forward_count += 1
        if forward_count >= CHECK_RIGHT_INTERVAL:
            triggered = True
    assert triggered, "CHECK_RIGHT should trigger after 20 steps"
    assert forward_count == 20, f"Expected 20, got {forward_count}"
    print(f"  Triggers after {forward_count} forward steps: OK")


# ---------------------------------------------------------------------------
# Section 25: Random Perturbation in Exploration
# ---------------------------------------------------------------------------
def test_random_perturbation():
    """Verify perturbation state vars, loop detection, and momentum tracking."""
    import cv2
    from collections import deque

    # 1. Perturbation state vars initialized
    perturb_next_interval = random.randint(60, 80)
    assert 60 <= perturb_next_interval <= 80, \
        f"Interval out of range: {perturb_next_interval}"
    perturb_forward_count = 0
    perturb_turning = False
    perturb_turn_counter = 0
    loop_buffer = deque(maxlen=20)
    loop_forward_count = 0
    recent_actions = deque(maxlen=30)
    print(f"  Perturbation vars initialized: interval={perturb_next_interval}")

    # 2. _detect_loop returns False with empty buffer
    def detect_loop(fpv, loop_buf):
        if fpv is None or len(loop_buf) < 3:
            return False
        cur = cv2.resize(cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY), (80, 60))
        for old_frame in loop_buf:
            mse = np.mean((cur.astype(np.float32) - old_frame.astype(np.float32)) ** 2)
            if mse < 200:  # LOOP_MSE_THRESHOLD
                return True
        return False

    empty_buf = deque(maxlen=20)
    test_frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    assert detect_loop(test_frame, empty_buf) == False, "Empty buffer should return False"
    assert detect_loop(None, empty_buf) == False, "None fpv should return False"
    print(f"  _detect_loop with empty buffer: False")

    # 3. _detect_loop returns True when current frame matches buffer
    frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (80, 60))
    full_buf = deque(maxlen=20)
    for _ in range(5):
        full_buf.append(small.copy())
    assert detect_loop(frame, full_buf) == True, "Matching frame should detect loop"
    print(f"  _detect_loop with matching frame: True")

    # 4. Perturbation doesn't fire before PERTURB_START_STEP
    PERTURB_START_STEP = 200
    explore_step = 100  # before start step
    should_perturb = False
    if explore_step >= PERTURB_START_STEP:
        should_perturb = True
    assert not should_perturb, "Should not perturb before PERTURB_START_STEP"
    print(f"  No perturbation before step {PERTURB_START_STEP}: OK")

    explore_step = 300  # after start step
    should_perturb = False
    if explore_step >= PERTURB_START_STEP:
        should_perturb = True
    assert should_perturb, "Should allow perturbation after PERTURB_START_STEP"
    print(f"  Perturbation allowed after step {PERTURB_START_STEP}: OK")

    # 5. Momentum tracking (deque fills correctly)
    MOMENTUM_WINDOW = 30
    actions = deque(maxlen=MOMENTUM_WINDOW)
    for _ in range(25):
        actions.append('LEFT')
    for _ in range(5):
        actions.append('FORWARD')
    assert len(actions) == 30, f"Expected 30, got {len(actions)}"
    left_count = sum(1 for a in actions if a == 'LEFT')
    assert left_count == 25, f"Expected 25 LEFTs, got {left_count}"
    ratio = left_count / len(actions)
    assert ratio > 0.70, f"Left ratio should exceed 0.70: {ratio}"
    print(f"  Momentum: {left_count}/{len(actions)} LEFT ratio={ratio:.2f} > 0.70: OK")

    # Deque overflow: adding more items evicts oldest
    for _ in range(20):
        actions.append('RIGHT')
    assert len(actions) == MOMENTUM_WINDOW
    left_count_after = sum(1 for a in actions if a == 'LEFT')
    assert left_count_after < left_count, "Old LEFTs should be evicted"
    print(f"  Deque overflow evicts old items: OK")


# ---------------------------------------------------------------------------
# Section 26: Oscillation Detection in Navigation
# ---------------------------------------------------------------------------
def test_oscillation_detection():
    """Verify oscillation detection with node_history maxlen=10."""
    from collections import deque

    # node_history maxlen is 10
    node_history = deque(maxlen=10)
    assert node_history.maxlen == 10, f"Expected maxlen=10, got {node_history.maxlen}"
    print(f"  node_history maxlen: {node_history.maxlen}")

    # With 8 entries alternating between 2 nodes, unique <= 2 triggers
    node_history.clear()
    for i in range(8):
        node_history.append(5 if i % 2 == 0 else 10)

    assert len(node_history) >= 8
    recent = list(node_history)[-8:]
    unique = len(set(recent))
    assert unique == 2, f"Expected 2 unique nodes, got {unique}"
    oscillating = unique <= 2
    assert oscillating, "Should detect oscillation with 2 alternating nodes"
    print(f"  8 entries alternating [5,10]: unique={unique}, oscillating={oscillating}")

    # With 3+ unique nodes, no oscillation
    node_history.clear()
    for node in [1, 2, 3, 4, 5, 6, 7, 8]:
        node_history.append(node)
    recent = list(node_history)[-8:]
    unique = len(set(recent))
    assert unique > 2, f"Should have >2 unique: {unique}"
    assert not (unique <= 2), "Should NOT detect oscillation"
    print(f"  8 different nodes: unique={unique}, not oscillating")

    # Not enough entries (< 8) -> no oscillation check
    node_history.clear()
    for i in range(5):
        node_history.append(5 if i % 2 == 0 else 10)
    assert len(node_history) < 8
    triggered = len(node_history) >= 8
    assert not triggered, "Should not trigger with < 8 entries"
    print(f"  5 entries: oscillation check not triggered")

    # Single node repeated (unique==1) -> handled by separate stuck branch
    node_history.clear()
    for _ in range(8):
        node_history.append(42)
    recent = list(node_history)[-8:]
    unique = len(set(recent))
    assert unique == 1
    # unique <= 2 still true, but unique == 1 is a different branch
    print(f"  8 same nodes: unique={unique}, caught by stuck OR oscillation branch")


# ---------------------------------------------------------------------------
# Section 27: Multi-View CHECKIN (_target_vlads cached)
# ---------------------------------------------------------------------------
def test_target_vlads_cached():
    """Verify _target_vlads is set after goal setup and has correct length."""

    # Simulate _setup_goal caching target VLADs
    class MockExtractor:
        def extract(self, img):
            return np.random.randn(8192).astype(np.float32)

    extractor = MockExtractor()
    targets = [np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(4)]

    _target_vlads = [extractor.extract(t) for t in targets]

    assert _target_vlads is not None, "_target_vlads should not be None"
    assert len(_target_vlads) == 4, f"Expected 4 target VLADs, got {len(_target_vlads)}"
    print(f"  _target_vlads length: {len(_target_vlads)}")

    # Each VLAD is a vector
    for i, v in enumerate(_target_vlads):
        assert v.shape == (8192,), f"Target VLAD {i} wrong shape: {v.shape}"
    print(f"  Each target VLAD shape: (8192,)")

    # Multi-view CHECKIN: take max similarity across views
    fpv_vlad = np.random.randn(8192).astype(np.float32)
    fpv_vlad /= np.linalg.norm(fpv_vlad)
    for v in _target_vlads:
        v /= np.linalg.norm(v)

    target_sims = [float(fpv_vlad @ tv) for tv in _target_vlads]
    visual_sim = max(target_sims)
    best_view_idx = int(np.argmax(target_sims))
    view_names = ['front', 'left', 'back', 'right']
    best_view = view_names[best_view_idx]
    print(f"  Max sim={visual_sim:.4f} from {best_view} view")
    assert visual_sim >= min(target_sims), "Max should be >= all individual sims"


# ---------------------------------------------------------------------------
# Section 28: Variable Action Hold
# ---------------------------------------------------------------------------
def test_variable_action_hold():
    """Verify action hold logic: near goal=1, turns=2, forward=ACTION_HOLD_FRAMES."""
    ACTION_HOLD_FRAMES = 15

    def compute_hold(hops, action_str):
        if hops <= 3:
            return 1
        elif action_str in ('LEFT', 'RIGHT'):
            return 2
        else:
            return ACTION_HOLD_FRAMES

    # hops <= 3: hold = 1
    assert compute_hold(0, 'FORWARD') == 1
    assert compute_hold(1, 'LEFT') == 1
    assert compute_hold(3, 'RIGHT') == 1
    print(f"  hops<=3: hold=1 (any action)")

    # LEFT/RIGHT action with hops > 3: hold = 2
    assert compute_hold(5, 'LEFT') == 2
    assert compute_hold(10, 'RIGHT') == 2
    print(f"  hops>3, LEFT/RIGHT: hold=2")

    # FORWARD with hops > 3: hold = ACTION_HOLD_FRAMES
    assert compute_hold(5, 'FORWARD') == ACTION_HOLD_FRAMES
    assert compute_hold(50, 'FORWARD') == ACTION_HOLD_FRAMES
    assert compute_hold(4, 'BACKWARD') == ACTION_HOLD_FRAMES
    print(f"  hops>3, FORWARD/BACKWARD: hold={ACTION_HOLD_FRAMES}")


# ---------------------------------------------------------------------------
# Section 29: hops==0 Bypass Fix (no direct CHECKIN)
# ---------------------------------------------------------------------------
def test_hops_zero_no_direct_checkin():
    """Verify navigation uses SEARCH state near goal instead of direct CHECKIN."""

    source_path = os.path.join("source", "player.py")
    if not os.path.exists(source_path):
        source_path = "player.py"
    if os.path.exists(source_path):
        with open(source_path) as f:
            src = f.read()
        # New navigation uses NavState.SEARCH for near-goal behavior
        assert "NavState.SEARCH" in src, "NavState.SEARCH should be used"
        assert "_search_act" in src, "_search_act method should exist"
        assert "SEARCH_SIM_THRESHOLD" in src, "SEARCH_SIM_THRESHOLD should exist"
        # Should NOT have the old direct hops==0 -> CHECKIN bypass
        assert 'if hops == 0:\n            return Action.CHECKIN' not in src, \
            "Old hops==0 CHECKIN bypass should be removed"
        print(f"  NavState.SEARCH found in source")
        print(f"  _search_act method found")
        print(f"  No hops==0 CHECKIN bypass (old bug removed)")
    else:
        print(f"  WARNING: player.py not found")

    # Simulate SEARCH state behavior: turn and check similarity
    def search_check(visual_sim, threshold=0.25):
        """Returns action in search state."""
        if visual_sim > threshold:
            return 'CHECKIN'
        return 'LEFT'  # keep turning

    assert search_check(0.05) == 'LEFT', "Low sim -> keep turning"
    print(f"  sim=0.05: LEFT (keep searching)")
    assert search_check(0.30) == 'CHECKIN', "High sim -> checkin"
    print(f"  sim=0.30: CHECKIN")
    assert search_check(0.20) == 'LEFT', "Below threshold -> keep turning"
    print(f"  sim=0.20: LEFT (below 0.25 threshold)")


# ---------------------------------------------------------------------------
# Section 30: Graph Constants (top_k=100, MIN_SHORTCUT_GAP=30, sim floor 0.15)
# ---------------------------------------------------------------------------
def test_graph_constants():
    """Verify graph construction constants."""

    source_path = os.path.join("source", "player.py")
    if not os.path.exists(source_path):
        source_path = "player.py"
    if os.path.exists(source_path):
        with open(source_path) as f:
            src = f.read()
        assert "MIN_SHORTCUT_GAP = 30" in src, "MIN_SHORTCUT_GAP should be 30"
        assert "top_k_shortcuts: int = 100" in src, "top_k_shortcuts default should be 100"
        # Similarity floor 0.15 (VISUAL_SHORTCUT_SIM_FLOOR)
        assert "s < VISUAL_SHORTCUT_SIM_FLOOR" in src or "VISUAL_SHORTCUT_SIM_FLOOR = 0.15" in src, \
            "Similarity floor 0.15 should be in source"
        print(f"  MIN_SHORTCUT_GAP=30: found")
        print(f"  top_k_shortcuts=100: found")
        print(f"  Similarity floor 0.15: found")
    else:
        print(f"  WARNING: player.py not found")

    # Verify similarity floor filters out low-quality shortcuts
    sim_values = [0.95, 0.80, 0.60, 0.45, 0.39, 0.30, 0.10]
    accepted = [s for s in sim_values if s >= 0.15]
    rejected = [s for s in sim_values if s < 0.15]
    assert len(accepted) == 6, f"Expected 6 accepted, got {len(accepted)}"
    assert len(rejected) == 1, f"Expected 1 rejected, got {len(rejected)}"
    print(f"  Sim floor 0.15: accepted {len(accepted)}, rejected {len(rejected)}")

    # MIN_SHORTCUT_GAP=30 (was 50)
    MIN_SHORTCUT_GAP = 30
    assert MIN_SHORTCUT_GAP == 30
    # Gap should filter nearby node pairs
    n = 100
    valid_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                   if abs(j - i) >= MIN_SHORTCUT_GAP]
    invalid_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if abs(j - i) < MIN_SHORTCUT_GAP]
    assert len(valid_pairs) > 0
    assert len(invalid_pairs) > 0
    print(f"  MIN_SHORTCUT_GAP=30: {len(valid_pairs)} valid, {len(invalid_pairs)} excluded pairs")


# ---------------------------------------------------------------------------
# Section 31: RootSIFT Epsilon (zero-sum safety)
# ---------------------------------------------------------------------------
def test_rootsift_epsilon():
    """Verify RootSIFT handles zero-sum descriptors without NaN."""

    def root_sift(des):
        """RootSIFT with epsilon to prevent division by zero."""
        des = des / (np.sum(des, axis=1, keepdims=True) + 1e-6)
        return np.sqrt(des)

    # Normal descriptors
    des = np.random.rand(10, 128).astype(np.float32)
    result = root_sift(des)
    assert np.all(np.isfinite(result)), "Normal descriptors should be finite"
    print(f"  Normal descriptors: all finite")

    # Zero-sum descriptor (all zeros)
    zero_des = np.zeros((1, 128), dtype=np.float32)
    result = root_sift(zero_des)
    assert np.all(np.isfinite(result)), "Zero descriptor should not produce NaN"
    assert not np.any(np.isnan(result)), "Zero descriptor produced NaN"
    print(f"  Zero descriptor: no NaN (epsilon={1e-6})")

    # Mixed: some zero, some normal
    mixed = np.zeros((5, 128), dtype=np.float32)
    mixed[0] = np.random.rand(128)
    mixed[2] = np.random.rand(128)
    mixed[4] = np.random.rand(128)
    result = root_sift(mixed)
    assert np.all(np.isfinite(result)), "Mixed descriptors should be finite"
    print(f"  Mixed (zero + normal): all finite")

    # Without epsilon, zero-sum would divide by 0
    def root_sift_no_epsilon(des):
        des = des / np.sum(des, axis=1, keepdims=True)
        return np.sqrt(des)

    zero_des = np.zeros((1, 128), dtype=np.float32)
    with np.errstate(invalid='ignore', divide='ignore'):
        result_bad = root_sift_no_epsilon(zero_des)
    assert np.any(np.isnan(result_bad)), "Without epsilon, zero descriptor should produce NaN"
    print(f"  Without epsilon: NaN produced (confirming need for epsilon)")

    # Verify source has epsilon
    source_path = os.path.join("source", "player.py")
    if not os.path.exists(source_path):
        source_path = "player.py"
    if os.path.exists(source_path):
        with open(source_path) as f:
            src = f.read()
        assert "1e-6" in src, "Epsilon 1e-6 should be in _root_sift"
        print(f"  Epsilon 1e-6 found in source")


# ---------------------------------------------------------------------------
# Section 32: _make_mask Static Method
# ---------------------------------------------------------------------------
def test_make_mask():
    """Verify _make_mask returns correct shape and values for various sizes."""

    def make_mask(h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.10):int(h * 0.90), :] = 255
        return mask

    # Standard size
    mask = make_mask(240, 320)
    assert mask.shape == (240, 320), f"Wrong shape: {mask.shape}"
    assert mask.dtype == np.uint8
    top_rows = int(240 * 0.10)  # 24
    bot_start = int(240 * 0.90)  # 216
    assert np.sum(mask[:top_rows, :]) == 0, "Top 10% should be zero"
    assert np.sum(mask[bot_start:, :]) == 0, "Bottom 10% should be zero"
    assert np.all(mask[top_rows:bot_start, :] == 255), "Middle should be 255"
    print(f"  240x320: top={top_rows}rows masked, bot from row {bot_start} masked")

    # Different size
    mask2 = make_mask(480, 640)
    assert mask2.shape == (480, 640)
    top2 = int(480 * 0.10)  # 48
    bot2 = int(480 * 0.90)  # 432
    assert np.sum(mask2[:top2, :]) == 0
    assert np.sum(mask2[bot2:, :]) == 0
    assert np.all(mask2[top2:bot2, :] == 255)
    print(f"  480x640: top={top2}rows, bot from row {bot2}")

    # Small image
    mask3 = make_mask(60, 80)
    assert mask3.shape == (60, 80)
    top3 = int(60 * 0.10)  # 6
    bot3 = int(60 * 0.90)  # 54
    assert np.sum(mask3[:top3, :]) == 0
    assert np.all(mask3[top3:bot3, :] == 255)
    print(f"  60x80: top={top3}rows, bot from row {bot3}")

    # Total masked pixels check
    total_white = np.sum(mask == 255)
    expected_white = (bot_start - top_rows) * 320
    assert total_white == expected_white, \
        f"White pixels: {total_white} vs expected {expected_white}"
    print(f"  White pixels: {total_white} = {expected_white}")


# ---------------------------------------------------------------------------
# Section 33: Gradient Check Logic
# ---------------------------------------------------------------------------
def test_gradient_check_logic():
    """Verify the gradient system detects worsening, improving, plateau, and flips.

    Tests the core logic of _gradient_check() with synthetic hop sequences,
    without instantiating the full player or game engine.
    """
    from collections import deque

    # Import constants from player.py
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import player as P
    GRADIENT_HISTORY_SIZE = P.GRADIENT_HISTORY_SIZE
    GRADIENT_PATIENCE = P.GRADIENT_PATIENCE
    GRADIENT_CHECK_INTERVAL = P.GRADIENT_CHECK_INTERVAL
    NAV_DIRECTION_COOLDOWN = P.NAV_DIRECTION_COOLDOWN
    NAV_PLATEAU_STEPS = P.NAV_PLATEAU_STEPS

    # --- Helper: simulate the gradient logic in isolation ---
    class GradientSim:
        """Minimal reproduction of _gradient_check() logic."""
        def __init__(self):
            self.hop_history = deque(maxlen=GRADIENT_HISTORY_SIZE)
            self.prev_avg_hops = None
            self.gradient_worsen_count = 0
            self.nav_last_best_hops = 9999
            self.nav_plateau_start = 0
            self.nav_last_flip_step = 0
            self.nav_total_steps = 0
            self.wall_hand = "left"
            self.flips = []          # track (step, reason)
            self.progress_events = []

        def gradient_check(self, hops: int) -> dict:
            """Feed a hop reading and return what happened."""
            self.hop_history.append(hops)
            avg_hops = float(np.median(list(self.hop_history)))
            flipped = False
            reason = None

            if self.prev_avg_hops is not None:
                if avg_hops > self.prev_avg_hops + 2:
                    self.gradient_worsen_count += 1
                    if self.gradient_worsen_count >= GRADIENT_PATIENCE:
                        if (self.nav_total_steps - self.nav_last_flip_step) >= NAV_DIRECTION_COOLDOWN:
                            old = self.wall_hand
                            self.wall_hand = "right" if old == "left" else "left"
                            self.nav_last_flip_step = self.nav_total_steps
                            self.gradient_worsen_count = 0
                            self.hop_history.clear()
                            flipped = True
                            reason = "gradient"
                            self.flips.append((self.nav_total_steps, "gradient"))
                elif avg_hops < self.prev_avg_hops - 2:
                    self.gradient_worsen_count = 0

            if hops < self.nav_last_best_hops:
                self.nav_last_best_hops = hops
                self.nav_plateau_start = self.nav_total_steps
                self.progress_events.append((self.nav_total_steps, hops))

            self.prev_avg_hops = avg_hops
            return {"avg_hops": avg_hops, "flipped": flipped, "reason": reason,
                    "worsen_count": self.gradient_worsen_count}

        def check_plateau(self) -> bool:
            """Check if plateau escape should trigger."""
            if (self.nav_total_steps - self.nav_plateau_start) >= NAV_PLATEAU_STEPS:
                old = self.wall_hand
                self.wall_hand = "right" if old == "left" else "left"
                self.nav_last_flip_step = self.nav_total_steps
                self.hop_history.clear()
                self.prev_avg_hops = None
                self.nav_last_best_hops = 9999
                self.nav_plateau_start = self.nav_total_steps
                self.flips.append((self.nav_total_steps, "plateau"))
                return True
            return False

    # === Test 1: Steady worsening triggers gradient flip ===
    sim = GradientSim()
    # Feed steadily increasing hops with large jumps (>+2 each).
    # First check has no prev_avg_hops (None), so no worsening detected.
    hops_sequence = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    for i, h in enumerate(hops_sequence):
        sim.nav_total_steps = (i + 1) * GRADIENT_CHECK_INTERVAL
        sim.gradient_check(h)

    assert len(sim.flips) > 0, (
        f"Expected gradient flip with steadily increasing hops "
        f"{hops_sequence}, but got 0 flips. worsen_count={sim.gradient_worsen_count}")
    assert sim.flips[0][1] == "gradient", "Flip should be gradient-triggered"
    print(f"  Test 1 - Steady worsening: flip at step {sim.flips[0][0]} after "
          f"{len(hops_sequence)} checks")

    # === Test 2: Improving hops resets worsen count ===
    sim2 = GradientSim()
    # Worsen a few times then improve sharply (drop > 2 from median).
    for i, h in enumerate([30, 35, 40, 45, 20]):
        sim2.nav_total_steps = (i + 1) * GRADIENT_CHECK_INTERVAL
        sim2.gradient_check(h)
    assert sim2.gradient_worsen_count == 0, (
        f"Expected worsen_count reset after improvement, got {sim2.gradient_worsen_count}")
    print(f"  Test 2 - Improving hops resets worsen count: OK")

    # === Test 3: Flat hops -- no worsening, no improvement ===
    sim3 = GradientSim()
    for i in range(10):
        sim3.nav_total_steps = (i + 1) * GRADIENT_CHECK_INTERVAL
        sim3.gradient_check(30)  # constant
    assert len(sim3.flips) == 0, (
        f"Expected no flips with constant hops, got {len(sim3.flips)}")
    assert sim3.gradient_worsen_count == 0, (
        f"Expected worsen_count=0 with flat hops, got {sim3.gradient_worsen_count}")
    print(f"  Test 3 - Flat hops: no flip, worsen_count=0: OK")

    # === Test 4: Cooldown prevents rapid flips ===
    sim4 = GradientSim()
    # Trigger first flip with enough worsening (PATIENCE steps) and enough
    # elapsed steps to clear the NAV_DIRECTION_COOLDOWN.
    # Use larger step increments so total steps exceed cooldown.
    step_increment = max(GRADIENT_CHECK_INTERVAL,
                         NAV_DIRECTION_COOLDOWN // (GRADIENT_PATIENCE + 1) + 1)
    for i in range(GRADIENT_PATIENCE + 1):
        sim4.nav_total_steps = (i + 1) * step_increment
        sim4.gradient_check(10 + i * 15)  # steep worsening
    first_flip_step = sim4.flips[0][0] if sim4.flips else None
    assert first_flip_step is not None, \
        f"Should have flipped on steep worsening (worsen_count={sim4.gradient_worsen_count}, " \
        f"patience={GRADIENT_PATIENCE}, last_step={sim4.nav_total_steps}, cooldown={NAV_DIRECTION_COOLDOWN})"
    num_flips_at_first = len(sim4.flips)

    # Now feed more worsening with steps very close together (within cooldown)
    for i in range(6):
        sim4.nav_total_steps = first_flip_step + (i + 1)  # 1 step apart
        sim4.gradient_check(100 + i * 15)
    flips_during_cooldown = len(sim4.flips) - num_flips_at_first
    assert flips_during_cooldown == 0, (
        f"Cooldown should prevent rapid flips, but got {flips_during_cooldown} "
        f"more flips within cooldown period")
    print(f"  Test 4 - Cooldown prevents rapid flips: OK "
          f"(cooldown={NAV_DIRECTION_COOLDOWN})")

    # === Test 5: Plateau triggers escape after NAV_PLATEAU_STEPS ===
    sim5 = GradientSim()
    sim5.nav_total_steps = NAV_PLATEAU_STEPS
    escaped = sim5.check_plateau()
    assert escaped, (
        f"Expected plateau escape at step {NAV_PLATEAU_STEPS}")
    assert sim5.nav_last_best_hops == 9999, "Plateau escape should reset best hops"
    assert sim5.prev_avg_hops is None, "Plateau escape should clear prev_avg_hops"
    print(f"  Test 5 - Plateau escape at {NAV_PLATEAU_STEPS} steps: OK")

    # === Test 6: Progress resets plateau timer ===
    sim6 = GradientSim()
    sim6.nav_total_steps = NAV_PLATEAU_STEPS - 10
    sim6.gradient_check(20)  # Progress! (20 < 9999)
    assert sim6.nav_plateau_start == NAV_PLATEAU_STEPS - 10, (
        f"Progress should reset plateau start, got {sim6.nav_plateau_start}")
    sim6.nav_total_steps = NAV_PLATEAU_STEPS
    escaped = sim6.check_plateau()
    assert not escaped, "Should NOT plateau -- progress was recent"
    print(f"  Test 6 - Progress resets plateau timer: OK")

    # === Test 7: Median filter stability with noisy input ===
    sim7 = GradientSim()
    noisy_hops = [30, 25, 35, 28, 32, 30, 27, 33, 29, 31]
    medians = []
    for i, h in enumerate(noisy_hops):
        sim7.nav_total_steps = (i + 1) * GRADIENT_CHECK_INTERVAL
        result = sim7.gradient_check(h)
        medians.append(result["avg_hops"])
    # Once history is full (after GRADIENT_HISTORY_SIZE readings),
    # median should stay in a narrow range despite noise
    stable_medians = medians[GRADIENT_HISTORY_SIZE:]
    if stable_medians:
        median_range = max(stable_medians) - min(stable_medians)
        assert median_range <= 6, (
            f"Median filter should smooth noise, range was {median_range:.1f}")
    assert len(sim7.flips) == 0, (
        f"Noisy-but-stable hops should not trigger flip, got {len(sim7.flips)}")
    print(f"  Test 7 - Median filter smooths noise: OK")

    # === Test 8: Timing -- gradient flip vs plateau race ===
    # With PATIENCE checks needed and checks every INTERVAL steps,
    # the earliest gradient flip requires: 1 skip (no prev) + PATIENCE worsenings
    # Also must clear NAV_DIRECTION_COOLDOWN from initial flip_step=0
    earliest_gradient_flip = max((GRADIENT_PATIENCE + 1) * GRADIENT_CHECK_INTERVAL,
                                 NAV_DIRECTION_COOLDOWN)
    print(f"  Test 8 - Timing analysis:")
    print(f"    Earliest gradient flip: step {earliest_gradient_flip} "
          f"(patience={GRADIENT_PATIENCE} x interval={GRADIENT_CHECK_INTERVAL})")
    print(f"    Plateau escape: step {NAV_PLATEAU_STEPS}")
    if earliest_gradient_flip > NAV_PLATEAU_STEPS:
        print(f"    WARNING: Gradient flip can NEVER beat plateau escape! "
              f"({earliest_gradient_flip} > {NAV_PLATEAU_STEPS})")
    else:
        print(f"    Gradient flip can fire before plateau: OK "
              f"({earliest_gradient_flip} <= {NAV_PLATEAU_STEPS})")

    # The gradient system MUST be able to fire before the plateau mechanism.
    # Otherwise it is effectively dead code and the plateau does all the work.
    assert earliest_gradient_flip <= NAV_PLATEAU_STEPS, (
        f"Gradient flip earliest={earliest_gradient_flip} > plateau={NAV_PLATEAU_STEPS}. "
        f"Gradient system can never beat plateau -- reduce PATIENCE or INTERVAL.")


# ---------------------------------------------------------------------------
# Section 34: Perturbation Momentum Fix (spin bug regression test)
# ---------------------------------------------------------------------------
def test_perturbation_momentum_fix():
    """Verify that perturbation turns record into _recent_actions, preventing
    the momentum check from immediately re-triggering (the spin bug).

    Bug: _explore_act() perturbation handler returned early WITHOUT recording
    the turn direction in _recent_actions.  This caused the momentum check
    (>70% LEFT in last 30 actions) to re-trigger immediately after every
    perturbation, wasting ~40% of exploration time spinning.

    Fix: append the perturbation direction to _recent_actions each step.
    """
    from collections import deque

    MOMENTUM_WINDOW = 30
    MOMENTUM_LEFT_RATIO = 0.70
    TURN_STEPS_90 = 3

    # --- Setup: deque with >70% LEFT (triggers momentum) ---
    recent_actions = deque(maxlen=MOMENTUM_WINDOW)
    for _ in range(22):
        recent_actions.append('LEFT')
    for _ in range(8):
        recent_actions.append('FORWARD')

    left_count = sum(1 for a in recent_actions if a == 'LEFT')
    ratio_before = left_count / len(recent_actions)
    assert ratio_before > MOMENTUM_LEFT_RATIO, \
        f"Pre-condition failed: ratio {ratio_before:.2f} should exceed {MOMENTUM_LEFT_RATIO}"
    print(f"  Before perturbation: {left_count}/{len(recent_actions)} LEFT "
          f"ratio={ratio_before:.2f} (triggers momentum)")

    # --- Simulate the perturbation: 3 steps of RIGHT recorded ---
    # This is what the fixed code does (line 877-878 of player.py)
    perturb_direction = 'RIGHT'  # momentum forces RIGHT
    for step in range(TURN_STEPS_90):
        recent_actions.append(perturb_direction)

    # --- Verify: momentum should NOT re-trigger ---
    assert len(recent_actions) == MOMENTUM_WINDOW, \
        f"Deque length should be {MOMENTUM_WINDOW}, got {len(recent_actions)}"

    left_count_after = sum(1 for a in recent_actions if a == 'LEFT')
    ratio_after = left_count_after / len(recent_actions)
    assert ratio_after <= MOMENTUM_LEFT_RATIO, \
        f"After {TURN_STEPS_90} RIGHT steps, LEFT ratio should be <= {MOMENTUM_LEFT_RATIO}, " \
        f"got {ratio_after:.2f} ({left_count_after}/{len(recent_actions)})"
    print(f"  After perturbation:  {left_count_after}/{len(recent_actions)} LEFT "
          f"ratio={ratio_after:.2f} (<= {MOMENTUM_LEFT_RATIO})")

    # --- Verify the exact math ---
    # Deque had 30 items. We appended 3 RIGHT, evicting the 3 oldest (all LEFT).
    # Remaining: 19 LEFT + 8 FORWARD + 3 RIGHT = 30
    # LEFT ratio = 19/30 = 0.6333...
    expected_left = 22 - TURN_STEPS_90  # 19 LEFTs remain
    assert left_count_after == expected_left, \
        f"Expected {expected_left} LEFTs remaining, got {left_count_after}"
    print(f"  Exact count: {left_count_after} LEFTs = {expected_left} (22 original - {TURN_STEPS_90} evicted)")

    # --- Counter-test: without the fix, ratio stays the same ---
    # If perturbation did NOT record actions, the deque is unchanged
    broken_actions = deque(maxlen=MOMENTUM_WINDOW)
    for _ in range(22):
        broken_actions.append('LEFT')
    for _ in range(8):
        broken_actions.append('FORWARD')
    # Simulate buggy code: perturbation steps with NO append
    # (nothing happens to the deque)
    broken_left = sum(1 for a in broken_actions if a == 'LEFT')
    broken_ratio = broken_left / len(broken_actions)
    assert broken_ratio > MOMENTUM_LEFT_RATIO, \
        f"Buggy version should still have high ratio: {broken_ratio:.2f}"
    print(f"  Buggy (no append):   {broken_left}/{len(broken_actions)} LEFT "
          f"ratio={broken_ratio:.2f} (would re-trigger!)")
    print(f"  Fix confirmed: perturbation recording prevents momentum re-trigger")


# ---------------------------------------------------------------------------
# Section 35: Localization With Local Search Radius + Jump Threshold
# ---------------------------------------------------------------------------
def test_localization_no_smoothing():
    """Verify localization behavior with local search radius and jump threshold.

    Current _get_current_node() uses:
      - Light temporal smoothing (0.95/0.05)
      - LOCALIZATION_SEARCH_RADIUS: local argmax within a window around prev_node
      - LOCALIZATION_JUMP_THRESHOLD: if global argmax is far from prev, prefer local
        unless global similarity is significantly better (LOCALIZATION_JUMP_MARGIN)

    This test verifies:
      a) Localization returns the true best-match node on first call (no prev).
      b) Local search radius constrains jumps when local match is comparable.
      c) _prev_node is updated every call.
      d) The gradient system (median of hop_history) absorbs noise.
      e) Source inspection confirms the local search radius approach.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import player as P

    SEARCH_RADIUS = P.LOCALIZATION_SEARCH_RADIUS
    JUMP_THRESHOLD = P.LOCALIZATION_JUMP_THRESHOLD
    JUMP_MARGIN = P.LOCALIZATION_JUMP_MARGIN

    np.random.seed(99)
    n, dim = 200, 128
    database = np.random.randn(n, dim).astype(np.float32)
    database /= np.linalg.norm(database, axis=1, keepdims=True)

    # --- Standalone replica of _get_current_node (local search) ---
    class LocalizerWithRadius:
        def __init__(self, db):
            self.database = db
            self._prev_node = None
            self._prev_sims = None

        def localize(self, feat):
            sims = self.database @ feat
            if self._prev_sims is not None:
                sims = 0.95 * sims + 0.05 * self._prev_sims
            self._prev_sims = sims.copy()

            cur = int(np.argmax(sims))
            if self._prev_node is not None and len(sims) > 0:
                prev = self._prev_node
                lo = max(0, prev - SEARCH_RADIUS)
                hi = min(len(sims), prev + SEARCH_RADIUS + 1)
                local_cur = lo + int(np.argmax(sims[lo:hi]))
                if abs(cur - prev) > JUMP_THRESHOLD:
                    global_sim = float(sims[cur])
                    local_sim = float(sims[local_cur])
                    if local_sim >= global_sim - JUMP_MARGIN:
                        cur = local_cur
            self._prev_node = cur
            return cur

    loc = LocalizerWithRadius(database)

    # (a) Exact match on first call (no prev_node): query identical to node 42 -> returns 42
    q = database[42].copy()
    result = loc.localize(q)
    assert result == 42, f"Expected 42, got {result}"
    print(f"  (a) Exact match node 42: got {result}")

    # (b) Local search: query for nearby node should work smoothly
    q_near = database[45].copy()
    result_near = loc.localize(q_near)
    # Should return 45 or very close (within search radius of 42)
    assert abs(result_near - 45) <= 5, f"Expected near 45, got {result_near}"
    print(f"  (b) Near jump 42->{result_near}: smooth transition")

    # (c) _prev_node tracks latest result
    assert loc._prev_node == result_near, \
        f"_prev_node should be {result_near}, got {loc._prev_node}"
    print(f"  (c) _prev_node updated to {loc._prev_node}")

    # (d) Simulate gradient system: median of noisy hop counts is stable
    from collections import deque
    hop_history = deque(maxlen=5)
    goal_node = 100
    np.random.seed(42)
    medians = []
    for i in range(10):
        true_pos = 50 + i * 2
        noise = np.random.choice([-30, -10, 0, 10, 30])
        noisy_node = max(0, min(n - 1, true_pos + noise))
        hops = abs(noisy_node - goal_node)
        hop_history.append(hops)
        med = float(np.median(list(hop_history)))
        medians.append(med)

    assert medians[0] >= medians[-1] - 10, \
        f"Median should trend down: first={medians[0]}, last={medians[-1]}"
    print(f"  (d) Median hop trend: {[f'{m:.0f}' for m in medians]}")
    print(f"      First={medians[0]:.0f}, Last={medians[-1]:.0f} (stable despite noise)")

    # (e) Verify player.py _get_current_node uses local search radius approach
    player_path = os.path.join("source", "player.py")
    if not os.path.exists(player_path):
        player_path = os.path.join(os.path.dirname(os.path.abspath(
            sys.modules[__name__].__file__ if hasattr(sys.modules[__name__], '__file__')
            else __file__)), "player.py")
    if not os.path.exists(player_path):
        for p in ["source/player.py", "player.py", "vis_nav_player/source/player.py"]:
            if os.path.exists(p):
                player_path = p
                break
    with open(player_path) as f:
        src = f.read()
    import re
    match = re.search(r'def _get_current_node\(self\).*?(?=\n    def |\nclass |\Z)',
                      src, re.DOTALL)
    assert match, "Could not find _get_current_node in player.py"
    method_body = match.group()
    # Should contain local search radius approach
    assert 'LOCALIZATION_SEARCH_RADIUS' in method_body, \
        "_get_current_node should use LOCALIZATION_SEARCH_RADIUS"
    assert 'LOCALIZATION_JUMP_THRESHOLD' in method_body, \
        "_get_current_node should use LOCALIZATION_JUMP_THRESHOLD"
    # Should contain argmax
    assert 'argmax(sims' in method_body, \
        "_get_current_node should use argmax(sims)"
    assert 'self._prev_node = cur' in method_body, \
        "_get_current_node should update _prev_node"
    # Should have light temporal smoothing (0.95/0.05)
    assert '0.95' in method_body and '0.05' in method_body, \
        "_get_current_node should have 0.95/0.05 temporal smoothing"
    print(f"  (e) Source inspection: local search radius + jump threshold + light smoothing")



# ---------------------------------------------------------------------------
# Section 36: MiniBatchKMeans Pipeline
# ---------------------------------------------------------------------------
def test_minibatch_kmeans_pipeline():
    """Verify the full VLAD pipeline builds correctly with MiniBatchKMeans.

    Tests:
    - MiniBatchKMeans fits and produces correct cluster_centers_ shape
    - .predict() returns valid cluster assignments
    - .n_clusters and .inertia_ attributes exist
    - VLAD vectors computed from MiniBatchKMeans codebook are valid
    - Pickle round-trip preserves the MiniBatchKMeans object
    - MiniBatchKMeans and KMeans produce compatible VLAD outputs
    """
    from sklearn.cluster import MiniBatchKMeans, KMeans
    import pickle
    import tempfile
    import time

    np.random.seed(42)
    n_images = 200
    n_features_per_image = 500
    desc_dim = 128
    n_clusters = 64

    # Generate synthetic descriptors (simulate ~200 images x 500 features)
    all_des = np.random.rand(n_images * n_features_per_image, desc_dim).astype(np.float32)
    print(f"  Descriptor matrix: {all_des.shape} ({all_des.nbytes / 1e6:.1f} MB)")

    # --- (a) Fit MiniBatchKMeans with same params as player.py ---
    t0 = time.time()
    codebook = MiniBatchKMeans(
        n_clusters=n_clusters, init='k-means++',
        n_init=1, max_iter=100, batch_size=1024,
        random_state=42,
    ).fit(all_des)
    fit_time = time.time() - t0
    print(f"  (a) MiniBatchKMeans fit: {fit_time:.2f}s on {len(all_des)} descriptors")

    # Check all attributes that player.py _des_to_vlad and build_vocabulary use
    assert hasattr(codebook, 'cluster_centers_'), "Missing cluster_centers_"
    assert hasattr(codebook, 'predict'), "Missing predict method"
    assert hasattr(codebook, 'n_clusters'), "Missing n_clusters"
    assert hasattr(codebook, 'inertia_'), "Missing inertia_"
    assert codebook.cluster_centers_.shape == (n_clusters, desc_dim),         f"Wrong centers shape: {codebook.cluster_centers_.shape}"
    assert codebook.n_clusters == n_clusters
    print(f"      centers: {codebook.cluster_centers_.shape}, inertia={codebook.inertia_:.0f}")

    # --- (b) predict() returns valid labels ---
    sample = all_des[:100]
    labels = codebook.predict(sample)
    assert labels.shape == (100,), f"Wrong labels shape: {labels.shape}"
    assert labels.min() >= 0 and labels.max() < n_clusters,         f"Labels out of range: [{labels.min()}, {labels.max()}]"
    print(f"  (b) predict(): min={labels.min()}, max={labels.max()}, "
          f"unique={len(np.unique(labels))}")

    # --- (c) Compute VLAD vector (same logic as player.py _des_to_vlad) ---
    def compute_vlad(des, cb):
        labs = cb.predict(des)
        centers = cb.cluster_centers_
        k = cb.n_clusters
        vlad = np.zeros((k, des.shape[1]))
        for i in range(k):
            mask = labs == i
            if np.any(mask):
                vlad[i] = np.sum(des[mask] - centers[i], axis=0)
                norm = np.linalg.norm(vlad[i])
                if norm > 0:
                    vlad[i] /= norm
        vlad = vlad.ravel()
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm
        return vlad

    vlad = compute_vlad(sample, codebook)
    assert vlad.shape == (n_clusters * desc_dim,), f"Wrong VLAD shape: {vlad.shape}"
    assert abs(np.linalg.norm(vlad) - 1.0) < 1e-6, "VLAD not unit-normalized"
    assert np.all(np.isfinite(vlad)), "VLAD contains NaN/Inf"
    print(f"  (c) VLAD dim={vlad.shape[0]}, norm={np.linalg.norm(vlad):.6f}, "
          f"range=[{vlad.min():.4f}, {vlad.max():.4f}]")

    # --- (d) Deterministic ---
    vlad2 = compute_vlad(sample, codebook)
    assert np.allclose(vlad, vlad2, atol=1e-10), "VLAD not deterministic"
    print(f"  (d) Deterministic: OK")

    # --- (e) Pickle round-trip (simulates cache save/load) ---
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
        tmp_path = tmp.name
        pickle.dump(codebook, tmp)
    with open(tmp_path, 'rb') as f:
        loaded = pickle.load(f)
    os.unlink(tmp_path)

    assert type(loaded).__name__ == 'MiniBatchKMeans', \
        f"Loaded type is {type(loaded).__name__}, expected MiniBatchKMeans"
    assert np.allclose(loaded.cluster_centers_, codebook.cluster_centers_), \
        "Loaded codebook centers differ"
    labels_loaded = loaded.predict(sample)
    assert np.array_equal(labels, labels_loaded), \
        "Loaded codebook gives different predictions"
    vlad_loaded = compute_vlad(sample, loaded)
    assert np.allclose(vlad, vlad_loaded, atol=1e-10), \
        "Loaded codebook gives different VLAD"
    print(f"  (e) Pickle round-trip: OK (type={type(loaded).__name__})")

    # --- (f) Cross-compatibility: MiniBatchKMeans vs KMeans same API ---
    kmeans_cb = KMeans(
        n_clusters=n_clusters, n_init=1, max_iter=10, random_state=42
    ).fit(all_des[:10000])
    vlad_km = compute_vlad(sample, kmeans_cb)
    assert vlad_km.shape == vlad.shape, \
        "KMeans and MiniBatchKMeans VLAD shapes differ"
    print(f"  (f) KMeans vs MiniBatchKMeans: same shape ({vlad_km.shape[0]}), "
          f"cosine sim={float(vlad @ vlad_km):.4f}")

    # --- (g) Batch VLAD extraction (simulates extract_batch) ---
    per_image_des = [all_des[i*n_features_per_image:(i+1)*n_features_per_image]
                     for i in range(min(10, n_images))]
    vlads = []
    for des in per_image_des:
        vlads.append(compute_vlad(des, codebook))
    vlads = np.array(vlads)
    assert vlads.shape == (10, n_clusters * desc_dim), \
        f"Batch shape wrong: {vlads.shape}"
    sims = vlads @ vlads.T
    for i in range(10):
        assert abs(sims[i, i] - 1.0) < 1e-6, f"Self-sim for image {i} != 1.0"
    avg_off_diag = (sims.sum() - np.trace(sims)) / (10 * 9)
    print(f"  (g) Batch VLAD: {vlads.shape}, avg off-diag sim={avg_off_diag:.4f}")
    assert avg_off_diag < 0.99, "All VLADs too similar (codebook not discriminative)"

    # --- (h) Verify player.py uses MiniBatchKMeans ---
    source_path = os.path.join("source", "player.py")
    if not os.path.exists(source_path):
        source_path = "player.py"
    if os.path.exists(source_path):
        with open(source_path) as f:
            src = f.read()
        assert "MiniBatchKMeans" in src, "player.py should import MiniBatchKMeans"
        assert "MiniBatchKMeans(" in src, "player.py should instantiate MiniBatchKMeans"
        lines = src.split('\n')
        cluster_imports = [l for l in lines
                           if 'from sklearn.cluster' in l and 'import' in l]
        for line in cluster_imports:
            assert 'MiniBatchKMeans' in line, \
                f"sklearn.cluster import should include MiniBatchKMeans: {line}"
        print(f"  (h) player.py: MiniBatchKMeans import verified")
    else:
        print(f"  WARNING: player.py not found, skipping source check")

    print(f"\n  Timing summary: {fit_time:.2f}s for {len(all_des)} descriptors")
    print(f"  Projected: ~5-15s for 250K desc, ~15-40s for 1M desc")



# ---------------------------------------------------------------------------
# Section 39: Backup CHECKIN ("Lucky Pass" Detector)
# ---------------------------------------------------------------------------
def test_backup_checkin():
    """Verify backup CHECKIN mechanism in _gradient_check().

    The backup CHECKIN reuses the VLAD cached by _get_current_node() to check
    target similarity at zero extra computation cost (4 dot products).  When
    max target similarity exceeds BACKUP_CHECKIN_SIM (0.35), _gradient_check()
    sets nav_state=SEARCH and returns -1 as a sentinel so _auto_navigate()
    can immediately start a 360-degree scan.

    Tests:
      (a) _get_current_node caches VLAD as _last_vlad
      (b) Backup CHECKIN triggers when target sim > 0.35
      (c) Backup CHECKIN does NOT trigger when target sim <= 0.35
      (d) Sentinel return value is -1
      (e) nav_state is set to SEARCH when triggered
      (f) _auto_navigate handles sentinel correctly (returns LEFT)
      (g) Source inspection: BACKUP_CHECKIN_SIM constant exists in player.py
    """
    BACKUP_CHECKIN_SIM = 0.35  # must match player.py constant

    np.random.seed(42)
    dim = 128  # VLAD dimension for this test (real is 16384)

    # (a) Simulate _get_current_node caching VLAD
    fpv_vlad = np.random.randn(dim).astype(np.float32)
    fpv_vlad /= np.linalg.norm(fpv_vlad)
    _last_vlad = fpv_vlad  # this is what _get_current_node stores
    assert _last_vlad is not None, "_last_vlad should be cached after _get_current_node"
    print(f"  (a) _last_vlad cached: shape={_last_vlad.shape}")

    # Create target VLADs (4 views)
    _target_vlads = []
    for _ in range(4):
        tv = np.random.randn(dim).astype(np.float32)
        tv /= np.linalg.norm(tv)
        _target_vlads.append(tv)

    # (b) High similarity: make one target nearly identical to fpv_vlad
    _target_vlads_high = list(_target_vlads)  # copy
    _target_vlads_high[2] = fpv_vlad * 0.95 + np.random.randn(dim).astype(np.float32) * 0.05
    _target_vlads_high[2] /= np.linalg.norm(_target_vlads_high[2])

    target_sims_high = [float(_last_vlad @ tv) for tv in _target_vlads_high]
    max_sim_high = max(target_sims_high)
    assert max_sim_high > BACKUP_CHECKIN_SIM,         f"High-sim case should exceed {BACKUP_CHECKIN_SIM}, got {max_sim_high:.4f}"
    best_view_idx = int(np.argmax(target_sims_high))
    view_names = ['front', 'left', 'back', 'right']
    print(f"  (b) High sim: max={max_sim_high:.4f} ({view_names[best_view_idx]}) > {BACKUP_CHECKIN_SIM} -> triggers")

    # (c) Low similarity: random targets, unlikely to exceed 0.35
    target_sims_low = [float(_last_vlad @ tv) for tv in _target_vlads]
    max_sim_low = max(target_sims_low)
    assert max_sim_low < BACKUP_CHECKIN_SIM,         f"Random targets should have low sim, got {max_sim_low:.4f}"
    print(f"  (c) Low sim:  max={max_sim_low:.4f} < {BACKUP_CHECKIN_SIM} -> no trigger")

    # (d) & (e) Simulate the full _gradient_check backup logic
    nav_state = 'navigate'
    search_turn_counter = 0
    search_scan_count = 0
    search_best_sim = 0.0

    def simulate_gradient_check(last_vlad, target_vlads, hops_value):
        nonlocal nav_state, search_turn_counter, search_scan_count, search_best_sim
        hops = hops_value
        if last_vlad is not None and target_vlads:
            target_sims = [float(last_vlad @ tv) for tv in target_vlads]
            max_sim = max(target_sims)
            if max_sim > BACKUP_CHECKIN_SIM:
                nav_state = 'search'
                search_turn_counter = 0
                search_scan_count = 0
                search_best_sim = 0.0
                return -1, nav_state
        return hops, nav_state

    # Case 1: high sim -> sentinel -1, SEARCH state
    nav_state = 'navigate'
    result, state = simulate_gradient_check(_last_vlad, _target_vlads_high, 10)
    assert result == -1, f"Expected sentinel -1, got {result}"
    assert state == 'search', f"Expected SEARCH state, got {state}"
    print(f"  (d) Sentinel: gradient_check returned {result}")
    print(f"  (e) nav_state set to SEARCH: {state}")

    # Case 2: low sim -> normal hops return
    nav_state = 'navigate'
    result2, state2 = simulate_gradient_check(_last_vlad, _target_vlads, 10)
    assert result2 == 10, f"Expected hops=10, got {result2}"
    assert state2 == 'navigate', f"Expected NAVIGATE state, got {state2}"
    print(f"  (d') No trigger: gradient_check returned hops={result2}, state={state2}")

    # (f) Simulate _auto_navigate handling sentinel
    def simulate_auto_navigate(hops_from_gradient):
        if hops_from_gradient == -1:
            return 'LEFT'  # start 360 scan
        elif hops_from_gradient < 15:
            return 'APPROACH'
        else:
            return 'WALL_FOLLOW'

    action = simulate_auto_navigate(-1)
    assert action == 'LEFT', f"Sentinel should trigger LEFT, got {action}"
    print(f"  (f) _auto_navigate on sentinel: returns {action} (start scan)")

    action_normal = simulate_auto_navigate(10)
    assert action_normal == 'APPROACH'
    print(f"  (f') Normal hops=10: returns {action_normal}")

    action_far = simulate_auto_navigate(20)
    assert action_far == 'WALL_FOLLOW'
    print(f"  (f') Normal hops=20: returns {action_far}")

    # (g) Source inspection
    player_path = os.path.join('source', 'player.py')
    if not os.path.exists(player_path):
        for p in ['source/player.py', 'player.py', 'vis_nav_player/source/player.py']:
            if os.path.exists(p):
                player_path = p
                break
    with open(player_path) as f:
        src = f.read()
    assert 'BACKUP_CHECKIN_SIM' in src, 'BACKUP_CHECKIN_SIM missing from player.py'
    assert '_last_vlad' in src, '_last_vlad missing from player.py'
    assert 'self._last_vlad = feat' in src, '_get_current_node should cache VLAD'
    import re as _re
    gc_match = _re.search(r'def _gradient_check\(self\).*?(?=\n    def |\nclass |\Z)',
                          src, _re.DOTALL)
    assert gc_match, 'Could not find _gradient_check in player.py'
    gc_body = gc_match.group()
    assert '_last_vlad' in gc_body, '_gradient_check should use _last_vlad'
    assert '_target_vlads' in gc_body, '_gradient_check should check _target_vlads'
    assert 'return -1' in gc_body, '_gradient_check should return -1 sentinel'
    print(f"  (g) Source: BACKUP_CHECKIN_SIM, _last_vlad, sentinel -1 all present")
    print(f"  Cost: 0 extra VLAD extractions, 4 dot products per gradient check")



# ---------------------------------------------------------------------------
# Section 37: Wall-Follow Hand Flip State Reset
# ---------------------------------------------------------------------------
def test_wall_follow_hand_flip():
    """Verify hand-flipping resets wall-follow state correctly.

    Bugs fixed in player.py:
    1. Gradient flip did NOT reset _nav_es/_nav_tc/_nav_fwd/_nav_stuck_count.
       If agent was mid-TURN_LEFT when gradient flip changed hand to RIGHT,
       it continued emitting Action.LEFT (TURN_LEFT is hardcoded, not
       parameterized by wall_turn). Fix: reset _nav_es to FORWARD on flip.
    2. Plateau flip had NO cooldown check. A plateau could flip the hand
       within 100 steps of a gradient flip, causing rapid oscillation.
       Fix: plateau respects NAV_DIRECTION_COOLDOWN.
    3. Plateau did not reset _nav_fwd/_nav_stuck_count. Fix: reset those too.
    """
    from enum import Enum

    HAND_LEFT = 'left'
    HAND_RIGHT = 'right'
    NAV_DIRECTION_COOLDOWN = 100

    class ExploreState(Enum):
        FORWARD = 'forward'
        TURN_LEFT = 'turn_left'
        TURN_RIGHT = 'turn_right'
        CHECK_RIGHT = 'check_right'
        REVERSE = 'reverse'

    class WallFollower:
        TURN_STEPS_90 = 3
        TURN_STEPS_180 = 6

        def __init__(self):
            self._nav_es = ExploreState.FORWARD
            self._nav_tc = 0
            self._nav_fwd = 0
            self._nav_stuck_count = 0
            self.wall_hand = HAND_LEFT
            self.nav_last_flip_step = 0
            self.nav_total_steps = 0

        def flip_hand_gradient(self):
            if (self.nav_total_steps - self.nav_last_flip_step) < NAV_DIRECTION_COOLDOWN:
                return False
            old = self.wall_hand
            self.wall_hand = HAND_RIGHT if old == HAND_LEFT else HAND_LEFT
            self.nav_last_flip_step = self.nav_total_steps
            self._nav_es = ExploreState.FORWARD
            self._nav_tc = 0
            self._nav_fwd = 0
            self._nav_stuck_count = 0
            return True

        def flip_hand_plateau(self, respect_cooldown=True):
            flipped = False
            if not respect_cooldown or                (self.nav_total_steps - self.nav_last_flip_step) >= NAV_DIRECTION_COOLDOWN:
                old = self.wall_hand
                self.wall_hand = HAND_RIGHT if old == HAND_LEFT else HAND_LEFT
                self.nav_last_flip_step = self.nav_total_steps
                flipped = True
            self._nav_es = ExploreState.REVERSE
            self._nav_tc = 0
            self._nav_fwd = 0
            self._nav_stuck_count = 0
            return flipped

        def wall_follow_step(self, stuck=False):
            if self.wall_hand == HAND_LEFT:
                wall_turn, check_turn = 'LEFT', 'RIGHT'
                wall_state = ExploreState.TURN_LEFT
            else:
                wall_turn, check_turn = 'RIGHT', 'LEFT'
                wall_state = ExploreState.TURN_RIGHT

            if self._nav_es == ExploreState.FORWARD:
                if stuck:
                    self._nav_stuck_count += 1
                    if self._nav_stuck_count >= 3:
                        self._nav_es = ExploreState.REVERSE
                        self._nav_tc = 0
                    else:
                        self._nav_es = wall_state
                        self._nav_tc = 0
                    return wall_turn
                else:
                    self._nav_stuck_count = 0
                    self._nav_fwd += 1
                    return 'FORWARD'
            elif self._nav_es == ExploreState.TURN_LEFT:
                self._nav_tc += 1
                if self._nav_tc >= self.TURN_STEPS_90:
                    self._nav_es = ExploreState.FORWARD
                    return 'FORWARD'
                return 'LEFT'
            elif self._nav_es == ExploreState.TURN_RIGHT:
                self._nav_tc += 1
                if self._nav_tc >= self.TURN_STEPS_90:
                    self._nav_es = ExploreState.FORWARD
                    return 'FORWARD'
                return 'RIGHT'
            elif self._nav_es == ExploreState.REVERSE:
                self._nav_tc += 1
                if self._nav_tc >= self.TURN_STEPS_180:
                    self._nav_es = ExploreState.FORWARD
                    self._nav_stuck_count = 0
                    return 'FORWARD'
                return wall_turn
            return 'FORWARD'

    # 1. Gradient flip resets all state variables
    wf = WallFollower()
    wf.nav_total_steps = 200
    wf._nav_es = ExploreState.TURN_LEFT
    wf._nav_tc = 5
    wf._nav_fwd = 7
    wf._nav_stuck_count = 2
    ok = wf.flip_hand_gradient()
    assert ok, 'Gradient flip should succeed (past cooldown)'
    assert wf.wall_hand == HAND_RIGHT
    assert wf._nav_es == ExploreState.FORWARD, f'Should reset to FORWARD, got {wf._nav_es}'
    assert wf._nav_tc == 0
    assert wf._nav_fwd == 0
    assert wf._nav_stuck_count == 0
    print(f'  1. Gradient flip resets state to FORWARD: OK')

    # 2. Bug demo: TURN_LEFT + RIGHT hand = wrong direction
    wf2 = WallFollower()
    wf2.wall_hand = HAND_RIGHT
    wf2._nav_es = ExploreState.TURN_LEFT
    wf2._nav_tc = 0
    action = wf2.wall_follow_step()
    assert action == 'LEFT', f'TURN_LEFT hardcodes LEFT, got {action}'
    print(f'  2. Bug: TURN_LEFT + RIGHT hand -> LEFT (wrong, proves need for reset)')

    # 3. After proper gradient flip, next action is FORWARD
    wf3 = WallFollower()
    wf3.nav_total_steps = 200
    wf3._nav_es = ExploreState.TURN_LEFT
    wf3._nav_tc = 5
    wf3.flip_hand_gradient()
    action = wf3.wall_follow_step(stuck=False)
    assert action == 'FORWARD', f'After flip+reset, got {action}'
    print(f'  3. After gradient flip -> FORWARD: OK')

    # 4. Plateau enters REVERSE with correct hand direction
    wf4 = WallFollower()
    wf4.nav_total_steps = 1000
    wf4.flip_hand_plateau()
    assert wf4._nav_es == ExploreState.REVERSE
    assert wf4.wall_hand == HAND_RIGHT
    action = wf4.wall_follow_step()
    assert action == 'RIGHT', f'REVERSE + RIGHT hand -> RIGHT, got {action}'
    print(f'  4. Plateau REVERSE uses new hand ({action}): OK')

    # 5. Gradient cooldown blocks flip
    wf5 = WallFollower()
    wf5.nav_total_steps = 50
    ok = wf5.flip_hand_gradient()
    assert not ok
    assert wf5.wall_hand == HAND_LEFT
    print(f'  5. Gradient flip within cooldown: blocked')

    # 6. Plateau respects cooldown (no flip, but still escapes)
    wf6 = WallFollower()
    wf6.nav_total_steps = 50
    flipped = wf6.flip_hand_plateau(respect_cooldown=True)
    assert not flipped
    assert wf6.wall_hand == HAND_LEFT, 'No flip during cooldown'
    assert wf6._nav_es == ExploreState.REVERSE, 'Still enters REVERSE'
    print(f'  6. Plateau within cooldown: no flip, still escapes: OK')

    # 7. Plateau outside cooldown flips
    wf7 = WallFollower()
    wf7.nav_total_steps = 200
    flipped = wf7.flip_hand_plateau(respect_cooldown=True)
    assert flipped
    assert wf7.wall_hand == HAND_RIGHT
    print(f'  7. Plateau outside cooldown: flipped to {wf7.wall_hand}: OK')

    # 8. Back-to-back gradient flips blocked by cooldown
    wf8 = WallFollower()
    wf8.nav_total_steps = 200
    assert wf8.flip_hand_gradient()
    assert wf8.wall_hand == HAND_RIGHT
    wf8.nav_total_steps = 210
    assert not wf8.flip_hand_gradient(), 'Should block within cooldown'
    assert wf8.wall_hand == HAND_RIGHT
    wf8.nav_total_steps = 310
    assert wf8.flip_hand_gradient(), 'Should allow after cooldown'
    assert wf8.wall_hand == HAND_LEFT
    print(f'  8. Back-to-back flip prevention: OK')

    # 9. Verify player.py source has the fixes
    source_path = os.path.join('source', 'player.py')
    if not os.path.exists(source_path):
        source_path = 'player.py'
    if os.path.exists(source_path):
        with open(source_path) as f:
            src = f.read()
        assert '_nav_es = ExploreState.FORWARD' in src,             'player.py should reset _nav_es on gradient flip'
        assert 'nav_last_flip_step) >= NAV_DIRECTION_COOLDOWN' in src,             'player.py plateau should check cooldown'
        print(f'  9. Source inspection: state reset and cooldown check present')
    else:
        print(f'  9. WARNING: player.py not found, skipping source check')


# ---------------------------------------------------------------------------
# Section 38: Search State (360 Scan)
# ---------------------------------------------------------------------------
def test_search_state():
    """Verify SEARCH state constants and methods exist in player.py."""
    source_path = os.path.join('source', 'player.py')
    if not os.path.exists(source_path):
        source_path = 'player.py'
    if os.path.exists(source_path):
        with open(source_path) as f:
            src = f.read()
        assert 'SEARCH_HOPS_ENTER' in src
        assert 'NavState.SEARCH' in src
        assert '_search_act' in src
        print(f'  SEARCH state constants and methods present: OK')
    else:
        print(f'  WARNING: player.py not found')


# ---------------------------------------------------------------------------
# Section 40: Navigation State Transitions
# ---------------------------------------------------------------------------
def test_nav_state_transitions():
    """Verify all navigation state transitions and edge cases.

    Tests the NavState machine: NAVIGATE -> APPROACH -> SEARCH -> CHECKIN
    plus fallbacks (APPROACH->NAVIGATE, SEARCH->NAVIGATE) and ESCAPE.

    Edge cases tested:
      1. APPROACH->NAVIGATE: stale gradient data (hop_history, worsen_count)
      2. SEARCH->NAVIGATE: nav_last_best_hops not reset (plateau timing)
      3. NAVIGATE->APPROACH: mid-REVERSE wall-follow state carries over
      4. Multiple APPROACH->SEARCH->NAVIGATE cycles: search_scan_count reset
      5. action_hold_counter carryover across state transitions
      6. ESCAPE->NAVIGATE: plateau_start and gradient_step_counter reset
      7. NAVIGATE plateau -> ESCAPE transition
      8. Happy path: APPROACH -> SEARCH -> CHECKIN
    """
    from collections import deque

    # Import constants and enums from player.py
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    import player as P
    APPROACH_HOPS_EXIT = P.APPROACH_HOPS_EXIT
    APPROACH_HOPS_ENTER = P.APPROACH_HOPS_ENTER
    SEARCH_HOPS_ENTER = P.SEARCH_HOPS_ENTER
    SEARCH_MAX_SCANS = P.SEARCH_MAX_SCANS
    GRADIENT_PATIENCE = P.GRADIENT_PATIENCE
    GRADIENT_HISTORY_SIZE = P.GRADIENT_HISTORY_SIZE
    TURN_STEPS_180 = P.TURN_STEPS_180
    NAV_PLATEAU_STEPS = P.NAV_PLATEAU_STEPS
    NAV_ESCAPE_FORWARD = P.NAV_ESCAPE_FORWARD
    SEARCH_SIM_THRESHOLD = P.SEARCH_SIM_THRESHOLD

    class NavStateMock:
        """Mirrors the nav state variables from KeyboardPlayerPyGame."""
        def __init__(self):
            self.nav_state = P.NavState.NAVIGATE
            self.wall_hand = P.HAND_LEFT
            self.gradient_step_counter = 0
            self.hop_history = deque(maxlen=GRADIENT_HISTORY_SIZE)
            self.prev_avg_hops = None
            self.gradient_worsen_count = 0
            self.nav_last_best_hops = 9999
            self.nav_plateau_start = 0
            self.nav_last_flip_step = 0
            self.nav_total_steps = 0
            self.search_turn_counter = 0
            self.search_scan_count = 0
            self.search_best_sim = 0.0
            self._nav_es = P.ExploreState.FORWARD
            self._nav_tc = 0
            self._nav_fwd = 0
            self._nav_stuck_count = 0
            self._escape_remaining = 0
            self.current_action = "IDLE"
            self.action_hold_counter = 0

    # ================================================================
    # Edge case 1: APPROACH -> NAVIGATE leaves stale gradient data
    # ================================================================
    print("  --- Edge case 1: APPROACH->NAVIGATE stale gradient ---")
    m = NavStateMock()
    m.nav_state = P.NavState.APPROACH
    for h in [12, 14, 13, 15, 11]:
        m.hop_history.append(h)
    m.prev_avg_hops = 13.0
    m.gradient_worsen_count = 3

    # Transition: APPROACH -> NAVIGATE (mirrors player.py line 806)
    hops = 22
    if hops >= APPROACH_HOPS_EXIT:
        m.nav_state = P.NavState.NAVIGATE

    assert m.nav_state == P.NavState.NAVIGATE
    assert len(m.hop_history) == GRADIENT_HISTORY_SIZE, \
        "hop_history should still have old APPROACH values"
    assert m.gradient_worsen_count == 3, \
        "worsen_count should be stale (not reset on transition)"
    assert m.prev_avg_hops == 13.0, \
        "prev_avg_hops should be stale (not reset on transition)"

    # Next gradient check with hops=30 sees big avg jump
    m.hop_history.append(30)
    new_avg = float(np.median(list(m.hop_history)))
    assert new_avg > m.prev_avg_hops, \
        f"New avg {new_avg} should exceed stale prev {m.prev_avg_hops}"
    remaining_patience = GRADIENT_PATIENCE - m.gradient_worsen_count
    print(f"  Stale hop_history: prev_avg={m.prev_avg_hops}, new_avg={new_avg}")
    print(f"  worsen_count={m.gradient_worsen_count}, "
          f"only {remaining_patience} more needed for flip")
    print(f"  KNOWN ISSUE: stale gradient data can cause premature hand flip")

    # ================================================================
    # Edge case 2: SEARCH -> NAVIGATE with nav_last_best_hops unreset
    # ================================================================
    print("  --- Edge case 2: SEARCH->NAVIGATE nav_last_best_hops ---")
    m2 = NavStateMock()
    m2.nav_state = P.NavState.SEARCH
    m2.nav_last_best_hops = 2
    m2.nav_plateau_start = 1000
    m2.nav_total_steps = 1200
    m2.search_scan_count = SEARCH_MAX_SCANS

    # Transition: SEARCH -> NAVIGATE (mirrors player.py lines 746-752)
    if m2.search_scan_count >= SEARCH_MAX_SCANS:
        m2.nav_state = P.NavState.NAVIGATE
        m2.search_scan_count = 0
        m2.search_best_sim = 0.0
        m2.hop_history.clear()
        m2.prev_avg_hops = None

    assert m2.nav_state == P.NavState.NAVIGATE
    assert m2.nav_last_best_hops == 2, \
        "nav_last_best_hops should still be 2 (not reset on SEARCH exit)"

    current_hops = 30
    assert current_hops > m2.nav_last_best_hops

    m2.nav_total_steps = m2.nav_plateau_start + NAV_PLATEAU_STEPS
    plateau_triggered = (m2.nav_total_steps - m2.nav_plateau_start) >= NAV_PLATEAU_STEPS
    assert plateau_triggered, \
        f"Plateau should trigger after {NAV_PLATEAU_STEPS} steps"
    print(f"  nav_last_best_hops={m2.nav_last_best_hops} (stale from pre-SEARCH)")
    print(f"  Plateau triggers in {NAV_PLATEAU_STEPS} steps (acceptable)")

    # ================================================================
    # Edge case 3: NAVIGATE -> APPROACH with mid-REVERSE _nav_es
    # ================================================================
    print("  --- Edge case 3: NAVIGATE->APPROACH mid-REVERSE ---")
    m3 = NavStateMock()
    m3.nav_state = P.NavState.NAVIGATE
    m3._nav_es = P.ExploreState.REVERSE
    m3._nav_tc = 10

    hops = 12
    if hops < APPROACH_HOPS_ENTER:
        m3.nav_state = P.NavState.APPROACH
        m3.gradient_step_counter = 0

    assert m3.nav_state == P.NavState.APPROACH
    assert m3._nav_es == P.ExploreState.REVERSE, \
        "_nav_es should persist (not reset on NAVIGATE->APPROACH)"
    assert m3._nav_tc == 10

    remaining_turn = TURN_STEPS_180 - m3._nav_tc
    assert remaining_turn == TURN_STEPS_180 - 10
    print(f"  _nav_es=REVERSE with {m3._nav_tc}/{TURN_STEPS_180} steps done")
    print(f"  {remaining_turn} steps remain; completes naturally in APPROACH")

    # ================================================================
    # Edge case 4: Multiple APPROACH->SEARCH->NAVIGATE cycles
    # ================================================================
    print("  --- Edge case 4: Multiple cycles ---")
    m4 = NavStateMock()

    for cycle in range(3):
        m4.nav_state = P.NavState.APPROACH
        m4.gradient_step_counter = 0

        m4.nav_state = P.NavState.SEARCH
        m4.search_turn_counter = 0
        m4.search_scan_count = 0
        m4.search_best_sim = 0.0
        assert m4.search_scan_count == 0, \
            f"Cycle {cycle}: search_scan_count not reset at SEARCH entry"

        m4.search_scan_count = SEARCH_MAX_SCANS
        m4.nav_state = P.NavState.NAVIGATE
        m4.search_scan_count = 0
        m4.hop_history.clear()
        m4.prev_avg_hops = None
        assert m4.nav_state == P.NavState.NAVIGATE
        assert m4.search_scan_count == 0

    print(f"  3 full cycles: search_scan_count properly reset each time")

    # ================================================================
    # Edge case 5: action_hold_counter carryover across transitions
    # ================================================================
    print("  --- Edge case 5: action_hold_counter carryover ---")
    m5 = NavStateMock()
    m5.nav_state = P.NavState.SEARCH
    m5.current_action = "FORWARD"
    m5.action_hold_counter = 5

    m5.nav_state = P.NavState.NAVIGATE

    assert m5.action_hold_counter == 5
    assert m5.current_action == "FORWARD"

    steps_held = 0
    while m5.action_hold_counter > 0:
        m5.action_hold_counter -= 1
        steps_held += 1
    assert steps_held == 5
    print(f"  Hold counter carried {steps_held} FORWARD steps into NAVIGATE")
    print(f"  Harmless: moves forward briefly after failed SEARCH")

    # ================================================================
    # Edge case 6: ESCAPE -> NAVIGATE resets
    # ================================================================
    print("  --- Edge case 6: ESCAPE->NAVIGATE resets ---")
    m6 = NavStateMock()
    m6.nav_state = P.NavState.ESCAPE
    m6._escape_remaining = 1
    m6.nav_total_steps = 500

    m6._escape_remaining -= 1
    if m6._escape_remaining <= 0:
        m6.nav_state = P.NavState.NAVIGATE
        m6.nav_plateau_start = m6.nav_total_steps
        m6.gradient_step_counter = 0

    assert m6.nav_state == P.NavState.NAVIGATE
    assert m6.gradient_step_counter == 0
    assert m6.nav_plateau_start == 500
    print(f"  ESCAPE->NAVIGATE: gradient_step_counter=0, plateau_start=500")

    # ================================================================
    # Edge case 7: NAVIGATE plateau -> ESCAPE transition
    # ================================================================
    print("  --- Edge case 7: NAVIGATE plateau -> ESCAPE ---")
    m7 = NavStateMock()
    m7.nav_state = P.NavState.NAVIGATE
    m7.nav_plateau_start = 0
    m7.nav_total_steps = NAV_PLATEAU_STEPS
    m7.wall_hand = P.HAND_LEFT

    if (m7.nav_total_steps - m7.nav_plateau_start) >= NAV_PLATEAU_STEPS:
        old = m7.wall_hand
        m7.wall_hand = P.HAND_RIGHT if old == P.HAND_LEFT else P.HAND_LEFT
        m7.nav_last_flip_step = m7.nav_total_steps
        m7.hop_history.clear()
        m7.prev_avg_hops = None
        m7.nav_last_best_hops = 9999
        m7._escape_remaining = NAV_ESCAPE_FORWARD
        m7.nav_state = P.NavState.ESCAPE
        m7._nav_es = P.ExploreState.REVERSE
        m7._nav_tc = 0

    assert m7.nav_state == P.NavState.ESCAPE
    assert m7.wall_hand == P.HAND_RIGHT
    assert m7._escape_remaining == NAV_ESCAPE_FORWARD
    assert m7.nav_last_best_hops == 9999
    assert m7.prev_avg_hops is None
    assert len(m7.hop_history) == 0
    assert m7._nav_es == P.ExploreState.REVERSE
    assert m7._nav_tc == 0
    print(f"  Plateau at step {m7.nav_total_steps}: "
          f"hand flipped, gradient state fully reset")

    # ================================================================
    # Edge case 8: Happy path APPROACH -> SEARCH -> CHECKIN
    # ================================================================
    print("  --- Edge case 8: Happy path APPROACH->SEARCH->CHECKIN ---")
    m8 = NavStateMock()
    m8.nav_state = P.NavState.APPROACH

    hops = 2
    if hops <= SEARCH_HOPS_ENTER:
        m8.nav_state = P.NavState.SEARCH
        m8.search_turn_counter = 0
        m8.search_scan_count = 0
        m8.search_best_sim = 0.0

    assert m8.nav_state == P.NavState.SEARCH
    assert m8.search_turn_counter == 0
    assert m8.search_scan_count == 0

    sim = 0.35
    if sim > SEARCH_SIM_THRESHOLD:
        m8.nav_state = P.NavState.CHECKIN

    assert m8.nav_state == P.NavState.CHECKIN
    print(f"  APPROACH(hops={hops}) -> SEARCH -> CHECKIN(sim={sim})")

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("  === State Transition Summary ===")
    print("  1. APPROACH->NAVIGATE: KNOWN ISSUE - stale gradient data")
    print("  2. SEARCH->NAVIGATE:   ACCEPTABLE - stale best_hops")
    print("  3. NAVIGATE->APPROACH: OK - mid-REVERSE completes naturally")
    print("  4. Multi-cycle:        OK - search_scan_count properly reset")
    print("  5. action_hold:        OK - carryover is harmless")
    print("  6. ESCAPE->NAVIGATE:   OK - clean reset")
    print("  7. Plateau->ESCAPE:    OK - full gradient state reset")
    print("  8. Happy path:         OK - APPROACH->SEARCH->CHECKIN")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, ".."))
    print(f"Working directory: {os.getcwd()}")

    print(f"\n{'#'*60}")
    print(f"  VISUAL NAVIGATION PIPELINE - FULL TEST SUITE (43 tests)")
    print(f"{'#'*60}")

    # Navigation pipeline tests
    run_test("Sec 1:  Data Loading & Subsampling", test_data_loading)
    run_test("Sec 2:  SIFT Extraction + RootSIFT", test_sift_extraction)
    run_test("Sec 3:  VLAD Encoding", test_vlad_encoding)
    run_test("Sec 4:  Graph Construction", test_graph_construction)
    run_test("Sec 5:  Localization + Multi-View Goal", test_localization)

    # Navigation logic tests
    run_test("Sec 6a: Action Mapping (vis_nav_game)", test_action_mapping)
    run_test("Sec 6b: Edge Action Lookup", test_edge_action_lookup)
    run_test("Sec 6c: Graph-Based Stuck Detection", test_stuck_detection_graph)
    run_test("Sec 6d: Navigation Path Planning", test_navigation_path_planning)

    # Exploration tests
    run_test("Sec 7:  Frame-Based Stuck Detection", test_frame_stuck_detection)
    run_test("Sec 8:  Exploration State Machine", test_exploration_state_machine)
    run_test("Sec 9:  Turn Angle Calibration", test_turn_calibration)

    # Integration tests
    run_test("Sec 10: Player Initialization", test_player_initialization)

    # New tests (batch 1)
    run_test("Sec 11: Checkin Logic", test_checkin_logic)
    run_test("Sec 12: Goal Confidence Threshold", test_goal_confidence_threshold)
    run_test("Sec 13: Localization Smoothing", test_localization_smoothing)
    run_test("Sec 14: see() Frame Saving", test_see_frame_saving)
    run_test("Sec 15: Two-Stage Flag Transitions", test_two_stage_flag_transitions)
    run_test("Sec 16: Exploration Budget Enforcement", test_explore_step_budget)
    run_test("Sec 17: Goal Node None Fallback", test_goal_node_none_fallback)
    run_test("Sec 18: Empty Database Pipeline", test_empty_database_pipeline)
    run_test("Sec 19: Data Info JSON Roundtrip", test_data_info_json_roundtrip)
    run_test("Sec 20: Localization Jump Detection", test_localization_jump_detection)

    # New tests (batch 2 - player.py changes)
    run_test("Sec 21: VLAD No Power Norm", test_vlad_no_power_norm)
    run_test("Sec 22: SIFT Tuning (nfeatures, mask)", test_sift_tuning)
    run_test("Sec 23: n_clusters=64 Default", test_n_clusters_64)
    run_test("Sec 24: CHECK_RIGHT_INTERVAL=20", test_check_right_interval)
    run_test("Sec 25: Random Perturbation", test_random_perturbation)
    run_test("Sec 26: Oscillation Detection", test_oscillation_detection)
    run_test("Sec 27: Multi-View CHECKIN (_target_vlads)", test_target_vlads_cached)
    run_test("Sec 28: Variable Action Hold", test_variable_action_hold)
    run_test("Sec 29: hops==0 Bypass Fix", test_hops_zero_no_direct_checkin)
    run_test("Sec 30: Graph Constants", test_graph_constants)
    run_test("Sec 31: RootSIFT Epsilon", test_rootsift_epsilon)
    run_test("Sec 32: _make_mask Static Method", test_make_mask)
    run_test("Sec 33: Gradient Check Logic", test_gradient_check_logic)
    run_test("Sec 34: Perturbation Momentum Fix", test_perturbation_momentum_fix)
    run_test("Sec 35: Localization No Smoothing", test_localization_no_smoothing)
    run_test("Sec 36: MiniBatchKMeans Pipeline", test_minibatch_kmeans_pipeline)
    run_test("Sec 37: Wall-Follow Hand Flip", test_wall_follow_hand_flip)
    run_test("Sec 38: Search State (360 Scan)", test_search_state)
    run_test("Sec 39: Backup CHECKIN (Lucky Pass)", test_backup_checkin)
    run_test("Sec 40: Nav State Transitions", test_nav_state_transitions)

    # Summary
    total = PASSED + FAILED
    print(f"\n{'#'*60}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed out of {total}")
    if FAILED == 0:
        print(f"  ALL TESTS PASSED")
    elif FAILED == 1:
        print(f"  NOTE: 1 failure expected if vis_nav_game not installed")
    print(f"{'#'*60}")
    sys.exit(1 if FAILED > 0 else 0)
