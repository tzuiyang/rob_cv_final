from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import json
import pickle
import networkx as nx
from sklearn.cluster import KMeans
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants 
# ---------------------------------------------------------------------------
CACHE_DIR = "cache"
DATA_DIR = "exploration_data_midterm"  # Pre-collected exploration data

# Graph construction
TEMPORAL_WEIGHT = 1.0
VISUAL_WEIGHT_BASE = 2.0
VISUAL_WEIGHT_SCALE = 3.0
MIN_SHORTCUT_GAP = 50
TOP_K_SHORTCUTS = 200  # more shortcuts = shorter paths, but slower to build
MAP_SIZE = 700         # pixel size of the map visualization
TURN_ANGLE_DEG = 7.5   # degrees per LEFT/RIGHT action

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# VLAD Feature Extraction
# ---------------------------------------------------------------------------
class VLADExtractor:

    def __init__(self, n_clusters: int = 128):
        self.n_clusters = n_clusters  # higher = more accurate but slower KMeans
        self.sift = cv2.SIFT_create()
        self.codebook = None
        self._sift_cache: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self.n_clusters * 128

    @staticmethod
    def _root_sift(des: np.ndarray) -> np.ndarray:
        des = des / (np.sum(des, axis=1, keepdims=True) + 1e-10)
        return np.sqrt(des)

    def _des_to_vlad(self, des: np.ndarray) -> np.ndarray:
        labels = self.codebook.predict(des)
        centers = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        vlad = np.zeros((k, des.shape[1]))
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                vlad[i] = np.sum(des[mask] - centers[i], axis=0)
                norm = np.linalg.norm(vlad[i])
                if norm > 0:
                    vlad[i] /= norm
        vlad = vlad.ravel()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm
        return vlad

    def load_sift_cache(self, file_list: list[str], subsample_rate: int):
        # Cache is keyed by subsample rate — delete cache if you change subsample
        cache_file = os.path.join(CACHE_DIR, f"sift_ss{subsample_rate}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached SIFT from {cache_file}")
            with open(cache_file, "rb") as f:
                self._sift_cache = pickle.load(f)
            if all(fname in self._sift_cache for fname in file_list):
                return
            print("  Cache incomplete, re-extracting...")

        print(f"Extracting SIFT for {len(file_list)} images...")
        self._sift_cache = {}
        for fname in tqdm(file_list, desc="SIFT"):
            img = cv2.imread(fname)
            if img is None:
                continue
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                self._sift_cache[fname] = self._root_sift(des)
        with open(cache_file, "wb") as f:
            pickle.dump(self._sift_cache, f)
        print(f"  Saved {len(self._sift_cache)} descriptors -> {cache_file}")

    def build_vocabulary(self, file_list: list[str]):
        # Cache is keyed by n_clusters — delete cache if you change n_clusters
        cache_file = os.path.join(CACHE_DIR, f"codebook_k{self.n_clusters}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached codebook from {cache_file}")
            with open(cache_file, "rb") as f:
                self.codebook = pickle.load(f)
            return

        all_des = np.vstack([self._sift_cache[f] for f in file_list
                             if f in self._sift_cache])

        # Subsample descriptors to speed up KMeans — reduce max_desc if too slow
        max_desc = 500_000
        if len(all_des) > max_desc:
            idx = np.random.choice(len(all_des), max_desc, replace=False)
            all_des = all_des[idx]
            print(f"Subsampled to {max_desc} descriptors for KMeans")

        print(f"Fitting KMeans (k={self.n_clusters}) on {len(all_des)} descriptors...")
        self.codebook = KMeans(
            n_clusters=self.n_clusters, init='k-means++',
            n_init=1,        # increase for better codebook, slower build
            max_iter=300, tol=1e-4, verbose=1, random_state=42,
        ).fit(all_des)
        with open(cache_file, "wb") as f:
            pickle.dump(self.codebook, f)

    def extract(self, img: np.ndarray) -> np.ndarray:
        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.dim)
        return self._des_to_vlad(self._root_sift(des))

    def extract_batch(self, file_list: list[str]) -> np.ndarray:
        vectors = []
        for fname in tqdm(file_list, desc="VLAD"):
            if fname in self._sift_cache and len(self._sift_cache[fname]) > 0:
                vectors.append(self._des_to_vlad(self._sift_cache[fname]))
            else:
                vectors.append(np.zeros(self.dim))
        return np.array(vectors)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class KeyboardPlayerPyGame(Player):

    def __init__(self, n_clusters: int = 128, subsample_rate: int = 2,
                 top_k_shortcuts: int = TOP_K_SHORTCUTS):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super().__init__()

        self.subsample_rate = subsample_rate  # lower = more frames = more accurate, slower
        self.top_k_shortcuts = top_k_shortcuts

        self.motion_frames = []
        self.file_list = []
        self.traj_boundaries = []

        # --- Load exploration data ---
        # Supports both multi-trajectory (traj_0/, traj_1/, ...) and legacy (images/) formats
        traj_dirs = sorted([
            d for d in os.listdir(DATA_DIR)
            if d.startswith('traj_') and os.path.isdir(os.path.join(DATA_DIR, d))
        ])

        if traj_dirs:
            all_motion = []
            for traj_dir_name in traj_dirs:
                traj_path = os.path.join(DATA_DIR, traj_dir_name)
                info_path = os.path.join(traj_path, 'data_info.json')
                if not os.path.exists(info_path):
                    continue
                with open(info_path) as f:
                    raw = json.load(f)
                traj_id = traj_dir_name
                pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
                traj_motion = [
                    {'step': d['step'], 'image': d['image'], 'action': d['action'][0],
                     'traj_id': traj_id, 'image_path': os.path.join(traj_path, d['image'])}
                    for d in raw
                    if len(d['action']) == 1 and d['action'][0] in pure
                ]
                start_idx = len(all_motion)
                all_motion.extend(traj_motion)
                end_idx = len(all_motion)
                self.traj_boundaries.append((start_idx, end_idx))
                print(f"  {traj_dir_name}: {len(traj_motion)} motion frames")

            self.motion_frames = all_motion[::subsample_rate]
            self.traj_boundaries = []
            prev_traj = None
            for idx, m in enumerate(self.motion_frames):
                if m['traj_id'] != prev_traj:
                    if prev_traj is not None:
                        self.traj_boundaries[-1] = (self.traj_boundaries[-1][0], idx)
                    self.traj_boundaries.append((idx, len(self.motion_frames)))
                    prev_traj = m['traj_id']
            if self.traj_boundaries:
                self.traj_boundaries[-1] = (self.traj_boundaries[-1][0], len(self.motion_frames))

            self.file_list = [m['image_path'] for m in self.motion_frames]
            print(f"Frames: {len(all_motion)} total, "
                  f"{len(self.motion_frames)} after {subsample_rate}x subsample, "
                  f"{len(self.traj_boundaries)} trajectories")
        else:
            # Legacy single-directory format
            legacy_info = os.path.join(DATA_DIR, 'data_info.json')
            legacy_img_dir = os.path.join(DATA_DIR, 'images')
            if os.path.exists(legacy_info):
                with open(legacy_info) as f:
                    raw = json.load(f)
                pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
                all_motion = [
                    {'step': d['step'], 'image': d['image'], 'action': d['action'][0],
                     'traj_id': 'traj_0',
                     'image_path': os.path.join(legacy_img_dir, d['image'])}
                    for d in raw
                    if len(d['action']) == 1 and d['action'][0] in pure
                ]
                self.motion_frames = all_motion[::subsample_rate]
                self.file_list = [m['image_path'] for m in self.motion_frames]
                self.traj_boundaries = [(0, len(self.motion_frames))]
                print(f"Frames (legacy): {len(all_motion)} total, "
                      f"{len(self.motion_frames)} after {subsample_rate}x subsample")

        self.extractor = VLADExtractor(n_clusters=n_clusters)
        self.database = None
        self.G = None
        self.goal_node = None

        # --- Map visualization ---
        self.node_px = None       # (N, 2) pixel coords for each node
        self.base_map = None      # pre-rendered maze layout image
        self.visited_nodes = []   # node indices visited during navigation

    # --- Game engine hooks ---
    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT:   Action.LEFT,
            pygame.K_RIGHT:  Action.RIGHT,
            pygame.K_UP:     Action.FORWARD,
            pygame.K_DOWN:   Action.BACKWARD,
            pygame.K_SPACE:  Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption("KeyboardPlayer:fpv — Q for info | SPACE to check in")

        if self._state and self._state[1] == Phase.NAVIGATION:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                self.display_next_best_view()
            if keys[pygame.K_m]:
                self._show_map()

        rgb = fpv[:, :, ::-1]
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], 'RGB')
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def set_target_images(self, images):
        super().set_target_images(images)
        self.show_target_images()

    # --- Offline phase (runs before navigation timer starts) ---
    def pre_exploration(self):
        super().pre_exploration()
        print("Building database and graph offline...")
        self._build_database()
        self._build_graph()
        print("Offline build complete.")

    # --- Navigation phase (timer starts here) ---
    def pre_navigation(self):
        super().pre_navigation()
        # Engine may skip exploration (NAV_START_TIME in the past), so build here too
        self._build_database()
        self._build_graph()
        self._setup_goal()
        self._build_map()

    # --- Offline: build VLAD database ---
    def _build_database(self):
        if self.database is not None:
            print("Database already computed, skipping.")
            return
        self.extractor.load_sift_cache(self.file_list, self.subsample_rate)
        self.extractor.build_vocabulary(self.file_list)
        self.database = self.extractor.extract_batch(self.file_list)
        print(f"Database: {self.database.shape}")

    # --- Offline: build navigation graph ---
    def _build_graph(self):
        if self.G is not None:
            print("Graph already built, skipping.")
            return

        n = len(self.database)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        # Temporal edges: connect consecutive frames within each trajectory
        for start, end in self.traj_boundaries:
            for i in range(start, end - 1):
                self.G.add_edge(i, i + 1, weight=TEMPORAL_WEIGHT, edge_type="temporal")

        # Visual shortcut edges: connect visually similar frames far apart in the trajectory
        print("Computing similarity matrix...")
        sim = self.database @ self.database.T
        np.fill_diagonal(sim, -2)

        for i in range(n):
            lo = max(0, i - MIN_SHORTCUT_GAP)
            hi = min(n, i + MIN_SHORTCUT_GAP + 1)
            sim[i, lo:hi] = -2
        sim[~np.triu(np.ones((n, n), dtype=bool), k=1)] = -2

        flat = sim.ravel()
        top_k = self.top_k_shortcuts
        top_idx = np.argpartition(flat, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        dists = []
        print(f"Top-{top_k} shortcuts (min_gap={MIN_SHORTCUT_GAP}):")
        for rank, fi in enumerate(top_idx):
            i, j = divmod(int(fi), n)
            s = float(flat[fi])
            d = float(np.sqrt(max(0, 2 - 2 * s)))
            self.G.add_edge(i, j,
                            weight=VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * d,
                            edge_type="visual")
            dists.append(d)
            if rank < 5:
                print(f"  #{rank+1}: {i}<->{j} gap={abs(j-i)} d={d:.4f}")

        kd = np.array(dists)
        print(f"  {top_k} visual edges, dist: [{kd.min():.3f}, {kd.max():.3f}]")
        print(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    def _setup_goal(self):
        if self.goal_node is not None:
            print("Goal already set, skipping.")
            return
        targets = self.get_target_images()
        if not targets:
            return
        # Use all 4 target views (front/right/back/left) for more robust goal matching
        sims = np.zeros(len(self.database))
        for img in targets:
            sims += self.database @ self.extractor.extract(img)
        self.goal_node = int(np.argmax(sims))
        d = float(np.sqrt(max(0, 2 - 2 * sims[self.goal_node] / len(targets))))
        print(f"Goal: node {self.goal_node} (d={d:.4f})")

    # --- Online: localization helpers ---
    def _get_current_node(self) -> int:
        feat = self.extractor.extract(self.fpv)
        return int(np.argmax(self.database @ feat))

    def _get_goal_dist(self) -> float:
        # VLAD distance from current frame to goal node — lower means closer
        feat = self.extractor.extract(self.fpv)
        sim = float(self.database[self.goal_node] @ feat)
        return float(np.sqrt(max(0, 2 - 2 * sim)))

    def _get_path(self, start: int) -> list[int]:
        try:
            return nx.shortest_path(self.G, start, self.goal_node, weight="weight")
        except nx.NetworkXNoPath:
            return [start]

    # --- Map visualization ---
    def _build_map(self):
        if self.base_map is not None:
            return
        # Dead-reckon 2D positions from action sequences
        # Apply to ALL motion frames before subsampling, then pick every Nth position
        # This is needed because LEFT/RIGHT frames change heading without moving
        turn = np.radians(TURN_ANGLE_DEG)
        x, y, heading = 0.0, 0.0, 0.0
        all_positions = []
        # Reconstruct from the full trajectory (before subsample)
        # We need the original all_motion list — re-derive it from data
        traj_dirs = sorted([
            d for d in os.listdir(DATA_DIR)
            if d.startswith('traj_') and os.path.isdir(os.path.join(DATA_DIR, d))
        ])
        for traj_dir_name in traj_dirs:
            traj_path = os.path.join(DATA_DIR, traj_dir_name)
            info_path = os.path.join(traj_path, 'data_info.json')
            if not os.path.exists(info_path):
                continue
            with open(info_path) as f:
                raw = json.load(f)
            pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
            for d in raw:
                if len(d['action']) == 1 and d['action'][0] in pure:
                    all_positions.append((x, y))
                    act = d['action'][0]
                    if act == 'FORWARD':
                        x += np.cos(heading)
                        y += np.sin(heading)
                    elif act == 'BACKWARD':
                        x -= np.cos(heading)
                        y -= np.sin(heading)
                    elif act == 'LEFT':
                        heading += turn
                    elif act == 'RIGHT':
                        heading -= turn

        # Subsample to match self.motion_frames
        pos = np.array(all_positions[::self.subsample_rate])
        n = len(self.motion_frames)
        pos = pos[:n]

        # Normalize to pixel coordinates
        margin = 30
        pmin = pos.min(axis=0)
        pmax = pos.max(axis=0)
        span = pmax - pmin
        span[span == 0] = 1
        self.node_px = ((pos - pmin) / span * (MAP_SIZE - 2 * margin) + margin).astype(int)

        # Pre-render base map with trajectory lines
        self.base_map = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8) + 20
        for i in range(n - 1):
            cv2.line(self.base_map, tuple(self.node_px[i]), tuple(self.node_px[i + 1]),
                     (60, 60, 60), 1)
        # Draw goal
        if self.goal_node is not None and self.goal_node < n:
            cv2.circle(self.base_map, tuple(self.node_px[self.goal_node]), 10, (0, 200, 0), -1)
        print(f"Map built: {n} nodes, goal at node {self.goal_node}")

    def _show_map(self):
        if self.base_map is None or self.database is None:
            return
        cur = self._get_current_node()
        self.visited_nodes.append(cur)

        img = self.base_map.copy()

        # Draw visited trail (blue)
        if len(self.visited_nodes) > 1:
            pts = np.array([self.node_px[n] for n in self.visited_nodes], dtype=np.int32)
            cv2.polylines(img, [pts], False, (200, 100, 0), 2)

        # Draw planned path (yellow)
        path = self._get_path(cur)
        if len(path) > 1:
            pts = np.array([self.node_px[n] for n in path], dtype=np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)

        # Draw current position (red)
        cv2.circle(img, tuple(self.node_px[cur]), 8, (0, 0, 255), -1)
        cv2.circle(img, tuple(self.node_px[cur]), 9, (255, 255, 255), 1)

        # Legend
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "M=map Q=info SPACE=checkin", (10, MAP_SIZE - 10),
                    font, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        hops = len(path) - 1
        cv2.putText(img, f"Node:{cur} Goal:{self.goal_node} Hops:{hops}",
                    (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Maze Map", img)
        cv2.waitKey(1)

    # --- Display ---
    def show_target_images(self):
        targets = self.get_target_images()
        if not targets:
            return
        top = cv2.hconcat(targets[:2])
        bot = cv2.hconcat(targets[2:])
        img = cv2.vconcat([top, bot])
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
        cv2.line(img, (0, h // 2), (w, h // 2), (0, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, pos in [('Front', (10, 25)), ('Right', (w//2+10, 25)),
                            ('Back', (10, h//2+25)), ('Left', (w//2+10, h//2+25))]:
            cv2.putText(img, label, pos, font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Target Images', img)
        cv2.waitKey(1)

    def display_next_best_view(self):
        # Shows current node, goal node, node difference, hops, and VLAD distance to goal
        # goal_d is the most useful metric — lower means you are closer to the target
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA

        cur = self._get_current_node()
        goal_d = self._get_goal_dist()
        path = self._get_path(cur)
        hops = len(path) - 1
        node_diff = abs(cur - self.goal_node)
        near = hops <= 5

        W, H = 700, 60
        bar = np.zeros((H, W, 3), dtype=np.uint8)
        bar[:] = (0, 120, 0) if near else (40, 40, 40)

        line1 = f"Current: {cur}  |  Goal: {self.goal_node}  |  Diff: {node_diff}"
        line2 = f"Hops: {hops}  |  Goal dist: {goal_d:.4f}  {'*** NEAR — PRESS SPACE ***' if near else '(lower = closer)'}"

        cv2.putText(bar, line1, (10, 22), FONT, 0.55, (255, 255, 255), 1, AA)
        cv2.putText(bar, line2, (10, 48), FONT, 0.55,
                    (0, 255, 255) if near else (200, 200, 200), 1, AA)

        cv2.imshow("Navigation Info", bar)
        cv2.waitKey(1)
        print(f"Node {cur} | Goal {self.goal_node} | Diff {node_diff} | "
              f"Hops {hops} | goal_d {goal_d:.4f}")


if __name__ == "__main__":
    import argparse
    import vis_nav_game
    import pybullet as _pb

    # Monkey-patch pybullet.getCameraImage: newer pybullet returns rgb/depth/seg
    # as tuples, but the vis_nav_game engine indexes them as numpy arrays.
    _orig_getCameraImage = _pb.getCameraImage
    def _patched_getCameraImage(*args, **kwargs):
        result = _orig_getCameraImage(*args, **kwargs)
        w, h = result[0], result[1]
        rgb = np.array(result[2], dtype=np.uint8).reshape(h, w, 4)
        depth = np.array(result[3], dtype=np.float32).reshape(h, w)
        seg = np.array(result[4], dtype=np.int32).reshape(h, w)
        return w, h, rgb, depth, seg
    _pb.getCameraImage = _patched_getCameraImage

    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=2,
                        help="Take every Nth motion frame (default: 2)")
    parser.add_argument("--n-clusters", type=int, default=128,
                        help="VLAD codebook size (default: 128)")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Number of global visual shortcut edges (default: 200)")
    args = parser.parse_args()

    vis_nav_game.play(the_player=KeyboardPlayerPyGame(
        n_clusters=args.n_clusters,
        subsample_rate=args.subsample,
        top_k_shortcuts=args.top_k,
    ))