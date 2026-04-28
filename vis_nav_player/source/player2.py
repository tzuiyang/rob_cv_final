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
DATA_DIR = "exploration_data_midterm"

# Graph construction
TEMPORAL_WEIGHT = 1.0
VISUAL_WEIGHT_BASE = 2.0
VISUAL_WEIGHT_SCALE = 3.0
MIN_SHORTCUT_GAP = 50
TOP_K_SHORTCUTS = 200

# HUD + Localization
LOCALIZE_INTERVAL = 10      # re-localize every N frames (~2x/sec at 20 FPS)
TREND_WINDOW = 5            # rolling window for hop trend

# Map
MAP_SIZE = 700
TURN_ANGLE_DEG = 7.5

# Speedrun
REPLAY_FILE = "speedrun.json"

os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# VLAD Feature Extraction (from player1.py — unchanged)
# ---------------------------------------------------------------------------
class VLADExtractor:

    def __init__(self, n_clusters: int = 128):
        self.n_clusters = n_clusters
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
        cache_file = os.path.join(CACHE_DIR, f"codebook_k{self.n_clusters}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached codebook from {cache_file}")
            with open(cache_file, "rb") as f:
                self.codebook = pickle.load(f)
            return

        all_des = np.vstack([self._sift_cache[f] for f in file_list
                             if f in self._sift_cache])
        max_desc = 500_000
        if len(all_des) > max_desc:
            idx = np.random.choice(len(all_des), max_desc, replace=False)
            all_des = all_des[idx]
            print(f"Subsampled to {max_desc} descriptors for KMeans")

        print(f"Fitting KMeans (k={self.n_clusters}) on {len(all_des)} descriptors...")
        self.codebook = KMeans(
            n_clusters=self.n_clusters, init='k-means++',
            n_init=1, max_iter=300, tol=1e-4, verbose=1, random_state=42,
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

        self.subsample_rate = subsample_rate
        self.top_k_shortcuts = top_k_shortcuts

        self.motion_frames = []
        self.file_list = []
        self.traj_boundaries = []

        # --- Load exploration data ---
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

        # --- Localization state (always-on HUD) ---
        self._frame_counter = 0
        self._cached_node = None
        self._cached_hops = None
        self._cached_dist = None
        self._cached_path = []
        self._hops_history = []         # last TREND_WINDOW hop values
        self._prev_sims = None          # temporal smoothing
        self._prev_node = None          # jump detection
        self._last_sims = None          # cached similarity vector

        # --- Bird's eye map ---
        self._node_px = None            # (N, 2) pixel coords
        self._map_base = None           # pre-rendered base map
        self._visited_nodes = []        # trail of visited nodes

        # --- Path recording + speedrun ---
        self._recording = []
        self._replay_actions = None
        self._replay_index = 0
        self._speedrun_mode = False

    # ------------------------------------------------------------------
    # Game engine hooks
    # ------------------------------------------------------------------
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
        # --- Speedrun replay mode ---
        if self._speedrun_mode and self._replay_actions is not None:
            if self._replay_index < len(self._replay_actions):
                action = Action(self._replay_actions[self._replay_index])
                self._replay_index += 1
                if self._replay_index % 100 == 0:
                    print(f"[REPLAY] {self._replay_index}/{len(self._replay_actions)}")
                return action
            else:
                self._speedrun_mode = False
                print("[REPLAY] Complete!")
                return Action.IDLE

        # --- Normal keyboard mode ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                elif event.key == pygame.K_t:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        action = self.last_act

        # --- Record action during navigation ---
        if (self._state and self._state[1] == Phase.NAVIGATION
                and not self._speedrun_mode):
            self._recording.append(int(action))
            if action & Action.CHECKIN:
                self._save_recording()

        return action

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv
        display_fpv = fpv.copy()

        if self._state and self._state[1] == Phase.NAVIGATION:
            self._frame_counter += 1

            # --- Periodic re-localization ---
            if (self._frame_counter % LOCALIZE_INTERVAL == 0
                    and self.database is not None
                    and self.goal_node is not None):
                cur = self._get_current_node()
                path = self._get_path(cur)
                hops = len(path) - 1
                sim = float(self._last_sims[self.goal_node])
                dist = float(np.sqrt(max(0, 2 - 2 * sim)))

                self._cached_node = cur
                self._cached_hops = hops
                self._cached_dist = dist
                self._cached_path = path
                self._hops_history.append(hops)
                if len(self._hops_history) > TREND_WINDOW:
                    self._hops_history.pop(0)
                self._visited_nodes.append(cur)

                # Update bird's eye map
                self._update_map()

            # --- Always draw HUD ---
            if self._cached_hops is not None:
                display_fpv = self._draw_hud(display_fpv)

        # --- Render to pygame ---
        if self.screen is None:
            h, w, _ = display_fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        caption = "Player2 | Arrows=move T=targets SPACE=checkin"
        if self._speedrun_mode:
            caption = f"Player2 [REPLAY {self._replay_index}/{len(self._replay_actions)}]"
        pygame.display.set_caption(caption)

        rgb = display_fpv[:, :, ::-1]
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], 'RGB')
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def set_target_images(self, images):
        super().set_target_images(images)
        self.show_target_images()

    # ------------------------------------------------------------------
    # Pipeline build (offline)
    # ------------------------------------------------------------------
    def pre_exploration(self):
        super().pre_exploration()
        print("Building database and graph offline...")
        self._build_database()
        self._build_graph()
        print("Offline build complete.")

    def pre_navigation(self):
        super().pre_navigation()
        self._build_database()
        self._build_graph()
        self._setup_goal()
        self._build_map()

    def _build_database(self):
        if self.database is not None:
            print("Database already computed, skipping.")
            return
        self.extractor.load_sift_cache(self.file_list, self.subsample_rate)
        self.extractor.build_vocabulary(self.file_list)
        self.database = self.extractor.extract_batch(self.file_list)
        print(f"Database: {self.database.shape}")

    def _build_graph(self):
        if self.G is not None:
            print("Graph already built, skipping.")
            return

        n = len(self.database)
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        for start, end in self.traj_boundaries:
            for i in range(start, end - 1):
                self.G.add_edge(i, i + 1, weight=TEMPORAL_WEIGHT, edge_type="temporal")

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
        sims = np.zeros(len(self.database))
        for img in targets:
            sims += self.database @ self.extractor.extract(img)
        self.goal_node = int(np.argmax(sims))
        d = float(np.sqrt(max(0, 2 - 2 * sims[self.goal_node] / len(targets))))
        print(f"Goal: node {self.goal_node} (d={d:.4f})")

    # ------------------------------------------------------------------
    # Localization (with temporal smoothing + jump detection)
    # ------------------------------------------------------------------
    def _get_current_node(self) -> int:
        feat = self.extractor.extract(self.fpv)
        sims = self.database @ feat

        # Temporal smoothing: 95% current + 5% previous
        if self._prev_sims is not None:
            sims = 0.95 * sims + 0.05 * self._prev_sims
        self._prev_sims = sims.copy()

        cur = int(np.argmax(sims))

        # Jump detection: prefer local match if jump is too large
        if self._prev_node is not None and len(sims) > 0:
            prev = self._prev_node
            if abs(cur - prev) > 150:
                lo = max(0, prev - 80)
                hi = min(len(sims), prev + 81)
                local_cur = lo + int(np.argmax(sims[lo:hi]))
                if float(sims[local_cur]) >= float(sims[cur]) - 0.05:
                    cur = local_cur

        self._prev_node = cur
        self._last_sims = sims
        return cur

    def _get_path(self, start: int) -> list[int]:
        try:
            return nx.shortest_path(self.G, start, self.goal_node, weight="weight")
        except nx.NetworkXNoPath:
            return [start]

    # ------------------------------------------------------------------
    # Bird's eye map
    # ------------------------------------------------------------------
    def _build_map(self):
        if self._map_base is not None:
            return

        # Dead-reckon 2D positions from full action sequence
        turn = np.radians(TURN_ANGLE_DEG)
        x, y, heading = 0.0, 0.0, 0.0
        all_positions = []

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
        self._node_px = ((pos - pmin) / span * (MAP_SIZE - 2 * margin) + margin).astype(int)

        # Pre-render base map
        self._map_base = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8) + 20
        for i in range(n - 1):
            cv2.line(self._map_base, tuple(self._node_px[i]), tuple(self._node_px[i + 1]),
                     (60, 60, 60), 1)

        # Draw goal
        if self.goal_node is not None and self.goal_node < n:
            gx, gy = self._node_px[self.goal_node]
            cv2.circle(self._map_base, (gx, gy), 10, (0, 200, 0), -1)
            cv2.putText(self._map_base, "GOAL", (gx + 12, gy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)

        print(f"Map built: {n} nodes, goal at node {self.goal_node}")

    def _update_map(self):
        if self._map_base is None or self._node_px is None:
            return

        img = self._map_base.copy()
        n = len(self._node_px)

        # Draw visited trail (blue-orange)
        if len(self._visited_nodes) > 1:
            valid = [nd for nd in self._visited_nodes if nd < n]
            if len(valid) > 1:
                pts = np.array([self._node_px[nd] for nd in valid], dtype=np.int32)
                cv2.polylines(img, [pts], False, (200, 100, 0), 2)

        # Draw Dijkstra path (yellow)
        if len(self._cached_path) > 1:
            valid = [nd for nd in self._cached_path if nd < n]
            if len(valid) > 1:
                pts = np.array([self._node_px[nd] for nd in valid], dtype=np.int32)
                cv2.polylines(img, [pts], False, (0, 255, 255), 2)

        # Draw current position (red circle with white ring)
        if self._cached_node is not None and self._cached_node < n:
            cx, cy = self._node_px[self._cached_node]
            cv2.circle(img, (cx, cy), 8, (0, 0, 255), -1)
            cv2.circle(img, (cx, cy), 9, (255, 255, 255), 1)

        # Info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        hops = self._cached_hops if self._cached_hops is not None else "?"
        dist = f"{self._cached_dist:.3f}" if self._cached_dist is not None else "?"
        cv2.putText(img, f"Node:{self._cached_node} Goal:{self.goal_node} Hops:{hops} Dist:{dist}",
                    (10, 20), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Legend
        cv2.putText(img, "Blue=trail Yellow=path Red=you Green=goal",
                    (10, MAP_SIZE - 10), font, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.imshow("Bird's Eye Map", img)
        cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Always-on HUD overlay
    # ------------------------------------------------------------------
    def _draw_hud(self, display_fpv: np.ndarray) -> np.ndarray:
        h, w = display_fpv.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        aa = cv2.LINE_AA

        hops = self._cached_hops
        dist = self._cached_dist
        node = self._cached_node
        n_nodes = len(self.motion_frames)

        # --- Compute trend ---
        if len(self._hops_history) >= 2:
            avg_prev = np.mean(self._hops_history[:-1])
            current = self._hops_history[-1]
            diff = current - avg_prev
            if diff <= -2:
                trend_text = ">> CLOSER"
                trend_color = (0, 255, 0)       # green
            elif diff >= 2:
                trend_text = "<< FURTHER"
                trend_color = (0, 0, 255)        # red
            else:
                trend_text = "-- STABLE"
                trend_color = (0, 255, 255)      # yellow
        else:
            trend_text = "..."
            trend_color = (180, 180, 180)

        # --- Semi-transparent bar at bottom ---
        bar_h = 50
        overlay = display_fpv.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_fpv, 0.3, 0, display_fpv)

        # Line 1: Hops + Distance + Trend
        line1 = f"Hops: {hops}  |  Dist: {dist:.3f}  |  {trend_text}"
        cv2.putText(display_fpv, line1, (10, h - 28),
                    font, 0.5, trend_color, 1, aa)

        # Line 2: Node info
        line2 = f"Node: {node} / {n_nodes}  |  Goal: {self.goal_node}"
        cv2.putText(display_fpv, line2, (10, h - 8),
                    font, 0.45, (200, 200, 200), 1, aa)

        # --- Near-goal alert at top ---
        if hops is not None and hops <= 5:
            alert = "NEAR TARGET -- PRESS SPACE"
            text_size = cv2.getTextSize(alert, font, 0.8, 2)[0]
            tx = (w - text_size[0]) // 2
            cv2.rectangle(display_fpv, (tx - 10, 5), (tx + text_size[0] + 10, 35), (0, 0, 0), -1)
            cv2.putText(display_fpv, alert, (tx, 28),
                        font, 0.8, (0, 0, 255), 2, aa)

        # --- Recording indicator ---
        if not self._speedrun_mode and len(self._recording) > 0:
            cv2.circle(display_fpv, (w - 20, 20), 6, (0, 0, 255), -1)
            cv2.putText(display_fpv, f"REC {len(self._recording)}",
                        (w - 100, 25), font, 0.4, (0, 0, 255), 1, aa)

        return display_fpv

    # ------------------------------------------------------------------
    # Path recording + speedrun replay
    # ------------------------------------------------------------------
    def _save_recording(self):
        data = {
            "actions": self._recording,
            "goal_node": self.goal_node,
            "total_frames": len(self._recording),
        }
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), REPLAY_FILE)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[REC] Saved {len(self._recording)} actions to {path}")

    def _load_replay(self, replay_path: str):
        with open(replay_path) as f:
            data = json.load(f)
        self._replay_actions = data["actions"]
        self._replay_index = 0
        self._speedrun_mode = True
        print(f"[REPLAY] Loaded {len(self._replay_actions)} actions from {replay_path}")

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------
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


if __name__ == "__main__":
    import argparse
    import vis_nav_game
    import pybullet as _pb

    # Monkey-patch pybullet.getCameraImage for newer pybullet compatibility
    _orig_getCameraImage = _pb.getCameraImage
    def _patched_getCameraImage(*args, **kwargs):
        result = _orig_getCameraImage(*args, **kwargs)
        w, h = result[0], result[1]
        rgb = np.array(result[2], dtype=np.uint8).reshape(h, w, 4)
        depth = np.array(result[3], dtype=np.float32).reshape(h, w)
        seg = np.array(result[4], dtype=np.int32).reshape(h, w)
        return w, h, rgb, depth, seg
    _pb.getCameraImage = _patched_getCameraImage

    parser = argparse.ArgumentParser(description="Player2: Manual nav with HUD + map + speedrun")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Take every Nth motion frame (default: 2)")
    parser.add_argument("--n-clusters", type=int, default=128,
                        help="VLAD codebook size (default: 128)")
    parser.add_argument("--top-k", type=int, default=200,
                        help="Number of visual shortcut edges (default: 200)")
    parser.add_argument("--replay", type=str, default=None,
                        help="Path to speedrun JSON for auto-replay")
    args = parser.parse_args()

    player = KeyboardPlayerPyGame(
        n_clusters=args.n_clusters,
        subsample_rate=args.subsample,
        top_k_shortcuts=args.top_k,
    )
    if args.replay:
        player._load_replay(args.replay)

    vis_nav_game.play(the_player=player)
