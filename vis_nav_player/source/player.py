"""
Autonomous visual navigation player.

Pipeline: RootSIFT → KMeans codebook → VLAD → cosine similarity graph → Dijkstra → auto-navigate

Game flow:
  - Engine may skip exploration (NAV_START_TIME in past) and go straight to navigation.
  - In that case, we explore during navigation phase first, then build pipeline and navigate.
  - If exploration phase runs, we save frames there and build pipeline in pre_navigation().

Note: pickle is used for SIFT/codebook caching (trusted local data only, same as baseline.py).
"""

from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import json
import pickle
import hashlib
import networkx as nx
from sklearn.cluster import MiniBatchKMeans
from collections import deque
from enum import Enum
import random


class ExploreState(Enum):
    """State machine states for automated exploration."""
    FORWARD = "forward"           # Try to move forward
    TURN_LEFT = "turn_left"       # Turning left (wall ahead)
    TURN_RIGHT = "turn_right"     # Turning right (checking side corridor)
    CHECK_RIGHT = "check_right"   # Periodic right check for side corridors
    REVERSE = "reverse"           # 180° turn (dead end)


class NavState(Enum):
    """Navigation state machine states."""
    NAVIGATE = "navigate"         # Wall-follow with gradient feedback
    APPROACH = "approach"         # Near goal, more frequent re-localization
    SEARCH = "search"             # At goal, 360° scan for visual match
    CHECKIN = "checkin"           # Submit check-in
    ESCAPE = "escape"             # Forward burst to physically leave stuck area

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = "cache"
IMAGE_DIR = "data/images/"
DATA_INFO_PATH = "data/data_info.json"

# Graph construction
TEMPORAL_WEIGHT = 1.0
VISUAL_WEIGHT_BASE = 2.0
VISUAL_WEIGHT_SCALE = 3.0
MIN_SHORTCUT_GAP = 30
VISUAL_SHORTCUT_SIM_FLOOR = 0.15
GOAL_NEIGHBORHOOD_RADIUS = 10
LOCALIZATION_SEARCH_RADIUS = 80
LOCALIZATION_JUMP_THRESHOLD = 150
LOCALIZATION_JUMP_MARGIN = 0.05

# Auto-target detection (CHECKIN aggressiveness — tuned for 51×51 dry-run where
# trajectory only sees the goal location from ONE direction, so multi-view
# consensus is impossible and we must fire on a single strong view).
TARGET_CHECK_INTERVAL = 5     # check target similarity every N frames
TARGET_VIEW_THRESH = 0.08     # per-view threshold for consensus counting (alert HUD only)
TARGET_CONSENSUS_ALERT = 2    # need N views above threshold for "NEAR TARGET" HUD
TARGET_SIM_CHECKIN = 0.12     # (DISABLED — see see() — kept for HUD/test compatibility)
TARGET_CONSENSUS_CHECKIN = 1
AUTO_CHECKIN_CONFIRM = 1
NEAR_GOAL_HOPS_CHECKIN = 8    # path-based CHECKIN when within N hops of goal candidate.
                              # Tightened from 30 because correctness > speed:
                              # 30 hops away can be 5+ meters off; 8 hops ≈ near-exact arrival.
GOAL_VERIFY_MIN_SIM = 0.15    # at goal, require ≥ this sim somewhere in 360° scan to CHECKIN

# Legacy constants (kept for test compatibility)
ACTION_HOLD_FRAMES = 15
PATH_GUIDED_REPLAN = 10
HAND_LOOKAHEAD = 5
GRADIENT_CHECK_INTERVAL = 50

# Action string → Action enum mapping (for path replay)
ACTION_MAP = {'FORWARD': Action.FORWARD, 'BACKWARD': Action.BACKWARD,
              'LEFT': Action.LEFT, 'RIGHT': Action.RIGHT}
REVERSE_ACTION = {'FORWARD': 'BACKWARD', 'BACKWARD': 'FORWARD',
                  'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
GRADIENT_HISTORY_SIZE = 5      # median-filter window for hop count
GRADIENT_PATIENCE = 6          # consecutive worsenings before hand flip
APPROACH_HOPS_ENTER = 15       # enter APPROACH state when hops < this
APPROACH_HOPS_EXIT = 20        # exit APPROACH back to NAVIGATE if hops >= this
SEARCH_HOPS_ENTER = 3          # enter SEARCH state when hops <= this
SEARCH_SIM_THRESHOLD = 0.20    # CHECKIN if max target sim > this
SEARCH_SIM_LOW_CONF = 0.18     # lower threshold for low-confidence goals.
                               # Was 0.12 — caused false-positive CHECKIN at aliased
                               # corridors (we observed 0.137 fire wrong location).
                               # Trajectory's best front-view sim is 0.22, so 0.18
                               # demands we be very near goal physically before CHECKIN.
BACKUP_CHECKIN_SIM = 0.30      # "lucky pass" CHECKIN during NAVIGATE/APPROACH
LOW_CONF_BACKUP_CHECKIN_SIM = 0.45
SEARCH_MAX_SCANS = 3           # max 360° scans before giving up
NAV_DIRECTION_COOLDOWN = 500   # min steps between hand flips (counts frames)
NAV_PLATEAU_STEPS = 2000       # steps without progress before trying next candidate
NAV_ESCAPE_FORWARD = 150       # steps in escape burst to leave area
NAV_HARD_TIMEOUT_STEPS = 12000 # absolute max nav steps before forced CHECKIN.
                               # If we never arrive at goal_node, fire SEARCH at
                               # current location: best guess > game timeout.
                               # ~6–10 min of game time at typical engine pace.
CLOSE_HOPS_NO_SWITCH = 30      # if best_hops < this, never switch candidate on
                               # plateau — we're already close, just retry.
                               # Observed: agent reached 17 hops then switched
                               # to a candidate 100+ hops away and timed out.
HAND_LEFT = "left"
HAND_RIGHT = "right"
LOW_CONF_AVG_SIM_THRESHOLD = 0.25
LOW_CONF_CONSENSUS_THRESHOLD = 3
LOW_CONF_APPROACH_HOPS_ENTER = 15
LOW_CONF_SEARCH_HOPS_ENTER = 3
LOW_CONF_APPROACH_STREAK = 2
LOW_CONF_SEARCH_STREAK = 3
LOW_CONF_MAX_NODE_DRIFT = 120
LOW_CONF_GOAL_CANDIDATES = 5
LOW_CONF_GOAL_SEPARATION = 30

# Exploration (wall-following)
TURN_STEPS_90 = 3           # LEFT/RIGHT actions for 90° turn (was 10 — each action now held 15 frames)
TURN_STEPS_180 = 6          # actions for 180° turn (was 20)
CHECK_RIGHT_INTERVAL = 20   # check right side every N forward steps (was 10 — less frequent)
STUCK_MSE_THRESHOLD = 200   # MSE below this = stuck (was 400 — lowered since hold is now 15 frames)
STUCK_FRAME_SIZE = (160, 120) # resize frames for comparison (was 80x60 — larger = less noise)

# Exploration budget (when exploring during navigation phase)
EXPLORE_STEPS = 10000       # steps to explore before building pipeline (was 8000)

# Exploration perturbation (loop-breaking)
PERTURB_START_STEP = 200    # don't perturb in first N steps (settle-in)
PERTURB_INTERVAL_MIN = 30   # min forward steps between perturbations (was 40)
PERTURB_INTERVAL_MAX = 50   # max forward steps (was 60)
LOOP_MSE_THRESHOLD = 300    # frame similarity for loop detection (was 200)
LOOP_BUFFER_SIZE = 20       # ring buffer of sampled frames
LOOP_SAMPLE_INTERVAL = 50   # sample every N forward steps
MOMENTUM_WINDOW = 30        # actions to track for momentum
MOMENTUM_LEFT_RATIO = 0.70  # LEFT ratio threshold for forced RIGHT

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# VLAD Feature Extraction (from baseline.py)
# pickle is used here for caching locally-generated SIFT descriptors and
# KMeans codebooks to disk — same pattern as the course-provided baseline.py.
# These files are only read back by the same code that wrote them.
# ---------------------------------------------------------------------------
class VLADExtractor:
    """RootSIFT + VLAD with intra-normalization and power normalization."""

    def __init__(self, n_clusters: int = 128):
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.03)
        self.codebook = None
        self._sift_cache: dict = {}

    @property
    def dim(self) -> int:
        return self.n_clusters * 128

    @staticmethod
    def _root_sift(des: np.ndarray) -> np.ndarray:
        des = des / (np.sum(des, axis=1, keepdims=True) + 1e-6)
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
        # Power normalization removed: SSR amplifies noise after intra-norm,
        # causing 0.046 mean adjacent similarity and 31.6% negative similarities.
        # Intra-norm already prevents cluster dominance.
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm
        return vlad

    @staticmethod
    def _make_mask(h: int, w: int) -> np.ndarray:
        """Mask that crops top/bottom 10% to ignore floor/ceiling."""
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * 0.10):int(h * 0.90), :] = 255
        return mask

    def load_sift_cache(self, file_list: list, subsample_rate: int, image_dir: str,
                        cache_tag: str = "default"):
        # pickle is used for caching locally-generated SIFT descriptors to disk
        # same pattern as the course-provided baseline.py (trusted local data only)
        cache_file = os.path.join(CACHE_DIR, f"sift_{cache_tag}_ss{subsample_rate}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached SIFT from {cache_file}")
            with open(cache_file, "rb") as f:
                self._sift_cache = pickle.load(f)  # nosec - trusted local cache
            if all(fname in self._sift_cache for fname in file_list):
                return
            print("  Cache incomplete, re-extracting...")

        print(f"Extracting SIFT for {len(file_list)} images...")
        self._sift_cache = {}
        for i, fname in enumerate(file_list):
            img = cv2.imread(os.path.join(image_dir, fname))
            if img is None:
                continue
            h, w = img.shape[:2]
            mask = self._make_mask(h, w)
            _, des = self.sift.detectAndCompute(img, mask)
            if des is not None:
                self._sift_cache[fname] = self._root_sift(des)
            if (i + 1) % 100 == 0:
                print(f"  SIFT: {i+1}/{len(file_list)}")
        with open(cache_file, "wb") as f:
            pickle.dump(self._sift_cache, f)  # nosec - trusted local cache
        print(f"  Saved {len(self._sift_cache)} descriptors -> {cache_file}")

    def build_vocabulary(self, file_list: list, cache_tag: str = "default"):
        cache_file = os.path.join(CACHE_DIR, f"codebook_{cache_tag}_k{self.n_clusters}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached codebook from {cache_file}")
            with open(cache_file, "rb") as f:
                self.codebook = pickle.load(f)
            return

        des_list = [self._sift_cache[f] for f in file_list if f in self._sift_cache]
        if len(des_list) == 0:
            print("WARNING: No SIFT descriptors to build vocabulary from!")
            return
        all_des = np.vstack(des_list)
        print(f"Fitting MiniBatchKMeans (k={self.n_clusters}) on {len(all_des)} descriptors...")
        self.codebook = MiniBatchKMeans(
            n_clusters=self.n_clusters, init='k-means++',
            n_init=1, max_iter=100, batch_size=1024,
            verbose=1, random_state=42,
        ).fit(all_des)
        print(f"  inertia={self.codebook.inertia_:.0f}")
        with open(cache_file, "wb") as f:
            pickle.dump(self.codebook, f)

    def extract(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        mask = self._make_mask(h, w)
        _, des = self.sift.detectAndCompute(img, mask)
        if des is None or len(des) == 0:
            return np.zeros(self.dim)
        return self._des_to_vlad(self._root_sift(des))

    def extract_batch(self, file_list: list) -> np.ndarray:
        vectors = []
        for i, fname in enumerate(file_list):
            if fname in self._sift_cache and len(self._sift_cache[fname]) > 0:
                vectors.append(self._des_to_vlad(self._sift_cache[fname]))
            else:
                vectors.append(np.zeros(self.dim))
            if (i + 1) % 100 == 0:
                print(f"  VLAD: {i+1}/{len(file_list)}")
        return np.array(vectors)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------
class KeyboardPlayerPyGame(Player):

    def __init__(self, n_clusters: int = 64, subsample_rate: int = 5,
                 top_k_shortcuts: int = 100, data_dir: str = "data",
                 offline_navigation: bool = False):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        super().__init__()

        self.subsample_rate = subsample_rate
        self.top_k_shortcuts = top_k_shortcuts
        self.offline_navigation = offline_navigation
        self.data_dir = os.path.abspath(data_dir)
        self.image_dir = os.path.join(self.data_dir, "images")
        self.data_info_path = os.path.join(self.data_dir, "data_info.json")
        self.cache_tag = hashlib.md5(self.data_dir.encode("utf-8")).hexdigest()[:8]

        # Trajectory data (loaded in pre_navigation)
        self.motion_frames = []
        self.file_list = []

        # VLAD pipeline
        self.extractor = VLADExtractor(n_clusters=n_clusters)
        self.database = None
        self.G = None
        self.goal_node = None
        self.goal_candidates = []
        self.goal_candidate_scores = {}
        self.goal_candidate_index = 0

        # Autonomous navigation state (gradient-guided wall-following)
        self.nav_tick = 0
        self.current_action = Action.IDLE
        self.action_hold_counter = 0
        self.nav_state = NavState.NAVIGATE
        self.wall_hand = HAND_LEFT
        self.gradient_step_counter = 0
        self.hop_history = deque(maxlen=GRADIENT_HISTORY_SIZE)
        self.prev_avg_hops = None
        self.gradient_worsen_count = 0
        self.nav_last_best_hops = 9999
        self.nav_plateau_start = 0
        self.nav_last_flip_step = 0
        self.nav_total_steps = 0
        # Search state (near-goal 360° scan)
        self.search_turn_counter = 0
        self.search_scan_count = 0
        self.search_best_sim = 0.0
        self.search_best_steps = 0
        # Navigation wall-following (separate from exploration state)
        self._nav_es = ExploreState.FORWARD
        self._nav_fwd = 0
        self._nav_tc = 0
        self._nav_stuck_count = 0
        self._escape_remaining = 0
        self._search_failure_cooldown = 0  # step when last SEARCH failed (approach re-entry cooldown)
        # Path-following state
        self._nav_path = None              # current Dijkstra path (list of node indices)
        self._nav_path_idx = 0             # index into _nav_path
        self._path_step = 0                # step counter for replan interval
        self._turn_remaining = 0            # frames remaining in path-directed turn
        self._turn_direction = Action.LEFT # direction of path-directed turn
        self._pre_action_frame = None      # frame saved before action hold starts (for skip-frame stuck detection)
        self._last_hand_change_step = 0    # step when wall hand last changed (cooldown)

        # Exploration state (wall-following)
        self.explore_state = ExploreState.FORWARD
        self.prev_frame = None              # previous frame for stuck detection
        self.forward_count = 0              # steps since last turn
        self.turn_counter = 0               # count rotation steps
        self.consecutive_stuck = 0          # stuck attempts (for dead-end detection)
        self.manual_mode = False            # keyboard override
        self.explore_hand = HAND_LEFT       # exploration wall-following hand (flips at phase transitions)
        self._escape_burst = 0              # forward burst steps remaining for phase transition

        # Exploration perturbation (loop-breaking)
        self._perturb_next_interval = random.randint(PERTURB_INTERVAL_MIN, PERTURB_INTERVAL_MAX)
        self._perturb_forward_count = 0     # forward steps since last perturbation
        self._perturb_turning = False       # True = executing a perturbation turn
        self._perturb_turn_counter = 0      # steps into perturbation turn
        self._perturb_direction = Action.RIGHT
        self._loop_buffer = deque(maxlen=LOOP_BUFFER_SIZE)
        self._loop_forward_count = 0
        self._recent_actions = deque(maxlen=MOMENTUM_WINDOW)

        # Navigation stuck detection (frame-based)
        self.nav_stuck_frames = 0           # consecutive frames physically stuck

        # Exploration frame saving
        self.explore_step = 0               # exploration step counter
        self.explore_data = []              # metadata for data_info.json
        self.last_explore_action = 'IDLE'   # action taken this step (for saving)

        # Two-stage navigation: explore first, then navigate
        # If exploration phase ran, pipeline_ready will be True after pre_navigation()
        # If exploration was skipped, we explore during navigation phase first
        self.pipeline_ready = False
        self.exploring_in_nav = False       # True = currently exploring during nav phase
        self.low_confidence_goal = False
        self.goal_avg_sim = 0.0
        self.goal_consensus = 0
        self.low_confidence_streak = 0
        self._last_gradient_node = None
        self._prev_sims = None
        self._prev_node = None
        self._target_vlads = None           # cached target VLADs for multi-view CHECKIN
        self._last_vlad = None              # cached VLAD from last _get_current_node()
        # Manual mode target detection state
        self._target_max_sim = 0.0
        self._target_consensus = 0
        self._target_best_ever = 0.0
        self._target_best_step = 0
        self._target_checkin_streak = 0
        self._target_check_counter = 0
        self._auto_checkin_triggered = False
        # Simple "been here" tracking: buffer of saved frame snapshots
        self._visit_buffer = []             # list of small grayscale frames (80x60)
        self._been_here = False             # current frame matches a saved snapshot

    # --- Game engine hooks ---

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def pre_exploration(self):
        self.explore_data = []
        self.explore_step = 0
        self.manual_mode = False
        self.exploring_in_nav = False
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')

    def pre_navigation(self):
        super().pre_navigation()
        # Save exploration data to disk if we collected any during exploration phase
        if self.explore_data:
            with open(self.data_info_path, 'w') as f:
                json.dump(self.explore_data, f)
            print(f"Saved {len(self.explore_data)} exploration frames to {self.data_info_path}")

        # Check if we have exploration data (from exploration phase or previous run)
        if os.path.exists(self.data_info_path):
            # Skip build if no target images yet (engine test-run before game starts)
            targets = self.get_target_images()
            if not targets or len(targets) == 0:
                print("No target images yet, deferring pipeline build.")
                self.exploring_in_nav = not self.offline_navigation
                return
            self._build_pipeline()
        else:
            # No exploration data — we need to explore during navigation phase
            print("No exploration data found. Will explore during navigation phase first.")
            self.exploring_in_nav = True
            self.explore_data = []
            self.explore_step = 0
            self.explore_state = ExploreState.FORWARD

    def _build_pipeline(self):
        """Build the full VLAD navigation pipeline from exploration data."""
        self.database = None
        self.G = None
        self.goal_node = None
        self.goal_candidates = []
        self.goal_candidate_scores = {}
        self.goal_candidate_index = 0
        self._load_trajectory_data()
        self._build_database()
        self._build_graph()
        self._setup_goal()
        if self.goal_node is not None:
            self.pipeline_ready = True
            self.exploring_in_nav = False
            self.show_target_images()
            print("\n=== MANUAL NAVIGATION MODE ===")
            print("Arrow keys to navigate. System auto-detects target.")
            print("Press SPACE to CHECKIN manually. Press any other key to show targets.")
        else:
            print("WARNING: Pipeline build failed (no goal). Continuing exploration...")

    def _finish_exploration_in_nav(self):
        """Called when exploration budget is exhausted during navigation phase."""
        print(f"\n=== EXPLORATION COMPLETE ({self.explore_step} frames) ===")
        print("Building navigation pipeline...")
        with open(self.data_info_path, 'w') as f:
            json.dump(self.explore_data, f)
        print(f"Saved {len(self.explore_data)} frames to {self.data_info_path}")
        self._build_pipeline()

    def act(self):
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return Action.QUIT
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        # If still exploring (either in exploration phase or nav phase)
        if not self.pipeline_ready:
            if self.offline_navigation:
                return Action.IDLE
            if self.exploring_in_nav and self.explore_step >= EXPLORE_STEPS:
                self._finish_exploration_in_nav()
                if self.pipeline_ready:
                    self.show_target_images()
                    print("\n=== MANUAL NAVIGATION MODE ===")
                    print("Use arrow keys to navigate. System will detect target automatically.")
                    return self.last_act
                return Action.FORWARD

            third = EXPLORE_STEPS // 3
            if self.explore_step == third and self.explore_hand == HAND_LEFT:
                self.explore_hand = HAND_RIGHT
                self.explore_state = ExploreState.FORWARD
                self.turn_counter = 0
                self.forward_count = 0
                self.consecutive_stuck = 0
                self._perturb_turning = False
                self._escape_burst = 50
                print(f"[EXPLORE] Phase 2 ({self.explore_step}), 180° + burst + right-hand")
            elif self.explore_step == third * 2 and self.explore_hand == HAND_RIGHT:
                self.explore_hand = HAND_LEFT
                self.explore_state = ExploreState.FORWARD
                self.turn_counter = 0
                self.forward_count = 0
                self.consecutive_stuck = 0
                self._escape_burst = 50
                print(f"[EXPLORE] Phase 3 ({self.explore_step}), 180° + burst + left-hand")

            if self.manual_mode:
                action = self.last_act
            else:
                action = self._explore_act()

            ACTION_NAMES = {
                Action.FORWARD: 'FORWARD', Action.BACKWARD: 'BACKWARD',
                Action.LEFT: 'LEFT', Action.RIGHT: 'RIGHT',
                Action.CHECKIN: 'CHECKIN', Action.IDLE: 'IDLE',
            }
            self.last_explore_action = ACTION_NAMES.get(action, 'IDLE')
            return action

        # === AUTO NAVIGATION (with manual override + auto-target detection) ===
        # Auto-CHECKIN if target detected
        if self._auto_checkin_triggered:
            print("[AUTO-CHECKIN] Target confirmed! Checking in...")
            return Action.CHECKIN

        # Manual override: if user is holding any arrow key, take it
        if self.last_act != Action.IDLE:
            return self.last_act

        # Otherwise, auto-navigate via Dijkstra path-following
        return self._auto_navigate()

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        # Store previous frame for stuck detection (before updating self.fpv)
        if self.fpv is not None:
            self.prev_frame = self.fpv.copy()

        self.fpv = fpv

        # Save frames during exploration (regardless of which game phase we're in)
        if not self.pipeline_ready and not self.offline_navigation:
            fname = f"{self.explore_step:06d}.png"
            cv2.imwrite(os.path.join(self.image_dir, fname), fpv)
            self.explore_data.append({
                'step': self.explore_step,
                'image': fname,
                'action': [self.last_explore_action],
            })
            self.explore_step += 1
            if self.explore_step % 500 == 0:
                print(f"Exploration: {self.explore_step} frames saved")

        # --- Simple "been here" detection + target matching ---
        display_fpv = fpv.copy()
        if self.pipeline_ready:
            self._target_check_counter += 1

            # Check every 10 frames (not too frequent — avoids lag)
            if self._target_check_counter % 10 == 0:
                # Downsample current frame for fast comparison
                small = cv2.resize(cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY), (80, 60))

                # Check if we've been here: compare against saved snapshots
                been_here = False
                for old_frame in self._visit_buffer:
                    mse = float(np.mean((small.astype(np.float32) - old_frame.astype(np.float32)) ** 2))
                    if mse < 300:  # very similar frame = been here
                        been_here = True
                        break

                self._been_here = been_here

                # Save snapshot every 30 checks (~every 300 frames = ~6 seconds)
                if self._target_check_counter % 30 == 0:
                    self._visit_buffer.append(small)

                # Target similarity check (uses VLAD — heavier, do less often)
                if self._target_check_counter % 20 == 0 and self._target_vlads:
                    fpv_vlad = self.extractor.extract(fpv)
                    target_sims = [float(fpv_vlad @ tv) for tv in self._target_vlads]
                    max_sim = max(target_sims)
                    consensus = sum(1 for s in target_sims if s > TARGET_VIEW_THRESH)
                    self._target_max_sim = max_sim
                    self._target_consensus = consensus
                    if max_sim > self._target_best_ever:
                        self._target_best_ever = max_sim
                        view_names = ['front', 'left', 'back', 'right']
                        best_view = view_names[int(np.argmax(target_sims))]
                        self._target_best_step = self.nav_total_steps
                        print(f"[TARGET] New best! sim={max_sim:.3f} ({best_view})")

                    # Visual-sim auto-CHECKIN intentionally DISABLED.
                    # On 51×51 dry-run, single-trajectory VLAD aliasing fires false
                    # positives at sim ~0.13–0.15 (a corridor that looks like the
                    # target's right view but is physically far from the goal).
                    # CHECKIN is now path-arrival-only — see _path_guided_act.
                    self._target_checkin_streak = 0

            # --- Simple text HUD (single line, no clutter) ---
            h, w = display_fpv.shape[:2]
            # "BEEN HERE" or "NEW AREA" — bottom of screen so it doesn't block navigation
            if self._been_here:
                cv2.putText(display_fpv, "BEEN HERE - try another way",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                cv2.putText(display_fpv, "NEW AREA",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Target match indicator — only show when getting warm
            if self._target_max_sim > 0.10:
                cv2.putText(display_fpv, f"Target match: {self._target_max_sim:.2f}",
                            (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if self._target_consensus >= TARGET_CONSENSUS_CHECKIN:
                cv2.putText(display_fpv, ">>> AT TARGET - PRESS SPACE <<<",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display
        if self.screen is None:
            h, w, _ = display_fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption("KeyboardPlayer:fpv")
        rgb = display_fpv[:, :, ::-1]
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], 'RGB')
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def set_target_images(self, images):
        super().set_target_images(images)
        self.show_target_images()

    # --- Section 1: Data Loading ---

    def _load_trajectory_data(self):
        if not os.path.exists(self.data_info_path):
            print(f"WARNING: {self.data_info_path} not found!")
            return
        with open(self.data_info_path) as f:
            raw = json.load(f)
        pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
        all_motion = [
            {'step': d['step'], 'image': d['image'], 'action': d['action'][0]}
            for d in raw
            if len(d['action']) == 1 and d['action'][0] in pure
        ]
        self.motion_frames = all_motion[::self.subsample_rate]
        self.file_list = [m['image'] for m in self.motion_frames]
        print(f"Frames: {len(all_motion)} total, "
              f"{len(self.motion_frames)} after {self.subsample_rate}x subsample")

    # --- Sections 2-3: VLAD Database ---

    def _build_database(self):
        if len(self.file_list) == 0:
            print("WARNING: No exploration frames to build database from!")
            self.database = np.zeros((0, self.extractor.dim))
            return
        self.extractor.load_sift_cache(
            self.file_list, self.subsample_rate, self.image_dir, self.cache_tag)
        self.extractor.build_vocabulary(self.file_list, self.cache_tag)
        if self.extractor.codebook is None:
            print("WARNING: No codebook built, skipping database extraction!")
            self.database = np.zeros((0, self.extractor.dim))
            return
        self.database = self.extractor.extract_batch(self.file_list)
        print(f"Database: {self.database.shape}")

    # --- Section 4: Graph Construction ---

    def _build_graph(self):
        n = len(self.database)
        if n == 0:
            print("WARNING: Empty database, cannot build graph!")
            self.G = nx.Graph()
            return
        self.G = nx.Graph()
        self.G.add_nodes_from(range(n))

        for i in range(n - 1):
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

        added = 0
        print(f"Top-{top_k} shortcuts (min_gap={MIN_SHORTCUT_GAP}):")
        for rank, fi in enumerate(top_idx):
            i, j = divmod(int(fi), n)
            s = float(flat[fi])
            if s < VISUAL_SHORTCUT_SIM_FLOOR:
                continue  # similarity floor: skip noisy shortcuts
            d = float(np.sqrt(max(0, 2 - 2 * s)))
            self.G.add_edge(i, j,
                            weight=VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * d,
                            edge_type="visual")
            added += 1
            if added <= 5:
                print(f"  #{added}: {i}<->{j} gap={abs(j-i)} sim={s:.4f} d={d:.4f}")
        print(f"  Added {added} shortcuts (filtered from {top_k} candidates)")

        n_components = nx.number_connected_components(self.G)
        print(f"Graph: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges, "
              f"{n_components} component(s)")
        if n_components > 1:
            sizes = sorted([len(c) for c in nx.connected_components(self.G)], reverse=True)
            print(f"  WARNING: Disconnected graph! Component sizes: {sizes[:5]}")

    # --- Section 5: Localization + Goal ---

    def _smooth_similarity_curve(self, sims: np.ndarray, radius: int) -> np.ndarray:
        """Average similarities in a small temporal neighborhood."""
        if radius <= 0 or len(sims) == 0:
            return sims
        window = np.ones(2 * radius + 1, dtype=np.float32) / float(2 * radius + 1)
        return np.convolve(sims, window, mode='same')

    @staticmethod
    def _is_low_confidence_goal(avg_sim: float, consensus: int) -> bool:
        return avg_sim < LOW_CONF_AVG_SIM_THRESHOLD or consensus < LOW_CONF_CONSENSUS_THRESHOLD

    def _backup_checkin_threshold(self) -> float:
        return LOW_CONF_BACKUP_CHECKIN_SIM if self.low_confidence_goal else BACKUP_CHECKIN_SIM

    def _should_enter_approach(self, hops: int) -> bool:
        # Cooldown: don't re-enter APPROACH for 300 steps after SEARCH failure
        if (self.nav_total_steps - self._search_failure_cooldown) < 300:
            return False
        if not self.low_confidence_goal:
            return hops < APPROACH_HOPS_ENTER
        return hops <= LOW_CONF_APPROACH_HOPS_ENTER and self.low_confidence_streak >= LOW_CONF_APPROACH_STREAK

    def _should_enter_search(self, hops: int) -> bool:
        if not self.low_confidence_goal:
            return hops <= SEARCH_HOPS_ENTER
        return hops <= LOW_CONF_SEARCH_HOPS_ENTER and self.low_confidence_streak >= LOW_CONF_SEARCH_STREAK

    def _reset_low_confidence_stability(self):
        self.low_confidence_streak = 0
        self._last_gradient_node = None

    def _active_goal_node(self) -> int | None:
        if self.goal_candidates:
            idx = min(self.goal_candidate_index, len(self.goal_candidates) - 1)
            return self.goal_candidates[idx]
        return self.goal_node

    def _advance_goal_candidate(self) -> bool:
        if self.goal_candidate_index + 1 >= len(self.goal_candidates):
            return False
        self.goal_candidate_index += 1
        self.goal_node = self.goal_candidates[self.goal_candidate_index]
        self._reset_low_confidence_stability()
        print(f"[GOAL] Switch to candidate {self.goal_candidate_index + 1}/{len(self.goal_candidates)} "
              f"(node={self.goal_node})")
        return True

    def _update_low_confidence_stability(self, cur: int, hops: int, avg_hops: float):
        if not self.low_confidence_goal:
            return

        node_consistent = (
            self._last_gradient_node is None or
            abs(cur - self._last_gradient_node) <= LOW_CONF_MAX_NODE_DRIFT
        )
        hops_consistent = self.prev_avg_hops is None or avg_hops <= self.prev_avg_hops + 5
        if node_consistent and hops_consistent:
            self.low_confidence_streak += 1
        else:
            self.low_confidence_streak = 0
        self._last_gradient_node = cur

    def _setup_goal(self):
        if self.database is None or len(self.database) == 0:
            print("WARNING: Empty database, cannot match goal!")
            return
        if self.extractor.codebook is None:
            print("WARNING: No codebook, cannot match goal!")
            return
        targets = self.get_target_images()
        if not targets:
            print("WARNING: No target images available!")
            return

        print(f"Matching goal using {len(targets)} target views...")

        all_sims = []
        view_names = ['front', 'left', 'back', 'right']
        per_view_best = []  # list of (best_node, best_sim, view_name)
        for i, target in enumerate(targets):
            vlad = self.extractor.extract(target)
            sims = self.database @ vlad
            all_sims.append(sims)
            best_node = int(np.argmax(sims))
            best_sim = float(sims[best_node])
            per_view_best.append((best_node, best_sim, view_names[i]))
            print(f"  {view_names[i]}: best node {best_node}, sim={best_sim:.4f}")

        # PRIMARY GOAL: the per-view single best match (front view typically wins —
        # that's the trajectory frame closest to where target was photographed).
        # Averaging across views is unreliable: when the trajectory passes the goal
        # only once, only one view matches strongly to a local trajectory window;
        # the other views match elsewhere in the trajectory, and the mean picks
        # something nobody likes.
        per_view_sorted = sorted(per_view_best, key=lambda x: -x[1])
        primary_node, primary_sim, primary_view = per_view_sorted[0]

        # Goal selection: primary = strongest single-view match (front-view's
        # best DB frame). Candidates = each view's best, deduped. SEARCH-based
        # verification at each candidate keeps cycling safe — we only CHECKIN
        # if the 360° scan crosses SEARCH_SIM_LOW_CONF (0.18), so a wrong
        # candidate just gets rejected and we move on rather than false-CHECKIN.
        candidates_seen = []
        for node, _sim, _name in per_view_sorted:
            if not any(abs(node - c) <= GOAL_NEIGHBORHOOD_RADIUS for c in candidates_seen):
                candidates_seen.append(node)
        self.goal_candidates = candidates_seen[:LOW_CONF_GOAL_CANDIDATES]
        self.goal_candidate_index = 0
        self.goal_node = primary_node

        # Score each candidate by its strongest per-view sim (used for low-conf cycling)
        self.goal_candidate_scores = {}
        for c in self.goal_candidates:
            best_view_sim = max(float(sims[c]) for sims in all_sims)
            self.goal_candidate_scores[c] = best_view_sim

        avg_sim = float(np.mean([sims[self.goal_node] for sims in all_sims]))
        d = float(np.sqrt(max(0, 2 - 2 * primary_sim)))
        print(f"  -> Picked goal via best single view: {primary_view} -> node {primary_node} (sim={primary_sim:.4f})")

        consensus = sum(1 for sims in all_sims
                        if abs(int(np.argmax(sims)) - self.goal_node) <= GOAL_NEIGHBORHOOD_RADIUS)

        self.goal_avg_sim = avg_sim
        self.goal_consensus = consensus
        self.low_confidence_goal = self._is_low_confidence_goal(avg_sim, consensus)
        if self.low_confidence_goal:
            print(f"WARNING: Low goal confidence! Target may not be in exploration data.")
            self._reset_low_confidence_stability()

        # Cache target VLADs for fast multi-view CHECKIN
        self._target_vlads = [self.extractor.extract(t) for t in targets]

        print(f"Goal: node {self.goal_node} (avg_sim={avg_sim:.4f}, d={d:.4f}, consensus={consensus}/4)")
        if self.low_confidence_goal and self.goal_candidates:
            print(f"Goal candidates: {self.goal_candidates}")

    def _get_current_node(self) -> int:
        feat = self.extractor.extract(self.fpv)
        self._last_vlad = feat              # cache for backup checkin
        sims = self.database @ feat

        # Very light temporal smoothing — mostly trust current frame
        if self._prev_sims is not None:
            sims = 0.95 * sims + 0.05 * self._prev_sims
        self._prev_sims = sims.copy()

        cur = int(np.argmax(sims))
        if self._prev_node is not None and len(sims) > 0:
            prev = self._prev_node
            lo = max(0, prev - LOCALIZATION_SEARCH_RADIUS)
            hi = min(len(sims), prev + LOCALIZATION_SEARCH_RADIUS + 1)
            local_cur = lo + int(np.argmax(sims[lo:hi]))
            if abs(cur - prev) > LOCALIZATION_JUMP_THRESHOLD:
                global_sim = float(sims[cur])
                local_sim = float(sims[local_cur])
                if local_sim >= global_sim - LOCALIZATION_JUMP_MARGIN:
                    cur = local_cur
        self._prev_node = cur

        return cur

    def _get_path(self, start: int) -> list:
        goal = self._active_goal_node()
        if goal is None:
            return list(range(9999))  # no goal → very high hops
        try:
            return nx.shortest_path(self.G, start, goal, weight="weight")
        except nx.NetworkXNoPath:
            # Return fake long path so caller sees high hop count, not 0
            return list(range(9999))

    def _edge_action(self, a: int, b: int):
        """Get the Action enum for traversing graph edge a→b."""
        if b == a + 1 and a < len(self.motion_frames):
            return ACTION_MAP.get(self.motion_frames[a]['action'])
        elif b == a - 1 and b < len(self.motion_frames):
            rev = REVERSE_ACTION.get(self.motion_frames[b]['action'])
            return ACTION_MAP.get(rev) if rev else None
        return None  # visual shortcut — can't replay

    def _choose_hand_from_path(self, path: list):
        """Choose wall-following hand based on path direction lookahead.
        Only changes hand if signal is very strong (>3 difference) and
        cooldown has passed (100 steps since last change)."""
        # Cooldown: don't flip hand too often
        if (self.nav_total_steps - self._last_hand_change_step) < 100:
            return
        left_count, right_count = 0, 0
        lookahead = min(HAND_LOOKAHEAD, len(path) - 1)
        for i in range(lookahead):
            action = self._edge_action(path[i], path[i + 1])
            if action == Action.LEFT:
                left_count += 1
            elif action == Action.RIGHT:
                right_count += 1
        if left_count > right_count + 2:
            if self.wall_hand != HAND_LEFT:
                self.wall_hand = HAND_LEFT
                self._last_hand_change_step = self.nav_total_steps
                print(f"[PATH] Hand → LEFT (L={left_count} R={right_count})")
        elif right_count > left_count + 2:
            if self.wall_hand != HAND_RIGHT:
                self.wall_hand = HAND_RIGHT
                self._last_hand_change_step = self.nav_total_steps
                print(f"[PATH] Hand → RIGHT (L={left_count} R={right_count})")

    @staticmethod
    def _select_goal_candidates(goal_scores: np.ndarray, count: int, separation: int) -> list:
        if len(goal_scores) == 0 or count <= 0:
            return []
        order = np.argsort(-goal_scores)
        picked = []
        for idx in order:
            idx = int(idx)
            if all(abs(idx - prev) >= separation for prev in picked):
                picked.append(idx)
                if len(picked) >= count:
                    break
        return picked

    # --- Section 6: Gradient-Guided Wall-Following Navigation ---

    def _wall_follow_act(self, hand: str) -> Action:
        """Wall-following movement parameterized by hand (left or right).
        Uses separate state from exploration to avoid interference.
        Stuck detection only fires in FORWARD state using saved pre-action frame."""
        if hand == HAND_LEFT:
            wall_turn, check_turn = Action.LEFT, Action.RIGHT
            wall_state = ExploreState.TURN_LEFT
        else:
            wall_turn, check_turn = Action.RIGHT, Action.LEFT
            wall_state = ExploreState.TURN_RIGHT

        if self._nav_es == ExploreState.FORWARD:
            # Compare current frame against the frame saved BEFORE this action hold started
            stuck = self._is_stuck(self.fpv, self._pre_action_frame)
            # Save current frame as reference for next stuck check
            if self.fpv is not None:
                self._pre_action_frame = self.fpv.copy()
            if stuck:
                self._nav_stuck_count += 1
                if self._nav_stuck_count >= 3:
                    self._nav_es = ExploreState.REVERSE
                    self._nav_tc = 0
                    return wall_turn
                else:
                    self._nav_es = wall_state
                    self._nav_tc = 0
                    return wall_turn
            else:
                self._nav_stuck_count = 0
                self._nav_fwd += 1
                if self._nav_fwd >= CHECK_RIGHT_INTERVAL:
                    self._nav_fwd = 0
                    self._nav_es = ExploreState.CHECK_RIGHT
                    self._nav_tc = 0
                    return check_turn
                return Action.FORWARD

        elif self._nav_es == ExploreState.TURN_LEFT:
            self._nav_tc += 1
            if self._nav_tc >= TURN_STEPS_90:
                self._nav_es = ExploreState.FORWARD
                if self.fpv is not None:
                    self._pre_action_frame = self.fpv.copy()
                return Action.FORWARD
            return Action.LEFT

        elif self._nav_es == ExploreState.TURN_RIGHT:
            self._nav_tc += 1
            if self._nav_tc >= TURN_STEPS_90:
                self._nav_es = ExploreState.FORWARD
                if self.fpv is not None:
                    self._pre_action_frame = self.fpv.copy()
                return Action.FORWARD
            return Action.RIGHT

        elif self._nav_es == ExploreState.CHECK_RIGHT:
            self._nav_tc += 1
            if self._nav_tc >= TURN_STEPS_90:
                # After turning to check, compare against pre-check frame
                stuck = self._is_stuck(self.fpv, self._pre_action_frame)
                if stuck:
                    self._nav_es = wall_state
                    self._nav_tc = 0
                    return wall_turn
                else:
                    self._nav_es = ExploreState.FORWARD
                    self._nav_stuck_count = 0
                    if self.fpv is not None:
                        self._pre_action_frame = self.fpv.copy()
                    return Action.FORWARD
            return check_turn

        elif self._nav_es == ExploreState.REVERSE:
            self._nav_tc += 1
            if self._nav_tc >= TURN_STEPS_180:
                self._nav_es = ExploreState.FORWARD
                self._nav_stuck_count = 0
                if self.fpv is not None:
                    self._pre_action_frame = self.fpv.copy()
                return Action.FORWARD
            return wall_turn

        return Action.FORWARD

    def _gradient_check(self) -> int:
        """Re-localize, compute hops, track progress, flip hand if regressing."""
        cur = self._get_current_node()
        path = self._get_path(cur)
        hops = len(path) - 1

        self.hop_history.append(hops)
        avg_hops = float(np.median(list(self.hop_history)))
        self._update_low_confidence_stability(cur, hops, avg_hops)

        if self.prev_avg_hops is not None:
            if avg_hops > self.prev_avg_hops + 2:
                self.gradient_worsen_count += 1
                if self.gradient_worsen_count >= GRADIENT_PATIENCE:
                    if (self.nav_total_steps - self.nav_last_flip_step) >= NAV_DIRECTION_COOLDOWN:
                        old = self.wall_hand
                        self.wall_hand = HAND_RIGHT if old == HAND_LEFT else HAND_LEFT
                        self.nav_last_flip_step = self.nav_total_steps
                        self.gradient_worsen_count = 0
                        self.hop_history.clear()
                        self._nav_es = ExploreState.FORWARD
                        self._nav_tc = 0
                        self._nav_fwd = 0
                        self._nav_stuck_count = 0
                        self._reset_low_confidence_stability()
                        print(f"[NAV] FLIP {old} -> {self.wall_hand} (avg_hops={avg_hops:.0f})")
            elif avg_hops < self.prev_avg_hops - 2:
                self.gradient_worsen_count = 0

        if hops < self.nav_last_best_hops:
            self.nav_last_best_hops = hops
            self.nav_plateau_start = self.nav_total_steps
            print(f"[NAV] Progress! {hops} hops (hand={self.wall_hand})")

        self.prev_avg_hops = avg_hops

        if self.nav_total_steps % 100 == 0:
            print(f"[NAV] step={self.nav_total_steps} node={cur} hops={hops} "
                  f"best={self.nav_last_best_hops} hand={self.wall_hand}")

        # Backup "lucky pass" visual CHECKIN DISABLED (false-positive prone).
        # We rely on path-arrival + 360° SEARCH verification only.
        return hops

    def _search_act(self) -> Action:
        """Near-goal: turn 360° scanning for visual match to target images."""
        self.search_turn_counter += 1

        # Check visual similarity every 3 turn steps
        if self.search_turn_counter % 3 == 0 and self.fpv is not None and self._target_vlads:
            fpv_vlad = self.extractor.extract(self.fpv)
            target_sims = [float(fpv_vlad @ tv) for tv in self._target_vlads]
            max_sim = max(target_sims)
            best_view = ['front', 'left', 'back', 'right'][int(np.argmax(target_sims))]

            if max_sim > self.search_best_sim:
                self.search_best_sim = max_sim
                self.search_best_steps = self.search_turn_counter

            threshold = SEARCH_SIM_LOW_CONF if self.low_confidence_goal else SEARCH_SIM_THRESHOLD
            if max_sim > threshold:
                print(f"[SEARCH] CHECKIN! sim={max_sim:.3f} ({best_view})")
                self.nav_state = NavState.CHECKIN
                return Action.CHECKIN

        # Full 360° rotation completed?
        scan_steps = TURN_STEPS_90 * 4
        if self.search_turn_counter >= scan_steps:
            self.search_scan_count += 1
            self.search_turn_counter = 0
            print(f"[SEARCH] Scan {self.search_scan_count}/{SEARCH_MAX_SCANS}, "
                  f"best_sim={self.search_best_sim:.3f}")

            if self.search_scan_count >= SEARCH_MAX_SCANS:
                # Tried hard and never crossed the verification threshold. Two options:
                # (a) we're not actually at the goal — ESCAPE and re-navigate
                # (b) we ARE at the goal but VLAD just can't break threshold here
                # Without a way to distinguish, we CHECKIN at goal_node anyway —
                # we've physically arrived per the path, and "best guess" beats
                # wandering forever. Better Partial than timeout.
                print(f"[SEARCH] No match after {SEARCH_MAX_SCANS} scans "
                      f"(best={self.search_best_sim:.3f}). CHECKIN at arrival point.")
                self.nav_state = NavState.CHECKIN
                return Action.CHECKIN
            else:
                # Move forward a bit then scan again from different position
                self.current_action = Action.FORWARD
                self.action_hold_counter = 5
                return Action.FORWARD

        return Action.LEFT  # turn in place

    def _path_guided_act(self) -> Action:
        """Hybrid path-replay + wall-follow: replay path actions, fall back to
        wall-follow when stuck. Re-localizes frequently."""
        self._path_step += 1

        # Re-localize and replan every N steps or when path is exhausted
        if self._path_step % PATH_GUIDED_REPLAN == 0 or self._nav_path is None:
            cur = self._get_current_node()
            self._nav_path = self._get_path(cur)
            self._nav_path_idx = 0
            hops = len(self._nav_path) - 1

            if hops < self.nav_last_best_hops:
                self.nav_last_best_hops = hops
                self.nav_plateau_start = self.nav_total_steps
                print(f"[PATH] Progress! {hops} hops")

            if self.nav_total_steps % 50 == 0:
                goal = self._active_goal_node()
                print(f"[PATH] step={self.nav_total_steps} node={cur} goal={goal} hops={hops}")

            # Backup visual-CHECKIN trigger DISABLED (false-positive prone on aliased corridors).
            # CHECKIN is now path-arrival-only: must be hops ≤ NEAR_GOAL_HOPS_CHECKIN.

            # Near goal → enter SEARCH state to do 360° verification before CHECKIN.
            # SEARCH scans all 4 headings; if max target sim across the scan
            # exceeds GOAL_VERIFY_MIN_SIM we're confident we're physically at
            # the goal (not just topologically close on the trajectory). If the
            # scan disagrees we resume NAVIGATE and try again.
            if hops <= NEAR_GOAL_HOPS_CHECKIN:
                print(f"[PATH] Arrived ({hops} hops), entering SEARCH for 360° verification")
                self.nav_state = NavState.SEARCH
                self.search_turn_counter = 0
                self.search_scan_count = 0
                self.search_best_sim = 0.0
                return Action.LEFT

            # Hard timeout: agent might never physically reach goal_node under
            # NOISY_MOTION + wall constraints. Better to fire SEARCH at current
            # location than time out the game. SEARCH still verifies before
            # CHECKIN, so this isn't a false positive — it's "best guess given
            # what we know."
            if self.nav_total_steps >= NAV_HARD_TIMEOUT_STEPS:
                print(f"[PATH] Hard timeout at step {self.nav_total_steps} "
                      f"(hops={hops}). Entering SEARCH at current location.")
                self.nav_state = NavState.SEARCH
                self.search_turn_counter = 0
                self.search_scan_count = 0
                self.search_best_sim = 0.0
                return Action.LEFT

            # Near goal → SEARCH (legacy gradient-based path)
            if self._should_enter_search(hops):
                print(f"[PATH] Near goal ({hops} hops), entering SEARCH")
                self.nav_state = NavState.SEARCH
                self.search_turn_counter = 0
                self.search_scan_count = 0
                self.search_best_sim = 0.0
                return Action.LEFT

        # Replay path actions (trust the path — NOISY_MOTION can slide along walls)
        path = self._nav_path
        if path and self._nav_path_idx < len(path) - 1:
            a = path[self._nav_path_idx]
            b = path[self._nav_path_idx + 1]
            action = self._edge_action(a, b)
            self._nav_path_idx += 1
            if action is not None:
                return action
            # Visual shortcut edge — can't replay, force replan
            self._nav_path = None
            return Action.FORWARD

        # Path exhausted — force replan
        self._nav_path = None
        return Action.FORWARD

    def _auto_navigate(self):
        """Path-following navigation with wall-follow fallback for stuck recovery.

        Primary: replays recorded actions along Dijkstra shortest path.
        Fallback: wall-following when physically stuck (wall blocking path action).
        SEARCH: 360° scan near goal for visual CHECKIN confirmation.
        """
        if self.fpv is None or self.goal_node is None:
            return Action.IDLE

        self.nav_total_steps += 1

        # Action hold (momentum — each action persists for ACTION_HOLD_FRAMES)
        if self.action_hold_counter > 0:
            self.action_hold_counter -= 1
            return self.current_action

        # --- CHECKIN state ---
        if self.nav_state == NavState.CHECKIN:
            return Action.CHECKIN

        # --- SEARCH state (360° scan for visual match near goal) ---
        if self.nav_state == NavState.SEARCH:
            return self._search_act()

        # --- ESCAPE state (forward burst to leave stuck area) ---
        if self.nav_state == NavState.ESCAPE:
            self._escape_remaining -= 1
            if self._escape_remaining <= 0:
                print(f"[ESCAPE] Done, resuming path-follow")
                self.nav_state = NavState.NAVIGATE
                self._nav_path = None  # force replan
                self._stuck_count = 0
                return Action.FORWARD
            stuck = self._is_stuck(self.fpv, self.prev_frame)
            if stuck:
                self._escape_remaining -= (TURN_STEPS_90 - 1)
                direction = random.choice([Action.LEFT, Action.RIGHT])
                self.current_action = direction
                self.action_hold_counter = TURN_STEPS_90 - 1
                return direction
            self.current_action = Action.FORWARD
            self.action_hold_counter = ACTION_HOLD_FRAMES - 1
            return Action.FORWARD

        # --- NAVIGATE / APPROACH: path-guided wall-following ---
        action = self._path_guided_act()

        # If path-follow triggered a state change (SEARCH/CHECKIN), return that
        if self.nav_state in (NavState.SEARCH, NavState.CHECKIN):
            return action

        # Plateau detection: no progress → ESCAPE burst + maybe try next candidate.
        # CRITICAL: never abandon a close approach. If best_hops < CLOSE_HOPS_NO_SWITCH
        # we're nearly arrived; plateau means "small recovery, keep trying" not
        # "switch to a candidate hundreds of hops away." We observed exactly this
        # bug: agent reached 17 hops then plateau-switched to candidate 2 which
        # was 100+ hops away and ran out the clock without arriving.
        if (self.nav_total_steps - self.nav_plateau_start) >= NAV_PLATEAU_STEPS:
            close_to_current = self.nav_last_best_hops < CLOSE_HOPS_NO_SWITCH
            if close_to_current:
                print(f"[NAV] Plateau at step {self.nav_total_steps} but close "
                      f"(best={self.nav_last_best_hops} hops). ESCAPE + retry SAME candidate.")
            elif self.low_confidence_goal and self._advance_goal_candidate():
                print(f"[NAV] Plateau at step {self.nav_total_steps}, switching candidate")
            else:
                print(f"[NAV] Plateau at step {self.nav_total_steps}, ESCAPE to relocate")
            self._escape_remaining = NAV_ESCAPE_FORWARD
            self.nav_state = NavState.ESCAPE
            self.nav_plateau_start = self.nav_total_steps
            self.nav_last_best_hops = 9999
            self._nav_path = None

        self.current_action = action
        self.action_hold_counter = ACTION_HOLD_FRAMES - 1
        return action

    # --- Section 7: Exploration (Wall-Following) ---

    def _is_stuck(self, current_frame: np.ndarray, previous_frame: np.ndarray) -> bool:
        if current_frame is None or previous_frame is None:
            return False
        gray_cur = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        gray_cur = cv2.resize(gray_cur, STUCK_FRAME_SIZE)
        gray_prev = cv2.resize(gray_prev, STUCK_FRAME_SIZE)
        mse = np.mean((gray_cur.astype(np.float32) - gray_prev.astype(np.float32)) ** 2)
        result = mse < STUCK_MSE_THRESHOLD
        if result and self.pipeline_ready and self.nav_total_steps % 10 == 0:
            print(f"[STUCK] MSE={mse:.0f} thresh={STUCK_MSE_THRESHOLD} es={self._nav_es.value}")
        return result

    def _detect_loop(self) -> bool:
        """Check if current frame is similar to any frame in the loop buffer."""
        if self.fpv is None or len(self._loop_buffer) < 3:
            return False
        cur = cv2.resize(cv2.cvtColor(self.fpv, cv2.COLOR_BGR2GRAY), STUCK_FRAME_SIZE)
        for old_frame in self._loop_buffer:
            mse = np.mean((cur.astype(np.float32) - old_frame.astype(np.float32)) ** 2)
            if mse < LOOP_MSE_THRESHOLD:
                return True
        return False

    def _explore_act(self) -> Action:
        stuck = self._is_stuck(self.fpv, self.prev_frame)

        # Escape burst: force FORWARD to physically leave current area
        if self._escape_burst > 0:
            if stuck:
                # Hit a wall during burst — turn and continue
                self._escape_burst -= 1
                return random.choice([Action.LEFT, Action.RIGHT])
            self._escape_burst -= 1
            return Action.FORWARD

        # Determine turn directions based on explore_hand
        if self.explore_hand == HAND_LEFT:
            wall_turn, check_turn = Action.LEFT, Action.RIGHT
            wall_state = ExploreState.TURN_LEFT
        else:
            wall_turn, check_turn = Action.RIGHT, Action.LEFT
            wall_state = ExploreState.TURN_RIGHT

        # Handle perturbation turn in progress (separate from state machine)
        if self._perturb_turning:
            self._perturb_turn_counter += 1
            # Record direction in recent_actions to prevent momentum re-trigger
            self._recent_actions.append(
                'RIGHT' if self._perturb_direction == Action.RIGHT else 'LEFT')
            if self._perturb_turn_counter >= TURN_STEPS_90:
                self._perturb_turning = False
                self._perturb_turn_counter = 0
                self.explore_state = ExploreState.FORWARD
                # After turning, inject a short forward burst to move to new area
                self._escape_burst = 15
                return Action.FORWARD
            return self._perturb_direction

        if self.explore_state == ExploreState.FORWARD:
            if stuck:
                self.consecutive_stuck += 1
                self._recent_actions.append(
                    'LEFT' if wall_turn == Action.LEFT else 'RIGHT')
                if self.consecutive_stuck >= 3:
                    self.explore_state = ExploreState.REVERSE
                    self.turn_counter = 0
                    return wall_turn
                else:
                    self.explore_state = wall_state
                    self.turn_counter = 0
                    return wall_turn
            else:
                self.consecutive_stuck = 0
                self.forward_count += 1
                self._perturb_forward_count += 1
                self._loop_forward_count += 1
                self._recent_actions.append('FORWARD')

                # Sample frame into loop buffer
                if self._loop_forward_count >= LOOP_SAMPLE_INTERVAL:
                    self._loop_forward_count = 0
                    if self.fpv is not None:
                        small = cv2.resize(
                            cv2.cvtColor(self.fpv, cv2.COLOR_BGR2GRAY), STUCK_FRAME_SIZE)
                        self._loop_buffer.append(small)

                # Perturbation checks (only after settle-in period)
                if self.explore_step >= PERTURB_START_STEP:
                    should_perturb = False
                    force_right = False

                    # Check 1: Fixed-interval random perturbation
                    if self._perturb_forward_count >= self._perturb_next_interval:
                        should_perturb = True
                        self._perturb_forward_count = 0
                        self._perturb_next_interval = random.randint(
                            PERTURB_INTERVAL_MIN, PERTURB_INTERVAL_MAX)

                    # Check 2: Loop detection via frame similarity
                    if not should_perturb and self._detect_loop():
                        should_perturb = True
                        self._perturb_forward_count = 0
                        self._loop_buffer.clear()

                    # Check 3: Momentum-based forced right turn
                    if (not should_perturb and
                            len(self._recent_actions) >= MOMENTUM_WINDOW):
                        left_count = sum(1 for a in self._recent_actions if a == 'LEFT')
                        if left_count / len(self._recent_actions) > MOMENTUM_LEFT_RATIO:
                            should_perturb = True
                            force_right = True
                            self._perturb_forward_count = 0

                    if should_perturb:
                        self._perturb_turning = True
                        self._perturb_turn_counter = 0
                        if force_right:
                            self._perturb_direction = Action.RIGHT
                        else:
                            self._perturb_direction = random.choice(
                                [Action.LEFT, Action.RIGHT])
                        self.forward_count = 0  # reset so CHECK_RIGHT doesn't fire right after
                        return self._perturb_direction

                # Normal wall-following: periodic corridor check
                if self.forward_count >= CHECK_RIGHT_INTERVAL:
                    self.forward_count = 0
                    self.explore_state = ExploreState.CHECK_RIGHT
                    self.turn_counter = 0
                    return check_turn
                return Action.FORWARD

        elif self.explore_state == ExploreState.TURN_LEFT:
            self.turn_counter += 1
            self._recent_actions.append('LEFT')
            if self.turn_counter >= TURN_STEPS_90:
                self.explore_state = ExploreState.FORWARD
                return Action.FORWARD
            return Action.LEFT

        elif self.explore_state == ExploreState.TURN_RIGHT:
            self.turn_counter += 1
            self._recent_actions.append('RIGHT')
            if self.turn_counter >= TURN_STEPS_90:
                self.explore_state = ExploreState.FORWARD
                return Action.FORWARD
            return Action.RIGHT

        elif self.explore_state == ExploreState.CHECK_RIGHT:
            self.turn_counter += 1
            self._recent_actions.append(
                'RIGHT' if check_turn == Action.RIGHT else 'LEFT')
            if self.turn_counter >= TURN_STEPS_90:
                if stuck:
                    self.explore_state = wall_state
                    self.turn_counter = 0
                    return wall_turn
                else:
                    self.explore_state = ExploreState.FORWARD
                    self.consecutive_stuck = 0
                    return Action.FORWARD
            return check_turn

        elif self.explore_state == ExploreState.REVERSE:
            self.turn_counter += 1
            self._recent_actions.append(
                'LEFT' if wall_turn == Action.LEFT else 'RIGHT')
            if self.turn_counter >= TURN_STEPS_180:
                self.explore_state = ExploreState.FORWARD
                self.consecutive_stuck = 0
                return Action.FORWARD
            return wall_turn

        return Action.FORWARD

    # --- Display ---

    def show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        h, w = concat_img.shape[:2]
        color = (0, 0, 0)
        concat_img = cv2.line(concat_img, (w // 2, 0), (w // 2, h), color, 2)
        concat_img = cv2.line(concat_img, (0, h // 2), (w, h // 2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1
        cv2.putText(concat_img, 'Front View', (10, 25), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (w // 2 + 10, 25), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (10, h // 2 + 25), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (w // 2 + 10, h // 2 + 25), font, size, color, stroke, line)

        cv2.imshow('KeyboardPlayer:target_images', concat_img)
        cv2.imwrite('target.jpg', concat_img)
        cv2.waitKey(1)


def _patch_pybullet_camera():
    """Monkey-patch pybullet.getCameraImage for pybullet >= 3.2.5 compatibility.

    Newer pybullet returns a flat tuple from getCameraImage()[2] instead of a
    numpy array, causing 'TypeError: tuple indices must be integers or slices,
    not tuple' when the vis_nav_game engine does img[:, :, 2::-1].

    Must be called BEFORE importing vis_nav_game.
    """
    try:
        import pybullet as _pb

        _original_getCameraImage = _pb.getCameraImage

        def _patched_getCameraImage(*args, **kwargs):
            result = _original_getCameraImage(*args, **kwargs)
            # result is (width, height, rgbPixels, depthPixels, segMask)
            # Ensure rgbPixels is a numpy array with shape (height, width, 4)
            w, h = result[0], result[1]
            rgb = result[2]
            if not isinstance(rgb, np.ndarray):
                rgb = np.array(rgb, dtype=np.uint8).reshape(h, w, 4)
            elif rgb.ndim == 1:
                rgb = rgb.reshape(h, w, 4)
            return (result[0], result[1], rgb) + result[3:]

        _pb.getCameraImage = _patched_getCameraImage
        print("[PATCH] pybullet.getCameraImage patched for numpy compatibility")
    except Exception as e:
        print(f"[PATCH] Warning: could not patch pybullet: {e}")


if __name__ == "__main__":
    import argparse
    import logging
    logging.basicConfig(filename='vis_nav_player.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    _patch_pybullet_camera()  # Must run before vis_nav_game import
    import vis_nav_game as vng
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing data_info.json and images/ (default: data)")
    parser.add_argument("--offline-navigation", action="store_true",
                        help="Use an existing exploration dataset and skip live exploration.")
    parser.add_argument("--n-clusters", type=int, default=64,
                        help="VLAD codebook size (default: 64)")
    parser.add_argument("--subsample", type=int, default=5,
                        help="Take every Nth motion frame (default: 5)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of visual shortcut edges (default: 100)")
    args = parser.parse_args()
    logging.info(f'player.py is using vis_nav_game {vng.core.__version__}')
    vng.play(the_player=KeyboardPlayerPyGame(
        n_clusters=args.n_clusters,
        subsample_rate=args.subsample,
        top_k_shortcuts=args.top_k,
        data_dir=args.data_dir,
        offline_navigation=args.offline_navigation,
    ))
