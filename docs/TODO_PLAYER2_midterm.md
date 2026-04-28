# TODO_PLAYER2 — Manual Navigation Player with Real-Time HUD, Bird's Eye Map & Speedrun Replay

## Context

**Why**: On challenge day (today), we need a manual-control player that gives real-time visual feedback so the human driver can navigate efficiently. After reaching the target once, we want to replay the exact action sequence as a "speedrun" for a faster time.

**What player2.py adds over player1.py**:
1. Always-on HUD overlay showing distance/hops/trend (no need to press Q)
2. Bird's eye map window showing dead-reckoned exploration path + current position + Dijkstra route
3. Action recording during manual navigation + speedrun replay mode

**Base**: Copy VLADExtractor, data loading, graph construction, goal matching from player1.py. Add temporal smoothing localization from player.py.

---

## Architecture Overview

```
player2.py (~650-750 lines)
├── Imports + Constants
├── VLADExtractor class          (copied from player1.py, unchanged)
├── KeyboardPlayerPyGame(Player)
│   ├── __init__()               Data loading + new state vars
│   ├── reset()                  Pygame init + keymap
│   ├── act()                    Keyboard OR speedrun replay + action recording
│   ├── see()                    Periodic localization + HUD overlay + map update
│   ├── pre_exploration()        Build pipeline (if exploration phase runs)
│   ├── pre_navigation()         Build pipeline + dead-reckon map + setup goal
│   ├── _build_database()        SIFT → KMeans → VLAD batch (from player1.py)
│   ├── _build_graph()           Temporal + visual shortcut edges (from player1.py)
│   ├── _setup_goal()            Multi-view VLAD goal matching (from player1.py)
│   ├── _get_current_node()      Localization WITH temporal smoothing + jump detection
│   ├── _get_path()              Dijkstra shortest path (from player1.py)
│   ├── _dead_reckon_map()       NEW: Compute (x,y) for all nodes from action sequence
│   ├── _render_base_map()       NEW: Pre-render gray dots + goal star on black canvas
│   ├── _draw_hud()              NEW: Overlay distance/hops/trend on FPV image
│   ├── _update_map()            NEW: Draw current pos + Dijkstra path on bird's eye map
│   ├── _record_action()         NEW: Append action to recording list
│   ├── _save_recording()        NEW: Write action sequence to JSON
│   ├── _load_replay()           NEW: Load action sequence for speedrun
│   ├── show_target_images()     (from player1.py)
│   └── display_next_best_view() REMOVED (replaced by always-on HUD)
└── main block                   Pybullet patch + argparse (--replay flag)
```

---

## Task Breakdown

### T1: Scaffold — Copy reusable code from player1.py

Copy verbatim (with minor constant changes):
- Imports and constants (lines 1-26), change `DATA_DIR` to `"exploration_data_real"`
- `VLADExtractor` class (lines 31-132)
- `__init__` data loading (lines 140-222)
- `reset()` (lines 228-240)
- `_build_database()`, `_build_graph()`, `_setup_goal()`, `_get_path()` (lines 300-387)
- `show_target_images()` (lines 390-405)
- Main block with pybullet patch + argparse (lines 437-466)

New constants to add:
```python
LOCALIZE_INTERVAL = 10       # Re-localize every N frames (~2x/sec at 20 FPS)
TREND_WINDOW = 5             # Rolling window for hop trend detection
MAP_SIZE = 600               # Bird's eye map canvas (pixels)
MAP_PADDING = 40             # Padding inside map edges
TURN_ANGLE_DEG = 7.5         # Degrees per LEFT/RIGHT action
REPLAY_FILE = "speedrun.json"
```

---

### T2: New state variables in `__init__`

After existing data loading, add:
```python
# Localization (always-on HUD)
self._frame_counter = 0
self._cached_node = None
self._cached_hops = None
self._cached_dist = None
self._hops_history = []          # last TREND_WINDOW hop values
self._prev_sims = None           # temporal smoothing (from player.py)
self._prev_node = None           # jump detection
self._last_sims = None           # cached similarity vector

# Bird's eye map
self._node_xy = None             # ndarray (N, 2) pixel coords
self._map_base = None            # pre-rendered base map image
self._all_raw_actions = []       # full unsubsampled action list for dead reckoning

# Path recording + speedrun
self._recording = []             # list of Action int values
self._replay_actions = None      # loaded replay (None = manual mode)
self._replay_index = 0
self._speedrun_mode = False
```

Also store raw actions for dead reckoning: load ALL entries from data_info.json (not just pure-motion subsampled ones) to reconstruct the full trajectory.

---

### T3: Localization with temporal smoothing — `_get_current_node()`

Port from player.py (lines 845-868) with one addition: cache `self._last_sims` to avoid double VLAD extraction when computing goal distance.

```python
def _get_current_node(self) -> int:
    feat = self.extractor.extract(self.fpv)
    sims = self.database @ feat
    # Temporal smoothing: 95% current + 5% previous
    if self._prev_sims is not None:
        sims = 0.95 * sims + 0.05 * self._prev_sims
    self._prev_sims = sims.copy()
    cur = int(np.argmax(sims))
    # Jump detection: reject jumps > 150 nodes unless significantly better
    if self._prev_node is not None:
        if abs(cur - self._prev_node) > 150:
            lo = max(0, self._prev_node - 80)
            hi = min(len(sims), self._prev_node + 81)
            local_cur = lo + int(np.argmax(sims[lo:hi]))
            if float(sims[local_cur]) >= float(sims[cur]) - 0.05:
                cur = local_cur
    self._prev_node = cur
    self._last_sims = sims  # cache for goal dist
    return cur
```

---

### T4: Dead reckoning map — `_dead_reckon_map()` + `_render_base_map()`

Called once in `pre_navigation()` after pipeline is built.

**`_dead_reckon_map()`**:
1. Walk through ALL raw actions (not subsampled) from data_info.json
2. Maintain `(x, y, theta)`. For each frame:
   - `FORWARD`: x += cos(theta), y += sin(theta)
   - `BACKWARD`: x -= cos(theta), y -= sin(theta)
   - `LEFT`: theta += 7.5 * pi/180
   - `RIGHT`: theta -= 7.5 * pi/180
   - `IDLE`: no change
3. Record (x, y) at every frame's step index
4. Map subsampled motion_frames to their dead-reckoned positions via step index
5. Normalize all positions to fit MAP_SIZE with MAP_PADDING
6. Store as `self._node_xy` (N x 2 array of pixel coordinates)

**`_render_base_map()`**:
1. Create black canvas (MAP_SIZE x MAP_SIZE x 3)
2. Draw all node positions as small gray dots (radius=1)
3. Draw goal node as a red star/circle (radius=6)
4. Store as `self._map_base`

---

### T5: Always-on HUD overlay — `_draw_hud(display_fpv)`

Called every frame in `see()`. Uses cached localization values (updated every LOCALIZE_INTERVAL frames).

**Layout** (bottom of FPV, semi-transparent dark bar):
```
┌─────────────────────────────────────────────┐
│  Hops: 23  │  Dist: 0.412  │  > CLOSER     │  <- green/red/yellow
│  Node: 1234 / 6896  │  Goal: 4567          │
└─────────────────────────────────────────────┘
```

**Trend logic**:
- Compare current hops to mean of `_hops_history` (last TREND_WINDOW values)
- Hops decreased by 2+ → GREEN "CLOSER" with up-arrow
- Hops increased by 2+ → RED "FURTHER" with down-arrow
- Otherwise → YELLOW "STABLE"

**Near-goal alert**: When hops <= 5, show large red text at top: `"NEAR TARGET -- PRESS SPACE"`

**Implementation**:
- Draw semi-transparent dark rectangle at bottom using `cv2.addWeighted` on sub-region
- Draw text with `cv2.putText`

---

### T6: Bird's eye map window — `_update_map(path)`

Called every LOCALIZE_INTERVAL frames alongside localization. Separate CV2 window "Bird's Eye Map".

1. Copy `self._map_base`
2. Draw Dijkstra path as green polyline connecting node positions
3. Draw current position as bright cyan circle (radius=5)
4. Draw trail of recent positions (last 10 localizations) as fading cyan dots
5. `cv2.imshow("Bird's Eye Map", img)` + `cv2.waitKey(1)`

---

### T7: Modified `see()` — orchestration

```python
def see(self, fpv):
    if fpv is None or len(fpv.shape) < 3:
        return
    self.fpv = fpv
    display_fpv = fpv.copy()

    if self._state and self._state[1] == Phase.NAVIGATION:
        self._frame_counter += 1
        # Periodic re-localization (every 10 frames)
        if (self._frame_counter % LOCALIZE_INTERVAL == 0
                and self.database is not None and self.goal_node is not None):
            cur = self._get_current_node()
            path = self._get_path(cur)
            hops = len(path) - 1
            dist = float(np.sqrt(max(0, 2 - 2 * float(self._last_sims[self.goal_node]))))
            self._cached_node = cur
            self._cached_hops = hops
            self._cached_dist = dist
            self._hops_history.append(hops)
            if len(self._hops_history) > TREND_WINDOW:
                self._hops_history.pop(0)
            self._update_map(path)
        # Always draw HUD (uses cached values)
        if self._cached_hops is not None:
            display_fpv = self._draw_hud(display_fpv)

    # Render pygame
    ...
```

Key optimization: `_get_current_node()` caches `self._last_sims`, so goal distance is computed from cached sims (no second VLAD extraction). Total amortized cost: ~5ms/frame.

---

### T8: Modified `act()` — keyboard + recording + replay

```python
def act(self):
    # Speedrun replay mode: return pre-recorded actions
    if self._speedrun_mode and self._replay_actions:
        if self._replay_index < len(self._replay_actions):
            action = Action(self._replay_actions[self._replay_index])
            self._replay_index += 1
            return action
        else:
            self._speedrun_mode = False
            print("Replay complete!")

    # Normal keyboard handling (from player1.py)
    for event in pygame.event.get():
        ... (keydown/keyup with bitwise OR/XOR)

    action = self.last_act
    # Record action during navigation
    if self._state and self._state[1] == Phase.NAVIGATION and not self._speedrun_mode:
        self._recording.append(int(action))
        if action & Action.CHECKIN:
            self._save_recording()
    return action
```

---

### T9: Recording save/load — `_save_recording()` + `_load_replay()`

**Save** (auto-triggered on CHECKIN):
```python
def _save_recording(self):
    data = {
        "actions": self._recording,
        "goal_node": self.goal_node,
        "total_frames": len(self._recording),
    }
    with open(REPLAY_FILE, "w") as f:
        json.dump(data, f)
```

**Load** (via --replay CLI flag):
```python
def _load_replay(self, path):
    with open(path) as f:
        data = json.load(f)
    self._replay_actions = data["actions"]
    self._speedrun_mode = True
```

---

### T10: Main block + argparse

Add `--replay` flag:
```python
parser.add_argument("--replay", type=str, default=None,
                    help="Path to speedrun JSON for auto-replay")
```

Include pybullet monkey-patch (from player1.py lines 443-452).

---

## Performance Budget

| Operation | Cost | Frequency | Per-frame amortized |
|-----------|------|-----------|-------------------|
| VLAD extract | ~50ms | Every 10 frames | ~5ms |
| DB dot product | ~2ms | Every 10 frames | ~0.2ms |
| Dijkstra | ~1ms | Every 10 frames | ~0.1ms |
| HUD overlay | <1ms | Every frame | <1ms |
| Map render | <1ms | Every 10 frames | <0.1ms |
| Action record | ~0us | Every frame | ~0 |
| **Total** | | | **~6ms** |

At 20 FPS (50ms/frame budget), this leaves ~44ms for the game engine.

---

## How to Run

```bash
cd vis_nav_player

# Manual navigation with HUD + map
python source/player2.py

# Tune parameters
python source/player2.py --subsample 2 --n-clusters 128 --top-k 200

# Speedrun replay (after reaching goal once)
python source/player2.py --replay speedrun.json
```

---

## Verification

1. Run `python source/player2.py` — confirm pipeline builds, FPV window shows HUD overlay, Bird's Eye Map window opens
2. Navigate with arrow keys — confirm hops/dist update every ~0.5s, trend shows green/red/yellow correctly
3. Press Space near goal — confirm `speedrun.json` is saved with action list
4. Run `python source/player2.py --replay speedrun.json` — confirm actions replay automatically
5. Verify map shows current position moving along the dead-reckoned path
