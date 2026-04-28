# CLAUDE.md — Visual Navigation Project

## Project Identity

- **Course**: NYU ROB-GY 6203 Robot Perception (Midterm Challenge)
- **Lab**: AI4CE Lab, NYU
- **Challenge Date**: April 1st, 2026, 10:00 AM - 8:00 PM
- **Task**: Build an autonomous visual navigation agent that explores a 3D maze using only RGB first-person vision, then navigates to a goal location identified by 4 target images
- **Constraint**: No GPS, no odometry, no depth sensor — only RGB frames

## Repository Structure

```
rob_cv_challenge/                    # Root project repo
├── CLAUDE.md                        # This file
├── INSTRUCTION.md                   # Challenge rules, rubric, deadlines
├── PLAN.md                          # Implementation approach analysis
├── TODO.md                          # Detailed implementation guide (all sections)
├── TODO_AUTO_EXPLORE.md             # Exploration automation design doc
├── README.md                        # Project overview and setup
├── ralph-wiggin.md                  # Ralph agent execution plan
├── .ralph/                          # Ralph agent config (DO NOT MODIFY)
│   ├── AGENT.md                     # Build/test/run instructions for Ralph
│   ├── PROMPT.md                    # Ralph development instructions
│   └── fix_plan.md                  # Ralph task tracking
├── .ralphrc                         # Ralph project config (DO NOT MODIFY)
├── .gitignore
├── .gitmodules                      # Points vis_nav_player to fork
└── vis_nav_player/                  # Git submodule (the actual game code)
    ├── source/
    │   ├── player.py                # THE PRIMARY FILE TO MODIFY
    │   ├── baseline.py              # Reference implementation (DO NOT MODIFY)
    │   └── test_pipeline.py         # Unit tests (23 tests)
    ├── data/
    │   ├── images/                  # Exploration frames (saved by player, NOT engine)
    │   └── data_info.json           # Frame metadata (step, image, action)
    ├── cache/                       # SIFT/codebook/VLAD caches
    ├── environment.yaml             # Conda environment definition
    ├── requirements.txt             # Pip dependencies
    ├── startup.json                 # Maze file IDs and MD5 keys
    └── target.jpg                   # Target images visualization
```

## Files to Modify

| File | Purpose |
|------|---------|
| `vis_nav_player/source/player.py` | **THE ONLY source file we modify and submit** |
| `vis_nav_player/source/test_pipeline.py` | Unit tests for the pipeline |

## Protected Files (DO NOT MODIFY)

- `vis_nav_player/source/baseline.py` — Reference implementation
- `vis_nav_game` package — Game engine (installed via pip)
- `.ralph/` directory and `.ralphrc` — Ralph agent configuration

## Environment Setup

```bash
# Create conda environment (one-time)
conda env create -f vis_nav_player/environment.yaml

# Activate before any work
conda activate game
```

### Dependencies (already in conda `game` env — DO NOT add new packages)

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.12 | Runtime |
| OpenCV | 4.8.0 | Image processing, SIFT feature extraction |
| scikit-learn | 1.3.0 | KMeans clustering |
| NetworkX | 3.4.2 | Graph construction and Dijkstra path planning |
| NumPy | 1.25.2 | Numerical operations |
| pygame | 2.5.1 | Display and keyboard input |
| pybullet | 3.24 | Physics engine (used by game) |
| vis-nav-game | (from test PyPI) | Game engine |

## Build and Run

```bash
# Run the game (auto-explore + auto-navigate)
cd vis_nav_player && python source/player.py

# Run unit tests
cd vis_nav_player && python source/test_pipeline.py

# Run baseline for comparison
cd vis_nav_player && python source/baseline.py
```

No build step — pure Python project.

## Game Engine API

```python
from vis_nav_game import Player, Action, Phase

# Actions:
Action.IDLE          # Do nothing
Action.FORWARD       # Move forward
Action.BACKWARD      # Move backward
Action.LEFT          # Rotate left IN PLACE (NOT strafing)
Action.RIGHT         # Rotate right IN PLACE (NOT strafing)
Action.CHECKIN       # Check in at goal location
Action.QUIT          # Quit game

# Phases:
Phase.EXPLORATION    # Agent collects images
Phase.NAVIGATION     # Agent navigates to target

# Player methods available via self:
self.get_state()                     # -> (bot_fpv, phase, step, time, fps, time_left)
self.get_camera_intrinsic_matrix()   # -> 3x3 numpy array K
self.get_target_images()             # -> list of 4 BGR images [front, left, back, right]
```

**Critical**: LEFT/RIGHT are **rotations in place**, not strafing. The game calls `act()` then `see()` each frame. `act()` must return an `Action` enum. `see(fpv)` receives the current first-person view BGR image.

## Critical: Game Engine Behavior

The game engine (`vis_nav_game`) has a `NAV_START_TIME` rule. When the current time is past `NAV_START_TIME`, **the engine skips the exploration phase entirely** and goes straight to navigation. The engine only calls `act()`/`see()` during `Phase.NAVIGATION`. The engine does NOT save exploration frames — the player must save them.

For the development maze, `NAV_START_TIME` is Oct 2024 (in the past), so exploration is always skipped. On challenge day (April 1, 2026), a new maze with a future `NAV_START_TIME` may enable real exploration.

**Our solution**: The player implements a **two-stage navigation** approach controlled by two flags: `pipeline_ready` (True when VLAD pipeline is built and navigation can begin) and `exploring_in_nav` (True when exploring during the navigation phase). When no exploration data exists, it first explores during the navigation phase (wall-following + frame saving for 5000 steps), then builds the VLAD pipeline and navigates.

## Architecture Overview

### Two-Stage Approach (handles both dev and challenge modes)

1. **If exploration data exists** (`data/data_info.json`): Build pipeline immediately in `pre_navigation()`, set `pipeline_ready=True`, start navigating
2. **If no exploration data**: Set `exploring_in_nav=True`, explore during navigation phase first (5000 frames via wall-following), save frames to disk, build pipeline, then navigate

### Exploration (Automated Wall-Following)

The agent moves through the maze automatically. **The player saves every frame** to `data/images/` and metadata to `data/data_info.json` (the engine does NOT do this). Wall detection uses frame-to-frame MSE comparison (if MSE < 100, the agent hit a wall).

**State machine** (`_explore_act()`):

| State | Behavior | Transition |
|-------|----------|------------|
| `FORWARD` | Move forward | Stuck -> `TURN_LEFT`; stuck 3x -> `REVERSE`; every 15 steps -> `CHECK_RIGHT` |
| `TURN_LEFT` | Rotate left 90deg (12 LEFT actions) | Complete -> `FORWARD` |
| `CHECK_RIGHT` | Rotate right 90deg to check for side corridors | Open -> `FORWARD`; wall -> `TURN_LEFT` |
| `REVERSE` | 180deg turn (24 LEFT actions) for dead ends | Complete -> `FORWARD` |
| `TURN_RIGHT` | Rotate right 90deg | Complete -> `FORWARD` |

**Keyboard override**: Pressing arrow keys during exploration switches to manual mode.

### Navigation (VLAD-Based Visual Place Recognition)

Pipeline:

1. **Data Loading** (`_load_trajectory_data`): Load `data_info.json`, filter to pure single-action motion frames, subsample every 5th frame
2. **SIFT Extraction** (`VLADExtractor.load_sift_cache`): Extract RootSIFT descriptors (L1-normalize + sqrt) from each subsampled image, cached to `cache/sift_ss5.pkl`
3. **Codebook** (`VLADExtractor.build_vocabulary`): KMeans(k=128) on all SIFT descriptors, cached to `cache/codebook_k128.pkl`
4. **VLAD Encoding** (`VLADExtractor.extract_batch`): Per-image VLAD vectors (128*128=16384 dims) with intra-normalization, power normalization, L2 normalization
5. **Graph Construction** (`_build_graph`): Temporal edges (consecutive frames, weight=1.0) + top-30 visual shortcut edges (similar but distant frames, weight=2.0+3.0*d)
6. **Goal Matching** (`_setup_goal`): Match all 4 target images via VLAD cosine similarity, average scores, find best node. Consensus check reports how many views agree. Sets `low_confidence_goal=True` if avg_sim < 0.25 (warns that target may not be in exploration data).
7. **Localization** (`_get_current_node`): Match live FPV to database via VLAD cosine similarity with **temporal smoothing** (0.6 current + 0.4 previous frame's similarities) and **jump detection** (rejects localization jumps > 50 nodes).
8. **Autonomous Navigation** (`_auto_navigate`): Re-localize -> Dijkstra path -> execute actions with hold-for-N-frames -> check-in when close. Check-in requires **both** proximity (hops <= 1) **and** visual similarity (> 0.25, or > 0.35 for low-confidence goals). Navigation has dual stuck detection: node-based (same node 5x) and frame-based (MSE < 100 for 10+ frames).

### Navigation Control Flow

```
act() called each frame
  -> If not pipeline_ready (still exploring):
       exploring_in_nav and explore_step >= EXPLORE_STEPS? -> build pipeline
       manual_mode? -> keyboard input
       else -> _explore_act() wall-following
  -> If pipeline_ready (navigating):
       action_hold_counter > 0? -> repeat last action
       else -> re-localize (smoothed) -> Dijkstra -> check-in (AND logic) -> execute action
```

### Key State Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `pipeline_ready` | bool | True when VLAD pipeline is built, controls explore vs navigate |
| `exploring_in_nav` | bool | True when exploring during navigation phase (engine skipped exploration) |
| `low_confidence_goal` | bool | True when goal avg_sim < 0.25, raises CHECKIN thresholds |
| `_prev_sims` | ndarray | Previous frame's similarity vector for localization smoothing |
| `_prev_node` | int | Previous localized node for jump detection |

## Key Constants

```python
# Exploration
TURN_STEPS_90 = 12              # LEFT/RIGHT actions for 90-degree turn (~7.5 deg/action)
TURN_STEPS_180 = 24             # Actions for 180-degree turn
CHECK_RIGHT_INTERVAL = 15       # Check right side every N forward steps
STUCK_MSE_THRESHOLD = 100       # MSE below this = stuck
STUCK_FRAME_SIZE = (80, 60)     # Resize for fast frame comparison
EXPLORE_STEPS = 5000            # Exploration budget when exploring during nav phase

# Navigation
REPLAN_INTERVAL = 10            # Re-localize every N frames
CHECKIN_THRESHOLD = 3           # Check in when within N hops
STUCK_THRESHOLD = 5             # Same node N times -> stuck
ACTION_HOLD_FRAMES = 5          # Hold each action for N frames (×2 for graph-stuck, ×3 for frame-stuck)

# Graph
TEMPORAL_WEIGHT = 1.0
VISUAL_WEIGHT_BASE = 2.0
VISUAL_WEIGHT_SCALE = 3.0
MIN_SHORTCUT_GAP = 50           # Min trajectory index gap for shortcuts
```

## Data Formats

### `data/data_info.json`
```json
[
  {"step": 0, "image": "000000.png", "action": ["FORWARD"]},
  {"step": 1, "image": "000001.png", "action": ["LEFT"]},
  {"step": 2, "image": "000002.png", "action": ["FORWARD", "LEFT"]}
]
```
- Only single pure-motion frames (`action` has exactly 1 element from {FORWARD, LEFT, RIGHT, BACKWARD}) are used
- Multi-action frames and CHECKIN/IDLE are filtered out

### VLAD Vectors
- Dimension: 16,384 (128 clusters * 128 SIFT descriptor dims)
- L2-normalized to unit vectors
- Similarity metric: cosine similarity (dot product)

## Submission Requirements (from INSTRUCTION.md)

| Deliverable | Format | Points |
|------------|--------|--------|
| Report | PDF (use provided template) | 3 pts |
| Game file (.npy) + code (.zip) | .npy saved to `./data/save` (press Space), code as .zip | 7 pts + 1 bonus |

### Competition Scoring

| Criterion | Points |
|-----------|--------|
| Participation | 1 pt |
| Position error < 0.1 | 2 pts |
| Position error 0.1-0.2 | 1 pt |
| Under 1 min navigation | 2 pts |
| Under 2 min | 1.5 pts |
| Under 5 min | 0.5 pts |
| Completely new solution | 2.0 pts |
| Modification of baseline | 1.5 pts |
| Baseline solution | 0.5 pts |
| **Fully automated solution (bonus)** | **1.0 pt** |

The .npy file is automatically saved to `./data/save` when finishing navigation (normally by pressing Space / CHECKIN).

### Evaluation Platform
- URL: https://ai4ce.github.io/vis_nav_player/
- Each student has a personal API key (sent via Slack DM)
- Platform verifies translation error and navigation time
- On challenge day: switch to a NEW maze with new exploration data

## Implementation Status

### Completed
- [x] VLADExtractor class (RootSIFT + VLAD encoding)
- [x] Data loading and subsampling
- [x] SIFT cache with disk persistence
- [x] KMeans codebook construction with caching
- [x] VLAD batch extraction
- [x] Graph construction (temporal + visual shortcut edges)
- [x] Multi-view goal matching (all 4 target images)
- [x] Localization via VLAD cosine similarity with **temporal smoothing** and **jump detection**
- [x] Dijkstra path planning
- [x] Autonomous navigation state machine with dual stuck detection (node + frame)
- [x] Visual similarity confirmation for check-in (**AND logic**, confidence-aware thresholds)
- [x] **Goal confidence detection** (`low_confidence_goal` flag when avg_sim < 0.25)
- [x] Frame-based stuck detection (MSE comparison)
- [x] Wall-following exploration state machine
- [x] Navigation frame-based stuck recovery
- [x] **Frame saving during exploration** (player saves to data/images/ and data_info.json)
- [x] **Two-stage navigation** (explore during nav phase when engine skips exploration, 5000 steps)
- [x] 23 unit tests all passing
- [x] **Full end-to-end integration test** — successfully explored, built pipeline, navigated, and checked in

### Known Bugs (Not Yet Fixed)
- [ ] **Bug 3**: `hops==0` at line 593 bypasses visual similarity check entirely (premature CHECKIN)
- [ ] **Bug 4**: `w, h` swapped in `show_target_images()` line 701 (cosmetic)

### Not Yet Implemented (see TODO.md for full details)
- [ ] **T1.1**: Remove VLAD power normalization (root cause of 0.046 adjacent similarity — SSR amplifies noise after intra-norm)
- [ ] **T1.2**: Random perturbation for exploration loop breaking (3-layer: interval + loop detect + momentum)
- [ ] **T1.3**: Reduce CHECK_RIGHT_INTERVAL from 15 to 8 (agent misses side corridors)
- [ ] **T1.4**: SIFT tuning (nfeatures=500, contrastThreshold=0.06, mask floor/ceiling, RootSIFT eps fix)
- [ ] **T1.5**: Reduce n_clusters from 128 to 64 (fewer sparse clusters = less noise)
- [ ] **T1.6**: Graph improvements (top_k=100, MIN_SHORTCUT_GAP=30, similarity floor 0.4)
- [ ] **T1.7**: Multi-view CHECKIN (all 4 target views) + variable action hold (1 near goal, 2 for turns)
- [ ] **T1.8**: Oscillation detection (2-node bounce in node_history)
- [ ] **T2.1**: Time budget manager (dynamic explore/navigate split using time_left)
- [ ] **T2.2**: Re-exploration mode for low-confidence goals (reversed wall-following + novelty detection)
- [ ] Turn angle auto-calibration (low urgency — current TURN_STEPS_90=12 works)

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Agent stands still during exploration | `_is_stuck()` always False, or `manual_mode` stuck True | Check STUCK_MSE_THRESHOLD, check `manual_mode` |
| Agent spins in circles | TURN_STEPS_90 wrong | Adjust up/down (default 12) |
| "No descriptors" crash | Not enough diverse exploration frames | Increase exploration time or check image saving |
| Goal consensus 0/4 | Exploration didn't cover goal area | Improve exploration coverage |
| Agent permanently stuck in navigation | Stuck recovery not triggering | Check nav_stuck_frames threshold (default 10) |
| CHECKIN too early | CHECKIN_THRESHOLD too high or localization error | Decrease threshold, raise visual sim requirement |
| KMeans too slow (>5 min) | Too many descriptors | Reduce n_clusters from 128 to 64 |
| Navigation oscillation | Agent goes back and forth | Increase ACTION_HOLD_FRAMES from 5 to 10 |
| Graph disconnected | Not enough visual shortcuts | Increase top_k_shortcuts from 30 to 50 |
| Memory error | Too many exploration images | Increase subsample_rate from 5 to 10+ |

## Coding Conventions

- All game logic is in a single class `KeyboardPlayerPyGame(Player)` in `player.py`
- Constants are module-level ALL_CAPS
- Methods prefixed with `_` are internal
- Caches go to `cache/` directory with descriptive filenames
- Print statements for pipeline progress (used as console logging)
- `ExploreState` enum for exploration state machine states
- Type hints on method signatures

## Challenge Day Workflow

On April 1st, the maze will change. The pipeline must handle switching to a new maze:

1. If exploration phase runs (new `NAV_START_TIME`): wall-following explores, frames saved by player to `data/images/`, pipeline built in `pre_navigation()`
2. If exploration skipped (past `NAV_START_TIME`): two-stage approach kicks in — explores during nav phase for 5000 steps, then builds pipeline
3. **Player saves frames** — the engine does NOT save them
4. SIFT/codebook caches will be rebuilt (new images, old caches don't match)
5. Navigation runs autonomously with localization smoothing and confidence-aware check-in

**Before challenge day**: Delete `cache/`, `data/images/`, and `data/data_info.json` to ensure fresh data for the new maze.

Use the "MidTerm Dry Run" on the evaluation platform to test maze switching before the challenge day.
