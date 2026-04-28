# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NYU ROB-UY 3203 **Final Challenge (Spring 2026)** — autonomous visual navigation in a 51×51 procedurally-generated maze using only RGB first-person frames. Submission is a single `Player` subclass evaluated by the `vis_nav_game` engine. See [FINAL_INSTRUCTION.md](FINAL_INSTRUCTION.md) for goals, deadlines, and the diff vs. the midterm. Architectural details from the midterm (still applicable) are in [docs/CLAUDE_midterm_reference.md](docs/CLAUDE_midterm_reference.md) — read it before substantive changes to the player.

## Common commands

All commands run from `final_challenge/vis_nav_player/`.

```bash
# Pinned engine version (different in 1.2.6 from system pip default)
pip install "vis_nav_game==1.2.6" --extra-index-url https://test.pypi.org/simple/

# Live game — opens a pygame window, plays the full game
python source/player.py                       # explore + navigate
python source/player.py --offline-navigation  # use existing data/, skip exploration

# Tunable hyperparameters
python source/player.py --offline-navigation --n-clusters 64 --subsample 5 --top-k 100

# Full unit test suite (43 tests; engine not required)
python source/test_pipeline.py

# Offline integration tests against the real exploration dataset
SDL_VIDEODRIVER=dummy python source/test_offline_navigation.py
SDL_VIDEODRIVER=dummy python source/test_offline_baseline.py

# Run a single test by importing it
python -c "import sys; sys.path.insert(0,'source'); from test_offline_navigation import test_load_trajectory_data_filters_idle as t; t()"

# Reference baseline (do not modify)
python source/baseline.py
```

`SDL_VIDEODRIVER=dummy` is required when running in headless contexts — the player imports `pygame` at module load.

## Data layout (load-bearing)

```
final_challenge/
├── exploration_data/                  # NOT committed (.gitignored, ~22k images)
│   ├── traj_0/
│   │   ├── data_info.json             # ground truth — list of {step, image, action[]}
│   │   └── *.jpg                      # FLAT, NOT in an images/ subdir
│   ├── data_info.json -> traj_0/data_info.json   # symlinks expected by tests
│   └── images -> traj_0
└── vis_nav_player/
    ├── data/                          # gitignored
    │   ├── data_info.json -> ../../exploration_data/traj_0/data_info.json
    │   └── images -> ../../exploration_data/traj_0
    ├── exploration_data -> ../exploration_data    # what test_offline_*.py expects
    └── cache/                         # gitignored, regenerated SIFT/codebook/VLAD
```

Two distinct path conventions exist:
- **Player code** (`player.py`) reads `<data_dir>/data_info.json` and `<data_dir>/images/<filename>` — `--data-dir` defaults to `data`.
- **Offline test scripts** read from `vis_nav_player/exploration_data/data_info.json` directly (top-level, no `images/` subdir indirection).

Both must work — keep both symlink chains intact. If running on a fresh machine: download the dataset from the dry-run platform, drop into `final_challenge/exploration_data/traj_0/`, then recreate the symlinks above.

When switching mazes (challenge day): `rm -rf vis_nav_player/cache vis_nav_player/data` so SIFT/codebook caches don't poison the new run. Cache filenames are MD5-hashed per `data_dir`, so stale caches *should* be ignored, but it's safer to wipe.

## Architecture

The submission is a single class `KeyboardPlayerPyGame(Player)` in [vis_nav_player/source/player.py](vis_nav_player/source/player.py). It overrides four engine hooks: `reset`, `pre_navigation`, `act`, `see`. The engine calls `see(fpv)` then `act()` each frame.

### Two-stage pipeline

The engine has a `NAV_START_TIME` constant: if "now" is past it, the engine **skips exploration entirely** and goes straight to navigation. The player handles both modes via two flags:

- `pipeline_ready` — VLAD database + graph + goal node all ready.
- `exploring_in_nav` — engine skipped exploration, so we wall-follow during the nav phase first (saves frames to `data/images/` ourselves; the engine does not save them), then build the pipeline at step `EXPLORE_STEPS` (5000).

If `data/data_info.json` already exists (e.g. `--offline-navigation`), pipeline is built in `pre_navigation()` and exploration is skipped.

### VLAD pipeline (`_build_pipeline`)

1. **`_load_trajectory_data`** — read `data_info.json`, keep only single-action pure-motion frames (`{FORWARD, LEFT, RIGHT, BACKWARD}`), subsample every Nth (default 5).
2. **`VLADExtractor.load_sift_cache`** — RootSIFT (L1-normalize then sqrt), cached `.pkl` per `(data_dir_md5, subsample)`.
3. **`VLADExtractor.build_vocabulary`** — `MiniBatchKMeans(k=n_clusters)` over all descriptors, cached.
4. **`VLADExtractor.extract_batch`** — VLAD aggregation → intra-norm → power-norm → L2-norm. Vector dim = `n_clusters * 128`.
5. **`_build_graph`** — NetworkX undirected graph. Temporal edges (consecutive frames, weight=`TEMPORAL_WEIGHT`) + top-K visual shortcut edges between similar but trajectory-distant frames (`MIN_SHORTCUT_GAP=30`, weight scales with cosine distance). On a healthy dataset this yields a single connected component.
6. **`_setup_goal`** — match all 4 target images via VLAD cosine similarity, average scores, pick best node. `low_confidence_goal=True` if avg_sim < 0.25 raises check-in thresholds.

Caching note: SIFT descriptors and the codebook are persisted to `cache/` via Python's `.pkl` format — this is trusted local data only, same as the reference baseline. Do not load `.pkl` files from untrusted sources.

### Navigation state machine (`_auto_navigate`, `act`)

`NavState` enum: `NAVIGATE` (wall-follow + Dijkstra) → `APPROACH` (near goal) → `SEARCH` (360° scan) → `CHECKIN` (submit) → `ESCAPE` (forward burst when stuck).

- `_get_current_node` localizes via VLAD cosine similarity with **temporal smoothing** (0.6 current + 0.4 prev) and **jump rejection** (>50 nodes between consecutive localizations is suppressed). Caches `_last_vlad` for the lucky-pass CHECKIN backup.
- `_should_enter_approach` gates the `NAVIGATE → APPROACH` transition using hop-count gradient over `hop_history`. **There are two known-failing tests around this method** — see "Known issues" below.
- Check-in requires **both** proximity (Dijkstra hops ≤ threshold) **and** visual similarity (>0.25, or >0.35 if `low_confidence_goal`). Dual stuck detection: same node 5× → `ESCAPE`; frame MSE < 100 for 10+ frames → forward burst.

### Module-level constants

Tunables live as module-level ALL_CAPS in [player.py:50-130](vis_nav_player/source/player.py). Modify here rather than passing through CLI. `EXPLORE_STEPS=5000` is the budget when exploring during nav phase — likely too small for a 51×51 maze, see open TODOs.

### `_patch_pybullet_camera`

Monkey-patches `pybullet.getCameraImage` for pybullet ≥ 3.2.5 compatibility — newer pybullet returns a flat tuple instead of a numpy array, breaking the engine's `img[:, :, 2::-1]` slice. Must run before `import vis_nav_game`. Already wired into `__main__`.

## Submodule history

`vis_nav_player/` was a git submodule in the midterm repo (origin: https://github.com/tzuiyang/vis_nav_player). It was flattened to a regular directory when copied into this repo on 2026-04-27 — the `.gitmodules` and submodule pointer are gone. Treat it as part of this tree.

## Player file inventory

- `source/player.py` — primary submission (1524 lines, full pipeline)
- `source/baseline.py` — reference implementation, **do not modify**
- `source/player1.py`, `source/player2.py` — alternate experiments from midterm; not currently in the build path
- `source/test_pipeline.py` — 43 unit tests covering VLAD, graph, state machines (no engine required)
- `source/test_offline_*.py` — integration tests against the real `exploration_data/` dataset
- `source/test_search_state.py` — focused on `SEARCH` state transitions
- `source/test_e2e_nav.py` — end-to-end navigation harness

## Known issues (carried from midterm)

- `test_offline_navigation.py` reports 3 stable failures regardless of dataset: `test_player_offline_paths_resolve_correctly` (symlink vs. `Path.resolve()`), `test_low_confidence_enters_approach_after_stable_gradient_checks`, `test_high_confidence_behavior_still_allows_normal_approach_transition`. The latter two are real `_should_enter_approach` threshold bugs — fix is on the open TODO list, but be aware they were already failing on the midterm code.
- Bug 3 (in midterm reference): `hops==0` at line 593 bypasses visual similarity check (premature CHECKIN).
- Bug 4: `w, h` swapped in `show_target_images()` line 701 (cosmetic).

Open improvement TODOs (T1.1–T2.2) are catalogued in [docs/TODO_midterm.md](docs/TODO_midterm.md) and summarized in [FINAL_INSTRUCTION.md](FINAL_INSTRUCTION.md).

## Conventions

- Caches go in `cache/` with descriptive names (`sift_<hash>_ss<N>.pkl`, `codebook_k<K>.pkl`).
- Internal methods prefixed `_`.
- Print statements (not logging) for pipeline progress — these become the console output during the live game.
- `ExploreState` and `NavState` enums own state-machine values; don't introduce string states.
