# ROB-UY 3203 — Final Challenge (Spring 2026)

> **Source repo (midterm baseline):** https://github.com/tzuiyang/rob_cv_challenge
> Files copied into [vis_nav_player/](vis_nav_player/) and [docs/](docs/) on 2026-04-27.

## What's changing vs. the midterm

| Item | Midterm | Final |
|---|---|---|
| Game engine | `vis_nav_game==1.2.5` | **`vis_nav_game==1.2.6`** (must upgrade) |
| Maze size | 31×31 | **51×51** (~2.7× more cells) |
| Exploration sequences provided | 1 | 1 |
| Goal threshold | Position error <0.1 m | **Perfect <0.1 m · Partial ≤0.2 m** |
| Challenge date | 2026-04-01 | **2026-04-29** |
| Dry-run deadline | n/a | **2026-04-28 23:59** |
| Grading rubric | (see [docs/INSTRUCTION_midterm.md](docs/INSTRUCTION_midterm.md)) | TBD — released ~1 week before final |
| Dry-run platform | https://ai4ce.github.io/vis_nav_player/ | https://ai4ce.github.io/vis_nav_player/#/challenges/311f2b2f556f |

## Top-level goals (in priority order)

1. **Submit something to the dry-run before 2026-04-28 23:59** so we know the pipeline runs end-to-end on the new 51×51 maze and 1.2.6 engine.
2. **Hit Perfect goal (<0.1 m translation error)** consistently — partial (≤0.2 m) is the floor.
3. **Bring nav time down** — Team 12 (rank #1) did 18.8 s with 0.0947 m. The TA baseline is 304 s with 0.0344 m. We don't need to win, but staying under ~2 min keeps us in any time-based rubric tier.
4. **Stay fully automated** — midterm rubric gave +1 pt bonus for full automation; assume the final keeps it.
5. **Watch out for new bugs** — the announcement says "we have fixed several bugs in the game engine" so 1.2.6 may behave differently from 1.2.5 (e.g., turn step calibration, NAV_START_TIME logic, frame saving).

## Setup checklist

- [ ] **Upgrade game engine**:
      ```bash
      pip install "vis_nav_game==1.2.6" --extra-index-url https://test.pypi.org/simple/
      ```
- [ ] **Verify** `python -c "import vis_nav_game; print(vis_nav_game.core.__version__)"` reports 1.2.6.
- [ ] **Verify `startup.json` keys** in [vis_nav_player/startup.json](vis_nav_player/startup.json) — values were transcribed from the dry-run page; re-check from the platform because OCR can confuse `0/O` and `1/I/l`. Currently saved:
  - `ESSENTIAL_FILE_ID`: `1CvIOxnKO8Z8NDBh-kKOpUuWEgPinLLSV`
  - `MAZE_FILE_ID`: `1vORG45yN2I_65Gr2rnIOWmizTFsnUMeO`
  - `MAZE_FILE_MD5_KEY`: `91c7829a684a1a21df2d31513f75f4O4` ← **suspicious**: MD5 is hex (0-9 a-f); the trailing `O` is almost certainly a `0`. Confirm against the dry-run page.
- [ ] **Download exploration data** from the dry-run page → unpack into `vis_nav_player/data/`.
- [ ] **Wipe stale caches from midterm**: `rm -rf vis_nav_player/cache vis_nav_player/data/images vis_nav_player/data/data_info.json` if reusing the dir.
- [ ] **Smoke test**: `cd vis_nav_player && python source/player.py --offline-navigation` and let it navigate end-to-end against the 51×51 dataset.

## What's already in `final_challenge/`

```
final_challenge/
├── FINAL_INSTRUCTION.md                # this file
├── vis_nav_player/
│   ├── source/
│   │   ├── player.py                   # primary submission — VLAD pipeline (the one we ship)
│   │   ├── baseline.py                 # reference (read-only)
│   │   ├── player1.py / player2.py     # alternate experiments from midterm
│   │   ├── test_pipeline.py            # 23 unit tests
│   │   ├── test_e2e_nav.py
│   │   ├── test_offline_baseline.py
│   │   ├── test_offline_navigation.py
│   │   └── test_search_state.py
│   ├── startup.json                    # dry-run keys (verify before running!)
│   ├── environment.yaml                # conda env definition
│   ├── requirements.txt
│   ├── .gitignore
│   └── README.md
└── docs/
    ├── CLAUDE_midterm_reference.md     # full midterm architecture writeup
    ├── INSTRUCTION_midterm.md          # midterm rubric (final TBD)
    ├── TODO_midterm.md                 # known-bugs / improvements list (T1.1–T2.2)
    ├── TODO_PLAYER2_midterm.md
    └── TODO_STUCK_NAV_midterm.md
```

The submodule pointer was dropped — `vis_nav_player/` is now a regular directory in this repo.

## Open work items (carried over from midterm — see [docs/TODO_midterm.md](docs/TODO_midterm.md))

These were unfinished going into the midterm and may matter more on a 51×51 maze:

- [ ] **T1.1**: Remove VLAD power normalization (root cause of ~0.046 adjacent-frame similarity).
- [ ] **T1.2**: Random perturbation for exploration loop-breaking (3-layer: interval + loop detect + momentum).
- [ ] **T1.3**: `CHECK_RIGHT_INTERVAL` 15 → 8 (catches more side corridors — bigger maze = more corridors to miss).
- [ ] **T1.4**: SIFT tuning (`nfeatures=500`, `contrastThreshold=0.06`, mask floor/ceiling, RootSIFT eps fix).
- [ ] **T1.5**: `n_clusters` 128 → 64 (player.py default is already 64; verify).
- [ ] **T1.6**: Graph (`top_k=100`, `MIN_SHORTCUT_GAP=30`, sim floor 0.4).
- [ ] **T1.7**: Multi-view CHECKIN over all 4 target views + variable action hold.
- [ ] **T1.8**: Oscillation detection (2-node bounce in `node_history`).
- [ ] **T2.1**: Time-budget manager (dynamic explore/navigate split via `time_left`).
- [ ] **T2.2**: Re-exploration mode for low-confidence goals.

## New work specific to the final (51×51 maze)

- [ ] **Memory budget**: 51×51 ≈ 2.7× the cells of 31×31. Re-check VLAD memory cost (16,384 dims × N_subsampled_frames). Subsampling at 5 may be too dense — try 8–10 if RAM gets tight.
- [ ] **Exploration coverage**: Wall-following on a bigger maze with the same 5,000-step budget will cover proportionally less of the map. Either (a) bump `EXPLORE_STEPS` (watch out for engine time-left), or (b) use the provided exploration sequence as the primary database and only run live exploration as a backup.
- [ ] **Codebook size**: With more diverse views, may need `n_clusters` 64 → 96/128. Re-tune.
- [ ] **Stuck detection on bigger maze**: `STUCK_THRESHOLD=5` and `nav_stuck_frames=10` were tuned for 31×31; verify they still fire correctly when paths are longer.
- [ ] **Validate against engine 1.2.6 changes**: Run `test_pipeline.py` once on the new engine to make sure no API broke.
- [ ] **Confirm grading rubric** when released (~2026-04-22). Update this file then.

## Decisions to make

1. **Which player file to ship?** `player.py` (1524 lines, full pipeline) is the tested one from the midterm. `player1.py` / `player2.py` are alt experiments. Default = `player.py` unless one of the alts beat it on dry-run scoring.
2. **Live exploration or rely on provided dataset?** Final and dry-run both provide one exploration sequence. The midterm two-stage approach (explore-during-nav-phase) is a fallback; given the bigger maze, the provided dataset is probably more reliable than a fresh wall-follow.
3. **Whether to attempt the leaderboard-leader's strategy** (18 s nav time). That's so far below TA baseline that they're probably skipping the visual-similarity confirmation entirely and trusting Dijkstra. Risky if the maze flips on us.

## Reference links

- Game engine: https://github.com/ai4ce/vis_nav_game_public
- Player starter: https://github.com/ai4ce/vis_nav_player
- Dry-run platform: https://ai4ce.github.io/vis_nav_player/#/challenges/311f2b2f556f
- Midterm repo (full history): https://github.com/tzuiyang/rob_cv_challenge
