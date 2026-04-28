# TODO — Manual Control Mode with Auto Target Detection

## New Direction

Abandon autonomous navigation. The user will manually walk through the maze using arrow keys. The system's job is:

1. **Show target images** so the user knows what to look for
2. **Continuously monitor** live FPV for visual similarity to targets
3. **Alert the user** when they're near the target (on-screen HUD + console)
4. **Auto-CHECKIN** when confidence is high enough (or let user press Space)

## Architecture

```
User presses arrow keys → game engine moves agent → see(fpv) receives frame
  → Every 5 frames: extract VLAD from FPV
  → Compare against 4 target VLADs (front/left/back/right)
  → Display HUD overlay: similarity score, "NEAR TARGET!" alert
  → If multi-view consensus high: auto-CHECKIN or prompt user
```

## Implementation Plan

### Phase 1: Switch to Manual Keyboard Control

- [ ] **1.1** In `act()`, when `pipeline_ready=True`, use keyboard input instead of `_auto_navigate()`
  - Reuse the existing keymap (arrow keys → FORWARD/LEFT/RIGHT/BACKWARD)
  - Space → CHECKIN
  - Keep pygame event handling as-is
  - File: `player.py` `act()` method (~line 437)

- [ ] **1.2** Show target images at navigation start
  - Call `show_target_images()` when pipeline finishes building
  - Display in a persistent OpenCV window so user can reference while navigating
  - File: `player.py` `pre_navigation()` (~line 395)

### Phase 2: Continuous Target Matching

- [ ] **2.1** Add a target match check every 5 frames in `see()` method
  - Extract VLAD from live FPV: `fpv_vlad = self.extractor.extract(self.fpv)`
  - Compare against 4 target VLADs: `target_sims = [fpv_vlad @ tv for tv in _target_vlads]`
  - Track `max_sim` and `view_consensus` (count of views > 0.10)
  - Cost: ~50-100ms per extraction, at every 5th frame = ~10Hz check rate
  - File: `player.py` `see()` method (~line 506)

- [ ] **2.2** Multi-view consensus detection
  - Instead of just `max(sims) > threshold`:
    - Count views above 0.08: `consensus = sum(1 for s in sims if s > 0.08)`
    - If `consensus >= 2` AND `max_sim > 0.10`: flag "NEAR TARGET"
    - If `consensus >= 3` AND `max_sim > 0.12`: flag "AT TARGET"
  - These thresholds are based on the observed similarity range (avg=0.07, max individual=0.16)

- [ ] **2.3** Track best-ever similarity
  - Keep `self._best_target_sim = 0.0` across the entire run
  - Update when new max is found
  - Display in HUD so user can see if they're getting warmer/colder

### Phase 3: On-Screen HUD

- [ ] **3.1** Overlay text on FPV window using cv2.putText before pygame blit
  - Top bar: `"Match: 0.12 | Best: 0.16 | Views: 2/4"`
  - When near target: green banner `"NEAR TARGET! Press SPACE"`
  - When at target: red flashing banner `"AT TARGET! AUTO-CHECKIN IN 3s"`
  - File: `player.py` `see()` method

- [ ] **3.2** Color-code the FPV border
  - Green tint when `max_sim > 0.10`
  - Bright green when `consensus >= 2`
  - Red pulse when auto-CHECKIN imminent

### Phase 4: Auto-CHECKIN Logic

- [ ] **4.1** Auto-CHECKIN trigger in `act()`
  - When `consensus >= 3` AND `max_sim > 0.12` for 3 consecutive checks (~1.5 seconds):
    - Print `"[AUTO-CHECKIN] Target detected!"`
    - Return `Action.CHECKIN`
  - OR: when `max_sim > 0.18` (single strong view match):
    - Immediate CHECKIN
  - The user can also press Space at any time

- [ ] **4.2** Safety: don't auto-CHECKIN too early
  - Only enable auto-CHECKIN after 100 steps of manual movement (give user time to orient)
  - Track a confirmation counter: must exceed threshold for N consecutive checks

### Phase 5: Optional Enhancements

- [ ] **5.1** Show best-matching database image alongside FPV (like baseline's navigation panel)
  - Activated by pressing 'Q' key
  - Shows: current node, hop count, path to goal, best match thumbnail

- [ ] **5.2** Console logging for diagnostics
  - Every 50 frames: `[NAV] step=N max_sim=0.12 consensus=2/4 best_ever=0.16`
  - On new best: `[NAV] New best sim=0.18 (front view)`

## Key Constants

```python
TARGET_CHECK_INTERVAL = 5     # check target similarity every N frames
TARGET_SIM_ALERT = 0.10       # show "NEAR TARGET" when max_sim > this
TARGET_CONSENSUS_ALERT = 2    # need N views above 0.08 for alert
TARGET_SIM_CHECKIN = 0.12     # auto-CHECKIN when max_sim > this
TARGET_CONSENSUS_CHECKIN = 3  # need N views above 0.08 for auto-CHECKIN
AUTO_CHECKIN_CONFIRM = 3      # need N consecutive checks above threshold
```

## Files to Modify

| File | Changes |
|------|---------|
| `source/player.py` | Main changes: act() for manual control, see() for HUD + target matching |

## How to Run

```bash
cd vis_nav_player
conda activate game
python source/player.py --data-dir exploration_data --offline-navigation
```

Arrow keys to move. Look at target images window. Walk toward matching scenery. System alerts when near target and auto-CHECKINs.
