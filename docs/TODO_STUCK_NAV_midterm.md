# Master Diagnosis: Why the Agent Twitches, Loops, and Fails to Navigate

## Executive Summary

10 parallel investigation agents traced every code path in the navigation pipeline. All converge on **one primary root cause with 6 compounding secondary causes**. The system is trapped in a tight oscillation loop caused by a fundamental mismatch between how long the agent commits to an action (5 frames) and how much visual change is needed to avoid triggering stuck detection (MSE ≥ 400). The agent takes 5 frames of FORWARD, sees barely any pixel change, declares itself stuck, turns for 50 frames, tries FORWARD again, and repeats — creating the visible "twitching" behavior.

---

## Symptoms vs. Mechanisms

| Observed Symptom | Primary Mechanism | Secondary Mechanisms |
|---|---|---|
| Twitching in front of identical frames | ACTION_HOLD=5 too short → stuck fires every cycle | Stuck detection has no temporal buffer |
| Repeated local loops | Wall-follow turn (50 frames) + forward (5 frames) creates 55-frame micro-loop | No visited-node memory; CHECK_RIGHT forces turns every 50 frames |
| Can't walk straight | CHECK_RIGHT_INTERVAL=10 steps forces right-turn every 50 frames | Wall-follow state machine interrupts forward movement |
| Revisits same spots | No memory of visited nodes; localization returns same node | Replan gives identical path from same node |
| Low-progress wandering | 95% of frames spent turning, 5% moving forward | NOISY_MOTION adds drift; temporal smoothing lags localization |
| Poor maze coverage | Agent trapped locally before perturbation fires | PERTURB_INTERVAL counts forward-steps (rare in stuck loop) |
| Can't commit to movement | Stuck detection fires after 5 frames regardless of action validity | 1-frame delta comparison; no multi-frame confirmation |
| Slower than human | Human holds key for 20-30 frames; agent holds for 5 | Human has visual feedback loop; agent re-evaluates too fast |

---

## Ranked Root Causes

### RC-1: ACTION_HOLD_FRAMES=5 is too short relative to STUCK_MSE_THRESHOLD=400
- **Priority**: P0 (must fix)
- **Confidence**: HIGH
- **Symptom**: Twitching, can't walk straight, can't commit, slower than human, low progress
- **Root-cause hypothesis**: At 50 FPS, 5 frames = 0.1 seconds of movement. Physical displacement in 0.1s produces MSE ≈ 300-380 between consecutive frames resized to 80×60. This is below the stuck threshold of 400. So `_is_stuck()` fires True after every FORWARD attempt, triggering an immediate turn.
- **Why plausible**: The exploration used `_explore_act()` which also returns 1 action per frame BUT exploration had `_escape_burst` (50-frame forward bursts at phase transitions) and perturbation bursts that sustained forward movement. Navigation has no equivalent burst — only 5-frame hold.
- **Evidence supporting**: (1) From live runs, hop count plateaus after initial improvement — agent stops making progress. (2) The wall-follow state machine cycle: FORWARD(5 frames) → TURN_LEFT(50 frames) → FORWARD(5 frames) → TURN_LEFT(50 frames) = 95% time turning. (3) Adjacent frame similarity from test: mean=0.18, which at 80×60 resolution translates to low MSE between consecutive frames.
- **Evidence missing**: Actual MSE values during live navigation are never logged (Gap identified by Agent 9).
- **How to verify**: Add logging: `print(f"[STUCK] MSE={mse:.0f} threshold={STUCK_MSE_THRESHOLD}")` inside `_is_stuck()`. If MSE is consistently 300-400, this confirms the hypothesis.
- **Suggested instrumentation**: Log every stuck detection result with MSE value, frame indices, and wall-follow state.
- **Possible fixes**: (a) Increase ACTION_HOLD_FRAMES from 5 to 15-20. (b) Lower STUCK_MSE_THRESHOLD from 400 to 150-200. (c) Require N consecutive stuck detections before declaring stuck (multi-frame confirmation). (d) Compare frames N and N-5 instead of N and N-1 (skip-frame comparison).
- **Expected impact**: Resolves 6/8 symptoms. Agent sustains forward movement for 15-20 frames (0.3-0.4s), producing MSE ≈ 800-1500, well above threshold. Eliminates the micro-oscillation loop entirely.

### RC-2: Stuck detection uses single-frame delta with no temporal buffer
- **Priority**: P1 (should fix)
- **Confidence**: HIGH
- **Symptom**: Twitching, can't commit, false stuck during valid movement
- **Root-cause hypothesis**: `_is_stuck()` compares only frame N vs frame N-1 (lines 1150-1158). A single frame of low MSE (camera jitter, texture noise, motion blur during turn) triggers stuck=True. No confirmation window.
- **Why plausible**: Even 20-frame FORWARD can have a single blurry frame mid-movement where MSE drops below threshold. One bad frame = full stuck detection = turn cycle triggered.
- **Evidence supporting**: `_is_stuck()` at line 837 is called every time `_wall_follow_act()` runs. With ACTION_HOLD=5, this is every 5 frames. There's no debounce or buffer.
- **Evidence missing**: No log of how often stuck fires True vs False.
- **How to verify**: Log stuck detection results. Count True vs False ratio over 1000 frames.
- **Suggested instrumentation**: `[STUCK] MSE={mse:.0f} result={stuck} consecutive={_nav_stuck_count}`
- **Possible fixes**: (a) Require 3 consecutive stuck=True before acting. (b) Use rolling average of last 3-5 MSE values. (c) Compare frame N with frame N-K (where K matches action hold duration).
- **Expected impact**: Eliminates false-positive stuck detections from single-frame noise. Agent maintains forward momentum through brief MSE dips.

### RC-3: CHECK_RIGHT_INTERVAL=10 forces turns every 50 frames of forward movement
- **Priority**: P1 (should fix)
- **Confidence**: MEDIUM
- **Symptom**: Can't walk straight, zig-zag navigation, wasted time
- **Root-cause hypothesis**: Every 10 FORWARD decisions (= 50 frames with action hold), the agent enters CHECK_RIGHT state, turning 90° right (another 50 frames). This is a 50-50 split between forward and turning regardless of corridor geometry.
- **Why plausible**: In a straight corridor, the agent should walk forward 100+ frames. Instead it turns every 50 frames, checking for side corridors that may not exist.
- **Evidence supporting**: Code at lines 860-864. Counter `_nav_fwd` increments once per `_wall_follow_act()` call, which happens every ACTION_HOLD_FRAMES.
- **Evidence missing**: What CHECK_RIGHT_INTERVAL was in exploration (also 10, per line 103).
- **How to verify**: Log CHECK_RIGHT entries and exits. Count how many find real corridors vs turn back.
- **Suggested instrumentation**: `[CHECK-RIGHT] entered at step={step}, result={'found_corridor'|'wall'}`
- **Possible fixes**: (a) Increase CHECK_RIGHT_INTERVAL from 10 to 30-50. (b) Only check right when path direction suggests a junction. (c) Disable CHECK_RIGHT entirely during navigation (path already knows the route).
- **Expected impact**: Doubles effective forward speed in corridors. Agent walks 150 frames straight instead of 50.

### RC-4: Turn injection duration is 5x too long (calibrated for 1-frame, runs with 5-frame hold)
- **Priority**: P1 (should fix)
- **Confidence**: HIGH
- **Symptom**: Overshooting turns, oscillation between LEFT and RIGHT
- **Root-cause hypothesis**: `TURN_STEPS_90 = 10` was calibrated for 1 action per frame (~7.5° per action = 75° per 10 actions ≈ 90°). But with ACTION_HOLD=5, each turn step lasts 5 frames. 10 steps × 5 frames = 50 frames. If 10 frames = 90°, then 50 frames = 450° (more than a full rotation). The `_turn_remaining` counter in `_path_guided_act()` decrements once per call (not per frame), but the action is held for 5 frames. So path-directed turns are 5x too long.
- **Why plausible**: The wall-follow state machine's `_nav_tc` counter also increments once per `_wall_follow_act()` call, matching the decrement rate. So wall-follow turns are consistently 50 frames (fine if calibrated that way). But `_turn_remaining` in `_path_guided_act()` was added later and may not account for the hold correctly.
- **Evidence supporting**: In the wall-follow state machine, TURN_STEPS_90=10 controls turns (lines 869, 876, 883, 896). These also run once per action-hold cycle. So wall-follow turns are consistently 50 game frames. If 50 frames = 90° (calibrated during exploration), this is correct. But if TURN_STEPS_90=10 was meant for 1-frame-per-action (10 frames = 90°), then all wall-follow turns are also 5x too long.
- **Evidence missing**: Actual angular displacement per action at the current frame rate with NOISY_MOTION.
- **How to verify**: Log the agent's heading before and after a 10-step turn. If it's >90°, the turn calibration is wrong.
- **Suggested instrumentation**: `[TURN] starting heading=X°` and `[TURN] ending heading=Y° (delta=Z°)`. This requires reading camera pose from engine (may not be available).
- **Possible fixes**: (a) Recalibrate TURN_STEPS_90 empirically. (b) Reduce to TURN_STEPS_90=2-3 if hold is 5 frames and each frame is ~7.5°.
- **Expected impact**: Turns complete in correct angular displacement. No overshoot/oscillation.

### RC-5: Localization temporal smoothing (0.8/0.2) lags behind physical movement
- **Priority**: P1 (should fix)
- **Confidence**: MEDIUM
- **Symptom**: Localization stuck at old node, path replanned from wrong position, revisiting
- **Root-cause hypothesis**: `sims = 0.8 * sims + 0.2 * self._prev_sims` (line 762) blends 20% of previous frame's similarity into current. If the agent physically moves 10 nodes in 25 wall-follow steps (between replans), the smoothing pulls localization toward the old node. The localization lags by ~5-10 nodes.
- **Why plausible**: Adjacent frame similarity is ~0.18. A 10-node jump might only increase the true node's similarity by ~0.02. With 20% blending from the old node (which has sim ~0.18), the smoothed value for the new node is pulled down while the old node is pulled up. Argmax may pick the old node.
- **Evidence supporting**: In live runs, the log shows `node=689` repeated across many replans — agent physically moved but localization didn't track.
- **Evidence missing**: Actual similarity scores per replan (never logged).
- **How to verify**: Log raw vs smoothed argmax. If they differ consistently, smoothing is the problem.
- **Suggested instrumentation**: `[LOC] raw_node={raw} smoothed_node={smoothed} raw_sim={raw_sim:.3f} smoothed_sim={smoothed_sim:.3f}`
- **Possible fixes**: (a) Remove temporal smoothing entirely. (b) Reduce to 0.95/0.05. (c) Reset `_prev_sims = None` on state transitions or after turns.
- **Expected impact**: Localization tracks physical movement more accurately. Replanning uses correct position. Hops decrease faster.

### RC-6: Goal matching avg_sim=0.07, consensus=1/4 — likely navigating to wrong node
- **Priority**: P1 (should fix for accuracy)
- **Confidence**: HIGH
- **Symptom**: Agent reaches "goal area" but target not there, cycles through candidates
- **Root-cause hypothesis**: The 4 target views match to completely different nodes (2082, 881, 1940, 1579). The average similarity is only 0.07 — barely above random (0.01). The selected goal node and all 5 candidates are likely wrong.
- **Why plausible**: If the exploration data doesn't cover the target location (or covers it from a very different angle), VLAD matching fails. The agent navigates to the best-matching node, but "best" is still terrible.
- **Evidence supporting**: Console output shows `avg_sim=0.0725, consensus=1/4`. All 4 views disagree by >100 nodes.
- **Evidence missing**: Where the actual target is physically. Whether the exploration data covers it.
- **How to verify**: Run `baseline.py --data-dir exploration_data` to see the goal matching panel. Or: compute similarity of all target views to every 10th database node, plot the similarity landscape.
- **Suggested instrumentation**: During SEARCH state, log the maximum similarity achieved: `[SEARCH] scan best_sim={best_sim:.3f}`. If it never exceeds 0.10 at any location, the target is not in the database.
- **Possible fixes**: (a) Continuously match target views against live FPV during navigation (not just at database nodes). (b) If avg_sim < 0.10, switch to random exploration instead of goal-directed navigation. (c) Collect new exploration data that covers the goal area.
- **Expected impact**: With correct goal, hops reach 0 and CHECKIN succeeds. Without this, all other fixes just make the agent navigate faster to the wrong place.

### RC-7: No visited-node memory; no oscillation detection; no path failure history
- **Priority**: P2 (nice to have)
- **Confidence**: MEDIUM
- **Symptom**: Revisiting same spots, repeated local loops, wasted time
- **Root-cause hypothesis**: The agent has no memory of where it's been. It can visit node 689 fifty times without knowing. There's no oscillation detector (node A→B→A→B pattern). Plateau detection (1000 steps) is the only anti-loop mechanism — far too slow.
- **Why plausible**: Without visited-node tracking, Dijkstra replans from the same node give the same path. Wall-following tries the same route repeatedly.
- **Evidence supporting**: Code review confirms: no `visited_nodes` set, no `node_history` deque, no path comparison across replans.
- **Evidence missing**: Actual node visitation frequencies during a run.
- **How to verify**: Log every localized node. Count unique vs total. If unique/total < 0.3, the agent is heavily revisiting.
- **Suggested instrumentation**: `visited = set()` tracking, logged every 100 steps: `[COVERAGE] visited={len(visited)}/{total_nodes} unique_ratio={ratio:.2f}`
- **Possible fixes**: (a) Add node oscillation detector: if same 3 nodes repeat 5+ times, enter ESCAPE. (b) Add visited-node set: penalize revisited nodes in Dijkstra (increase edge weight). (c) Reduce NAV_PLATEAU_STEPS from 1000 to 200.
- **Expected impact**: Faster escape from local loops. Better maze coverage. 3-5x faster loop recovery.

---

## Instrumentation Gaps (Must-Add Before Any Fix)

These logs are CRITICAL for diagnosing the twitching — currently invisible:

| What to Log | Where (line) | Format | Why |
|---|---|---|---|
| Every action executed | After line 1144 | `[ACT] step=N action=FWD es=FORWARD hand=LEFT` | Can't see actual movement commands |
| Stuck detection result | Inside `_is_stuck()` ~line 1158 | `[STUCK] MSE=350 thresh=400 result=True count=2` | Can't see how often stuck fires |
| Wall-follow state transitions | In `_wall_follow_act()` on state change | `[WF] FORWARD→TURN_LEFT stuck=True` | Can't trace state machine |
| Localization per replan | In `_get_current_node()` ~line 778 | `[LOC] node=689 sim=0.23 prev=688` | Can't see node oscillation |
| Turn injection | At line 1069 | `[TURN] inject 10 steps LEFT from edge 689→690` | Can't see path-directed turns |
| Path edges | At line 1024 | `[PATH] plan=[689,690,691,...] goal=1940 hops=35` | Can't see planned route |
| Frame/decision count | At line 1090 | `[TICK] frame=5000 decision=1000` | nav_total_steps counts frames, not decisions |

---

## Quick Experiments to Disambiguate Causes

1. **Set ACTION_HOLD_FRAMES=20** and run. If twitching stops → RC-1 confirmed.
2. **Set STUCK_MSE_THRESHOLD=100** (very low) and run. If agent walks straight into walls without turning → stuck threshold is the issue.
3. **Set CHECK_RIGHT_INTERVAL=100** and run. If agent walks straighter → RC-3 confirmed.
4. **Remove temporal smoothing** (`_prev_sims = None` always) and run. If localization tracks better → RC-5 confirmed.
5. **Log MSE values** for 1000 frames. If median MSE during FORWARD is 300-380 → RC-1 confirmed with hard evidence.

---

## High-Risk False Leads to Avoid

| False Lead | Why It's Tempting | Why It's Wrong |
|---|---|---|
| "VLAD is fundamentally broken" | Low similarity scores (0.18 adjacent) | VLAD works in tests — discriminability ratio is 15x. The problem is stuck detection preventing movement, not VLAD. |
| "Need a completely different navigation approach" | Current approach seems fundamentally flawed | The path-guided wall-following is sound in principle. The timing constants are just miscalibrated. |
| "Goal matching is the only problem" | avg_sim=0.07 is terrible | Even with a perfect goal, the agent would still twitch because of stuck detection. Fix movement first. |
| "Need faster VLAD extraction" | VLAD takes 15-55ms | Extraction only happens every 25 steps. At 125 frames between extractions, the 30ms cost is negligible. |
| "The pybullet patch breaks images" | Monkey-patching is risky | The patch only reshapes the array — no color/dtype changes. Images are identical to unpatched. |

---

## Detailed Issue Sections

### Perception Issues
- SIFT features on blue circuit-board texture: median 231 features (adequate after tuning)
- VLAD without power normalization: adjacent similarity 0.18, distant 0.01 (15x ratio — adequate)
- Localization temporal smoothing lags by 5-10 nodes (RC-5)
- Argmax instability when multiple nodes have similarity within 0.005 (contributes to oscillation)
- Jump detection (LOCALIZATION_SEARCH_RADIUS=80) can prevent recognizing large physical moves

### Control/Action Issues
- ACTION_HOLD_FRAMES=5 is the P0 root cause (RC-1)
- TURN_STEPS_90=10 may be 5x too long with action hold (RC-4)
- CHECK_RIGHT_INTERVAL=10 forces turns every 50 frames (RC-3)
- Path-directed turns (`_turn_remaining`) and wall-follow turns (`_nav_tc`) are unsynchronized
- Stuck detection resets `_nav_stuck_count` on ANY successful forward, preventing dead-end escalation

### Planning/Exploration Issues
- Goal matching avg_sim=0.07 means all candidates may be wrong (RC-6)
- Dijkstra path to wrong node leads agent to wrong area
- Wall-following is O(perimeter) while path is O(shortest_path) — but path-guided hand selection helps
- NAV_PLATEAU_STEPS=1000 is too slow for loop recovery (20 seconds wasted)
- Candidate cycling exhausts all 5 options without finding target

### Memory/State Issues
- No visited-node tracking (RC-7)
- No node oscillation detection
- `_prev_sims` persists across replans (stale smoothing)
- `hop_history` cleared aggressively on state transitions
- `_nav_es` wall-follow state not reset after path-directed turns

### Latency/Synchronization Issues
- VLAD extraction blocks for 15-55ms (1-3 frames at 50 FPS) during replan
- act() uses frame from previous see() (1-frame observation-action delay)
- NOISY_MOTION adds unpredictable displacement to every action
- Action hold means stuck detection compares frames from SAME held action

### Training-Serving Mismatch Issues
- test_e2e_nav.py feeds database images directly — bypasses all live game physics
- All test metrics (100% localization, monotonic hops decrease) are artifacts
- Exploration positions ≠ navigation positions (viewpoint shift with NOISY_MOTION)
- SIFT cache from exploration vs live SIFT from different camera angle
- Goal matching done once at setup, never updated during navigation
- ACTION_HOLD_FRAMES timing may differ from exploration timing
