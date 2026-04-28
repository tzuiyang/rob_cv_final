"""
Unit test for the SEARCH state 360-degree scan logic in _search_act().

Run: conda activate game && cd vis_nav_player && python source/test_search_state.py

Verifies:
  1. Full 360 scan = 48 LEFT steps, similarity checked every 3 steps (16 checks)
  2. CHECKIN fires when similarity exceeds threshold (normal confidence)
  3. Low confidence goal blocks CHECKIN below higher threshold
  4. Low confidence goal allows CHECKIN above higher threshold
  5. Gives up after SEARCH_MAX_SCANS failed scans -> NAVIGATE
  6. best_sim tracks highest similarity across scan
  7. Extract called exactly 16 times per scan (every 3 steps)
  8. Scan granularity: 22.5 degrees between checks
  9. Reposition: 6 FORWARD steps between scans (1 returned + 5 held)
"""

import numpy as np
import os
import sys

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
        import traceback
        traceback.print_exc()


def test_search_state():
    """Verify _search_act scan logic with synthetic similarity values."""
    from unittest.mock import MagicMock
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from player import (
        KeyboardPlayerPyGame, NavState, TURN_STEPS_90,
        SEARCH_SIM_THRESHOLD, SEARCH_SIM_LOW_CONF, SEARCH_MAX_SCANS,
    )
    try:
        from vis_nav_game import Action
    except ImportError:
        from enum import Enum
        class Action(Enum):
            IDLE = 0; FORWARD = 1; BACKWARD = 2
            LEFT = 3; RIGHT = 4; CHECKIN = 5; QUIT = 6

    scan_steps = TURN_STEPS_90 * 4  # 48
    checks_per_scan = scan_steps // 3  # 16
    print(f"  scan_steps={scan_steps}, checks_per_scan={checks_per_scan}")
    assert scan_steps == 48, f"Expected 48 scan steps, got {scan_steps}"
    assert checks_per_scan == 16, f"Expected 16 checks per scan, got {checks_per_scan}"

    # --- Helper: build a minimal mock player with _search_act wired up ---
    def make_player(low_confidence=False):
        """Create a mock player with search state initialized."""
        import types

        player = MagicMock()
        player.nav_state = NavState.SEARCH
        player.search_turn_counter = 0
        player.search_scan_count = 0
        player.search_best_sim = 0.0
        player.search_best_steps = 0
        player.low_confidence_goal = low_confidence
        player.fpv = np.zeros((240, 320, 3), dtype=np.uint8)
        player.current_action = None
        player.action_hold_counter = 0
        player.hop_history = MagicMock()
        player.prev_avg_hops = None

        # Unit vector for controlled dot-product similarity
        unit_v = np.zeros(16384, dtype=np.float32)
        unit_v[0] = 1.0
        player._target_vlads = [unit_v.copy() for _ in range(4)]

        # Default extractor: returns zero vector (sim=0.0)
        player.extractor = MagicMock()
        player.extractor.extract = lambda img: np.zeros(16384, dtype=np.float32)

        # Bind the real _search_act method
        player._search_act = types.MethodType(
            KeyboardPlayerPyGame._search_act, player
        )
        return player

    def set_sim_at_step(player, target_step, sim_value):
        """Configure extractor to return sim_value at the given turn counter step."""
        def controlled_extract(img):
            v = np.zeros(16384, dtype=np.float32)
            if player.search_turn_counter == target_step:
                v[0] = sim_value
            return v
        player.extractor.extract = controlled_extract

    # --- Test 1: Full scan with no match -> scan_count increments ---
    print("  Test 1: Full 360 scan with no match")
    player = make_player()
    actions = []
    for i in range(scan_steps):
        actions.append(player._search_act())

    assert player.search_scan_count == 1, \
        f"Expected scan_count=1, got {player.search_scan_count}"
    assert player.search_turn_counter == 0, \
        f"Expected turn_counter reset to 0, got {player.search_turn_counter}"
    assert actions[-1] == Action.FORWARD, \
        f"Expected FORWARD after scan, got {actions[-1]}"
    assert player.action_hold_counter == 5, \
        f"Expected action_hold_counter=5, got {player.action_hold_counter}"
    left_count = sum(1 for a in actions[:47] if a == Action.LEFT)
    assert left_count == 47, \
        f"Expected 47 LEFT in first 47 steps, got {left_count}"
    print(f"    scan_count={player.search_scan_count}, "
          f"hold={player.action_hold_counter}")

    # --- Test 2: CHECKIN fires on high similarity (normal confidence) ---
    print("  Test 2: CHECKIN on high sim (normal confidence)")
    player = make_player(low_confidence=False)
    checkin_step = 12  # 12 % 3 == 0, similarity checked here
    set_sim_at_step(player, checkin_step, 0.30)  # Above SEARCH_SIM_THRESHOLD=0.25

    checkin_found = False
    for i in range(scan_steps):
        a = player._search_act()
        if a == Action.CHECKIN:
            checkin_found = True
            break
    assert checkin_found, \
        f"Expected CHECKIN with sim=0.30 > {SEARCH_SIM_THRESHOLD}"
    print(f"    CHECKIN fired at turn_counter={player.search_turn_counter}")

    # --- Test 3: Low confidence requires higher threshold ---
    print("  Test 3: Low confidence goal blocks low sim")
    player = make_player(low_confidence=True)
    set_sim_at_step(player, checkin_step, 0.30)  # Below SEARCH_SIM_LOW_CONF=0.35

    checkin_found = False
    for i in range(scan_steps):
        a = player._search_act()
        if a == Action.CHECKIN:
            checkin_found = True
            break
    assert not checkin_found, \
        f"Should NOT checkin: sim=0.30 < {SEARCH_SIM_LOW_CONF}"
    print(f"    Correctly skipped (0.30 < {SEARCH_SIM_LOW_CONF})")

    # --- Test 4: Low confidence DOES checkin above higher threshold ---
    print("  Test 4: Low confidence checks in above higher threshold")
    player = make_player(low_confidence=True)
    set_sim_at_step(player, checkin_step, 0.40)  # Above SEARCH_SIM_LOW_CONF=0.35

    checkin_found = False
    for i in range(scan_steps):
        a = player._search_act()
        if a == Action.CHECKIN:
            checkin_found = True
            break
    assert checkin_found, \
        f"Expected CHECKIN with sim=0.40 > {SEARCH_SIM_LOW_CONF}"
    print(f"    CHECKIN at sim=0.40 > {SEARCH_SIM_LOW_CONF}")

    # --- Test 5: After SEARCH_MAX_SCANS failed scans -> NAVIGATE ---
    print(f"  Test 5: Give up after {SEARCH_MAX_SCANS} failed scans")
    player = make_player()

    for scan in range(SEARCH_MAX_SCANS):
        for step in range(scan_steps):
            player._search_act()
        if scan < SEARCH_MAX_SCANS - 1:
            assert player.action_hold_counter == 5, \
                f"Expected hold=5 after scan {scan+1}"
            player.action_hold_counter = 0  # simulate _auto_navigate consuming hold

    assert player.nav_state == NavState.NAVIGATE, \
        f"Expected NAVIGATE after {SEARCH_MAX_SCANS} scans, got {player.nav_state}"
    assert player.search_scan_count == 0, \
        f"Expected scan_count reset to 0"
    print(f"    Returned to NAVIGATE after {SEARCH_MAX_SCANS} failed scans")

    # --- Test 6: best_sim tracks highest similarity across scan ---
    print("  Test 6: best_sim tracks highest similarity")
    player = make_player()
    sim_at = {3: 0.10, 6: 0.05, 9: 0.20, 12: 0.15, 15: 0.22}

    def varying_extract(img):
        v = np.zeros(16384, dtype=np.float32)
        step = player.search_turn_counter
        if step in sim_at:
            v[0] = sim_at[step]
        return v
    player.extractor.extract = varying_extract

    for i in range(scan_steps):
        player._search_act()

    expected_best = max(sim_at.values())  # 0.22
    assert abs(player.search_best_sim - expected_best) < 1e-6, \
        f"Expected best_sim={expected_best}, got {player.search_best_sim}"
    print(f"    best_sim={player.search_best_sim:.3f} == {expected_best}")

    # --- Test 7: Similarity only checked every 3 steps ---
    print("  Test 7: Similarity checked only at step % 3 == 0")
    player = make_player()
    extract_calls = [0]

    def counting_extract(img):
        extract_calls[0] += 1
        return np.zeros(16384, dtype=np.float32)
    player.extractor.extract = counting_extract

    for i in range(scan_steps):
        player._search_act()

    # counter goes 1..48, checks at 3,6,9,...,48 -> 16 checks
    assert extract_calls[0] == checks_per_scan, \
        f"Expected {checks_per_scan} extract calls, got {extract_calls[0]}"
    print(f"    {extract_calls[0]} extract calls in {scan_steps} steps")

    # --- Test 8: Scan granularity analysis ---
    print("  Test 8: Scan granularity analysis")
    deg_per_step = 360.0 / scan_steps
    deg_between = deg_per_step * 3
    print(f"    {deg_per_step:.1f} deg/step, "
          f"{deg_between:.1f} deg between checks, "
          f"{checks_per_scan} checks per 360 scan")
    assert deg_between == 22.5
    assert checks_per_scan == 16

    # --- Test 9: Reposition distance between scans ---
    print("  Test 9: Reposition between scans")
    player = make_player()
    for i in range(scan_steps):
        player._search_act()

    assert player.search_scan_count == 1
    assert player.action_hold_counter == 5, \
        f"Expected 5 held FORWARD steps"
    total_fwd = 1 + player.action_hold_counter
    print(f"    Reposition: 1+{player.action_hold_counter}="
          f"{total_fwd} FORWARD steps between scans")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.join(script_dir, ".."))
    print(f"Working directory: {os.getcwd()}")

    print(f"\n{'#'*60}")
    print(f"  SEARCH STATE TEST SUITE")
    print(f"{'#'*60}")

    run_test("Search State (360 Scan Logic)", test_search_state)

    # Summary
    total = PASSED + FAILED
    print(f"\n{'#'*60}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed out of {total}")
    if FAILED == 0:
        print(f"  ALL TESTS PASSED")
    print(f"{'#'*60}")
    sys.exit(1 if FAILED > 0 else 0)
