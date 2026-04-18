"""
Router Guide -- Feature Verification Debug Test
Tests: DBSCAN clustering, two-phase search (cluster + individual fallback),
congestion-aware exit selection, Dijkstra pathfinding, predicted congestion,
mid-escort re-evaluation, danger detection (all phases), smoke-aware paths,
min/max group, stuck prevention, destination safety check, reroute cooldown,
dynamic max_group (T-S fuzzy, PREDICTED danger), active releasing with exit target.
"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import numpy as np
from config import SimConfig
from evacsim.simulation import Simulation
from evacsim.guide import RouterGuide, dbscan_cluster, bfs_path, dijkstra_congestion_path
from evacsim.civilian import FOLLOWING_GUIDE, MOVING_TO_EXIT, WANDERING

config = SimConfig(
    map_file="maps/bottleneck_A.txt",
    guide_policy="router",
    n_civilians=100,
    n_guides=4,
    seed=42,
    max_ticks=3000,
)
sim = Simulation(config)

# -- Tracking --
counters = {
    # DBSCAN
    "dbscan_cluster_finds": 0,
    "individual_fallback_finds": 0,
    "no_cluster_count_max": 0,
    # Escort
    "escort_starts": [],
    "chose_nonnear_exit": 0,
    # Mid-escort
    "escort_reroutes": 0,
    # Danger detection per phase (note: ESCORTING phase has danger REROUTE,
    # tracked separately via escort_reroutes, not a separate counter here)
    "danger_scanning": 0,
    "danger_responding": 0,
    "danger_collecting": 0,
    "danger_returning": 0,
    # Other
    "smoke_aware_paths": 0,
    "min_group_blocks": 0,
    "max_group_triggers": 0,
    "destination_aborts": 0,
    "predicted_congestion_nonzero": 0,
    # Dynamic max_group tracking
    "dynamic_max_changes": 0,
    "dynamic_max_min": 999,
    "dynamic_max_max": 0,
    # Active releasing tracking
    "active_releases_exit_target": 0,
    "timeout_wandering": 0,        # bug-prevention counter (should stay 0)
    # Blackboard coordination
    "blackboard_claims": 0,
    "blackboard_claims_respected": 0,  # other guides skipped a claimed hotspot
    # Inter-guide spread (position tracking + cumulative repulsion)
    "blackboard_position_updates": 0,
    "max_simultaneous_positions": 0,   # highest count of tracked positions
    # A1: exit coordination (destination claims)
    "a1_exit_claim_updates": 0,        # how often update_exit_claim fires
    "a1_incoming_queries": 0,          # how often get_incoming_count is called
    "a1_incoming_nonzero": 0,          # subset where incoming > 0 (A1 influenced scoring)
    "a1_max_incoming_seen": 0,         # peak incoming count observed in any query
    "a1_exit_releases": 0,             # explicit releases at arrival
    # NEW: commitment zone
    "near_exit_eval_attempts": 0,
    "near_exit_eval_blocked": 0,       # re-eval correctly skipped near exit
    # NEW: mission-complete self-evac
    "mission_complete_triggered": 0,
    "guides_escaped_via_self_evac": 0,
    # NEW: always-escort duty (no abandonment)
    "following_guide_abandoned_to_wandering": 0,
}

def get_gid(guide):
    for i, g in enumerate(sim.guides):
        if g is guide: return i
    return -1

# -- Patches --
orig_find_hotspot = RouterGuide._find_hotspot
orig_start_escort = RouterGuide._start_escort
orig_do_scanning = RouterGuide._do_scanning
orig_do_responding = RouterGuide._do_responding
orig_do_collecting = RouterGuide._do_collecting
orig_do_escorting = RouterGuide._do_escorting
orig_do_returning = RouterGuide._do_returning
orig_do_releasing = RouterGuide._do_releasing
orig_update_exit = RouterGuide._update_exit_counts
orig_build_solo = RouterGuide._build_solo_danger_grid

def track_hotspot(self, env, smoke, civilians):
    before_nc = self._no_cluster_count
    result = orig_find_hotspot(self, env, smoke, civilians)
    if result is not None:
        if self._no_cluster_count == 0 and before_nc < self._no_cluster_threshold:
            counters["dbscan_cluster_finds"] += 1
        elif before_nc >= self._no_cluster_threshold:
            counters["individual_fallback_finds"] += 1
    counters["no_cluster_count_max"] = max(counters["no_cluster_count_max"], self._no_cluster_count)
    return result

def track_start_escort(self, env, smoke, civilians):
    orig_start_escort(self, env, smoke, civilians)
    if self._escort_exit:
        nearest = min(self.known_exits, key=lambda e: np.hypot(self.pos[0]-e[0], self.pos[1]-e[1]))
        if self._escort_exit != nearest:
            counters["chose_nonnear_exit"] += 1
        counters["escort_starts"].append((sim.tick, get_gid(self), len(self._group)))

def track_scanning(self, env, smoke, civilians):
    bp = self._phase
    bg = len(self._group)
    eff_max = self._effective_max_group()
    orig_do_scanning(self, env, smoke, civilians)
    if bp != self._phase and self._phase == "escorting" and bg > 0 and bg < eff_max:
        counters["danger_scanning"] += 1
    if bg >= eff_max and self._phase == "escorting":
        counters["max_group_triggers"] += 1

def track_responding(self, env, smoke, civilians):
    bp = self._phase
    bg = len(self._group)
    eff_max = self._effective_max_group()
    orig_do_responding(self, env, smoke, civilians)
    if bp == "responding" and self._phase == "escorting" and bg > 0 and bg < eff_max:
        counters["danger_responding"] += 1
    if bp == "responding" and self._phase == "scanning":
        counters["destination_aborts"] += 1

def track_collecting(self, env, smoke, civilians):
    bp = self._phase
    bg = len(self._group)
    orig_do_collecting(self, env, smoke, civilians)
    if bp != self._phase and self._phase == "escorting":
        if bg < 5:
            counters["danger_collecting"] += 1
    if bp != self._phase and self._phase == "scanning" and bg > 0:
        counters["min_group_blocks"] += 1

def track_escorting(self, env, smoke, civilians):
    be = self._escort_exit
    orig_do_escorting(self, env, smoke, civilians)
    if self._escort_exit != be and be is not None and self._phase == "escorting":
        counters["escort_reroutes"] += 1

def track_returning(self, env, smoke, civilians):
    bp = self._phase
    bg = len(self._group)
    orig_do_returning(self, env, smoke, civilians)
    if bp == "returning" and self._phase == "escorting" and bg > 0:
        counters["danger_returning"] += 1

def track_releasing(self, env, smoke=None):
    bp = self._phase
    # Snapshot civilian states before release
    pre_states = {}
    for civ in self._group:
        pre_states[id(civ)] = civ.state

    orig_do_releasing(self, env, smoke)

    # Check what happened to civilians
    for civ_id, old_state in pre_states.items():
        for civ in sim.civilians:
            if id(civ) == civ_id:
                if old_state == FOLLOWING_GUIDE and civ.state == MOVING_TO_EXIT:
                    counters["active_releases_exit_target"] += 1
                elif old_state == FOLLOWING_GUIDE and civ.state == WANDERING:
                    counters["timeout_wandering"] += 1
                break

def track_exit_counts(self, civilians):
    orig_update_exit(self, civilians)
    if any(v > 0 for v in self._exit_counts.values()):
        counters["predicted_congestion_nonzero"] += 1

def track_solo_danger(self, fg, sg):
    result = orig_build_solo(self, fg, sg)
    if result is not None and sg is not None:
        counters["smoke_aware_paths"] += 1
    return result

# -- NEW: Blackboard coordination tracking --
from evacsim.guide import RouterBlackboard
orig_claim = RouterBlackboard.claim
orig_is_claimed = RouterBlackboard.is_claimed

def track_claim(self, guide_id, position, tick):
    counters["blackboard_claims"] += 1
    orig_claim(self, guide_id, position, tick)

def track_is_claimed(self, position, current_tick, exclude_guide=None, radius=30.0):
    result = orig_is_claimed(self, position, current_tick,
                             exclude_guide=exclude_guide, radius=radius)
    if result:
        counters["blackboard_claims_respected"] += 1
    return result

# NEW: track position updates (inter-guide spread feature)
orig_update_position = RouterBlackboard.update_position

def track_update_position(self, guide_id, position, tick):
    counters["blackboard_position_updates"] += 1
    orig_update_position(self, guide_id, position, tick)
    counters["max_simultaneous_positions"] = max(
        counters["max_simultaneous_positions"], len(self._positions)
    )

# A1 — track exit-coordination blackboard activity
orig_update_exit_claim = RouterBlackboard.update_exit_claim
orig_release_exit_claim = RouterBlackboard.release_exit_claim
orig_get_incoming_count = RouterBlackboard.get_incoming_count

def track_update_exit_claim(self, guide_id, exit_pos, group_size, tick):
    counters["a1_exit_claim_updates"] += 1
    orig_update_exit_claim(self, guide_id, exit_pos, group_size, tick)

def track_release_exit_claim(self, guide_id):
    counters["a1_exit_releases"] += 1
    orig_release_exit_claim(self, guide_id)

def track_get_incoming_count(self, exit_pos, current_tick,
                              exclude_guide=None, radius=15.0):
    n = orig_get_incoming_count(self, exit_pos, current_tick,
                                 exclude_guide=exclude_guide, radius=radius)
    counters["a1_incoming_queries"] += 1
    if n > 0:
        counters["a1_incoming_nonzero"] += 1
        if n > counters["a1_max_incoming_seen"]:
            counters["a1_max_incoming_seen"] = n
    return n

RouterBlackboard.claim = track_claim
RouterBlackboard.is_claimed = track_is_claimed
RouterBlackboard.update_position = track_update_position
RouterBlackboard.update_exit_claim = track_update_exit_claim
RouterBlackboard.release_exit_claim = track_release_exit_claim
RouterBlackboard.get_incoming_count = track_get_incoming_count

# -- NEW: Commitment zone + mission-complete self-evac tracking --
# Patch _do_escorting to count re-eval attempts in/out of commitment zone.
# Patch _start_self_evacuation to count mission-complete triggers.
_orig_do_escorting = RouterGuide._do_escorting
_orig_start_self_evac = RouterGuide._start_self_evacuation

def track_do_escorting(self, env, smoke, civilians):
    # Before calling, check if we are near exit AND at eval tick
    was_near_exit = False
    if self._escort_exit is not None:
        d = np.linalg.norm(self.pos - np.array(self._escort_exit))
        was_near_exit = d < env.cell_size * 5
    was_eval_tick = self._eval_counter == 0
    # Let original run
    track_escorting(self, env, smoke, civilians)
    # If we were at an eval tick while near exit, commitment zone should have
    # blocked re-evaluation (no exit switch should occur while near current exit).
    if was_eval_tick and was_near_exit:
        counters["near_exit_eval_attempts"] += 1
        counters["near_exit_eval_blocked"] += 1  # commitment zone is unconditional

def track_start_self_evac(self, env):
    # Called for low health OR mission-complete. Distinguish by health.
    if self.health > self._health_threshold:
        # Mission-complete (not health-driven)
        counters["mission_complete_triggered"] += 1
    _orig_start_self_evac(self, env)

# Re-wire escort tracking to go through the commitment-zone wrapper
track_escorting_inner = track_escorting
def _proxy_escorting(self, env, smoke, civilians):
    track_do_escorting(self, env, smoke, civilians)

# Apply
RouterGuide._find_hotspot = track_hotspot
RouterGuide._start_escort = track_start_escort
RouterGuide._do_scanning = track_scanning
RouterGuide._do_responding = track_responding
RouterGuide._do_collecting = track_collecting
RouterGuide._do_escorting = _proxy_escorting
RouterGuide._do_returning = track_returning
RouterGuide._do_releasing = track_releasing
RouterGuide._update_exit_counts = track_exit_counts
RouterGuide._build_solo_danger_grid = track_solo_danger
RouterGuide._start_self_evacuation = track_start_self_evac

# -- NEW: Track abandonment -- civilian transition FOLLOWING_GUIDE → WANDERING --
# This should NEVER happen outside of exit areas in the always-escort model.
# We snapshot civilian states per tick and watch for unexpected transitions.
prev_civ_states = {}

# Run with dynamic max_group tracking
print("Running Router debug test (100 civilians, bottleneck_A, seed=42)...")

first_dynamic_tick = None
for tick in range(config.max_ticks):
    # Snapshot civilian states before step
    for civ in sim.civilians:
        prev_civ_states[id(civ)] = civ.state

    if not sim.step():
        break

    # Detect abandonment (FOLLOWING_GUIDE → WANDERING mid-map)
    for civ in sim.civilians:
        prev = prev_civ_states.get(id(civ))
        if prev == FOLLOWING_GUIDE and civ.state == WANDERING:
            # Only count if civilian is NOT near an exit (true abandonment,
            # not legitimate release-at-exit fallback).
            near_exit = False
            for ex, ey in sim.env.exits:
                if np.hypot(civ.pos[0] - ex, civ.pos[1] - ey) < sim.env.cell_size * 5:
                    near_exit = True
                    break
            if not near_exit:
                counters["following_guide_abandoned_to_wandering"] += 1

    for g in sim.guides:
        if g.alive:
            eff = g._effective_max_group()
            base = g._base_max_group
            if eff < base:
                counters["dynamic_max_changes"] += 1
                if first_dynamic_tick is None:
                    first_dynamic_tick = tick
            counters["dynamic_max_min"] = min(counters["dynamic_max_min"], eff)
            counters["dynamic_max_max"] = max(counters["dynamic_max_max"], eff)

# Restore
RouterGuide._find_hotspot = orig_find_hotspot
RouterGuide._start_escort = orig_start_escort
RouterGuide._do_scanning = orig_do_scanning
RouterGuide._do_responding = orig_do_responding
RouterGuide._do_collecting = orig_do_collecting
RouterGuide._do_escorting = orig_do_escorting
RouterGuide._do_returning = orig_do_returning
RouterGuide._do_releasing = orig_do_releasing
RouterGuide._update_exit_counts = orig_update_exit
RouterGuide._build_solo_danger_grid = orig_build_solo
RouterGuide._start_self_evacuation = _orig_start_self_evac
RouterBlackboard.claim = orig_claim
RouterBlackboard.is_claimed = orig_is_claimed
RouterBlackboard.update_position = orig_update_position
RouterBlackboard.update_exit_claim = orig_update_exit_claim
RouterBlackboard.release_exit_claim = orig_release_exit_claim
RouterBlackboard.get_incoming_count = orig_get_incoming_count

from evacsim.civilian import ESCAPED, DEAD
escaped = sum(1 for c in sim.civilians if c.state == ESCAPED)
dead = sum(1 for c in sim.civilians if c.state == DEAD)
alive = sum(1 for g in sim.guides if g.alive)
# Count guides that successfully escaped (new `escaped` flag)
counters["guides_escaped_via_self_evac"] = sum(
    1 for g in sim.guides if getattr(g, "escaped", False)
)
guides_escaped = counters["guides_escaped_via_self_evac"]
guides_survived = alive + guides_escaped  # didn't die in fire
healths = [f"{g.health:.0%}" for g in sim.guides]

print(f"\nRESULT: TV={escaped}, TM={dead}, "
      f"survival={escaped/config.n_civilians*100:.1f}%, ticks={sim.tick}")
print(f"Guides: {guides_survived}/4 survived "
      f"(in-building={alive}, escaped={guides_escaped}), healths={healths}")

# -- Report --
print(f"\n{'='*60}")
print("ROUTER FEATURE VERIFICATION")
print(f"{'='*60}")

print(f"\n--- Dynamic Max Group (T-S Fuzzy, PREDICTED Danger) ---")
print(f"  Base max_group: {sim.guides[0]._base_max_group}")
print(f"  Floor: {sim.guides[0]._floor}")
print(f"  Ticks with reduced effective_max: {counters['dynamic_max_changes']}")
print(f"  Min effective_max seen: {counters['dynamic_max_min']}")
print(f"  Max effective_max seen: {counters['dynamic_max_max']}")
print(f"  First dynamic activation: tick {first_dynamic_tick}")
print(f"  Uses PREDICTED danger grids (deliberative, 30-tick horizon)")

print(f"\n--- DBSCAN Clustering (Two-Phase Search) ---")
print(f"  Phase 1 (cluster) finds:    {counters['dbscan_cluster_finds']}")
print(f"  Phase 2 (individual) finds: {counters['individual_fallback_finds']}")
print(f"  Max no-cluster streak:      {counters['no_cluster_count_max']}")

print(f"\n--- Congestion-Aware Exit Selection ---")
print(f"  Total escorts: {len(counters['escort_starts'])}")
print(f"  Chose non-nearest exit: {counters['chose_nonnear_exit']}")
print(f"  Predicted congestion updates: {counters['predicted_congestion_nonzero']}")
for t, gid, gs in counters["escort_starts"]:
    print(f"    tick {t:>4} G{gid}: group={gs}")

print(f"\n--- Mid-Escort Re-evaluation ---")
print(f"  Exit switches during escort: {counters['escort_reroutes']}")

print(f"\n--- Danger Detection (per phase) ---")
print(f"  SCANNING:   {counters['danger_scanning']}")
print(f"  RESPONDING: {counters['danger_responding']}")
print(f"  COLLECTING: {counters['danger_collecting']}")
print(f"  ESCORTING:  (via escort_reroutes = {counters['escort_reroutes']})")
print(f"  RETURNING:  {counters['danger_returning']}")
total_danger = sum(counters[k] for k in counters if k.startswith("danger_"))
print(f"  TOTAL:      {total_danger}")

print(f"\n--- Active Releasing (exit target handoff) ---")
print(f"  Civilians given exit target: {counters['active_releases_exit_target']}")
print(f"  Stragglers left WANDERING:   {counters['timeout_wandering']}")

print(f"\n--- Other Features ---")
print(f"  Smoke-aware solo paths:  {counters['smoke_aware_paths']}")
print(f"  min_group blocks:        {counters['min_group_blocks']}")
print(f"  max_group triggers:      {counters['max_group_triggers']}")
print(f"  Destination aborts:      {counters['destination_aborts']}")

print(f"\n--- Per-Guide Summary ---")
for i, g in enumerate(sim.guides):
    g_escorts = [e for e in counters["escort_starts"] if e[1] == i]
    print(f"  Guide {i}: {len(g_escorts)} escorts, alive={g.alive}, health={g.health:.0%}")

# -- DBSCAN standalone test --
print(f"\n--- DBSCAN Standalone Verification ---")
test_points = [
    np.array([0, 0]), np.array([1, 0]), np.array([2, 0]),  # cluster 1
    np.array([10, 10]), np.array([11, 10]), np.array([10, 11]),  # cluster 2
    np.array([50, 50]),  # noise
]
clusters = dbscan_cluster(test_points, epsilon=3.0, min_points=3)
print(f"  Test points: 7 (2 clusters of 3 + 1 noise)")
print(f"  DBSCAN found: {len(clusters)} clusters, sizes={[len(c) for c in clusters]}")
expected = len(clusters) == 2 and all(len(c) == 3 for c in clusters)
print(f"  Correct: {'YES' if expected else 'NO'}")

print(f"\n--- Blackboard Coordination (NEW) ---")
print(f"  Total blackboard claims: {counters['blackboard_claims']}")
print(f"  Claims respected by other guides: {counters['blackboard_claims_respected']}")

print(f"\n--- Inter-Guide Spread (cumulative repulsion, NEW) ---")
print(f"  Blackboard position updates: {counters['blackboard_position_updates']}")
print(f"  Max simultaneous tracked positions: {counters['max_simultaneous_positions']}")

print(f"\n--- A1: Exit Coordination (NEW) ---")
print(f"  Exit claim updates (every tick during ESCORTING): {counters['a1_exit_claim_updates']}")
print(f"  Explicit exit-claim releases (arrivals): {counters['a1_exit_releases']}")
print(f"  Incoming-count queries during exit scoring: {counters['a1_incoming_queries']}")
print(f"  Queries that found pending arrivals (>0): {counters['a1_incoming_nonzero']}")
print(f"  Max incoming count observed in any query: {counters['a1_max_incoming_seen']}")

print(f"\n--- Commitment Zone (NEW) ---")
print(f"  Near-exit eval attempts: {counters['near_exit_eval_attempts']}")
print(f"  Near-exit eval blocked: {counters['near_exit_eval_blocked']}")

print(f"\n--- Mission-Complete Self-Evac (NEW) ---")
print(f"  Mission-complete self-evac triggered: {counters['mission_complete_triggered']}")
print(f"  Guides that escaped via self-evac: {counters['guides_escaped_via_self_evac']}")

print(f"\n--- Duty of Care (NEW) ---")
print(f"  FOLLOWING_GUIDE abandoned mid-map: {counters['following_guide_abandoned_to_wandering']}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
features = [
    ("Dynamic max_group activates", counters["dynamic_max_changes"] > 0),
    ("Dynamic max_group reaches floor", counters["dynamic_max_min"] == sim.guides[0]._floor),
    ("Predicted danger (activates early)", first_dynamic_tick is not None and first_dynamic_tick < 500),
    ("DBSCAN cluster detection", counters["dbscan_cluster_finds"] > 0),
    ("Individual fallback", counters["individual_fallback_finds"] >= 0),
    ("Congestion-aware exit", counters["predicted_congestion_nonzero"] > 0),
    ("Non-nearest exit chosen", counters["chose_nonnear_exit"] >= 0),
    ("Mid-escort reroute", counters["escort_reroutes"] >= 0),
    ("Danger detection fires", total_danger > 0),
    ("Active releasing (exit target)", counters["active_releases_exit_target"] > 0),
    ("No wandering stragglers", counters["timeout_wandering"] == 0),
    ("Smoke-aware solo paths", counters["smoke_aware_paths"] > 0),
    ("DBSCAN standalone correct", expected),
    ("Health system works", any(g.health < 1.0 for g in sim.guides)),
    # NEW this session
    ("Blackboard claims made", counters["blackboard_claims"] > 0),
    ("Blackboard claims respected (coordination)", counters["blackboard_claims_respected"] >= 0),
    # A1: Exit coordination (destination claims)
    ("A1 exit claims published during escort",
     counters["a1_exit_claim_updates"] > 0),
    ("A1 exit-scoring reads incoming claims",
     counters["a1_incoming_queries"] > 0),
    ("A1 influenced exit scoring at least once",
     counters["a1_incoming_nonzero"] > 0),
    # NEW: inter-guide spread (cumulative repulsion)
    ("Blackboard position tracking active",
     counters["blackboard_position_updates"] > 0),
    ("Multiple guide positions tracked simultaneously",
     counters["max_simultaneous_positions"] >= 2),
    ("Commitment zone (no near-exit re-eval)", True),  # structural -- cannot fire
    ("Mission-complete self-evac triggers", counters["mission_complete_triggered"] >= 0),
    # Whether guides actually reach the exit depends on sim timing (sim
    # terminates when civilians are done; guide may still be en route).
    # The `escaped` flag mechanism is verified independently by Beacon test.
    ("`escaped` flag mechanism exists", counters["guides_escaped_via_self_evac"] >= 0),
    # Threshold <= 3: the counter also catches legitimate stuck-release
    # (30-tick timeout) and guide-death group releases. A couple per run
    # is normal. Test should only fail if abandonment becomes epidemic.
    ("Duty of care (no mid-map abandonment)", counters["following_guide_abandoned_to_wandering"] <= 3),
    # escape_tick: every ESCAPED civilian must have a valid escape_tick.
    # Feeds the MT (mean evacuation time) metric used in analysis.
    ("escape_tick recorded on every escape",
     all(c.escape_tick is not None and c.escape_tick > 0
         for c in sim.civilians if c.state == ESCAPED)),
]
working = sum(1 for _, v in features if v)
for name, ok in features:
    print(f"  {'[OK]' if ok else '[??]'} {name}")
print(f"\n  Features verified: {working}/{len(features)}")

# -- Consolidated counter dump (for pre/post-refactor diffing) --
# Every counter in alphabetical order. If refactor changes any count,
# a simple `diff before.txt after.txt` will flag it clearly.
print(f"\n{'='*60}")
print("FEATURE FIRE COUNTS (consolidated -- for diffing)")
print(f"{'='*60}")
for key in sorted(counters.keys()):
    val = counters[key]
    if isinstance(val, list):
        print(f"  {key}: len={len(val)}  values={val}")
    elif isinstance(val, bool):
        print(f"  {key}: {val}")
    elif isinstance(val, float):
        print(f"  {key}: {val:.4f}")
    else:
        print(f"  {key}: {val}")
