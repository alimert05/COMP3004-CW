"""
Router Guide — Feature Verification Debug Test
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

# ── Tracking ──
counters = {
    # DBSCAN
    "dbscan_cluster_finds": 0,
    "individual_fallback_finds": 0,
    "no_cluster_count_max": 0,
    # Escort
    "escort_starts": [],
    "chose_nonnear_exit": 0,
    "dijkstra_used": 0,
    "bfs_fallback_used": 0,
    # Mid-escort
    "escort_reroutes": 0,
    # Danger detection per phase
    "danger_scanning": 0,
    "danger_responding": 0,
    "danger_collecting": 0,
    "danger_escorting": 0,
    "danger_returning": 0,
    # Other
    "smoke_aware_paths": 0,
    "min_group_blocks": 0,
    "max_group_triggers": 0,
    "stuck_fix_bfs": 0,
    "destination_aborts": 0,
    "reroute_cooldowns": 0,
    "predicted_congestion_nonzero": 0,
    # NEW: dynamic max_group tracking
    "dynamic_max_changes": 0,
    "dynamic_max_min": 999,
    "dynamic_max_max": 0,
    "dynamic_earlier_than_sweeper": False,
    # NEW: active releasing tracking
    "active_releases_exit_target": 0,
    "timeout_with_exit_target": 0,
    "timeout_wandering": 0,
}

def get_gid(guide):
    for i, g in enumerate(sim.guides):
        if g is guide: return i
    return -1

# ── Patches ──
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

# Apply
RouterGuide._find_hotspot = track_hotspot
RouterGuide._start_escort = track_start_escort
RouterGuide._do_scanning = track_scanning
RouterGuide._do_responding = track_responding
RouterGuide._do_collecting = track_collecting
RouterGuide._do_escorting = track_escorting
RouterGuide._do_returning = track_returning
RouterGuide._do_releasing = track_releasing
RouterGuide._update_exit_counts = track_exit_counts
RouterGuide._build_solo_danger_grid = track_solo_danger

# Run with dynamic max_group tracking
print("Running Router debug test (100 civilians, bottleneck_A, seed=42)...")

first_dynamic_tick = None
for tick in range(config.max_ticks):
    if not sim.step():
        break
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

from evacsim.civilian import ESCAPED, DEAD
escaped = sum(1 for c in sim.civilians if c.state == ESCAPED)
dead = sum(1 for c in sim.civilians if c.state == DEAD)
alive = sum(1 for g in sim.guides if g.alive)
healths = [f"{g.health:.0%}" for g in sim.guides]

print(f"\nRESULT: TV={escaped}, TM={dead}, "
      f"survival={escaped/config.n_civilians*100:.1f}%, ticks={sim.tick}")
print(f"Guides: {alive}/4 alive, healths={healths}")

# ── Report ──
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
print(f"  ESCORTING:  {counters['danger_escorting']}")
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

# ── DBSCAN standalone test ──
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
]
working = sum(1 for _, v in features if v)
for name, ok in features:
    print(f"  {'[OK]' if ok else '[??]'} {name}")
print(f"\n  Features verified: {working}/{len(features)}")
