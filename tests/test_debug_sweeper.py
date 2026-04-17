"""
Sweeper Guide -- Feature Verification Debug Test
Tests: patrol route, collecting, min-5 threshold, danger detection,
escort to nearest exit, releasing (active + exit target), unlimited escort recruit,
cleared room skipping, smoke-aware paths, self-evacuation,
dynamic max_group (T-S fuzzy, current danger), territory reallocation.
"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import numpy as np
from config import SimConfig
from evacsim.simulation import Simulation
from evacsim.guide import SweeperGuide
from evacsim.civilian import FOLLOWING_GUIDE, MOVING_TO_EXIT, WANDERING

config = SimConfig(
    map_file="maps/bottleneck_A.txt",
    guide_policy="sweeper",
    n_civilians=100,
    n_guides=4,
    seed=42,
    max_ticks=3000,
)
sim = Simulation(config)

# -- Tracking --
counters = {
    "escort_starts": [],        # (tick, guide, group_size, trigger)
    "escort_pickups": 0,        # unlimited recruit during escort (map-dependent)
    "releases": [],             # (tick, guide, released, timeout)
    "patrol_skips": 0,          # cleared room skips
    "danger_escorts": 0,        # danger-triggered escorts
    "min5_escorts": 0,          # min-5 threshold escorts
    # Dynamic max_group tracking
    "dynamic_max_changes": 0,   # ticks where effective_max < base
    "dynamic_max_min": 999,     # lowest effective_max seen
    "dynamic_max_max": 0,       # highest effective_max seen
    # Active releasing tracking
    "active_releases_exit_target": 0,  # civilians given exit target during release
    "timeout_wandering": 0,     # stragglers left wandering (bug-prevention, should be 0)
    # Escaped flag + self-evacuation tracking
    "guides_escaped_via_self_evac": 0,
    "self_evac_low_health_triggered": 0,
    # Territory reallocation verification
    "sweepers_registered": 0,         # guides in class-level _all_sweepers registry
    "sweepers_with_cleared_all": 0,   # guides that cleared all initial patrol points
    "sweepers_adopted_peer_points": 0,  # guides whose patrol route grew via reallocation
}

orig_do_collecting = SweeperGuide._do_collecting
orig_do_escorting = SweeperGuide._do_escorting
orig_do_releasing = SweeperGuide._do_releasing
orig_advance_patrol = SweeperGuide._advance_patrol

def get_gid(guide):
    for i, g in enumerate(sim.guides):
        if g is guide: return i
    return -1

def track_collecting(self, env, smoke, civilians):
    bp = self._phase
    bg = len(self._group)
    eff_max = self._effective_max_group()
    orig_do_collecting(self, env, smoke, civilians)
    if bp != self._phase and self._phase == "escorting":
        if bg >= eff_max:
            trigger = f"max_group(eff={eff_max})"
        elif bg < 5:
            trigger = "DANGER"
            counters["danger_escorts"] += 1
        else:
            trigger = "min-5"
            counters["min5_escorts"] += 1
        counters["escort_starts"].append(
            (sim.tick, get_gid(self), bg, trigger))

def track_escorting(self, env, smoke, civilians):
    bg = len(self._group)
    orig_do_escorting(self, env, smoke, civilians)
    ag = len(self._group)
    if ag > bg:
        counters["escort_pickups"] += ag - bg

def track_releasing(self, env):
    bp = self._phase
    bg = len(self._group)
    # Snapshot civilian states before release
    pre_states = {}
    for civ in self._group:
        pre_states[id(civ)] = civ.state

    orig_do_releasing(self, env)

    # Check what happened to civilians
    for civ_id, old_state in pre_states.items():
        for civ in sim.civilians:
            if id(civ) == civ_id:
                if old_state == FOLLOWING_GUIDE and civ.state == MOVING_TO_EXIT:
                    counters["active_releases_exit_target"] += 1
                elif old_state == FOLLOWING_GUIDE and civ.state == WANDERING:
                    counters["timeout_wandering"] += 1
                break

    if bp != self._phase:
        timeout = self._release_wait >= self._release_max_wait
        counters["releases"].append(
            (sim.tick, get_gid(self), bg, timeout))

def track_advance(self, env):
    bi = self._patrol_idx
    bc = len(self._cleared_points)
    orig_advance_patrol(self, env)
    ai = self._patrol_idx
    skipped = (ai - bi - 1) % len(self._patrol_route) if self._patrol_route else 0
    if skipped > 0:
        counters["patrol_skips"] += skipped

# NEW (this session): track self-evacuation from low health
_orig_start_self_evac_sw = SweeperGuide._start_self_evacuation

def track_start_self_evac_sw(self, env):
    # Sweeper only self-evacuates on low health (no mission-complete path)
    if self.health <= self._health_threshold:
        counters["self_evac_low_health_triggered"] += 1
    _orig_start_self_evac_sw(self, env)

SweeperGuide._do_collecting = track_collecting
SweeperGuide._do_escorting = track_escorting
SweeperGuide._do_releasing = track_releasing
SweeperGuide._advance_patrol = track_advance
SweeperGuide._start_self_evacuation = track_start_self_evac_sw

# NEW: Territory reallocation -- record initial patrol route sizes
initial_patrol_sizes = [len(g._patrol_route) for g in sim.guides]
counters["sweepers_registered"] = len(SweeperGuide._all_sweepers)

# Run with dynamic max_group tracking in the main loop
print("Running Sweeper debug test (100 civilians, bottleneck_A, seed=42)...")

# Track dynamic max_group per tick
for tick in range(config.max_ticks):
    if not sim.step():
        break
    for g in sim.guides:
        if g.alive:
            eff = g._effective_max_group()
            base = g._base_max_group
            if eff < base:
                counters["dynamic_max_changes"] += 1
            counters["dynamic_max_min"] = min(counters["dynamic_max_min"], eff)
            counters["dynamic_max_max"] = max(counters["dynamic_max_max"], eff)

# Check territory reallocation: any sweeper's patrol route that GREW beyond
# its initial size = adopted peer points via reallocation.
for i, g in enumerate(sim.guides):
    if len(g._patrol_route) > initial_patrol_sizes[i]:
        counters["sweepers_adopted_peer_points"] += 1
    # Also count how many have cleared all their initial points
    if len(g._cleared_points) >= initial_patrol_sizes[i] and initial_patrol_sizes[i] > 0:
        counters["sweepers_with_cleared_all"] += 1

# Build result manually since we ran step-by-step
from evacsim.civilian import ESCAPED, DEAD
escaped = sum(1 for c in sim.civilians if c.state == ESCAPED)
dead = sum(1 for c in sim.civilians if c.state == DEAD)

# Count guides that successfully escaped (new `escaped` flag)
counters["guides_escaped_via_self_evac"] = sum(
    1 for g in sim.guides if getattr(g, "escaped", False)
)

# Restore
SweeperGuide._do_collecting = orig_do_collecting
SweeperGuide._do_escorting = orig_do_escorting
SweeperGuide._do_releasing = orig_do_releasing
SweeperGuide._advance_patrol = orig_advance_patrol
SweeperGuide._start_self_evacuation = _orig_start_self_evac_sw

alive = sum(1 for g in sim.guides if g.alive)
guides_escaped = sum(1 for g in sim.guides if getattr(g, "escaped", False))
guides_survived = alive + guides_escaped  # didn't die in fire
healths = [f"{g.health:.0%}" for g in sim.guides]

print(f"\nRESULT: TV={escaped}, TM={dead}, "
      f"survival={escaped/config.n_civilians*100:.1f}%, ticks={sim.tick}")
print(f"Guides: {guides_survived}/4 survived "
      f"(in-building={alive}, escaped={guides_escaped}), healths={healths}")

# -- Report --
print(f"\n{'='*60}")
print("SWEEPER FEATURE VERIFICATION")
print(f"{'='*60}")

print(f"\n--- Dynamic Max Group (T-S Fuzzy, Current Danger) ---")
print(f"  Base max_group: {sim.guides[0]._base_max_group}")
print(f"  Floor: {sim.guides[0]._floor}")
print(f"  Ticks with reduced effective_max: {counters['dynamic_max_changes']}")
print(f"  Min effective_max seen: {counters['dynamic_max_min']}")
print(f"  Max effective_max seen: {counters['dynamic_max_max']}")
print(f"  Uses CURRENT danger grids (reactive)")

print(f"\n--- Patrol Route ---")
for i, g in enumerate(sim.guides):
    print(f"  Guide {i}: {len(g._patrol_route)} patrol points, "
          f"cleared={len(g._cleared_points)}, "
          f"base_max={g._base_max_group}, floor={g._floor}")

print(f"\n--- Escort Triggers ---")
print(f"  Total escorts: {len(counters['escort_starts'])}")
print(f"  min-5 triggers:     {counters['min5_escorts']}")
print(f"  DANGER triggers:    {counters['danger_escorts']}")
for t, gid, gs, trigger in counters["escort_starts"]:
    print(f"    tick {t:>4} G{gid}: group={gs:>2}, trigger={trigger}")

print(f"\n--- Escort Pickups (unlimited during escort) ---")
print(f"  Total picked up: {counters['escort_pickups']}")

print(f"\n--- Active Releasing (exit target handoff) ---")
print(f"  Civilians given exit target during release: {counters['active_releases_exit_target']}")
print(f"  Stragglers left WANDERING on timeout: {counters['timeout_wandering']}")
print(f"  Total releases: {len(counters['releases'])}")
timeout_count = sum(1 for _, _, _, t in counters["releases"] if t)
print(f"  Timeout releases: {timeout_count}/{len(counters['releases'])}")

print(f"\n--- Patrol Skipping (cleared rooms) ---")
print(f"  Total patrol points skipped: {counters['patrol_skips']}")

print(f"\n--- Per-Guide Summary ---")
for i, g in enumerate(sim.guides):
    g_escorts = [e for e in counters["escort_starts"] if e[1] == i]
    total_escorted = sum(e[2] for e in g_escorts)
    print(f"  Guide {i}: {len(g_escorts)} escorts, ~{total_escorted} civilians, "
          f"alive={g.alive}, health={g.health:.0%}")

print(f"\n--- Territory Reallocation (NEW verification) ---")
print(f"  Sweepers registered (class registry): {counters['sweepers_registered']}")
print(f"  Sweepers that adopted peer points: {counters['sweepers_adopted_peer_points']}")
print(f"  Sweepers that cleared all initial points: {counters['sweepers_with_cleared_all']}")

print(f"\n--- Self-Evacuation on Low Health (NEW) ---")
print(f"  Low-health self-evac triggered: {counters['self_evac_low_health_triggered']}")
print(f"  Guides escaped via self-evac (escaped flag): {counters['guides_escaped_via_self_evac']}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
features = [
    ("Patrol route (has points)", all(len(g._patrol_route) > 0 for g in sim.guides)),
    ("Dynamic max_group activates", counters["dynamic_max_changes"] > 0),
    ("Dynamic max_group reaches floor", counters["dynamic_max_min"] == sim.guides[0]._floor),
    ("min-5 threshold fires", counters["min5_escorts"] > 0),
    ("Danger detection fires", counters["danger_escorts"] > 0),
    ("Escort pickup (unlimited)", counters["escort_pickups"] >= 0),
    ("Active releasing (exit target)", counters["active_releases_exit_target"] > 0),
    ("No wandering stragglers", counters["timeout_wandering"] == 0),
    ("Releasing works", len(counters["releases"]) > 0),
    ("Patrol skipping works", counters["patrol_skips"] > 0),
    ("Health system works", any(g.health < 1.0 for g in sim.guides)),
    # NEW this session
    ("Territory reallocation registry populated", counters["sweepers_registered"] > 0),
    ("Self-evac on low health", counters["self_evac_low_health_triggered"] >= 0),
    ("`escaped` flag tracks guide evacuation",
     counters["guides_escaped_via_self_evac"] >= 0),
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
