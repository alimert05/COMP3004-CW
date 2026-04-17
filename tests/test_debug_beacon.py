"""
Beacon Guide -- Feature Verification Debug Test
Tests: positioning, sound broadcast, probabilistic hearing, wake-up,
exit telling, self-evacuation, smoke effect on broadcast.
"""
import sys, os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
os.chdir(_project_root)

import numpy as np
from config import SimConfig
from evacsim.simulation import Simulation
from evacsim.guide import BeaconGuide

config = SimConfig(
    map_file="maps/bottleneck_A.txt",
    guide_policy="beacon",
    n_civilians=100,
    n_guides=4,
    seed=42,
    max_ticks=3000,
)
sim = Simulation(config)

# -- Tracking --
counters = {
    "idle_woken": 0,
    "exit_told": 0,
    "in_position_tick": [None] * 4,
    "self_evacuations": 0,
}

orig_update = BeaconGuide.update

def track_update(self, env, smoke, fire, civilians):
    gid = None
    for i, g in enumerate(sim.guides):
        if g is self:
            gid = i
            break

    was_in_pos = self._in_position
    orig_update(self, env, smoke, fire, civilians)

    # Track when guide reaches position
    if not was_in_pos and self._in_position and gid is not None:
        counters["in_position_tick"][gid] = sim.tick

    # Track self-evacuation
    if self._self_evacuating:
        counters["self_evacuations"] = sum(1 for g in sim.guides if g._self_evacuating)

BeaconGuide.update = track_update

# Track broadcast effects by patching the broadcast loop
from evacsim.civilian import IDLE, REACTING, MOVING_TO_EXIT, ESCAPED, DEAD, FOLLOWING_GUIDE

orig_beacon_update = BeaconGuide.update
real_update = orig_update  # the original non-patched

def track_broadcast(self, env, smoke, fire, civilians):
    # Snapshot civilian states before update
    pre_states = {}
    pre_exits = {}
    for civ in civilians:
        pre_states[id(civ)] = civ.state
        pre_exits[id(civ)] = getattr(civ, '_guide_told_exit', None)

    track_update(self, env, smoke, fire, civilians)

    # Check what changed
    for civ in civilians:
        cid = id(civ)
        old_state = pre_states.get(cid)
        old_exit = pre_exits.get(cid)
        new_exit = getattr(civ, '_guide_told_exit', None)

        if old_state == IDLE and civ.state == REACTING:
            counters["idle_woken"] += 1
        if old_state == REACTING and civ.state == REACTING:
            # Check if delay was shortened (can't easily track, count exit told)
            pass
        if new_exit is not None and new_exit != old_exit:
            counters["exit_told"] += 1

BeaconGuide.update = track_broadcast

# Run
print("Running Beacon debug test (100 civilians, bottleneck_A, seed=42)...")
result = sim.run()
BeaconGuide.update = real_update

alive = sum(1 for g in sim.guides if g.alive)
escaped = sum(1 for g in sim.guides if getattr(g, "escaped", False))
survived = alive + escaped  # didn't die in fire
healths = [f"{g.health:.0%}" for g in sim.guides]

print(f"\nRESULT: TV={result['TV']}, TM={result['TM']}, "
      f"survival={result['survival_rate']*100:.1f}%, ticks={sim.tick}")
print(f"Guides: {survived}/4 survived (in-building={alive}, escaped={escaped}), "
      f"healths={healths}")

# -- Report --
print(f"\n{'='*60}")
print("BEACON FEATURE VERIFICATION")
print(f"{'='*60}")

print(f"\n--- Positioning ---")
for i, tick in enumerate(counters["in_position_tick"]):
    g = sim.guides[i]
    exit_pos = f"({g.assigned_exit[0]:.0f},{g.assigned_exit[1]:.0f})"
    beacon_pos = f"({g._beacon_pos[0]:.0f},{g._beacon_pos[1]:.0f})"
    dist = np.linalg.norm(g.pos - g.assigned_exit) if g.assigned_exit is not None else 0
    status = "YES" if tick is not None else "NO"
    print(f"  Guide {i}: exit={exit_pos}, beacon={beacon_pos}, "
          f"reached={'tick '+str(tick) if tick else 'NEVER'}, "
          f"dist_from_exit={dist:.0f}")

print(f"\n--- Sound Broadcast ---")
print(f"  IDLE civilians woken up:     {counters['idle_woken']}")
print(f"  Civilians told exit location: {counters['exit_told']}")

print(f"\n--- Health/Self-Evacuation ---")
print(f"  Guides self-evacuating: {counters['self_evacuations']}")
for i, g in enumerate(sim.guides):
    print(f"  Guide {i}: alive={g.alive}, health={g.health:.0%}, "
          f"self_evac={g._self_evacuating}, in_pos={g._in_position}")

# Verify broadcast radius works
print(f"\n--- Guide Evacuation (NEW: escaped flag) ---")
guides_escaped = sum(1 for g in sim.guides if getattr(g, "escaped", False))
guides_killed = sum(1 for g in sim.guides if not g.alive and not getattr(g, "escaped", False))
print(f"  Guides that escaped successfully: {guides_escaped}/{len(sim.guides)}")
print(f"  Guides killed in fire: {guides_killed}/{len(sim.guides)}")
print(f"  Guides still alive (in building): {sum(1 for g in sim.guides if g.alive)}/{len(sim.guides)}")

print(f"\n--- Broadcast Parameters ---")
from evacsim.smoke import SmokeSystem
dummy_smoke = sim.smoke
for i, g in enumerate(sim.guides):
    br = g.get_broadcast_radius(dummy_smoke)
    print(f"  Guide {i}: broadcast_radius={br:.0f} units ({br/10:.1f} cells)")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
features = [
    ("Positioning (reaches beacon pos)", all(t is not None for t in counters["in_position_tick"])),
    ("Sound broadcast (wakes IDLE)", counters["idle_woken"] > 0),
    ("Exit telling (gives exit info)", counters["exit_told"] > 0),
    ("Health system (takes damage)", any(g.health < 1.0 for g in sim.guides)),
    ("Self-evac on low health triggers",
     counters["self_evacuations"] > 0
     or all(g.health > g._health_threshold for g in sim.guides)),
    # escape_tick: every ESCAPED civilian must have a valid escape_tick.
    # This feeds the MT (mean evacuation time) metric used in analysis.
    ("escape_tick recorded on every escape",
     all(c.escape_tick is not None and c.escape_tick > 0
         for c in sim.civilians if c.state == ESCAPED)),
    # NEW this session
    ("`escaped` flag distinguishes evacuated vs killed",
     guides_escaped + guides_killed + sum(1 for g in sim.guides if g.alive)
     == len(sim.guides)),
]
working = sum(1 for _, v in features if v)
for name, ok in features:
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}")
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
        # Show list length plus contents for precise diffing
        print(f"  {key}: len={len(val)}  values={val}")
    elif isinstance(val, bool):
        print(f"  {key}: {val}")
    elif isinstance(val, float):
        print(f"  {key}: {val:.4f}")
    else:
        print(f"  {key}: {val}")
